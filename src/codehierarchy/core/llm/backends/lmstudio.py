"""
LM Studio backend for LLM inference.

This module provides a backend that manages LM Studio via its CLI,
with optional xvfb automation for headless environments.
"""

import atexit
import glob
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import requests

from codehierarchy.config.schema import LLMConfig
from .base import BaseLLMBackend


class LMStudioBackend(BaseLLMBackend):
    """
    Backend using LM Studio for inference.

    Manages LM Studio lifecycle including:
    - AppImage discovery and xvfb-wrapped startup
    - CLI bootstrapping
    - Server and model management

    Attributes:
        config: LLMConfig with lmstudio settings.
        lms_path: Path to lms CLI binary.
        _xvfb_process: xvfb-run process handle (if used).
        _model_id: Currently loaded model identifier.
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the LM Studio backend."""
        self.config = config
        self._lmstudio_config = config.lmstudio
        self._port = self._lmstudio_config.port
        self._xvfb_process: Optional[subprocess.Popen] = None
        self._model_id: str = ""
        self.lms_path: Optional[str] = None

    def setup(self) -> None:
        """
        Initialize LM Studio.

        1. Start AppImage with xvfb if configured
        2. Resolve and bootstrap lms CLI
        3. Wait for server to be ready
        """
        # Start AppImage with xvfb if needed
        if self._lmstudio_config.use_xvfb:
            self._start_with_xvfb()

        # Resolve lms CLI path
        self.lms_path = self._resolve_lms_path()
        if not self.lms_path:
            raise RuntimeError(
                "LM Studio CLI (lms) not found. Install LM Studio first."
            )

        # Bootstrap CLI
        self._bootstrap_lms()
        atexit.register(self.shutdown)

    def load_model(self) -> Optional[str]:
        """
        Load the configured model.

        Returns:
            Model identifier if successful, None otherwise.
        """
        if not self._is_server_healthy():
            logging.info("LM Studio server not running, starting...")
            if not self._start_server():
                logging.error("Failed to start LM Studio server.")
                return None

        # Check if already loaded
        target_id = self._get_target_identifier()
        loaded = self._get_loaded_model_ids()

        if target_id in loaded or self.config.model_name in loaded:
            logging.info(f"Model '{target_id}' already loaded.")
            self._model_id = target_id
            return target_id

        # Load via CLI
        logging.info(f"Loading model '{self.config.model_name}'...")
        try:
            cmd = [
                self.lms_path, "load", self.config.model_name,
                "--identifier", target_id,
                "--context-length", str(self.config.context_window or 2048)
            ]

            gpu_ratio = self.config.gpu_offload_ratio
            if gpu_ratio is not None and gpu_ratio > 0:
                cmd.extend(["--gpu", str(float(gpu_ratio))])

            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Verify load
            for _ in range(5):
                if target_id in self._get_loaded_model_ids():
                    self._model_id = target_id
                    logging.info(f"Model '{target_id}' loaded successfully.")
                    return target_id
                time.sleep(1)

            logging.error("Model load succeeded but not visible via API.")
            return None

        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to load model: {e.stderr}")
            return None

    def is_healthy(self) -> bool:
        """Check if LM Studio server is responding."""
        return self._is_server_healthy()

    def shutdown(self) -> None:
        """Shutdown LM Studio processes."""
        if self._xvfb_process:
            logging.info("Stopping xvfb-wrapped LM Studio...")
            try:
                self._xvfb_process.terminate()
                self._xvfb_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._xvfb_process.kill()
            self._xvfb_process = None

    @property
    def base_url(self) -> str:
        """OpenAI-compatible API base URL."""
        return f"http://localhost:{self._port}/v1"

    @property
    def model_id(self) -> str:
        """Currently loaded model identifier."""
        return self._model_id

    def get_extra_body(self) -> Dict[str, Any]:
        """Return LM Studio specific extra parameters."""
        return {
            "top_k": self.config.top_k,
            "repeat_penalty": self.config.repeat_penalty,
            "min_p": self.config.min_p,
            "context_overflow_policy": self.config.context_overflow_policy,
        }

    # --- Private Methods ---

    def _start_with_xvfb(self) -> None:
        """Start LM Studio AppImage with xvfb-run."""
        appimage_path = self._find_appimage()
        if not appimage_path:
            logging.warning(
                "AppImage not found, assuming LM Studio is already running."
            )
            return

        # Check if xvfb-run is available
        if not shutil.which("xvfb-run"):
            raise RuntimeError(
                "xvfb-run not found. Install with: sudo apt install xvfb"
            )

        logging.info(f"Starting LM Studio with xvfb: {appimage_path}")
        self._xvfb_process = subprocess.Popen(
            ["xvfb-run", "--auto-servernum", str(appimage_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for app to initialize
        time.sleep(5)

    def _find_appimage(self) -> Optional[Path]:
        """Find LM Studio AppImage."""
        # Check configured path
        if self._lmstudio_config.appimage_path:
            path = Path(self._lmstudio_config.appimage_path)
            if path.is_file():
                return path

        # Search common locations
        search_patterns = [
            str(Path.home() / "*.AppImage"),
            str(Path.home() / "Downloads" / "LM*.AppImage"),
            str(Path.home() / "Applications" / "LM*.AppImage"),
            "/opt/LM*.AppImage",
        ]

        for pattern in search_patterns:
            matches = glob.glob(pattern)
            for match in matches:
                if "LMStudio" in match or "lm-studio" in match.lower():
                    return Path(match)

        return None

    def _resolve_lms_path(self) -> Optional[str]:
        """Locate lms CLI binary."""
        in_path = shutil.which("lms")
        if in_path:
            return in_path

        home = Path.home()
        system = platform.system()

        if system == "Windows":
            candidates = [home / ".lmstudio" / "bin" / "lms.exe"]
        else:
            candidates = [home / ".lmstudio" / "bin" / "lms"]

        for p in candidates:
            if p.exists():
                return str(p)

        return None

    def _bootstrap_lms(self) -> None:
        """Run lms bootstrap and verify CLI."""
        try:
            logging.info("Bootstrapping LM Studio CLI...")
            subprocess.run(
                [self.lms_path, "bootstrap"],
                check=True, capture_output=True, text=True
            )

            # Add to PATH
            bin_dir = os.path.dirname(self.lms_path)
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + bin_dir

            # Verify
            subprocess.run(
                [self.lms_path, "version"],
                check=True, capture_output=True
            )
            logging.info("LM Studio CLI ready.")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to bootstrap lms: {e.stderr}")

    def _is_server_healthy(self) -> bool:
        """Check if API is responsive."""
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=2)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def _start_server(self) -> bool:
        """Start LM Studio server via CLI."""
        try:
            subprocess.Popen(
                [self.lms_path, "server", "start", "--port", str(self._port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            for i in range(15):
                if self._is_server_healthy():
                    return True
                time.sleep(2)
                logging.debug(f"Waiting for server... ({i+1}/15)")

            return False
        except Exception as e:
            logging.error(f"Error starting server: {e}")
            return False

    def _get_loaded_model_ids(self) -> list:
        """Get list of loaded model IDs via API."""
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=5)
            if resp.status_code == 200:
                return [m['id'] for m in resp.json().get('data', [])]
        except requests.RequestException:
            pass
        return []

    def _get_target_identifier(self) -> str:
        """Generate clean model identifier."""
        return (
            self.config.model_name
            .lower()
            .replace(".gguf", "")
            .split("/")[-1]
        )
