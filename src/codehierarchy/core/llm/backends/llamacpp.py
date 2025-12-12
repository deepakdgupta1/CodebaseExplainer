"""
llama.cpp backend for LLM inference.

This module provides a backend implementation that directly manages
a llama-server process, enabling fully headless LLM inference without
requiring a GUI application.
"""

import atexit
import logging
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any

import requests

from codehierarchy.config.schema import LLMConfig
from .base import BaseLLMBackend


class LlamaCppBackend(BaseLLMBackend):
    """
    Backend using llama.cpp's llama-server for inference.

    Manages the llama-server process lifecycle and provides an
    OpenAI-compatible API for chat completions.

    Attributes:
        config: LLMConfig with llamacpp settings.
        _process: Subprocess handle for llama-server.
        _model_id: Currently loaded model identifier.
    """

    def __init__(self, config: LLMConfig) -> None:
        """
        Initialize the llama.cpp backend.

        Args:
            config: LLMConfig with llamacpp nested config.
        """
        self.config = config
        self._llamacpp_config = config.llamacpp
        self._process: Optional[subprocess.Popen] = None
        self._model_id: str = ""
        self._port = self._llamacpp_config.port

    def setup(self) -> None:
        """
        Start the llama-server process.

        Locates the server binary, starts it with configured options,
        and waits for it to become healthy.

        Raises:
            RuntimeError: If server binary not found or fails to start.
        """
        server_path = self._resolve_server_path()
        if not server_path:
            raise RuntimeError(
                f"llama-server not found. Install llama.cpp or set "
                f"llm.llamacpp.server_path in config."
            )

        model_path = self._resolve_model_path()
        if not model_path:
            raise RuntimeError(
                f"Model file not found. Set llm.llamacpp.model_path in config."
            )

        # Build command
        cmd = [
            server_path,
            "--model", str(model_path),
            "--port", str(self._port),
            "--n-gpu-layers", str(self._llamacpp_config.n_gpu_layers),
            "--ctx-size", str(self.config.context_window),
        ]

        if self.config.flash_attention:
            cmd.append("--flash-attn")

        logging.info(f"Starting llama-server: {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Register cleanup
            atexit.register(self.shutdown)

            # Wait for server to be ready
            if not self._wait_for_healthy(timeout=120):
                self.shutdown()
                raise RuntimeError("llama-server failed to start")

            self._model_id = Path(model_path).stem
            logging.info(f"llama-server started on port {self._port}")

        except Exception as e:
            logging.error(f"Failed to start llama-server: {e}")
            raise RuntimeError(f"Failed to start llama-server: {e}")

    def load_model(self) -> Optional[str]:
        """
        Return the loaded model identifier.

        llama-server loads the model at startup, so this just returns
        the model ID if the server is healthy.

        Returns:
            Model identifier or None if not healthy.
        """
        if self.is_healthy():
            return self._model_id
        return None

    def is_healthy(self) -> bool:
        """
        Check if llama-server is responding.

        Returns:
            True if server responds to health check.
        """
        try:
            resp = requests.get(
                f"http://localhost:{self._port}/health",
                timeout=5
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def shutdown(self) -> None:
        """
        Terminate the llama-server process.
        """
        if self._process:
            logging.info("Shutting down llama-server...")
            try:
                self._process.terminate()
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    @property
    def base_url(self) -> str:
        """OpenAI-compatible API base URL."""
        return f"http://localhost:{self._port}/v1"

    @property
    def model_id(self) -> str:
        """Currently loaded model identifier."""
        return self._model_id

    def get_extra_body(self) -> Dict[str, Any]:
        """
        Return llama.cpp compatible extra parameters.

        Note: llama.cpp doesn't support context_overflow_policy.
        """
        return {
            "top_k": self.config.top_k,
            "repeat_penalty": self.config.repeat_penalty,
            "min_p": self.config.min_p,
        }

    def _resolve_server_path(self) -> Optional[str]:
        """Find llama-server executable."""
        configured = self._llamacpp_config.server_path

        # Check if it's already a valid path
        if Path(configured).is_file():
            return configured

        # Check PATH
        in_path = shutil.which(configured)
        if in_path:
            return in_path

        # Check common locations
        common_paths = [
            Path.home() / ".local" / "bin" / "llama-server",
            Path("/usr/local/bin/llama-server"),
            Path("/usr/bin/llama-server"),
        ]
        for p in common_paths:
            if p.is_file():
                return str(p)

        return None

    def _resolve_model_path(self) -> Optional[Path]:
        """Resolve model file path."""
        model_path = self._llamacpp_config.model_path

        if model_path and model_path.is_file():
            return model_path

        # Try to find by model name in common locations
        model_name = self.config.model_name
        search_dirs = [
            Path.home() / ".cache" / "lm-studio" / "models",
            Path.home() / "models",
            Path("/models"),
        ]

        for search_dir in search_dirs:
            if search_dir.is_dir():
                matches = list(search_dir.rglob(f"*{model_name}*"))
                if matches:
                    return matches[0]

        return None

    def _wait_for_healthy(self, timeout: int = 60) -> bool:
        """Wait for server to become healthy with backoff."""
        start = time.time()
        delay = 1.0

        while time.time() - start < timeout:
            if self.is_healthy():
                return True
            time.sleep(delay)
            delay = min(delay * 1.5, 10)  # Exponential backoff, max 10s

        return False
