import logging
import subprocess
import shutil
import time
import os
import sys
import platform
import requests
from pathlib import Path
from codehierarchy.config.schema import LLMConfig

# Configure logging if not already configured in main app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelManager:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.port = 1234
        self.base_url = f"http://localhost:{self.port}/v1"
        
        # 1. Resolve LMS Binary Path
        self.lms_path = self._resolve_lms_path()
        if not self.lms_path:
            logging.critical("LM Studio executable not found. Please install LM Studio.")
            sys.exit(1)

        # 2. Idempotent Bootstrap & Environment Setup
        self._bootstrap_lms()

    def _resolve_lms_path(self) -> str | None:
        """
        Locates the LMS executable. Checks system PATH first, then default install locations.
        """
        # Check if already in PATH
        path_in_env = shutil.which("lms")
        if path_in_env:
            return path_in_env

        # Check default install locations based on OS
        home = Path.home()
        system = platform.system()
        
        possible_paths = []
        if system == "Windows":
            possible_paths.append(home / ".lmstudio" / "bin" / "lms.exe")
        else:
            possible_paths.append(home / ".lmstudio" / "bin" / "lms")

        for p in possible_paths:
            if p.exists():
                return str(p)
        
        return None

    def _bootstrap_lms(self):
        """
        Performs idempotent bootstrapping. 
        Runs 'lms bootstrap' and injects the bin path into the current process environment
        to simulate 'restarting the terminal'.
        """
        try:
            # Run bootstrap blindly; it is safe/idempotent.
            logging.info("Ensuring LM Studio CLI is bootstrapped...")
            subprocess.run([self.lms_path, "bootstrap"], 
                         check=True, capture_output=True, text=True)
            
            # CRITICAL: Add to PATH for the current process
            # This programmatically solves "WARNING... Run ... once" without a restart.
            bin_dir = os.path.dirname(self.lms_path)
            if bin_dir not in os.environ["PATH"]:
                os.environ["PATH"] += os.pathsep + bin_dir
                logging.debug(f"Injected {bin_dir} into process PATH")

            # Verify CLI is responsive (FIXED: Use 'version' not '--version')
            try:
                subprocess.run([self.lms_path, "version"], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                # Fallback if 'version' fails, try 'help' just to ensure binary runs
                subprocess.run([self.lms_path, "help"], check=True, capture_output=True)
                
            logging.info("LM Studio CLI initialized and verified.")

        except subprocess.CalledProcessError as e:
            # If bootstrap itself fails, we must exit
            logging.error(f"Failed to bootstrap LM Studio: {e.stderr}")
            sys.exit(1)

    def load_model(self) -> str | None:
        """
        Orchestrates the full lifecycle: 
        Server Check -> Server Start (if needed) -> Model Check -> Model Load (if needed).
        """
        # 3. Ensure Server is Running
        if not self._is_server_healthy():
            logging.info("LM Studio server is not running. Starting server...")
            if not self._start_server():
                logging.error("Failed to start LM Studio server.")
                return None

        # 4. Check if Model is already loaded
        # We define a clean identifier for our specific model to ensure consistency
        target_identifier = self.config.model_name.lower().replace(".gguf", "").split("/")[-1]
        
        loaded_models = self._get_loaded_model_ids()
        
        # Check if our target ID or the raw config name is already present
        if target_identifier in loaded_models or self.config.model_name in loaded_models:
            logging.info(f"Model '{target_identifier}' is already loaded and ready.")
            return target_identifier

        # 5. Load Model via CLI
        logging.info(f"Loading model '{self.config.model_name}' as '{target_identifier}'...")
        try:
            cmd = [
                self.lms_path, "load", self.config.model_name,
                "--identifier", target_identifier,
                "--context-length", str(self.config.context_window or 2048)
            ]
            
            # --- CORRECTION APPLIED HERE ---
            # Only include the --gpu flag if gpu_offload_ratio is explicitly set
            # and non-zero, and ensure it's converted to a valid string format (number).
            gpu_ratio = self.config.gpu_offload_ratio
            if gpu_ratio is not None and gpu_ratio > 0:
                # Assuming gpu_offload_ratio is a float (e.g., 0.8)
                cmd.extend(["--gpu", str(float(gpu_ratio))])
            # If the value is 0 or None, omit the flag to use default/CPU behavior.
            # --- END CORRECTION ---

            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # 6. Final Verification (Wait for API to reflect the load)
            retries = 5
            while retries > 0:
                if target_identifier in self._get_loaded_model_ids():
                    logging.info(f"Successfully verified model '{target_identifier}' via API.")
                    return target_identifier
                time.sleep(1)
                retries -= 1
            
            logging.error("Model load command succeeded, but API does not report model as loaded.")
            return None

        except subprocess.CalledProcessError as e:
            logging.error(f"CLI Error loading model: {e.stderr}")
            return None

    def _is_server_healthy(self) -> bool:
        """Checks if the API endpoint is responsive."""
        try:
            # Short timeout to avoid hanging if server is totally dead
            response = requests.get(f"{self.base_url}/models", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _start_server(self) -> bool:
        """Starts the server in a detached process and waits for health check."""
        try:
            # Start server as a detached process
            # stdout/stderr to DEVNULL prevents it from spamming our console
            subprocess.Popen(
                [self.lms_path, "server", "start", "--port", str(self.port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait loop
            max_retries = 15  # 30 seconds
            for i in range(max_retries):
                if self._is_server_healthy():
                    return True
                time.sleep(2)
                logging.debug(f"Waiting for server... ({i+1}/{max_retries})")
            
            return False
        except Exception as e:
            logging.error(f"Unexpected error starting server: {e}")
            return False

    def _get_loaded_model_ids(self) -> list[str]:
        """Returns a list of 'id' strings for currently loaded models via API."""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m['id'] for m in data.get('data', [])]
        except requests.exceptions.RequestException:
            pass
        return []
        