import logging
import subprocess
import shutil
from typing import Optional
from codehierarchy.config.schema import LLMConfig

class ModelManager:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.lms_path = shutil.which("lms")
        
    def load_model(self) -> bool:
        """
        Attempt to load the model using 'lms' CLI if available.
        Returns True if successful or if manual loading is assumed.
        """
        if not self.lms_path:
            logging.warning("LM Studio CLI 'lms' not found. Please ensure the model is loaded manually in LM Studio.")
            logging.info(f"Expected Model: {self.config.model_name}")
            logging.info(f"Settings: Context={self.config.context_window}, GPU Offload={self.config.gpu_offload_ratio}")
            return False
            
        try:
            logging.info(f"Loading model {self.config.model_name} via lms...")
            # Construct lms load command
            # lms load <model> --gpu-offload <ratio> --context-length <len> ...
            # Note: The exact flags for lms might vary, this is a best-effort implementation based on common CLI patterns.
            cmd = [
                self.lms_path, "load", self.config.model_name,
                "--gpu-offload", str(self.config.gpu_offload_ratio),
                "--context-length", str(self.config.context_window),
            ]
            
            if self.config.use_mmap:
                cmd.append("--mmap")
                
            # Add other flags if supported by lms CLI
            
            subprocess.run(cmd, check=True, capture_output=True)
            logging.info("Model loaded successfully.")
            return True
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to load model via lms: {e.stderr.decode()}")
            return False
        except Exception as e:
            logging.error(f"Error managing model: {e}")
            return False
