import json
from pathlib import Path
from typing import Dict
import logging

def save_checkpoint(summaries: Dict[str, str], checkpoint_file: Path) -> None:
    """
    Save current summaries to a JSON checkpoint file.
    """
    try:
        # Create parent dir if needed
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_file, 'w') as f:
            json.dump(summaries, f, indent=2)
        logging.info(f"Checkpoint saved to {checkpoint_file} ({len(summaries)} summaries)")
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")

def load_checkpoint(checkpoint_file: Path) -> Dict[str, str]:
    """
    Load summaries from a JSON checkpoint file.
    """
    if not checkpoint_file.exists():
        return {}
        
    try:
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {k: str(v) for k, v in data.items()}
            return {}
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        return {}
