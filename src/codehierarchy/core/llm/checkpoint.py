"""
Checkpoint management for LLM summarization.

This module provides functions to save and load summarization progress
as JSON checkpoint files. This enables resumable summarization runs,
allowing the system to continue from where it left off if interrupted.

Functions:
    save_checkpoint: Persist summaries to a JSON file.
    load_checkpoint: Restore summaries from a JSON file.
"""

import json
from pathlib import Path
from typing import Dict
import logging


def save_checkpoint(summaries: Dict[str, str], checkpoint_file: Path) -> None:
    """
    Save current summaries to a JSON checkpoint file.

    Creates the parent directory if it doesn't exist. The checkpoint
    file is overwritten on each save.

    Args:
        summaries: Dictionary mapping node IDs to their summary text.
        checkpoint_file: Path to the checkpoint JSON file.

    Note:
        Errors are logged but not raised to avoid interrupting the
        main summarization workflow.
    """
    try:
        # Create parent dir if needed
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_file, 'w') as f:
            json.dump(summaries, f, indent=2)
        logging.info(
            f"Checkpoint saved to {checkpoint_file} ({
                len(summaries)} summaries)")
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")


def load_checkpoint(checkpoint_file: Path) -> Dict[str, str]:
    """
    Load summaries from a JSON checkpoint file.

    Reads a previously saved checkpoint and returns the summaries
    dictionary. Returns an empty dict if the file doesn't exist or
    if an error occurs.

    Args:
        checkpoint_file: Path to the checkpoint JSON file.

    Returns:
        Dictionary mapping node IDs to their summary text.
        Empty dict if checkpoint doesn't exist or is invalid.
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
