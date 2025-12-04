from pathlib import Path
from typing import Optional

def detect_language(file_path: Path) -> Optional[str]:
    """
    Detect the programming language of a file based on its extension.
    Returns 'python', 'typescript', or None if not supported.
    """
    suffix = file_path.suffix.lower()
    if suffix == '.py':
        return 'python'
    elif suffix in ['.ts', '.tsx', '.js', '.jsx']:
        # We group JS/TS/JSX/TSX under 'typescript' for simplicity in the high-level architecture,
        # though the parser might need to distinguish them internally.
        return 'typescript'
    return None
