from pathlib import Path
from typing import List, Optional, Set
import pathspec
from ..config.schema import ParsingConfig

class FileScanner:
    def __init__(self, config: ParsingConfig):
        self.config = config
        # Map languages to extensions
        self.extensions: Set[str] = set()
        if 'python' in config.languages:
            self.extensions.add('.py')
        if 'typescript' in config.languages:
            self.extensions.update({'.ts', '.tsx', '.js', '.jsx'})
            
        # Compile exclude patterns
        self.exclude_spec = pathspec.PathSpec.from_lines('gitwildmatch', config.exclude_patterns)

    def scan_directory(self, root: Path) -> List[Path]:
        """
        Scan directory for supported files, respecting .gitignore and config exclusions.
        """
        # Load .gitignore if present
        gitignore_spec = self._load_gitignore(root)
        
        files = []
        # Use rglob for recursive search
        # Note: rglob('*') can be slow for massive trees, but we filter as we go
        for path in root.rglob('*'):
            if not path.is_file():
                continue
                
            # Check extension
            if path.suffix.lower() not in self.extensions:
                continue
                
            # Check file size
            try:
                if path.stat().st_size > self.config.max_file_size_mb * 1024 * 1024:
                    continue
            except OSError:
                continue
                
            # Get relative path for matching
            try:
                rel_path = path.relative_to(root)
            except ValueError:
                continue
                
            # Check .gitignore
            if gitignore_spec and gitignore_spec.match_file(str(rel_path)):
                continue
                
            # Check config exclusions
            if self.exclude_spec.match_file(str(rel_path)):
                continue
                
            files.append(path)
            
        return files

    def _load_gitignore(self, root: Path) -> Optional[pathspec.PathSpec]:
        gitignore_path = root / '.gitignore'
        if gitignore_path.exists():
            try:
                with open(gitignore_path, 'r') as f:
                    return pathspec.PathSpec.from_lines('gitwildmatch', f)
            except Exception:
                pass
        return None
