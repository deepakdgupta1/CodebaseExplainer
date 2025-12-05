"""
Tests for FileScanner module.
"""

import pytest
from pathlib import Path
from codehierarchy.analysis.scanner.file_scanner import FileScanner
from codehierarchy.config.schema import ParsingConfig

@pytest.fixture
def scanner_config():
    return ParsingConfig(
        languages=['python'],
        exclude_patterns=['*.txt', 'venv/'],
        max_file_size_mb=1.0
    )

@pytest.fixture
def scanner(scanner_config):
    return FileScanner(scanner_config)

def test_scan_directory_basic(scanner, tmp_path):
    """Test basic file scanning."""
    # Create dummy files
    (tmp_path / "test.py").touch()
    (tmp_path / "ignore.txt").touch()
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.py").touch()
    
    files = scanner.scan_directory(tmp_path)
    
    # Should find .py files but ignore .txt
    assert len(files) == 2
    filenames = {f.name for f in files}
    assert "test.py" in filenames
    assert "nested.py" in filenames
    assert "ignore.txt" not in filenames

def test_scan_with_gitignore(scanner, tmp_path):
    """Test that .gitignore is respected."""
    (tmp_path / ".gitignore").write_text("ignored.py\nbuild/")
    
    (tmp_path / "included.py").touch()
    (tmp_path / "ignored.py").touch()
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "build_artifact.py").touch()
    
    files = scanner.scan_directory(tmp_path)
    
    filenames = {f.name for f in files}
    assert "included.py" in filenames
    assert "ignored.py" not in filenames
    assert "build_artifact.py" not in filenames

def test_language_filtering(tmp_path):
    """Test filtering by language."""
    config = ParsingConfig(languages=['typescript'])
    scanner = FileScanner(config)
    
    (tmp_path / "test.ts").touch()
    (tmp_path / "test.py").touch()
    
    files = scanner.scan_directory(tmp_path)
    
    assert len(files) == 1
    assert files[0].name == "test.ts"

def test_max_file_size(scanner, tmp_path):
    """Test skipping large files."""
    large_file = tmp_path / "large.py"
    # Create a file slightly larger than 1MB (1024*1024 + 1 bytes)
    with open(large_file, "wb") as f:
        f.seek(1024 * 1024)
        f.write(b"\0")
        
    small_file = tmp_path / "small.py"
    small_file.touch()
    
    files = scanner.scan_directory(tmp_path)
    
    assert len(files) == 1
    assert files[0].name == "small.py"
