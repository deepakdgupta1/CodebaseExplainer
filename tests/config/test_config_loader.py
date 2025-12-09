import pytest
from pathlib import Path
from codehierarchy.config.loader import load_config
from codehierarchy.config.schema import Config

def test_load_default_config():
    # Should load the default config.yaml we created
    config = load_config()
    assert isinstance(config, Config)
    assert config.system.max_memory_gb == 26.0

def test_load_nonexistent_config():
    # Should return default values
    config = load_config(Path("nonexistent.yaml"))
    assert isinstance(config, Config)
