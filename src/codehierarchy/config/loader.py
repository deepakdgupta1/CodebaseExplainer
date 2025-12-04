import yaml
from pathlib import Path
from typing import Optional
from importlib import resources
from .schema import Config

def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from a YAML file.
    If no path is provided, tries to load from package resources or returns default config.
    """
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return Config(**config_data)
    
    # Try loading from package resources
    try:
        # For Python 3.9+, use files() API
        if hasattr(resources, 'files'):
            config_file = resources.files('codehierarchy.config').joinpath('config.yaml')
            with config_file.open('r') as f:
                config_data = yaml.safe_load(f)
            return Config(**config_data)
        else:
            # Fallback for older Python versions
            with resources.open_text('codehierarchy.config', 'config.yaml') as f:
                config_data = yaml.safe_load(f)
            return Config(**config_data)
    except Exception:
        pass
        
    # Return default config if no file found
    return Config()

def load_prompt_template(template_name: str) -> str:
    """
    Load a prompt template from the package resources.
    """
    if not template_name.endswith('.txt'):
        template_name = f"{template_name}.txt"
    
    try:
        # For Python 3.9+
        if hasattr(resources, 'files'):
            prompts_dir = resources.files('codehierarchy.config').joinpath('prompts')
            template_file = prompts_dir.joinpath(template_name)
            return template_file.read_text()
        else:
            # Fallback for older Python
            return resources.read_text('codehierarchy.config.prompts', template_name)
    except Exception as e:
        raise FileNotFoundError(f"Prompt template not found: {template_name}") from e
