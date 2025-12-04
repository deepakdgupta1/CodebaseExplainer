import yaml
from pathlib import Path
from typing import Optional
from .schema import Config

def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from a YAML file.
    If no path is provided, tries to load from default locations or returns default config.
    """
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return Config(**config_data)
    
    # Try default location
    default_path = Path("config/config.yaml")
    if default_path.exists():
        with open(default_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return Config(**config_data)
        
    # Return default config if no file found
    return Config()

def load_prompt_template(template_name: str) -> str:
    """
    Load a prompt template from the config/prompts directory.
    """
    prompt_path = Path(f"config/prompts/{template_name}")
    if not prompt_path.exists():
        # Fallback to checking if the name includes extension
        prompt_path = Path(f"config/prompts/{template_name}.txt")
        
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_name}")
        
    with open(prompt_path, 'r') as f:
        return f.read()
