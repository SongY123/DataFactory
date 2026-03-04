import os
import yaml
from typing import Any, Dict

class ConfigLoader:
    _config: Dict[str, Any] = {}

    @classmethod
    def load_config(cls, config_path: str):
        """Load a YAML configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cls._config = yaml.safe_load(f)

        return cls._config

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get the config object. Raise if it has not been loaded."""
        if not cls._config:
            raise RuntimeError("Config is not initialized. Call load_config(path) first.")
        return cls._config

    @classmethod
    def get(cls, key_path: str, default: Any = None) -> Any:
        """Get a config value by dotted path, e.g. 'server.port'."""
        config = cls.get_config()
        keys = key_path.split(".")
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

# Global helper function
def get_config(key_path: str, default: Any = None) -> Any:
    return ConfigLoader.get(key_path, default)
