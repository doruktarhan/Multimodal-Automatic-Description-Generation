import json
import os
import yaml

class ConfigManager:
    """
    Manages Conffigurations for the evaluation process.
    """
    def __init__(self,config_path):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        """
        Load the configuration from a file.
        """
        with open(self.config_path, 'r') as f:
            if self.config_path.endswith('.json'):
                return json.load(f)
            elif self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                return yaml.safe_load(f)
            
    def get_directory(self, dir_name):
        """
        Get the directory path from the configuration.
        """
        return self.config.get("directories", {}).get(dir_name)
    
    def get_metrics_config(self):
        """
        Get the metrics configuration from the configuration file.
        """
        return self.config.get("metrics", [])

    def get_built_in_metrics_config(self):
        """
        Get the built-in metrics configuration from the configuration file.
        """
        return self.config.get("built_in_metrics", [])

    def resolve_path(self, base_dir_name, *subpaths):
        """Resolve a full path by combining base directory with subpaths."""
        base = self.get_directory(base_dir_name)
        if not base:
            raise ValueError(f"Directory '{base_dir_name}' not found in configuration")
        return os.path.join(base, *subpaths)