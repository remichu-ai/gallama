# src/gallama/config_manager.py

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List


class ConfigManager:
    def __init__(self):
        self.configs: Dict[str, Any] = {}
        self.load_model_configs()
        self.default_model_list = self.load_default_model_list()

    @property
    def get_library_path(self):
        return Path(__file__).parent.parent


    @property
    def get_data_dir(self) -> Path:
        """Get the absolute path to the data directory."""
        return Path(__file__).parent.parent / 'data'

    @property
    def get_gallama_user_config_folder(self) -> Path:
        """Get the absolute path to the Gallama user config folder."""
        home_dir = Path.home()
        config_dir = home_dir / "gallama"
        return config_dir

    @property
    def get_gallama_user_config_file_path(self) -> Path:
        """Get the absolute path to the Gallama user config folder."""
        home_dir = Path.home()
        config_dir = home_dir / "gallama" / "model_config.yaml"
        return config_dir

    def load_model_configs(self):
        """Load all YAML files (both .yaml and .yml) from the data directory and combine them."""
        # data_dir = self.get_data_dir()
        data_dir = self.get_gallama_user_config_folder
        for yaml_file in data_dir.glob('model_config.yaml'):  # This pattern matches both .yaml and .yml
            with open(yaml_file, 'r') as file:
                yaml_data = yaml.safe_load(file)
                # Merge the loaded data with existing configs
                if yaml_data:
                    self.configs.update(yaml_data)
                else:
                    raise ValueError(f'No model config in YAML files found in {yaml_file}')

    def load_default_model_list(self):
        data_dir = self.get_data_dir
        with open(data_dir / "default_model_list.yaml", 'r') as file:
            yaml_data = yaml.safe_load(file)
            return yaml_data

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        return self.configs.get(model_name, {})

    def get_all_model_names(self) -> List[str]:
        """Get a list of all available model names."""
        return list(self.configs.keys())

    def get_model_parameter(self, model_name: str, parameter: str) -> Any:
        """Get a specific parameter for a model."""
        model_config = self.get_model_config(model_name)
        return model_config.get(parameter)

    def get_models_by_parameter(self, parameter: str, value: Any) -> List[str]:
        """Get all models that have a specific parameter value."""
        return [model for model, config in self.configs.items()
                if config.get(parameter) == value]

    def get_models_by_yaml_file(self, yaml_filename: str) -> List[str]:
        """Get all models defined in a specific YAML file."""
        yaml_path = self.get_data_dir / yaml_filename
        if not yaml_path.exists():
            return []
        with open(yaml_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        return list(yaml_data.keys())
