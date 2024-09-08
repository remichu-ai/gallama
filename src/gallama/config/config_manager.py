# src/gallama/config_manager.py

import yaml
import re
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
from gallama.logger import logger
from operator import itemgetter
import os


class ConfigManager:
    def __init__(self):
        self.configs: Dict[str, Any] = {}
        try:
            self.load_model_configs()
        except Exception as e:
            logger.info("~/gallama/model_config.yaml is empty file")
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
        gallama_home = os.environ.get('GALLAMA_HOME_PATH')
        if gallama_home:
            config_dir = Path(gallama_home)
        else:
            home_dir = Path.home()
            config_dir = home_dir / "gallama"
        return config_dir

    @property
    def get_gallama_user_config_file_path(self) -> Path:
        """Get the absolute path to the Gallama user config file."""
        return self.get_gallama_user_config_folder / "model_config.yaml"

    def load_model_configs(self):
        """Load all YAML files (both .yaml and .yml) from the data directory and combine them."""
        config_file = self.get_gallama_user_config_file_path
        if config_file.exists():
            with open(config_file, 'r') as file:
                yaml_data = yaml.safe_load(file)
                if yaml_data:
                    self.configs.update(yaml_data)
                else:
                    raise ValueError(f'No model config in YAML file found in {config_file}')
        else:
            logger.info(f"Config file not found at {config_file}")

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

    @staticmethod
    def extract_bpw(details, model_name):
        model_id = details.get('model_id', '')
        bpw = details.get('bpw', '')
        quant = details.get('quant', '')

        # Check if bpw is in the model name (exact format)
        model_bpw_match = re.search(r'(\d+(\.\d+)?)(bpw|BPW)', model_id.split('/')[-1])
        if model_bpw_match:
            return model_bpw_match.group(1)

        if bpw:
            return str(bpw)

        # Check if bpw is in the model_id
        bpw_match = re.search(r'(\d+(\.\d+)?)(bpw|BPW)', model_id)
        if bpw_match:
            return bpw_match.group(1)

        if quant:
            return str(quant)

        # Last resort: check if the model name contains 'bpw' and extract nearby numbers
        if 'bpw' in model_name.lower():
            bpw_last_resort = re.search(r'(\d+(\.\d+)?)\s*bpw', model_name.lower())
            if bpw_last_resort:
                return bpw_last_resort.group(1)

        return ''  # Return an empty string if no bpw information found

    @property
    def list_available_models_table(self):
        return self.generate_model_table()

    @property
    def list_downloaded_models_table(self):
        # Create a list of tuples (model, backend, bpw)
        model_data = []
        for model, details in self.configs.items():
            backend = details.get('backend', '')
            bpw = self.extract_bpw(details, model)
            model_data.append((model, backend, bpw))

        # Sort the list by backend first, then by model name
        sorted_data = sorted(model_data, key=itemgetter(1, 0))

        # Create the table
        table = "| Model | Backend | Quantizations (bpw) |\n|-------|---------|---------------------|\n"
        for model, backend, bpw in sorted_data:
            table += f"| {model} | {backend} | {bpw} |\n"

        return table

    def generate_model_table(self, specific_backend=None):
        # Parse the YAML data
        data = self.default_model_list
        # Start the table
        table = "| Model | Backend | Available Quantizations (bpw) |\n|-------|---------|-------------------------------|\n"

        # Iterate through the models
        for model, info in data.items():
            if 'repo' in info:
                # Group quantizations by backend
                backend_quants = defaultdict(set)
                for repo in info['repo']:
                    backend = repo.get('backend', 'unknown')
                    quants = repo.get('quant', [])
                    backend_quants[backend].update(quants)

                # Filter by specific backend if provided
                if specific_backend:
                    if specific_backend in backend_quants:
                        backend_quants = {specific_backend: backend_quants[specific_backend]}
                    else:
                        continue  # Skip this model if it doesn't have the specified backend

                # Sort backends and their quantizations
                sorted_backends = sorted(backend_quants.keys())
                for i, backend in enumerate(sorted_backends):
                    quants = backend_quants[backend]
                    quants_formatted = [f"`{q}`" for q in sorted(quants)]

                    # For the first backend, include the model name
                    if i == 0:
                        table += f"| {model} | {backend} | {', '.join(quants_formatted)} |\n"
                    else:
                        # For subsequent backends, leave the model name cell empty
                        table += f"| | {backend} | {', '.join(quants_formatted)} |\n"

        return table

    def generate_model_dict(self, specific_backend=None) -> List[Dict[str, Any]]:
        # Parse the YAML data
        data = self.default_model_list

        # List to hold the dictionaries
        model_list = []

        # Iterate through the models
        for model, info in data.items():
            if 'repo' in info:
                # Group quantizations by backend
                backend_quants = defaultdict(set)
                for repo in info['repo']:
                    backend = repo.get('backend', 'unknown')
                    quants = repo.get('quant', [])
                    backend_quants[backend].update(quants)

                # Filter by specific backend if provided
                if specific_backend:
                    if specific_backend in backend_quants:
                        backend_quants = {specific_backend: backend_quants[specific_backend]}
                    else:
                        continue  # Skip this model if it doesn't have the specified backend

                # Sort backends and their quantizations
                sorted_backends = sorted(backend_quants.keys())
                for backend in sorted_backends:
                    quants = backend_quants[backend]
                    quants_formatted = [q for q in sorted(quants)]
                    model_list.append({
                        "model": model,
                        "backend": backend,
                        "available_quantizations": quants_formatted
                    })

        return model_list


    @property
    def list_available_models_dict(self) -> List[Dict[str, Any]]:
        return self.generate_model_dict()

    @property
    def list_downloaded_models_dict(self) -> List[Dict[str, Any]]:
        # Create a list of dictionaries (model, backend, bpw)
        model_data = []
        for model, details in self.configs.items():
            # print(details)
            # backend = details.get('backend', '')
            # bpw = self.extract_bpw(details, model)
            # model_data.append({"model": model, "backend": backend, "bpw": bpw})
            model_data.append({**{"model": model}, **details})

        # Sort the list by backend first, then by model name
        sorted_data = sorted(model_data, key=itemgetter("backend", "model"))

        return sorted_data
