# src/gallama/config_manager.py

import yaml
import re
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
from gallama.logger import logger
from operator import itemgetter
import os


GLOBAL_CONFIG_KEY = "_global"


LOCAL_MODEL_PATH_PATTERN = re.compile(r"^(~|/|\.{1,2}([/\\]|$)|[A-Za-z]:[\\/])")
TOP_LEVEL_MODEL_KEY_PATTERN = re.compile(r'^([^\s#][^:]*)\s*:\s*(?:#.*)?$')


class ConfigManager:
    def __init__(self):
        self.configs: Dict[str, Any] = {}
        self.raw_config: Dict[str, Any] = {}
        self.global_config: Dict[str, Any] = {}
        try:
            self.load_model_configs()
        except Exception as e:
            logger.info("~/gallama/model_config.yaml is empty file")
        self.default_model_list = self.load_default_model_list()

    def __str__(self):
        return str(self.configs)

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
        self.configs = {}
        self.raw_config = {}
        self.global_config = {}
        if config_file.exists():
            with open(config_file, 'r') as file:
                yaml_data = yaml.safe_load(file)
                if yaml_data:
                    if not isinstance(yaml_data, dict):
                        raise ValueError(f"Top-level YAML structure in {config_file} must be a mapping")
                    self.raw_config.update(yaml_data)
                    global_config = yaml_data.get(GLOBAL_CONFIG_KEY, {})
                    if global_config is not None and not isinstance(global_config, dict):
                        raise ValueError(f"'{GLOBAL_CONFIG_KEY}' in {config_file} must be a mapping")
                    self.global_config = global_config or {}
                    self.configs.update(
                        {
                            key: value
                            for key, value in yaml_data.items()
                            if key != GLOBAL_CONFIG_KEY
                        }
                    )
                else:
                    raise ValueError(f'No model config in YAML file found in {config_file}')
        else:
            logger.info(f"Config file not found at {config_file}")

    @staticmethod
    def is_local_model_path(model_id: str) -> bool:
        if not model_id:
            return False

        if LOCAL_MODEL_PATH_PATTERN.match(model_id):
            return True

        return Path(model_id).exists()

    def find_missing_local_model_paths(self) -> List[Dict[str, str]]:
        missing_models = []

        for model_name, details in self.configs.items():
            if not isinstance(details, dict):
                continue

            model_id = details.get("model_id")
            if not isinstance(model_id, str) or not model_id.strip():
                continue

            expanded_path = Path(model_id).expanduser()
            if self.is_local_model_path(model_id) and not expanded_path.exists():
                missing_models.append({
                    "model_name": model_name,
                    "model_id": model_id,
                })

        return missing_models

    def comment_out_missing_models(self) -> List[Dict[str, str]]:
        config_file = self.get_gallama_user_config_file_path
        if not config_file.exists():
            logger.info(f"Config file not found at {config_file}")
            return []

        missing_models = self.find_missing_local_model_paths()
        if not missing_models:
            return []

        missing_by_name = {
            item["model_name"]: item["model_id"]
            for item in missing_models
        }

        original_lines = config_file.read_text().splitlines(keepends=True)
        newline = "\n"
        if original_lines:
            if original_lines[0].endswith("\r\n"):
                newline = "\r\n"
            elif original_lines[0].endswith("\n"):
                newline = "\n"

        rewritten_lines: List[str] = []
        current_model_name: str | None = None
        current_block: List[str] = []

        def flush_current_block():
            nonlocal current_block
            nonlocal current_model_name

            if current_model_name in missing_by_name:
                rewritten_lines.append(
                    f"# Disabled by gallama clean: missing model path {missing_by_name[current_model_name]}{newline}"
                )
                for line in current_block:
                    if not line.strip():
                        rewritten_lines.append(line)
                    elif line.lstrip().startswith("#"):
                        rewritten_lines.append(line)
                    else:
                        rewritten_lines.append(f"# {line}")
            else:
                rewritten_lines.extend(current_block)

            current_block = []
            current_model_name = None

        for line in original_lines:
            match = TOP_LEVEL_MODEL_KEY_PATTERN.match(line.rstrip("\r\n"))
            if match:
                flush_current_block()
                current_model_name = match.group(1).strip().strip("'\"")
                current_block = [line]
            else:
                current_block.append(line)

        flush_current_block()

        config_file.write_text("".join(rewritten_lines))
        try:
            self.load_model_configs()
        except ValueError:
            self.configs = {}
            self.raw_config = {}
            self.global_config = {}
        return missing_models

    def load_default_model_list(self):
        data_dir = self.get_data_dir
        with open(data_dir / "default_model_list.yaml", 'r') as file:
            yaml_data = yaml.safe_load(file)
            return yaml_data

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        return self.configs.get(model_name, {})

    def get_effective_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model config merged with supported global defaults."""
        model_config = self.get_model_config(model_name)
        if not model_config:
            return {}

        effective_config = model_config.copy()
        global_env = self.get_global_env()
        if global_env or "env" in effective_config:
            model_env = effective_config.get("env") or {}
            if model_env is not None and not isinstance(model_env, dict):
                raise ValueError(f"'env' for model '{model_name}' must be a mapping")
            effective_config["env"] = {**global_env, **model_env}

        return effective_config

    def get_global_env(self) -> Dict[str, Any]:
        env = self.global_config.get("env", {})
        if env is None:
            return {}
        if not isinstance(env, dict):
            raise ValueError(f"'env' under '{GLOBAL_CONFIG_KEY}' must be a mapping")
        return env.copy()

    def get_full_config(self) -> Dict[str, Any]:
        return self.raw_config.copy()

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
