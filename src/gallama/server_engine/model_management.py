import yaml
import os
import subprocess
from gallama.config import ConfigManager
from gallama.logger import logger
from fastapi import HTTPException
from huggingface_hub import snapshot_download, hf_hub_download
from pathlib import Path
from typing import Dict
from gallama.data_classes import ModelInfo, ModelDownloadSpec


def represent_list(self, data):
    """ Custom representation for lists for writing to yaml file"""

    # Check if the list contains only integers
    if all((isinstance(item, int) or isinstance(item, float) or isinstance(item, str))for item in data):
        # Represent as a flow-style (inline) sequence
        return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    # For other lists, use the default representation
    return self.represent_sequence('tag:yaml.org,2002:seq', data)


# Add the custom representation to SafeDumper
yaml.SafeDumper.add_representer(list, represent_list)


def update_model_yaml(
    model_name: str,
    model_path: str,
    backend: str,
    prompt_template: str,
    quant: str,
    cache_quant: str = None,
    config_manager: ConfigManager = ConfigManager()
):
    config = config_manager.configs      # the default list of model

    # Update or add the model configuration
    config[model_name] = {
        'model_id': model_path,
        'backend': backend,
        'gpus': "auto",
        'prompt_template': prompt_template,
        'quant': quant,
    }

    # Optional update cache quant
    if cache_quant is not None:
        config[model_name]['cache_quant'] = cache_quant

    # Write the updated configuration back to the YAML file
    with open(config_manager.get_gallama_user_config_file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, Dumper=yaml.SafeDumper)

    logger.info(f"Updated model configuration in {config_manager.get_gallama_user_config_file_path}")


def download_model_from_hf(model_spec: ModelDownloadSpec):
    config_manager: ConfigManager = ConfigManager()
    config = config_manager.default_model_list

    model_name = model_spec.model_name
    quant = model_spec.quant
    backend = model_spec.backend

    if model_name not in config:
        raise HTTPException(status_code=404, detail=f"Error: Model '{model_name}' not found in configuration.")

    model_config = config[model_name]

    if quant is None:
        quant = model_config.get('default_quant')
        if quant is None:
            raise HTTPException(status_code=400, detail=f"Error: No default quantization specified for model '{model_name}'.")

    # Filter repo_info based on both quant and backend
    repo_info = next((repo for repo in model_config['repo'] if quant in repo['quant'] and repo['backend'] == backend), None)

    if repo_info is None:
        raise HTTPException(status_code=400, detail=f"Error: Quantization {quant} with backend {backend} not available for model '{model_name}'.")

    repo_id = repo_info['repo']
    branch = next(branch for branch, q in zip(repo_info['branch'], repo_info['quant']) if q == quant)

    download_dir = str(Path.home() / "gallama" / "models" / f"{model_name}-{quant}bpw-{backend}")

    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

    logger.info(f"Downloading {model_name} (quantization: {quant}, backend: {backend})...")
    try:
        if backend == "llama_cpp" and 'url' in repo_info:
            # Single file download for GGUF models
            filename = repo_info['url'].split('/')[-1]
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=download_dir,
                revision=branch
            )
            logger.info(f"Download complete. Model saved as {file_path}")

            update_model_yaml(
                model_name=f"{model_name}_llama_cpp",
                model_path=f"{str(download_dir)}/{filename}",
                backend=backend,
                prompt_template=model_config.get('prompt_template', None),
                quant=quant,
                cache_quant=model_config.get('default_cache_quant', "Q4"),
                config_manager=config_manager,
            )
        else:
            # Full repository download for other models
            snapshot_download(
                repo_id=repo_id,
                revision=branch,
                local_dir=download_dir,
            )
            logger.info(f"Download complete. Model saved in {download_dir}")

            update_model_yaml(
                model_name=model_name,
                model_path=str(download_dir),
                backend=backend,
                prompt_template=model_config.get('prompt_template', None),
                quant=quant,
                cache_quant=model_config.get('default_cache_quant', "Q4"),
                config_manager=config_manager,
            )

        return {"status": "success", "message": f"Model downloaded and configured: {model_name}"}
    except Exception as e:
        logger.error(f"Error during download: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during download: {str(e)}")


def get_gpu_memory_info():

    try:
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=index,memory.used,memory.free,memory.total',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')
        lines = result.strip().split('\n')

        total_used = 0
        total_free = 0
        total_memory = 0

        gpu_memory_info = []
        for line in lines:
            index, used, free, total = map(int, line.split(','))
            gpu_memory_info.append(
                f"GPU {index}: "
                f"Used:  {used/1024:.1f}GB, "
                f"Free:  {free/1024:.1f}GB, "
                f"Total:  {total/1024:.1f}GB"
            )
            total_used += used
            total_free += free
            total_memory += total

        total_line = (f"Total: "
                      f"Used: {total_used/1024:.1f}GB, "
                      f"Free: {total_free/1024:.1f}GB, "
                      f"Total: {total_memory/1024:.1f}GB")

        if len(gpu_memory_info) == 1:
            return "\n".join(gpu_memory_info)   # no total as only 1 GPU
        else:
            return "\n".join(gpu_memory_info +
                             ["------+-------------+-------------+------------------------------------------+"] +
                             [total_line])    # add total line as more than 1 GPU
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Unable to retrieve GPU information"


def log_model_status(models: Dict[str, ModelInfo], custom_logger: "logger" =None):
    total_models = len(models)
    total_instances = sum(len(model_info.instances) for model_info in models.values())

    # Prepare model details
    model_details = []
    for model_name, model_info in models.items():
        instances = ["{0}".format(inst.port) for inst in model_info.instances]
        model_details.append("| {0:<20} | {1:>2} | {2:<30} ".format(model_name, len(instances), ', '.join(instances)))

    # Prepare GPU info
    gpu_info = get_gpu_memory_info().split('\n')
    formatted_gpu_info = ''.join("| {0}\n".format(line) for line in gpu_info)

    # Construct the log message
    log_message = """```
+------------------------------------------------------------------------------+
| Current Status: {0} model(s) loaded with {1} total instance(s)               
+------------------------------------------------------------------------------+
| Model Name           | # | Ports                                             
+----------------------+---+---------------------------------------------------+
{2}
+------------------------------------------------------------------------------+
| GPU Memory Information                                                       
+-------+-------------+-------------+------------------------------------------+
| GPU   | Used        | Free        | Total       
+-------+-------------+-------------+------------------------------------------+
{3}+-------+-------------+-------------+------------------------------------------+
```""".format(
        total_models,
        total_instances,
        '\n'.join(model_details),
        formatted_gpu_info
    )

    if custom_logger:
        custom_logger.info(log_message)
    else:
        logger.info(log_message)

