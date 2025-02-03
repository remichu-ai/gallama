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
import httpx
import zipfile
import io
import shutil

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


def replace_home_dir_with_model_path(backend_extra_args: Dict, model_path: str) -> Dict:
    """Replace <<HOME_DIR>> in backend_extra_args with the actual model_path."""
    updated_args = {}
    for key, value in backend_extra_args.items():
        if isinstance(value, str):
            updated_args[key] = value.replace("<<HOME_DIR>>", model_path)
        else:
            updated_args[key] = value
    return updated_args

def update_model_yaml(
    model_name: str,
    model_path: str,
    backend: str,
    prompt_template: str,
    quant: str,
    cache_quant: str = None,
    backend_extra_args: Dict = None,     # additional argument for transformers loading
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

    if backend_extra_args:
        config[model_name]["backend_extra_args"] = backend_extra_args

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

    # get the list of backend available in the yaml for download
    available_backends_for_this_model = [repo['backend'] for repo in model_config["repo"] if quant in repo['quant']]

    # if there is only 1 backend available, set to that backend
    if len(available_backends_for_this_model) == 1:
        backend_to_download = available_backends_for_this_model[0]
    elif len(available_backends_for_this_model) > 1:
        # default to exllama if it is available as one of the backend
        if "exllama" in available_backends_for_this_model:
            backend_to_download = "exllama"
        else:   # just pick any
            backend_to_download = available_backends_for_this_model[0]
    else:
        raise HTTPException(status_code=400, detail=f"Error: No backend available for model '{model_name}'.")

    backend = backend_to_download

    # Filter repo_info based on both quant and backend
    repo_info = next((repo for repo in model_config['repo'] if quant in repo['quant'] and repo['backend'] == backend), None)

    if repo_info is None:
        raise HTTPException(status_code=400, detail=f"Error: Quantization {quant} with backend {backend} not available for model '{model_name}'.")

    repo_id = repo_info['repo']
    branch = next(branch for branch, q in zip(repo_info['branch'], repo_info['quant']) if q == quant)

    download_dir = str(Path.home() / "gallama" / "models" / f"{model_name}-{quant}bpw-{backend}")

    backend_extra_args = repo_info.get('backend_extra_args', None)
    if backend_extra_args:
        backend_extra_args = replace_home_dir_with_model_path(backend_extra_args, download_dir)

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
                backend_extra_args=backend_extra_args,
                quant=quant,
                cache_quant=model_config.get('default_cache_quant', None),
                config_manager=config_manager,
            )
        elif backend == "gpt_sovits":
            # Full repository download
            snapshot_download(
                repo_id=repo_id,
                revision=branch,
                local_dir=download_dir,
            )

            # Fetch the China voice URL from the extra_url field in the config
            if 'extra_url' in repo_info and 'china_voice_url' in repo_info['extra_url']:
                china_voice_url = repo_info['extra_url']['china_voice_url']
                logger.info(f"Downloading China voice model from {china_voice_url}...")

                # Download China voice model using httpx
                with httpx.Client() as client:
                    response = client.get(china_voice_url)
                    response.raise_for_status()  # Ensure the download was successful

                    # Unzip the file
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                        zip_ref.extractall(download_dir)

                    # Locate the G2PWModel_1.1 folder
                    g2pw_folder = os.path.join(download_dir, "G2PWModel_1.1")
                    if os.path.exists(g2pw_folder):
                        # Rename the folder to G2PWModel
                        new_g2pw_folder = os.path.join(download_dir, "G2PWModel")
                        os.rename(g2pw_folder, new_g2pw_folder)

                        # Move the renamed folder to the /text folder
                        text_folder = os.path.join(download_dir, "text")
                        os.makedirs(text_folder, exist_ok=True)
                        shutil.move(new_g2pw_folder, text_folder)

                logger.info(f"China voice model downloaded and extracted to {text_folder}.")
            else:
                logger.warning("No 'china_voice_url' found in 'extra_url' field. Skipping China voice model download.")

            logger.info(f"Download complete. Model saved in {download_dir}")

            update_model_yaml(
                model_name=model_name,
                model_path=str(download_dir),
                backend=backend,
                prompt_template=model_config.get('prompt_template', None),
                backend_extra_args=backend_extra_args,
                quant=quant,
                cache_quant=model_config.get('default_cache_quant', None),
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
            if backend == "transformers":
                update_model_yaml(
                    model_name=f"{model_name}_transformers",
                    model_path=str(download_dir),
                    backend=backend,
                    prompt_template=model_config.get('prompt_template', None),
                    backend_extra_args=backend_extra_args,
                    quant=quant,
                    cache_quant=model_config.get('default_cache_quant', None),
                    config_manager=config_manager,
                )

            else:
                update_model_yaml(
                    model_name=model_name,
                    model_path=str(download_dir),
                    backend=backend,
                    prompt_template=model_config.get('prompt_template', None),
                    backend_extra_args=backend_extra_args,
                    quant=quant,
                    cache_quant=model_config.get('default_cache_quant', None),
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

