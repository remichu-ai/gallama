import argparse
import shutil
from pathlib import Path
from gallama.config import ConfigManager
from gallama.logger.logger import logger
from gallama.server import run_from_script, download_model
from gallama.data_classes.data_class import ModelDownloadSpec
from rich.markdown import Markdown
from rich.console import Console


def get_data_dir() -> Path:
    """Get the absolute path to the data directory."""
    return Path(__file__).parent / 'data'

def get_data_dir_dev() -> Path:
    """Get the absolute path to the data directory."""
    return Path(__file__).parent / 'gallama' / 'data'


def find_config_file(config_dir, filename):
    """Find the configuration file with either .yml or .yaml extension."""
    yml_file = config_dir / f"{filename}.yml"
    yaml_file = config_dir / f"{filename}.yaml"
    logger.debug(f"Searching for config files: {yml_file}, {yaml_file}")
    return yml_file if yml_file.exists() else yaml_file if yaml_file.exists() else None


def ensure_config_file():
    """Ensure the user-specific configuration file exists."""
    home_dir = Path.home()
    config_dir = home_dir / "gallama"
    model_dir = home_dir / "gallama" / "models"
    config_filename = "model_config"

    config_dir_default = get_data_dir()
    config_dir_default_dev = get_data_dir_dev()     # dev mode file path is different

    try:
        if not config_dir.exists():
            config_dir.mkdir(parents=True)
            logger.info(f"Created configuration directory: {config_dir}")

        if not model_dir.exists():
            model_dir.mkdir(parents=True)
            logger.info(f"Created configuration directory: {model_dir}")

        config_file = find_config_file(config_dir, config_filename)
        if not config_file:

            default_config = find_config_file(config_dir_default, config_filename)
            default_config_dev = find_config_file(config_dir_default_dev, config_filename)

            if default_config:
                config_file = config_dir / default_config.name
                shutil.copy(default_config, config_file)
                logger.info(f"Created default configuration file: {config_file}")
            elif default_config_dev:
                config_file = config_dir / default_config_dev.name
                shutil.copy(default_config_dev, config_file)
                logger.info(f"Created default configuration file: {default_config_dev}")
            else:
                raise FileNotFoundError(f"Default configuration file not found in: {config_dir}")

        return config_file
    except Exception as e:
        logger.error(f"Error creating configuration: {str(e)}")
        logger.info("Please ensure the default configuration file exists in the package's data directory.")
        return None


def parse_dict(arg):
    """Parses a key=value string and returns a dictionary."""
    result = {}
    for pair in arg.split():

        key, value = pair.split('=')
        result[key] = value.strip("'")  # Strip single quotes here as well
    return result


def run_server(host, port):
    import uvicorn
    from .app import app
    uvicorn.run(app, host=host, port=port)


def main_cli():
    arg_parser = argparse.ArgumentParser(description="Launch multi model src instance")
    arg_parser.add_argument('-v', "--verbose", action='store_true', help="Turn on more verbose logging")

    subparsers = arg_parser.add_subparsers(dest="command")

    # Add 'serve' subcommand
    serve_parser = subparsers.add_parser("run", help="Run the FastAPI server_engine")
    serve_parser.add_argument("model_name", nargs='?', help="Model name to run (simplified version)")
    serve_parser.add_argument("--strict_mode", action="store_true", default=False,
                              help="Enable strict mode for routing non-embedding requests to matching model names")
    serve_parser.add_argument("-id", "--model_id", action='append', type=parse_dict, default=None,
                              help="Model configuration. Can be used multiple times for different models. "
                                   "Format: -id model_id=<model_name> [gpu=<vram1>,<vram2>,...] [gpu<n>=<vram>] [cache=<size>] [model_type=exllama|embedding]"
                                   "Examples:\n"
                                   "1. Specify all GPUs: -id model_id=/path/llama3-8b gpu=4,2,0,3 cache=8192\n"
                                   "2. Specify individual GPUs: -id model_id=/path/llama3-8b gpu0=8 gpu1=8 cache=8192\n"
                                   "3. Mix formats: -id model_id=gpt4 gpu=4,2 gpu3=3 cache=8192\n"
                                   "4. Embedding model: -id model_id=/path/gte-large-en-v1.5 gpu=0,0,0,3"
                                   "cache_size is defaulted to model context length if not specified"
                                   "cache_size can be more than context length, which will help model perform better in batched generation"
                                   "cache_size is not application to embedding model"
                                   "VRAM is specified in GB. Cache size is integer which is the context length to cache."
                                   "VRAM for embedding will simple set env parameter to allow infinity_embedding to view the specific GPU and can not enforce VRAM size restriction")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="The host to bind to.")
    serve_parser.add_argument('-p', "--port", type=int, default=8000, help="The port to bind to.")

    # Add 'download' subcommand
    download_parser = subparsers.add_parser("download", help="Download a model from Hugging Face")
    download_parser.add_argument("model_spec", type=str, help="Model specification in the format 'model_name:quant'")
    download_parser.add_argument("--backend", type=str, default="exllama", choices=['exllama', 'llama_cpp', 'embedding'],
                                 help="The backend to download model with. One of exllama, llama_cpp, embedding")

    # Modify 'list' subcommand
    list_parser = subparsers.add_parser("list", help="List models")
    list_parser.add_argument("type", nargs='?', choices=['available', 'downloaded'], default='downloaded',
                             help="List downloaded model (default) or list available to show the list of model that can be downloaded")

    args = arg_parser.parse_args()

    # ensure config file is there
    ensure_config_file()

    # set logger level
    if args.verbose:
        logger.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")

    if args.command == "run" or args.command is None:
        if args.model_name and not args.model_id:
            # Convert simplified version to the full version
            args.model_id = [{"model_id": args.model_name}]
        run_from_script(args)  # Pass all arguments to run_from_script

    elif args.command == "download":
        if ":" in args.model_spec:
            model_name, quant = args.model_spec.split(':')
        else:
            # default download mode
            model_name = args.model_spec
            quant = None

        download_model(ModelDownloadSpec(
            model_name=model_name,
            quant=float(quant) if quant else None,
            backend=args.backend,
        ))

    elif args.command == "list":
        config_manager = ConfigManager()
        console = Console()
        if args.type == "available":
            md = Markdown(config_manager.list_available_models_table)
            console.print(md)
        elif args.type == "downloaded" or args.type is None:
            md = Markdown(config_manager.list_downloaded_models_table.lstrip())
            console.print(md)
        elif args.type == "running":
            # Implement logic to list running models
            pass


if __name__ == "__main__":
    main_cli()