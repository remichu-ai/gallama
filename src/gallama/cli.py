import argparse
from pydantic import BaseModel, Field
from torch import multiprocessing as mp
from typing import List, Optional
from enum import Enum
from gallama.data_classes import ModelSpec, ServerSetting
from gallama.app import make_server
import typer
from gallama.config.config_manager import ConfigManager


cli = typer.Typer()

# import shutil
# from pathlib import Path
# from gallama.config import ConfigManager
# from gallama.logger.logger import logger
# from gallama.server import run_from_script, download_model
# from gallama.data_classes.data_class import ModelDownloadSpec
# from rich.markdown import Markdown
# from rich.console import Console
#
#
# def get_data_dir() -> Path:
#     """Get the absolute path to the data directory."""
#     return Path(__file__).parent / 'data'
#
# def get_data_dir_dev() -> Path:
#     """Get the absolute path to the data directory."""
#     return Path(__file__).parent / 'gallama' / 'data'
#
#
# def find_config_file(config_dir, filename):
#     """Find the configuration file with either .yml or .yaml extension."""
#     yml_file = config_dir / f"{filename}.yml"
#     yaml_file = config_dir / f"{filename}.yaml"
#     logger.debug(f"Searching for config files: {yml_file}, {yaml_file}")
#     return yml_file if yml_file.exists() else yaml_file if yaml_file.exists() else None
#
#
# def ensure_config_file():
#     """Ensure the user-specific configuration file exists."""
#     home_dir = Path.home()
#     config_dir = home_dir / "gallama"
#     model_dir = home_dir / "gallama" / "models"
#     config_filename = "model_config"
#
#     config_dir_default = get_data_dir()
#     config_dir_default_dev = get_data_dir_dev()     # dev mode file path is different
#
#     try:
#         if not config_dir.exists():
#             config_dir.mkdir(parents=True)
#             logger.info(f"Created configuration directory: {config_dir}")
#
#         if not model_dir.exists():
#             model_dir.mkdir(parents=True)
#             logger.info(f"Created configuration directory: {model_dir}")
#
#         config_file = find_config_file(config_dir, config_filename)
#         if not config_file:
#
#             default_config = find_config_file(config_dir_default, config_filename)
#             default_config_dev = find_config_file(config_dir_default_dev, config_filename)
#
#             if default_config:
#                 config_file = config_dir / default_config.name
#                 shutil.copy(default_config, config_file)
#                 logger.info(f"Created default configuration file: {config_file}")
#             elif default_config_dev:
#                 config_file = config_dir / default_config_dev.name
#                 shutil.copy(default_config_dev, config_file)
#                 logger.info(f"Created default configuration file: {default_config_dev}")
#             else:
#                 raise FileNotFoundError(f"Default configuration file not found in: {config_dir}")
#
#         return config_file
#     except Exception as e:
#         logger.error(f"Error creating configuration: {str(e)}")
#         logger.info("Please ensure the default configuration file exists in the package's data directory.")
#         return None
#
#

def parse_dict(arg):
    """Parses a key=value string and returns a dictionary."""
    result = {}
    for pair in arg.split():

        key, value = pair.split('=')
        result[key] = value.strip("'")  # Strip single quotes here as well
    return result


@cli.command()
def run(
    model_name: Optional[str] = typer.Argument(
        None,
        help="Model name or path for simple mode (e.g., 'llama3-8b')"
    ),
    model_id: Optional[List[str]] = typer.Option(
        None,
        "--model-id",
        "-id",
        help="""Model configuration. Format: model_id=<name> [gpu=vram1,vram2,...] [cache=size]
             Examples:
             1. -id "model_id=/path/llama3-8b gpu=4,2,0,3 cache=8192"
             2. -id "model_id=/path/llama3-8b gpu0=8 gpu1=8 cache=8192"
             """
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Turn on more verbose logging"),
    host: str = typer.Option("127.0.0.1", help="The host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="The port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload")
):
    """Launch the OpenAI-compatible model server"""
    model_list = []

    # Handle simple mode if model_name is provided
    if model_name and not model_id:
        spec = f"model_name={model_name}"
        model_spec = ModelSpec.from_dict(parse_dict(spec))
        model_list.append(model_spec)

    # standard mode
    if model_id:
        for spec in model_id:
            model_spec = ModelSpec.from_dict(parse_dict(spec))
            model_list.append(model_spec)

    settings = ServerSetting(
        host=host,
        port=port,
        verbose=verbose,
        model_specs=model_list
    )

    make_server(settings)
#     arg_parser = argparse.ArgumentParser(description="Launch multi model src instance")
#     arg_parser.add_argument('-v', "--verbose", action='store_true', help="Turn on more verbose logging")
#
#     subparsers = arg_parser.add_subparsers(dest="command")
#
#     # Add 'serve' subcommand
#     serve_parser = subparsers.add_parser("run", help="Run the FastAPI server_engine")
#     serve_parser.add_argument("model_name", nargs='?', help="Model name to run (simplified version)")
#     serve_parser.add_argument("--strict_mode", action="store_true", default=False,
#                               help="Enable strict mode for routing non-embedding requests to matching model names")
#     serve_parser.add_argument("-id", "--model_id", action='append', type=parse_dict, default=None,
#                               help="Model configuration. Can be used multiple times for different models. "
#                                    "Format: -id model_id=<model_name> [gpu=<vram1>,<vram2>,...] [gpu<n>=<vram>] [cache=<size>] [model_type=exllama|embedding]"
#                                    "Examples:\n"
#                                    "1. Specify all GPUs: -id model_id=/path/llama3-8b gpu=4,2,0,3 cache=8192\n"
#                                    "2. Specify individual GPUs: -id model_id=/path/llama3-8b gpu0=8 gpu1=8 cache=8192\n"
#                                    "3. Mix formats: -id model_id=gpt4 gpu=4,2 gpu3=3 cache=8192\n"
#                                    "4. Embedding model: -id model_id=/path/gte-large-en-v1.5 gpu=0,0,0,3"
#                                    "cache_size is defaulted to model context length if not specified"
#                                    "cache_size can be more than context length, which will help model perform better in batched generation"
#                                    "cache_size is not application to embedding model"
#                                    "VRAM is specified in GB. Cache size is integer which is the context length to cache."
#                                    "VRAM for embedding will simple set env parameter to allow infinity_embedding to view the specific GPU and can not enforce VRAM size restriction")
#     serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="The host to bind to.")
#     serve_parser.add_argument('-p', "--port", type=int, default=8000, help="The port to bind to.")
#
#     # Add 'download' subcommand
#     download_parser = subparsers.add_parser("download", help="Download a model from Hugging Face")
#     download_parser.add_argument("model_spec", type=str, help="Model specification in the format 'model_name:quant'")
#     download_parser.add_argument("--backend", type=str, default="exllama", choices=['exllama', 'llama_cpp', 'embedding', 'transformers'],
#                                  help="The backend to download model with. One of exllama, llama_cpp, embedding or transformers")
#
#     # Modify 'list' subcommand
#     list_parser = subparsers.add_parser("list", help="List models")
#     list_parser.add_argument("type", nargs='?', choices=['available', 'downloaded'], default='downloaded',
#                              help="List downloaded model (default) or list available to show the list of model that can be downloaded")
#
#     args = arg_parser.parse_args()
#
#     # ensure config file is there
#     ensure_config_file()
#
#     # set logger level
#     if args.verbose:
#         logger.setLevel("DEBUG")
#     else:
#         logger.setLevel("INFO")
#
#     if args.command == "run" or args.command is None:
#         if args.model_name and not args.model_id:
#             # Convert simplified version to the full version
#             args.model_id = [{"model_id": args.model_name}]
#         run_from_script(args)  # Pass all arguments to run_from_script
#
#     elif args.command == "download":
#         if ":" in args.model_spec:
#             model_name, quant = args.model_spec.split(':')
#         else:
#             # default download mode
#             model_name = args.model_spec
#             quant = None
#
#         download_model(ModelDownloadSpec(
#             model_name=model_name,
#             quant=float(quant) if quant else None,
#             backend=args.backend,
#         ))
#
#     elif args.command == "list":
#         config_manager = ConfigManager()
#         console = Console()
#         if args.type == "available":
#             md = Markdown(config_manager.list_available_models_table)
#             console.print(md)
#         elif args.type == "downloaded" or args.type is None:
#             md = Markdown(config_manager.list_downloaded_models_table.lstrip())
#             console.print(md)
#         elif args.type == "running":
#             # Implement logic to list running models
#             pass

@cli.command()
def list(
    show_all: bool = typer.Option(False, "--all", "-a", help="Show all available models, not just downloaded ones")
):
    """List available models"""
    if show_all:
        typer.echo("Showing all available models:")
        # Add your code to show all models
    else:
        typer.echo("Showing downloaded models:")
        # Add your code to show downloaded models

class Backend(str, Enum):
    EXLLAMA = "exllama"
    LLAMA_CPP = "llama_cpp"
    TRANSFORMERS = "transformers"


@cli.command()
def download(
    model_name: str = typer.Argument(..., help="Name of the model to download"),
    backend: Backend = typer.Option(
        Backend.EXLLAMA,
        "--backend",
        help="Backend to use"
    )
):
    """Download a model"""
    typer.echo(f"Downloading {model_name} using {backend.value} backend...")
    # Add your download logic here



def main():
    cli()

if __name__ == "__main__":
    mp.freeze_support()
    main()
