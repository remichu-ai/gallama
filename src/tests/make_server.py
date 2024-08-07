from argparse import Namespace
from typing import List, Dict


def setup_make_server_args(
        model_id: List[Dict[str, str]] = None,
        verbose: bool = False,
        detached: bool = False,
        host: str = "127.0.0.1",
        port: int = 8000,
        reload: bool = False
) -> Namespace:
    """
    Create an argparse.Namespace object with the given parameters.

    Args:
        model_id (List[Dict[str, str]], optional): List of dictionaries containing model IDs. Defaults to None.
        verbose (bool, optional): Turn on more verbose logging. Defaults to False.
        detached (bool, optional): Log to ZeroMQ. Defaults to False.
        host (str, optional): The host to bind to. Defaults to "127.0.0.1".
        port (int, optional): The port to bind to. Defaults to 8000.
        reload (bool, optional): Enable auto-reload. Defaults to False.

    Returns:
        Namespace: An argparse.Namespace object with the specified attributes.
    """
    args = Namespace(
        model_id=model_id,
        verbose=verbose,
        detached=detached,
        host=host,
        port=port,
        reload=reload
    )
    return args


# Example usage:
if __name__ == "__main__":
    # Example model_id
    model_id = [{"model_id": "Mixtral-8x7B", "gpus": "1.0", "cache_size": "2048"}]

    # Create args
    args = setup_make_server_args(
        model_id=model_id,
        verbose=True,
        port=8080
    )

    # Call make_server with the created args
    make_server(args)