__all__ = [
    "make_server",
    "ConfigManager",
    "model_manager",
    "get_server_manager",
    "get_server_logger",
    "DEFAULT_ZMQ_URL",
]


def __getattr__(name):
    if name == "make_server":
        from .app import make_server

        return make_server
    if name == "ConfigManager":
        from .config import ConfigManager

        return ConfigManager
    if name == "model_manager":
        from .dependencies import model_manager

        return model_manager
    if name in {"get_server_manager", "get_server_logger", "DEFAULT_ZMQ_URL"}:
        from .dependencies_server import (
            get_server_manager,
            get_server_logger,
            DEFAULT_ZMQ_URL,
        )

        return {
            "get_server_manager": get_server_manager,
            "get_server_logger": get_server_logger,
            "DEFAULT_ZMQ_URL": DEFAULT_ZMQ_URL,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
