from gallama.server_engine import ServerManager
from gallama.server_engine.responses_ws_bridge import ResponsesWebSocketHub
import time
from gallama.logger.logger import basic_log_extra, get_logger
import logging
import zmq
import json
import threading
import traceback


server_manager = ServerManager()
responses_websocket_hub = ResponsesWebSocketHub()
receiver_log_file = None

def get_server_manager():
    if server_manager is None:
        raise RuntimeError("ServerManager not initialized")
    return server_manager


def get_responses_websocket_hub():
    if responses_websocket_hub is None:
        raise RuntimeError("ResponsesWebSocketHub not initialized")
    return responses_websocket_hub



# set up logging
def start_log_receiver(zmq_url, log_file: str | None = None):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5555")

    # Initialize the logger for the receiver
    receiver_logger = get_logger(
        name="log_receiver",
        log_file=log_file or "./log/llm_response.log",
        to_console=True,
        to_file=bool(log_file),
        to_zmq=False
    )

    receiver_logger.info(f"Log receiver started on {zmq_url}", extra=basic_log_extra())

    def receive_logs():
        while True:
            try:
                message = socket.recv_json()
                log_level = getattr(logging, message['level'].upper(), logging.INFO)

                log_kwargs = {}
                if message.get("gallama_basic"):
                    log_kwargs["extra"] = basic_log_extra()

                receiver_logger.log(
                    level=log_level,
                    msg=f"{message['model']}:{message['port']} | {message['log']}",
                    **log_kwargs,
                )
            except zmq.Again:
                # No message available, sleep for a short time
                time.sleep(0.1)
            except zmq.ZMQError as e:
                receiver_logger.error(f"ZMQ Error in log receiver: {e}")
            except json.JSONDecodeError as e:
                receiver_logger.error(f"JSON Decode Error in log receiver: {e}")
            except Exception as e:
                receiver_logger.error(f"Unexpected error in log receiver: {e}")
                receiver_logger.error(traceback.format_exc())

    # Start the receiver in a separate thread
    receiver_thread = threading.Thread(target=receive_logs, daemon=True)
    receiver_thread.start()

    return receiver_thread


# Start the log receiver in a separate thread
DEFAULT_ZMQ_URL = "tcp://127.0.0.1:5555"  # Using 5559 as a standard port for logging


# Initialize the logger for the manager
# Set to_console=True and to_zmq=False to avoid duplication
server_logger = get_logger(name="manager", to_console=True, to_zmq=False)


def configure_server_logging(log_file: str | None = None):
    global server_logger
    global receiver_log_file

    receiver_log_file = log_file
    server_logger = get_logger(
        name="manager",
        log_file=log_file or "./log/llm_response.log",
        to_console=True,
        to_file=bool(log_file),
        to_zmq=False
    )
    return server_logger

def get_server_logger():
    if server_logger is None:
        raise RuntimeError("ServerManager not initialized")
    return server_logger
