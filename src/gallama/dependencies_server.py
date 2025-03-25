from gallama.server_engine import ServerManager
import time
from gallama.logger.logger import get_logger
import logging
import zmq
import json
import threading
import traceback


server_manager = ServerManager()

def get_server_manager():
    if server_manager is None:
        raise RuntimeError("ServerManager not initialized")
    return server_manager



# set up logging
def start_log_receiver(zmq_url):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5555")

    # Initialize the logger for the receiver
    receiver_logger = get_logger(name="log_receiver", to_console=True, to_file=False, to_zmq=False)

    receiver_logger.info(f"Log receiver started on {zmq_url}")

    def receive_logs():
        while True:
            try:
                message = socket.recv_json()
                log_level = getattr(logging, message['level'].upper(), logging.INFO)

                receiver_logger.log(
                    level=log_level,
                    msg=f"{message['model']}:{message['port']} | {message['log']}"
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

def get_server_logger():
    if server_logger is None:
        raise RuntimeError("ServerManager not initialized")
    return server_logger
