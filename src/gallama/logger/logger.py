import logging
from logging.config import dictConfig
from logging.handlers import RotatingFileHandler
import os
import sys
import textwrap
from colorama import Fore, Back, Style, init
import json
import re
from pygments import highlight
from pygments.lexers import PythonLexer, MarkdownLexer, JsonLexer
from pygments.formatters import TerminalFormatter
from pygments.token import Token
from pygments.lexer import RegexLexer
from pydantic import BaseModel, validator
from typing import Optional, Union
import zmq

# Initialize colorama
init(autoreset=True)

# Aggressively reset all logging
logging.root.handlers = []
logging.root.setLevel(logging.NOTSET)


DEFAULT_ZMQ_URL = "tcp://127.0.0.1:5555"  # Using 5559 as a standard port for logging


class PromptLexer(RegexLexer):
    SPECIAL_TOKENS = [
        '[INST]', '[/INST]',
        '<s>', '</s>',
        '[AVAILABLE_TOOLS]', '[/AVAILABLE_TOOLS]', '[TOOL_RESULTS]', '[/TOOL_RESULTS]',
        '<|begin_of_text|>', '<|end_of_text|>', '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>',
        '<|assistant|>', '<|user|>', '<|end|>',
        '<|im_start|>', '<|im_end|>',
        '<start_of_turn>', '<end_of_turn>',
    ]

    tokens = {
        'root': [
            (r'|'.join(map(re.escape, SPECIAL_TOKENS)), Token.Keyword),
            (r'.*?\n', Token.Text),
        ]
    }


class ColorTabularFormatter(logging.Formatter):
    def __init__(self, max_width=50):
        super().__init__()
        self.max_width = max_width
        self.COLORS = {
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Back.WHITE
        }
        self.python_lexer = PythonLexer()
        self.markdown_lexer = MarkdownLexer()
        self.json_lexer = JsonLexer()
        self.prompt_lexer = PromptLexer()
        self.terminal_formatter = TerminalFormatter()

    def format(self, record):
        level = record.levelname
        filename = record.filename
        message = record.getMessage()

        color = self.COLORS.get(level, '')
        reset = Style.RESET_ALL

        message = self.highlight_content(message)

        message_lines = message.splitlines()

        formatted_lines = []
        for i, line in enumerate(message_lines):
            wrapped_lines = textwrap.wrap(line, self.max_width)

            for j, wrapped_line in enumerate(wrapped_lines):
                if i == 0 and j == 0:
                    formatted_lines.append(
                        f"{color}{level:<8}{reset} | {wrapped_line:<{self.max_width}} | {Fore.BLUE}{filename}{reset}")
                else:
                    formatted_lines.append(f"{' ':8}   {wrapped_line:<{self.max_width}}   ")

        record.plain_message = record.getMessage()

        return "\n".join(formatted_lines)

    def highlight_content(self, message):
        def highlight_code(match):
            lang = match.group(1) or 'python'
            code = match.group(2)

            if lang in ('python', 'py', ''):
                highlighted = highlight(code, self.python_lexer, self.terminal_formatter)
            elif lang in ('md', 'markdown'):
                highlighted = highlight(code, self.markdown_lexer, self.terminal_formatter)
            elif lang == 'json':
                try:
                    parsed = json.loads(code)
                    formatted = json.dumps(parsed, indent=2)
                    highlighted = highlight(formatted, self.json_lexer, self.terminal_formatter)
                except json.JSONDecodeError:
                    highlighted = code
            else:
                highlighted = highlight(code, self.python_lexer, self.terminal_formatter)

            return f"```{lang}\n{highlighted}```"

        pattern = r'```(\w*)\n(.*?)```'
        message = re.sub(pattern, highlight_code, message, flags=re.DOTALL)

        highlighted_prompt = highlight(message, self.prompt_lexer, self.terminal_formatter)
        return highlighted_prompt


class PlainTextFormatter(logging.Formatter):
    def format(self, record):
        return record.plain_message if hasattr(record, 'plain_message') else record.getMessage()


class ZeroMQHandler(logging.Handler):
    def __init__(self, zmq_url=DEFAULT_ZMQ_URL):
        super().__init__()
        self.zmq_url = zmq_url
        self._context = None
        self._socket = None
        self._initialize_socket()

    def _initialize_socket(self):
        """Initialize or reinitialize the ZMQ socket"""
        try:
            if self._socket:
                self._socket.close()
            if self._context:
                self._context.term()

            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.PUSH)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.connect(self.zmq_url)
        except Exception as e:
            print(f"Error initializing ZMQ socket: {e}")
            self._socket = None
            self._context = None

    def emit(self, record):
        if not self._socket:
            self._initialize_socket()
            if not self._socket:
                return  # Skip logging if socket initialization failed

        log_entry = self.format(record)
        try:
            self._socket.send_json({
                'log': log_entry,
                'level': record.levelname,
                'model': os.environ.get('MODEL_NAME', 'unknown'),
                'port': os.environ.get('MODEL_PORT', 'unknown')
            }, flags=zmq.NOBLOCK)
        except zmq.Again:
            print("Warning: ZMQ socket buffer full, log message dropped")
        except zmq.error.ZMQError as e:
            print(f"ZMQ Error in emit: {e}")
            self._initialize_socket()  # Try to reinitialize socket
        except Exception as e:
            print(f"Error in ZeroMQHandler: {e}")
            self.handleError(record)

    def close(self):
        """Properly close the ZMQ socket and context"""
        try:
            if self._socket:
                self._socket.close()
            if self._context:
                self._context.term()
        except Exception as e:
            print(f"Error closing ZMQ handler: {e}")
        finally:
            self._socket = None
            self._context = None
            super().close()


class LogConfig(BaseModel):
    LOGGER_NAME: str = "logger"
    LOG_LEVEL: Union[str, int] = "DEBUG"
    LOG_FILE: Optional[str] = None
    ZMQ_URL: Optional[str] = None
    TO_CONSOLE: bool = True
    TO_FILE: bool = False
    TO_ZMQ: bool = False
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    BACKUP_COUNT: int = 5

    version: int = 1
    disable_existing_loggers: bool = False
    formatters: dict = {
        "color_tabular": {
            "()": ColorTabularFormatter,
            "max_width": 250,
        },
        "plain": {
            "()": PlainTextFormatter,
        },
    }
    handlers: dict = {}
    loggers: dict = {}

    @validator('LOG_FILE')
    def validate_log_file(cls, v):
        if v is not None:
            if not ensure_dir_exists(v):
                raise ValueError(f"Cannot create directory for log file: {v}")
        return v



def ensure_dir_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except PermissionError:
            print(f"Error: Permission denied when trying to create directory {directory}")
            return False
    return True


def setup_logger(config: LogConfig):
    handlers = []

    if config.TO_CONSOLE:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColorTabularFormatter(max_width=250))
        handlers.append(console_handler)

    if config.TO_FILE and config.LOG_FILE:
        if ensure_dir_exists(config.LOG_FILE):
            file_handler = RotatingFileHandler(
                filename=config.LOG_FILE,
                maxBytes=config.MAX_FILE_SIZE,
                backupCount=config.BACKUP_COUNT
            )
            file_handler.setFormatter(PlainTextFormatter())
            handlers.append(file_handler)
        else:
            print(f"Warning: Unable to create log file {config.LOG_FILE}. File logging will be disabled.")
            config.TO_FILE = False

    if config.TO_ZMQ and config.ZMQ_URL:
        zmq_handler = ZeroMQHandler(zmq_url=config.ZMQ_URL)
        zmq_handler.setFormatter(PlainTextFormatter())
        handlers.append(zmq_handler)

    logger = logging.getLogger(config.LOGGER_NAME)
    logger.setLevel(config.LOG_LEVEL)
    logger.handlers = []  # Remove any existing handlers
    for handler in handlers:
        logger.addHandler(handler)

    return logger

def get_logger(
    name='logger',
    log_file="./log/llm_response.log",
    log_level=None,
    to_console=True,
    to_file=False,
    to_zmq=True,
    zmq_url=DEFAULT_ZMQ_URL
):
    # Set verbosity
    logger_verbose = os.getenv('LOCAL_OPEN_AI_VERBOSE', '1')

    if not log_level:
        if logger_verbose == '1':
            log_level = logging.INFO
        elif logger_verbose == '2':
            log_level = logging.DEBUG

    config = LogConfig(
        LOGGER_NAME=name,
        LOG_LEVEL=log_level,
        LOG_FILE=log_file,
        ZMQ_URL=zmq_url,
        TO_CONSOLE=to_console,
        TO_FILE=to_file,
        TO_ZMQ=to_zmq
    )
    return setup_logger(config)

logger = get_logger()