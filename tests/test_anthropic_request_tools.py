import os
import sys
import types


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for path in (ROOT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def device_count():
            return 0

    torch_stub.cuda = _Cuda()
    sys.modules["torch"] = torch_stub

if "colorama" not in sys.modules:
    colorama_stub = types.ModuleType("colorama")
    colorama_stub.Fore = types.SimpleNamespace(CYAN="", GREEN="", YELLOW="", RED="", BLUE="")
    colorama_stub.Back = types.SimpleNamespace(WHITE="")
    colorama_stub.Style = types.SimpleNamespace(RESET_ALL="")
    colorama_stub.init = lambda autoreset=True: None
    sys.modules["colorama"] = colorama_stub

if "zmq" not in sys.modules:
    zmq_stub = types.ModuleType("zmq")

    class _Again(Exception):
        pass

    class _ZMQError(Exception):
        pass

    class _Socket:
        def setsockopt(self, *args, **kwargs):
            return None

        def connect(self, *args, **kwargs):
            return None

        def close(self):
            return None

        def send_json(self, *args, **kwargs):
            return None

    class _Context:
        def socket(self, *args, **kwargs):
            return _Socket()

        def term(self):
            return None

    zmq_stub.Context = _Context
    zmq_stub.PUSH = 0
    zmq_stub.LINGER = 0
    zmq_stub.NOBLOCK = 0
    zmq_stub.Again = _Again
    zmq_stub.error = types.SimpleNamespace(ZMQError=_ZMQError)
    sys.modules["zmq"] = zmq_stub

if "fastapi" not in sys.modules:
    fastapi_stub = types.ModuleType("fastapi")
    fastapi_stub.Query = lambda default=None, **kwargs: default
    sys.modules["fastapi"] = fastapi_stub

from gallama.data_classes.data_class import AnthropicHostedTool, AnthropicMessagesRequest


def test_anthropic_messages_request_accepts_hosted_tools_and_skips_local_conversion():
    request = AnthropicMessagesRequest.model_validate(
        {
            "model": "minimax",
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": "search for the coming ed sheeran concert",
                }
            ],
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 8,
                }
            ],
        }
    )

    assert len(request.tools) == 1
    assert isinstance(request.tools[0], AnthropicHostedTool)

    query = request.get_ChatMLQuery()

    assert query.tools is None
