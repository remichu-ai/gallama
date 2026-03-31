import importlib.util
import os
import sys
from types import SimpleNamespace


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
MODULE_PATH = os.path.join(SRC_DIR, "gallama", "server_engine", "request_routing.py")
MODULE_SPEC = importlib.util.spec_from_file_location("request_routing_test_module", MODULE_PATH)
REQUEST_ROUTING_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(REQUEST_ROUTING_MODULE)

instance_supports_vision = REQUEST_ROUTING_MODULE.instance_supports_vision
prefer_vision_instances = REQUEST_ROUTING_MODULE.prefer_vision_instances
request_requires_vision = REQUEST_ROUTING_MODULE.request_requires_vision


def _instance(name: str, modalities: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=name,
        port=8000,
        pid=1234,
        status="running",
        model_type="llm",
        strict=False,
        modalities=modalities,
    )


def test_request_requires_vision_for_chat_completions_image_url():
    payload = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                ],
            }
        ],
    }

    assert request_requires_vision(payload) is True


def test_request_requires_vision_for_anthropic_image_block():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "abc",
                        },
                    }
                ],
            }
        ]
    }

    assert request_requires_vision(payload) is True


def test_request_requires_vision_for_responses_input_image():
    payload = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "https://example.com/cat.png"},
                ],
            }
        ]
    }

    assert request_requires_vision(payload) is True


def test_request_requires_vision_returns_false_for_text_only_payload():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            }
        ]
    }

    assert request_requires_vision(payload) is False


def test_request_requires_vision_ignores_tool_schema_image_url_properties():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "can u search internet?"}],
            }
        ],
        "tools": [
            {
                "name": "understand_image",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "image_url": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                },
            }
        ],
    }

    assert request_requires_vision(payload) is False


def test_request_requires_vision_handles_non_string_type_values():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            }
        ],
        "response_format": {
            "json_schema": {
                "schema": {
                    "properties": {
                        "title": {
                            "type": {"kind": "string"},
                        }
                    }
                }
            }
        },
    }

    assert request_requires_vision(payload) is False


def test_prefer_vision_instances_prefers_vision_when_available():
    text_instance = _instance("text-only", ["text"])
    vision_instance = _instance("vision", ["text", "image"])

    preferred = prefer_vision_instances(
        [text_instance, vision_instance],
        vision_required=True,
    )

    assert preferred == [vision_instance]
    assert instance_supports_vision(vision_instance) is True
    assert instance_supports_vision(text_instance) is False


def test_prefer_vision_instances_falls_back_when_no_vision_model_exists():
    text_instance = _instance("text-only", ["text"])

    preferred = prefer_vision_instances(
        [text_instance],
        vision_required=True,
    )

    assert preferred == [text_instance]
