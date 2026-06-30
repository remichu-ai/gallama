from gallama.data_classes.data_class import ModelSpec
from gallama.data_classes.server_dataclass import ModelInfo, ModelInstanceInfo
from gallama.model_manager.ModelManager import ModelManager
from gallama.server_engine.server_manager import ServerManager


def test_model_spec_accepts_singular_and_multiple_aliases():
    singular = ModelSpec.model_validate(
        {
            "model_name": "local-qwen",
            "alias": "gpt-4o-mini",
            "model_id": "/models/qwen",
            "backend": "exllamav3",
        }
    )
    multiple = ModelSpec.from_dict(
        {
            "model_name": "local-llama",
            "aliases": ["llama", "local-chat"],
            "model_id": "/models/llama",
            "backend": "exllamav3",
        }
    )

    assert singular.aliases == ["gpt-4o-mini"]
    assert multiple.aliases == ["llama", "local-chat"]


def test_model_manager_resolves_strict_model_aliases_without_duplicate_model_keys():
    fake_model = object()
    manager = ModelManager()
    model_spec = ModelSpec(
        model_name="local-qwen",
        aliases=["gpt-4o-mini", "local-chat"],
        model_id="/models/qwen",
        backend="exllamav3",
        model_type="llm",
        strict=True,
    )

    manager._update_model("local-qwen", model_spec, fake_model)

    assert list(manager.llm_dict.keys()) == ["local-qwen"]
    assert manager.get_model("local-qwen", _type="llm") is fake_model
    assert manager.get_model("gpt-4o-mini", _type="llm") is fake_model
    assert manager.get_model("local-chat", _type="llm") is fake_model
    assert manager.get_model("unknown", _type="llm") is None
    assert manager.list_model_ids() == ["local-qwen", "gpt-4o-mini", "local-chat"]


def test_server_manager_lists_and_routes_aliases():
    manager = ServerManager()
    instance = ModelInstanceInfo(
        model_name="local-qwen",
        aliases=["gpt-4o-mini"],
        port=8001,
        pid=123,
        status="running",
        model_type="llm",
        strict=True,
    )
    manager.models["local-qwen"] = ModelInfo(instances=[instance])

    assert manager.resolve_model_name("local-qwen") == "local-qwen"
    assert manager.resolve_model_name("gpt-4o-mini") == "local-qwen"
    assert manager.list_model_ids() == ["local-qwen", "gpt-4o-mini"]
    assert manager.get_instance("llm", "gpt-4o-mini") is instance
    assert manager.get_instance("llm", "unknown") is None
