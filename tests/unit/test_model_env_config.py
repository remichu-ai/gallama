from gallama.config import ConfigManager
from gallama.data_classes.data_class import ModelSpec
from gallama.server_engine.model_management import update_model_yaml


def test_global_env_is_not_treated_as_model_and_merges_into_model_config(monkeypatch, tmp_path):
    config_file = tmp_path / "model_config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "_global:",
                "  env:",
                "    CUDA_VISIBLE_DEVICES: '1,0'",
                "    HF_HOME: '/tmp/hf-global'",
                "vision-model:",
                "  model_id: '/models/vision'",
                "  backend: 'exllama'",
                "  env:",
                "    HF_HOME: '/tmp/hf-vision'",
                "text-model:",
                "  model_id: '/models/text'",
                "  backend: 'exllama'",
                "",
            ]
        )
    )

    monkeypatch.setenv("GALLAMA_HOME_PATH", str(tmp_path))
    manager = ConfigManager()

    assert manager.get_all_model_names() == ["vision-model", "text-model"]
    assert manager.get_model_config("vision-model")["env"] == {"HF_HOME": "/tmp/hf-vision"}
    assert manager.get_effective_model_config("vision-model")["env"] == {
        "CUDA_VISIBLE_DEVICES": "1,0",
        "HF_HOME": "/tmp/hf-vision",
    }
    assert manager.get_effective_model_config("text-model")["env"] == {
        "CUDA_VISIBLE_DEVICES": "1,0",
        "HF_HOME": "/tmp/hf-global",
    }
    assert "_global" in manager.get_full_config()


def test_global_warmup_prompt_merges_into_model_config(monkeypatch, tmp_path):
    config_file = tmp_path / "model_config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "_global:",
                "  warmup_prompt:",
                "    path: warmups/claude.yaml",
                "    max_completion_tokens: 64",
                "vision-model:",
                "  model_id: '/models/vision'",
                "  backend: 'exllama'",
                "  warmup_prompt:",
                "    reasoning_effort: minimal",
                "text-model:",
                "  model_id: '/models/text'",
                "  backend: 'exllama'",
                "disabled-model:",
                "  model_id: '/models/disabled'",
                "  backend: 'exllama'",
                "  warmup_prompt: false",
                "",
            ]
        )
    )

    monkeypatch.setenv("GALLAMA_HOME_PATH", str(tmp_path))
    manager = ConfigManager()

    assert manager.get_effective_model_config("vision-model")["warmup_prompt"] == {
        "path": "warmups/claude.yaml",
        "max_completion_tokens": 64,
        "reasoning_effort": "minimal",
    }
    assert manager.get_effective_model_config("text-model")["warmup_prompt"] == {
        "path": "warmups/claude.yaml",
        "max_completion_tokens": 64,
    }
    assert manager.get_effective_model_config("disabled-model")["warmup_prompt"] is False


def test_update_model_yaml_preserves_global_config(monkeypatch, tmp_path):
    config_file = tmp_path / "model_config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "_global:",
                "  env:",
                "    CUDA_VISIBLE_DEVICES: '1,0'",
                "",
            ]
        )
    )

    monkeypatch.setenv("GALLAMA_HOME_PATH", str(tmp_path))
    manager = ConfigManager()

    update_model_yaml(
        model_name="new-model",
        model_path="/models/new-model",
        backend="exllama",
        prompt_template="Llama3",
        quant="4.5",
        config_manager=manager,
    )

    written = config_file.read_text()
    reloaded_manager = ConfigManager()

    assert "_global:" in written
    assert "new-model:" in written
    assert reloaded_manager.get_global_env() == {"CUDA_VISIBLE_DEVICES": "1,0"}
    assert reloaded_manager.get_model_config("new-model")["model_id"] == "/models/new-model"


def test_build_child_env_maps_explicit_gpu_split_against_visible_device_order():
    model_spec = ModelSpec.from_dict(
        {
            "model_name": "vision-model",
            "model_id": "/models/vision",
            "backend": "exllama",
            "gpus": [12.0, 0.0, 10.0, 0.0],
            "env": {
                "CUDA_VISIBLE_DEVICES": "3,1,0,2",
                "REMOVE_ME": None,
            },
        }
    )

    child_env = model_spec.build_child_env(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "REMOVE_ME": "yes",
            "KEEP_ME": "1",
        }
    )

    assert child_env["CUDA_VISIBLE_DEVICES"] == "3,0"
    assert child_env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert child_env["KEEP_ME"] == "1"
    assert "REMOVE_ME" not in child_env


def test_build_child_env_preserves_visible_device_order_for_auto_gpu_mode():
    model_spec = ModelSpec.from_dict(
        {
            "model_name": "vision-model",
            "model_id": "/models/vision",
            "backend": "exllama",
            "gpus": "auto",
            "env": {
                "CUDA_VISIBLE_DEVICES": "1,0",
            },
        }
    )

    child_env = model_spec.build_child_env({"CUDA_VISIBLE_DEVICES": "0,1"})

    assert child_env["CUDA_VISIBLE_DEVICES"] == "1,0"


def test_model_spec_from_dict_preserves_warmup_prompt():
    model_spec = ModelSpec.from_dict(
        {
            "model_name": "vision-model",
            "model_id": "/models/vision",
            "backend": "exllama",
            "warmup_prompt": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with OK.",
                    }
                ]
            },
        }
    )

    assert model_spec.warmup_prompt == {
        "messages": [
            {
                "role": "user",
                "content": "Reply with OK.",
            }
        ]
    }
