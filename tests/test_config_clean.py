from pathlib import Path

from gallama.config import ConfigManager


def test_comment_out_missing_models_comments_only_missing_local_paths(monkeypatch, tmp_path):
    existing_model_path = tmp_path / "existing-model"
    existing_model_path.mkdir()
    missing_model_path = tmp_path / "missing-model"

    config_file = tmp_path / "model_config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "# test config",
                "missing-model:",
                f"  model_id: '{missing_model_path}'",
                "  backend: 'exllama'",
                "",
                "remote-model:",
                "  model_id: 'Alibaba-NLP/gte-large-en-v1.5'",
                "  backend: 'embedding'",
                "",
                "present-model:",
                f"  model_id: '{existing_model_path}'",
                "  backend: 'exllama'",
                "",
            ]
        )
    )

    monkeypatch.setenv("GALLAMA_HOME_PATH", str(tmp_path))
    manager = ConfigManager()

    cleaned_models = manager.comment_out_missing_models()
    updated_config = config_file.read_text()

    assert cleaned_models == [
        {
            "model_name": "missing-model",
            "model_id": str(missing_model_path),
        }
    ]
    assert "# Disabled by gallama clean: missing model path" in updated_config
    assert "# missing-model:" in updated_config
    assert "#   model_id:" in updated_config
    assert "\nremote-model:\n" in updated_config
    assert "\npresent-model:\n" in updated_config
