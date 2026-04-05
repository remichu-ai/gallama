from gallama.cli import find_config_file, parse_dict


def test_find_config_file_prefers_yml_over_yaml(tmp_path):
    yml_path = tmp_path / "model_config.yml"
    yaml_path = tmp_path / "model_config.yaml"
    yml_path.write_text("preferred: true\n")
    yaml_path.write_text("preferred: false\n")

    assert find_config_file(tmp_path, "model_config") == yml_path


def test_find_config_file_falls_back_to_yaml(tmp_path):
    yaml_path = tmp_path / "model_config.yaml"
    yaml_path.write_text("backend: exllama\n")

    assert find_config_file(tmp_path, "model_config") == yaml_path


def test_find_config_file_returns_none_when_missing(tmp_path):
    assert find_config_file(tmp_path, "model_config") is None


def test_parse_dict_supports_nested_keys_and_quoted_values():
    parsed = parse_dict("model_id=my-model backend=exllama sampling.top_k=32 prompt='hello'")

    assert parsed == {
        "model_id": "my-model",
        "backend": "exllama",
        "sampling": {
            "top_k": "32",
        },
        "prompt": "hello",
    }
