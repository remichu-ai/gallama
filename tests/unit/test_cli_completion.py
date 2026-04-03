from gallama.cli import (
    load_model_names_for_completion,
    render_bash_completion_script,
    render_zsh_completion_script,
)


def test_load_model_names_for_completion_reads_user_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("GALLAMA_HOME_PATH", str(tmp_path))
    (tmp_path / "model_config.yaml").write_text(
        "mistral:\n"
        "  backend: exllama\n"
        "qwen2-72B:\n"
        "  backend: exllamav3\n"
        "123:\n"
        "  backend: exllama\n"
    )

    assert load_model_names_for_completion() == ["mistral", "qwen2-72B"]
    assert load_model_names_for_completion("q") == ["qwen2-72B"]


def test_load_model_names_for_completion_returns_empty_without_config(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GALLAMA_HOME_PATH", str(tmp_path))

    assert load_model_names_for_completion() == []


def test_bash_completion_script_uses_hidden_model_completion_command():
    script = render_bash_completion_script("gallama")

    assert "__complete_model_names" in script
    assert "complete -o default -F _gallama_completion gallama" in script


def test_zsh_completion_script_uses_hidden_model_completion_command():
    script = render_zsh_completion_script("gallama")

    assert "#compdef gallama" in script
    assert "__complete_model_names" in script
    assert "local -a commands" not in script
    assert "local -a gallama_subcommands models" in script
