from gallama.warmup import build_warmup_query, resolve_warmup_prompt_config


def test_resolve_warmup_prompt_from_external_file_with_override(tmp_path):
    warmup_dir = tmp_path / "warmups"
    warmup_dir.mkdir()
    warmup_file = warmup_dir / "claude.yaml"
    warmup_file.write_text(
        "\n".join(
            [
                "messages:",
                "  - role: developer",
                "    content: You are Claude Code.",
                "  - role: user",
                "    content: Reply with OK.",
                "reasoning_effort: high",
                "",
            ]
        )
    )

    resolved = resolve_warmup_prompt_config(
        {
            "path": "warmups/claude.yaml",
            "reasoning_effort": "minimal",
            "max_completion_tokens": 32,
        },
        base_dir=tmp_path,
    )

    assert resolved == {
        "messages": [
            {"role": "developer", "content": "You are Claude Code."},
            {"role": "user", "content": "Reply with OK."},
        ],
        "reasoning_effort": "minimal",
        "max_completion_tokens": 32,
    }


def test_build_warmup_query_applies_defaults(tmp_path):
    query = build_warmup_query(
        model_name="claude-code-model",
        warmup_prompt={
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with OK.",
                }
            ]
        },
        base_dir=tmp_path,
    )

    assert query.model == "claude-code-model"
    assert query.stream is False
    assert query.max_tokens == 64
    assert query.messages[0].content == "Reply with OK."
