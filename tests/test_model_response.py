from gallama.api_response.model_response import build_models_response, wants_anthropic_models_response


def test_wants_anthropic_models_response_detects_anthropic_headers_case_insensitively():
    headers = {
        "Anthropic-Version": "2023-06-01",
    }

    assert wants_anthropic_models_response(headers) is True


def test_build_models_response_returns_openai_shape_by_default():
    response = build_models_response(["model-a", "model-b"], {})
    payload = response.model_dump()

    assert payload == {
        "object": "list",
        "data": [
            {
                "id": "model-a",
                "object": "model",
                "owned_by": "remichu",
                "created_by": 1686935002,
            },
            {
                "id": "model-b",
                "object": "model",
                "owned_by": "remichu",
                "created_by": 1686935002,
            },
        ],
    }


def test_build_models_response_returns_anthropic_shape_when_requested():
    response = build_models_response(
        ["claude-local-a", "claude-local-b"],
        {"anthropic-version": "2023-06-01"},
    )
    payload = response.model_dump()

    assert payload["first_id"] == "claude-local-a"
    assert payload["last_id"] == "claude-local-b"
    assert payload["has_more"] is False
    assert payload["data"] == [
        {
            "id": "claude-local-a",
            "type": "model",
            "display_name": "claude-local-a",
            "created_at": "2023-06-16T19:03:22Z",
        },
        {
            "id": "claude-local-b",
            "type": "model",
            "display_name": "claude-local-b",
            "created_at": "2023-06-16T19:03:22Z",
        },
    ]


def test_build_models_response_returns_empty_anthropic_list_shape():
    response = build_models_response([], {"anthropic-beta": "tools-2024-04-04"})
    payload = response.model_dump()

    assert payload == {
        "data": [],
        "first_id": None,
        "has_more": False,
        "last_id": None,
    }
