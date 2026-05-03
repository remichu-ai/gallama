import pytest

from gallama.data_classes import BaseMessage, ChatMLQuery, ModelSpec
from gallama.sampling import resolve_sampling_overrides


def _query(**kwargs) -> ChatMLQuery:
    payload = {
        "model": "demo-model",
        "messages": [BaseMessage(role="user", content="hello")],
    }
    payload.update(kwargs)
    return ChatMLQuery.model_validate(payload)


def _default_sampling(*rules):
    model_spec = ModelSpec.model_validate(
        {
            "model_name": "demo-model",
            "model_id": "/tmp/demo-model",
            "backend": "transformers",
            "default_sampling": list(rules),
        }
    )
    return model_spec.default_sampling


def test_model_spec_accepts_default_sampling_rules():
    model_spec = ModelSpec.model_validate(
        {
            "model_name": "demo-model",
            "model_id": "/tmp/demo-model",
            "backend": "transformers",
            "default_sampling": [
                {"temperature": 0.7, "top_p": 0.85, "top_k": 20},
                {"condition": "thinking", "temperature": 1.0, "presence_penalty": 1.5},
            ],
        }
    )

    assert len(model_spec.default_sampling) == 2
    assert model_spec.default_sampling[0].condition is None
    assert model_spec.default_sampling[0].top_k == 20
    assert model_spec.default_sampling[1].condition == "thinking"
    assert model_spec.default_sampling[1].presence_penalty == 1.5


def test_model_spec_rejects_duplicate_default_sampling_conditions():
    with pytest.raises(ValueError, match="Duplicate default_sampling rule"):
        ModelSpec.model_validate(
            {
                "model_name": "demo-model",
                "model_id": "/tmp/demo-model",
                "backend": "transformers",
                "default_sampling": [
                    {"temperature": 0.7},
                    {"top_p": 0.9},
                ],
            }
        )


def test_default_sampling_applies_when_request_omits_fields():
    query = _query()
    resolved = resolve_sampling_overrides(
        query=query,
        default_sampling=_default_sampling(
            {
                "condition": None,
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                "presence_penalty": 0.5,
            }
        ),
        condition=None,
    )

    assert resolved["temperature"] == 0.2
    assert resolved["top_p"] == 0.9
    assert resolved["top_k"] == 40
    assert resolved["presence_penalty"] == 0.5


def test_condition_thinking_overrides_default_sampling():
    query = _query()
    resolved = resolve_sampling_overrides(
        query=query,
        default_sampling=_default_sampling(
            {"temperature": 0.2, "top_p": 0.9, "top_k": 20},
            {"condition": "thinking", "temperature": 1.0, "presence_penalty": 1.5},
        ),
        condition="thinking",
    )

    assert resolved["temperature"] == 1.0
    assert resolved["top_p"] == 0.9
    assert resolved["top_k"] == 20
    assert resolved["presence_penalty"] == 1.5


def test_request_sampling_overrides_yaml_per_field():
    query = _query(temperature=0.6, top_k=8)
    resolved = resolve_sampling_overrides(
        query=query,
        default_sampling=_default_sampling(
            {"temperature": 0.2, "top_p": 0.9, "top_k": 20},
            {"condition": "thinking", "temperature": 1.0, "top_p": 0.95},
        ),
        condition="thinking",
    )

    assert resolved["temperature"] == 0.6
    assert resolved["top_p"] == 0.95
    assert resolved["top_k"] == 8


def test_legacy_query_defaults_remain_when_model_has_no_sampling_rules():
    query = _query()
    resolved = resolve_sampling_overrides(
        query=query,
        default_sampling=[],
        condition=None,
    )

    assert resolved["temperature"] == 0.7
    assert resolved["top_p"] == 0.85
    assert resolved["presence_penalty"] == 0
