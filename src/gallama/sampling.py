from __future__ import annotations

from typing import Any, Literal, Optional

from .data_classes import ChatMLQuery


SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "seed",
)


def resolve_sampling_overrides(
    query: ChatMLQuery,
    default_sampling: list,
    condition: Optional[Literal["thinking"]] = None,
) -> dict[str, Any]:
    default_rule = next((rule for rule in default_sampling if rule.condition is None), None)
    conditioned_rule = next((rule for rule in default_sampling if rule.condition == condition), None)
    resolved: dict[str, Any] = {}

    for field in SAMPLING_FIELDS:
        query_value = getattr(query, field, None)
        if field in query.model_fields_set and query_value is not None:
            resolved[field] = query_value
            continue

        conditioned_value = getattr(conditioned_rule, field, None) if conditioned_rule else None
        if conditioned_value is not None:
            resolved[field] = conditioned_value
            continue

        default_value = getattr(default_rule, field, None) if default_rule else None
        if default_value is not None:
            resolved[field] = default_value
            continue

        if query_value is not None:
            resolved[field] = query_value
            continue

        field_default = query.__class__.model_fields[field].default
        if field_default is not None:
            resolved[field] = field_default

    return resolved
