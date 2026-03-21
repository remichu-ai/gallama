from __future__ import annotations

from copy import deepcopy
from math import ceil, floor
from typing import Any

_NUMERIC_BOUND_KEYS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
)
_MAX_INTEGER_ENUM_SPAN = 256


def normalize_json_schema_for_formatron(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize JSON Schema before handing it to Formatron.

    Formatron's JSON generator crashes on numeric bound metadata produced by
    Pydantic/JSON Schema, such as integer `minimum`/`maximum` constraints. To
    keep structured generation working:
    - small bounded integer ranges are rewritten as explicit `enum` values
    - unsupported numeric constraints are otherwise removed

    The schema is deep-copied so request data is not mutated in place.
    """
    normalized = deepcopy(schema)
    return _normalize_node(normalized)


def _normalize_node(node: Any) -> Any:
    if isinstance(node, dict):
        for key, value in list(node.items()):
            node[key] = _normalize_node(value)

        _normalize_numeric_constraints(node)
        return node

    if isinstance(node, list):
        return [_normalize_node(item) for item in node]

    return node


def _normalize_numeric_constraints(node: dict[str, Any]) -> None:
    node_types = _get_node_types(node)
    if not node_types or not any(key in node for key in _NUMERIC_BOUND_KEYS):
        return

    if "integer" in node_types and node_types.issubset({"integer", "null"}):
        # Preserve exact semantics when the bounded integer range is small enough
        # to represent directly. This avoids losing constraints for common cases
        # like 1..5 ratings while still sidestepping Formatron's metadata crash.
        integer_enum = _bounded_integer_enum(node, allow_null="null" in node_types)
        if integer_enum is not None:
            node["enum"] = integer_enum

    # Remove the original numeric metadata because Formatron cannot compile it
    # into its JSON grammar.
    for key in _NUMERIC_BOUND_KEYS:
        node.pop(key, None)


def _get_node_types(node: dict[str, Any]) -> set[str]:
    node_type = node.get("type")
    if isinstance(node_type, str):
        return {node_type}
    if isinstance(node_type, list):
        return {item for item in node_type if isinstance(item, str)}
    return set()


def _bounded_integer_enum(node: dict[str, Any], allow_null: bool) -> list[int | None] | None:
    minimum = node.get("minimum")
    maximum = node.get("maximum")
    exclusive_minimum = node.get("exclusiveMinimum")
    exclusive_maximum = node.get("exclusiveMaximum")

    lower_bound = None
    upper_bound = None

    if minimum is not None:
        lower_bound = ceil(minimum)
    elif exclusive_minimum is not None:
        lower_bound = floor(exclusive_minimum) + 1

    if maximum is not None:
        upper_bound = floor(maximum)
    elif exclusive_maximum is not None:
        upper_bound = ceil(exclusive_maximum) - 1

    if lower_bound is None or upper_bound is None or lower_bound > upper_bound:
        return None

    if (upper_bound - lower_bound + 1) > _MAX_INTEGER_ENUM_SPAN:
        # Large ranges would bloat the schema if converted to enum, so in those
        # cases we fall back to keeping only the base type.
        return None

    enum_values: list[int | None] = list(range(lower_bound, upper_bound + 1))
    if allow_null:
        enum_values.append(None)
    return enum_values
