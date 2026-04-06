from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "gallama" / "backend" / "llm" / "json_schema_utils.py"
_SPEC = spec_from_file_location("json_schema_utils", _MODULE_PATH)
_MODULE = module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

normalize_json_schema_for_formatron = _MODULE.normalize_json_schema_for_formatron


def test_normalize_json_schema_for_formatron_converts_small_integer_ranges_to_enum():
    schema = {
        "type": "object",
        "properties": {
            "rating": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
            }
        },
        "required": ["rating"],
    }

    normalized = normalize_json_schema_for_formatron(schema)

    assert normalized["properties"]["rating"] == {
        "type": "integer",
        "enum": [1, 2, 3, 4, 5],
    }
    assert schema["properties"]["rating"]["minimum"] == 1
    assert schema["properties"]["rating"]["maximum"] == 5


def test_normalize_json_schema_for_formatron_strips_large_integer_ranges_without_enum():
    schema = {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000,
            }
        },
    }

    normalized = normalize_json_schema_for_formatron(schema)

    assert normalized["properties"]["count"] == {"type": "integer"}


def test_normalize_json_schema_for_formatron_handles_nullable_nested_integer_bounds():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": ["integer", "null"],
                    "exclusiveMinimum": 0,
                    "maximum": 2,
                },
            }
        },
    }

    normalized = normalize_json_schema_for_formatron(schema)

    assert normalized["properties"]["items"]["items"] == {
        "type": ["integer", "null"],
        "enum": [1, 2, None],
    }
