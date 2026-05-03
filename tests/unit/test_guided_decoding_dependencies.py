import ast
import os


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
FORMAT_ENFORCER_PATH = os.path.join(
    ROOT_DIR,
    "src",
    "gallama",
    "backend",
    "llm",
    "format_enforcer.py",
)


def _load_format_enforcer_symbols():
    with open(FORMAT_ENFORCER_PATH, encoding="utf-8") as f:
        source = f.read()

    module_ast = ast.parse(source, filename=FORMAT_ENFORCER_PATH)
    selected_nodes = [
        node
        for node in module_ast.body
        if isinstance(node, ast.ClassDef) and node.name in {"SGLangFormatter", "FormatEnforcer"}
    ]

    namespace = {
        "json": __import__("json"),
        "logger": type("Logger", (), {"info": staticmethod(lambda *_args, **_kwargs: None)})(),
        "Tools": type(
            "ToolsStub",
            (),
            {"replace_refs_with_definitions_v2": staticmethod(lambda schema: schema)},
        ),
        "BaseModel": object,
        "FormatterBuilder": None,
        "ClassSchema": None,
        "JsonSchemaParser": None,
        "RegexParser": None,
        "TokenEnforcerTokenizerData": None,
        "GuidedDecodingParams": None,
        "Literal": __import__("typing").Literal,
        "Dict": __import__("typing").Dict,
        "Union": __import__("typing").Union,
        "Optional": __import__("typing").Optional,
        "Any": __import__("typing").Any,
        "FilterEngine": __import__("typing").Literal["formatron", "lm-format-enforcer", "sglang_formatter"],
        "FilterEngineOption": __import__("typing").Union[
            __import__("typing").Literal["formatron", "lm-format-enforcer", "sglang_formatter"],
            __import__("typing").Literal["auto"],
        ],
    }
    exec(compile(ast.Module(body=selected_nodes, type_ignores=[]), FORMAT_ENFORCER_PATH, "exec"), namespace)
    return namespace


def test_exllama_auto_prefers_formatron_when_available():
    namespace = _load_format_enforcer_symbols()
    namespace["FormatterBuilder"] = type("FormatterBuilder", (), {})
    namespace["ClassSchema"] = type("ClassSchema", (), {})
    namespace["JsonSchemaParser"] = type("JsonSchemaParser", (), {})
    namespace["RegexParser"] = type("RegexParser", (), {})
    namespace["TokenEnforcerTokenizerData"] = type("TokenEnforcerTokenizerData", (), {})

    format_enforcer = namespace["FormatEnforcer"]

    assert format_enforcer.get_default_engine(backend="exllama", preference="auto") == "formatron"


def test_llama_cpp_requires_lmfe_extra_when_missing():
    namespace = _load_format_enforcer_symbols()
    format_enforcer = namespace["FormatEnforcer"]

    try:
        format_enforcer.get_default_engine(backend="llama_cpp", preference="auto")
    except RuntimeError as exc:
        assert "gallama[llama-cpp]" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when lm-format-enforcer is unavailable")


def test_exllamav3_rejects_non_formatron_backend_choice():
    namespace = _load_format_enforcer_symbols()
    namespace["FormatterBuilder"] = type("FormatterBuilder", (), {})
    namespace["ClassSchema"] = type("ClassSchema", (), {})
    format_enforcer = namespace["FormatEnforcer"]

    try:
        format_enforcer.get_default_engine(backend="exllamav3", preference="lm-format-enforcer")
    except RuntimeError as exc:
        assert "only supports" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for unsupported exllamav3 guided decoding engine")


def test_tools_formatron_import_is_guarded():
    tools_path = os.path.join(
        ROOT_DIR,
        "src",
        "gallama",
        "backend",
        "llm",
        "tools.py",
    )

    with open(tools_path, encoding="utf-8") as f:
        source = f.read()

    assert "try:\n    from formatron.schemas.pydantic import ClassSchema, Schema" in source
    assert "def require_formatron" in source
