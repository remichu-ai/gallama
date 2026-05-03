import ast
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_invalid_raise_patterns_do_not_exist_in_python_sources():
    repo_root = _repo_root()
    source_root = repo_root / "src" / "gallama"
    patterns = ('raise "', "raise NotImplemented(")

    offenders = []
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if pattern in text:
                offenders.append(f"{path.relative_to(repo_root)}: contains {pattern!r}")

    assert offenders == []


def test_ws_llm_stream_passes_openai_provider():
    path = _repo_root() / "src" / "gallama" / "routes" / "ws_llm.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    provider_value = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "process_generation_stream":
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and getattr(child.func, "id", None) == "chat_completion_response_stream":
                    for keyword in child.keywords:
                        if keyword.arg == "provider" and isinstance(keyword.value, ast.Constant):
                            provider_value = keyword.value.value
                            break

    assert provider_value == "openai"
