import importlib.util
import os
import sys
from types import SimpleNamespace


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODULE_PATH = os.path.join(ROOT_DIR, "examples", "mcp", "server.py")
MODULE_SPEC = importlib.util.spec_from_file_location("examples_mcp_server_test_module", MODULE_PATH)
MCP_SERVER_MODULE = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = MCP_SERVER_MODULE
MODULE_SPEC.loader.exec_module(MCP_SERVER_MODULE)

_append_domain_filters_to_query = MCP_SERVER_MODULE._append_domain_filters_to_query
_call_unified_search = MCP_SERVER_MODULE._call_unified_search
_get_provider_attempt_order = MCP_SERVER_MODULE._get_provider_attempt_order
_record_provider_use = MCP_SERVER_MODULE._record_provider_use


def _settings(tmp_path, **overrides):
    defaults = {
        "search_usage_file": str(tmp_path / "search_usage.json"),
        "search_provider_order": ("exa", "tavily", "brave"),
        "exa_api_key": "exa-key",
        "tavily_api_key": "tavily-key",
        "brave_api_key": "brave-key",
        "exa_monthly_request_limit": None,
        "tavily_monthly_request_limit": None,
        "brave_monthly_request_limit": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_provider_attempt_order_rotates_after_last_success(tmp_path):
    settings = _settings(tmp_path)

    _record_provider_use(settings, "exa")

    attempt_order = _get_provider_attempt_order(settings, preferred_provider="auto")

    assert attempt_order == ["tavily", "brave", "exa"]


def test_provider_attempt_order_skips_provider_at_monthly_limit(tmp_path):
    settings = _settings(tmp_path, exa_monthly_request_limit=1)

    _record_provider_use(settings, "exa")

    attempt_order = _get_provider_attempt_order(settings, preferred_provider="auto")

    assert attempt_order == ["tavily", "brave"]


def test_append_domain_filters_to_query_formats_site_filters():
    query = _append_domain_filters_to_query(
        query="python mcp server",
        include_domains=["docs.python.org", "exa.ai"],
        exclude_domains=["reddit.com"],
    )

    assert query == "python mcp server site:docs.python.org site:exa.ai -site:reddit.com"


def test_unified_search_returns_provider_and_usage_metadata(tmp_path, monkeypatch):
    settings = _settings(tmp_path, tavily_api_key="", brave_api_key="")

    def fake_exa_search(**kwargs):
        return {
            "provider": "exa",
            "query": kwargs["query"],
            "results": [],
            "source": "Exa Search API",
        }

    monkeypatch.setattr(MCP_SERVER_MODULE, "_call_exa_search", fake_exa_search)

    result = _call_unified_search(query="latest python release", settings=settings)

    assert result["provider"] == "exa"
    assert result["provider_request_count_this_month"] == 1
    assert "provider_monthly_limit" in result
