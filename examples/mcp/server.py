from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
from mcp.server.fastmcp import FastMCP


SCRIPT_DIR = Path(__file__).resolve().parent
ENV_CANDIDATES = (
    SCRIPT_DIR / ".env",
    SCRIPT_DIR / "experiment" / "mcp" / ".env",
)
DEFAULT_USAGE_STATE_PATH = Path(__file__).with_name(".search_provider_usage.json")
MAX_IMAGE_BYTES = 20 * 1024 * 1024
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
SEARCH_PROVIDER_NAMES = ("exa", "tavily", "brave")
SEARCH_STATE_LOCK = threading.Lock()


def load_dotenv(path: Path) -> bool:
    if not path.exists():
        return False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return True


ACTIVE_ENV_PATH = next((path for path in ENV_CANDIDATES if load_dotenv(path)), ENV_CANDIDATES[0])


@dataclass(frozen=True)
class Settings:
    exa_api_key: str
    exa_api_url: str
    exa_search_type: str
    exa_num_results: int
    exa_user_location: str
    exa_monthly_request_limit: int | None
    tavily_api_key: str
    tavily_api_url: str
    tavily_monthly_request_limit: int | None
    brave_api_key: str
    brave_api_url: str
    brave_country: str
    brave_search_lang: str
    brave_ui_lang: str
    brave_safesearch: str
    brave_monthly_request_limit: int | None
    search_provider_order: tuple[str, ...]
    search_usage_file: str
    local_vision_base_url: str
    local_vision_endpoint: str
    local_vision_api_key: str
    local_vision_model: str
    local_vision_timeout_seconds: float
    local_vision_fetch_remote: bool
    local_vision_system_prompt: str
    mcp_host: str
    mcp_port: int
    mcp_path: str


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int_optional(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None

    parsed = int(value.strip())
    return parsed if parsed > 0 else None


def _resolve_local_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return str(path)


def _parse_provider_order(value: str | None) -> tuple[str, ...]:
    ordered: list[str] = []
    raw_value = value or ",".join(SEARCH_PROVIDER_NAMES)

    for item in raw_value.split(","):
        provider = item.strip().lower()
        if provider in SEARCH_PROVIDER_NAMES and provider not in ordered:
            ordered.append(provider)

    for provider in SEARCH_PROVIDER_NAMES:
        if provider not in ordered:
            ordered.append(provider)

    return tuple(ordered)


def get_settings() -> Settings:
    default_country = (os.getenv("EXA_USER_LOCATION", "US").strip() or "US").upper()
    return Settings(
        exa_api_key=os.getenv("EXA_API_KEY", "").strip(),
        exa_api_url=os.getenv("EXA_API_URL", "https://api.exa.ai/search").strip(),
        exa_search_type=os.getenv("EXA_SEARCH_TYPE", "auto").strip() or "auto",
        exa_num_results=max(1, int(os.getenv("EXA_NUM_RESULTS", "5"))),
        exa_user_location=default_country,
        exa_monthly_request_limit=_env_int_optional("EXA_MONTHLY_REQUEST_LIMIT"),
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
        tavily_api_url=os.getenv("TAVILY_API_URL", "https://api.tavily.com/search").strip(),
        tavily_monthly_request_limit=_env_int_optional("TAVILY_MONTHLY_REQUEST_LIMIT"),
        brave_api_key=os.getenv("BRAVE_API_KEY", "").strip(),
        brave_api_url=os.getenv("BRAVE_API_URL", "https://api.search.brave.com/res/v1/web/search").strip(),
        brave_country=(os.getenv("BRAVE_COUNTRY", default_country).strip() or default_country).upper(),
        brave_search_lang=os.getenv("BRAVE_SEARCH_LANG", "en").strip() or "en",
        brave_ui_lang=os.getenv("BRAVE_UI_LANG", "en-US").strip() or "en-US",
        brave_safesearch=os.getenv("BRAVE_SAFESEARCH", "moderate").strip() or "moderate",
        brave_monthly_request_limit=_env_int_optional("BRAVE_MONTHLY_REQUEST_LIMIT"),
        search_provider_order=_parse_provider_order(os.getenv("SEARCH_PROVIDER_ORDER")),
        search_usage_file=_resolve_local_path(
            os.getenv("SEARCH_USAGE_FILE", str(DEFAULT_USAGE_STATE_PATH.name)).strip()
            or str(DEFAULT_USAGE_STATE_PATH.name)
        ),
        local_vision_base_url=os.getenv("LOCAL_VISION_BASE_URL", "http://127.0.0.1:8000/v1").strip(),
        local_vision_endpoint=os.getenv("LOCAL_VISION_ENDPOINT", "/chat/completions").strip()
        or "/chat/completions",
        local_vision_api_key=os.getenv("LOCAL_VISION_API_KEY", "test").strip(),
        local_vision_model=os.getenv("LOCAL_VISION_MODEL", "").strip(),
        local_vision_timeout_seconds=float(os.getenv("LOCAL_VISION_TIMEOUT_SECONDS", "120")),
        local_vision_fetch_remote=_env_bool("LOCAL_VISION_FETCH_REMOTE", True),
        local_vision_system_prompt=os.getenv(
            "LOCAL_VISION_SYSTEM_PROMPT",
            (
                "You are a precise vision assistant. "
                "Answer only from visible evidence in the image. "
                "If something is uncertain or unreadable, say so explicitly."
            ),
        ).strip(),
        mcp_host=os.getenv("MCP_HOST", "0.0.0.0").strip() or "0.0.0.0",
        mcp_port=int(os.getenv("MCP_PORT", "18011")),
        mcp_path=os.getenv("MCP_PATH", "/mcp").strip() or "/mcp",
    )


def _require_value(value: str, env_name: str) -> str:
    if value:
        return value
    raise RuntimeError(f"{env_name} is not set. Update {ACTIVE_ENV_PATH}.")


def _is_remote_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _is_data_url(value: str) -> bool:
    return value.startswith("data:image/")


def _guess_mime_type(path_value: str, content_type: str | None = None) -> str:
    if content_type:
        mime_type = content_type.split(";", 1)[0].strip().lower()
        if mime_type.startswith("image/"):
            return mime_type

    guessed_type, _ = mimetypes.guess_type(path_value)
    if guessed_type and guessed_type.startswith("image/"):
        return guessed_type

    raise ValueError(f"Unsupported image type for {path_value!r}. Use JPEG, PNG, GIF, or WebP.")


def _validate_image_size(size_bytes: int) -> None:
    if size_bytes > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image is too large ({size_bytes} bytes). MiniMax-like MCP behavior only supports images up to 20MB."
        )


def _to_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _load_local_image_as_data_url(image_path: str) -> tuple[str, dict[str, Any]]:
    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Local image path does not exist: {image_path}")
    if not path.is_file():
        raise ValueError(f"Local image path is not a file: {image_path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError("Local image must be JPEG, PNG, GIF, or WebP.")

    image_bytes = path.read_bytes()
    _validate_image_size(len(image_bytes))
    mime_type = _guess_mime_type(str(path))
    return _to_data_url(image_bytes, mime_type), {
        "input_mode": "local_file",
        "resolved_path": str(path.resolve()),
        "mime_type": mime_type,
        "size_bytes": len(image_bytes),
    }


def _download_remote_image_as_data_url(image_url: str, timeout_seconds: float) -> tuple[str, dict[str, Any]]:
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        response = client.get(image_url)
        response.raise_for_status()
        image_bytes = response.content
        _validate_image_size(len(image_bytes))
        mime_type = _guess_mime_type(
            image_url,
            content_type=response.headers.get("content-type"),
        )
        return _to_data_url(image_bytes, mime_type), {
            "input_mode": "remote_url_fetched",
            "resolved_url": str(response.url),
            "mime_type": mime_type,
            "size_bytes": len(image_bytes),
        }


def _resolve_image_payload(image_url: str, settings: Settings) -> tuple[str, dict[str, Any]]:
    if _is_data_url(image_url):
        return image_url, {"input_mode": "data_url"}

    if _is_remote_url(image_url):
        if settings.local_vision_fetch_remote:
            try:
                return _download_remote_image_as_data_url(
                    image_url,
                    timeout_seconds=settings.local_vision_timeout_seconds,
                )
            except httpx.HTTPError as exc:
                return image_url, {
                    "input_mode": "remote_url_direct_fallback",
                    "resolved_url": image_url,
                    "warning": f"Remote image fetch failed: {exc}",
                }
        return image_url, {"input_mode": "remote_url_direct", "resolved_url": image_url}

    return _load_local_image_as_data_url(image_url)


def _extract_chat_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("Local vision model returned no choices.")

    message = (choices[0] or {}).get("message") or {}
    content = message.get("content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                text = item.get("text", "")
                if text:
                    parts.append(text)
        joined = "".join(parts).strip()
        if joined:
            return joined

    raise RuntimeError("Unable to extract text content from local vision model response.")


def _call_local_vision_model(prompt: str, image_payload: str, settings: Settings) -> dict[str, Any]:
    base_url = _require_value(settings.local_vision_base_url, "LOCAL_VISION_BASE_URL")
    model = _require_value(settings.local_vision_model, "LOCAL_VISION_MODEL")
    endpoint = settings.local_vision_endpoint
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"

    url = f"{base_url.rstrip('/')}{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.local_vision_api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": settings.local_vision_system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_payload}},
                ],
            },
        ],
        "temperature": 0.2,
    }

    with httpx.Client(timeout=settings.local_vision_timeout_seconds) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

    return {
        "model": model,
        "answer": _extract_chat_text(result),
        "usage": result.get("usage"),
        "source": "local openai-compatible vision llm",
    }


def _compact_exa_result(result: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "title": result.get("title"),
        "url": result.get("url"),
        "published_date": result.get("publishedDate"),
        "author": result.get("author"),
        "summary": result.get("summary"),
        "text": result.get("text"),
        "highlights": result.get("highlights") or [],
    }
    if result.get("favicon"):
        compact["favicon"] = result.get("favicon")
    return compact


def _build_related_suggestions(query: str, results: list[dict[str, Any]]) -> list[str]:
    suggestions: list[str] = []
    seen: set[str] = set()

    for result in results[:5]:
        title = (result.get("title") or "").strip()
        if title and title.lower() != query.strip().lower() and title.lower() not in seen:
            suggestions.append(title)
            seen.add(title.lower())

    if not suggestions:
        suggestions.extend(
            [
                f"{query} official documentation",
                f"{query} latest updates",
                f"{query} examples",
            ]
        )

    return suggestions[:5]


def _current_month_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _empty_usage_state() -> dict[str, Any]:
    return {
        "month": _current_month_key(),
        "last_provider": None,
        "counts": {provider: 0 for provider in SEARCH_PROVIDER_NAMES},
    }


def _load_usage_state(settings: Settings) -> dict[str, Any]:
    path = Path(settings.search_usage_file)
    state = _empty_usage_state()

    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                state.update({key: loaded.get(key, state[key]) for key in ("month", "last_provider")})
                loaded_counts = loaded.get("counts") or {}
                if isinstance(loaded_counts, dict):
                    for provider in SEARCH_PROVIDER_NAMES:
                        raw_count = loaded_counts.get(provider, 0)
                        state["counts"][provider] = raw_count if isinstance(raw_count, int) and raw_count >= 0 else 0
        except (OSError, json.JSONDecodeError):
            state = _empty_usage_state()

    if state["month"] != _current_month_key():
        state = _empty_usage_state()

    return state


def _save_usage_state(settings: Settings, state: dict[str, Any]) -> None:
    path = Path(settings.search_usage_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _get_provider_limit(settings: Settings, provider: Literal["exa", "tavily", "brave"]) -> int | None:
    if provider == "exa":
        return settings.exa_monthly_request_limit
    if provider == "tavily":
        return settings.tavily_monthly_request_limit
    return settings.brave_monthly_request_limit


def _provider_is_configured(settings: Settings, provider: Literal["exa", "tavily", "brave"]) -> bool:
    if provider == "exa":
        return bool(settings.exa_api_key)
    if provider == "tavily":
        return bool(settings.tavily_api_key)
    return bool(settings.brave_api_key)


def _is_provider_available(
    settings: Settings,
    provider: Literal["exa", "tavily", "brave"],
    state: dict[str, Any],
) -> bool:
    if not _provider_is_configured(settings, provider):
        return False

    limit = _get_provider_limit(settings, provider)
    if limit is None:
        return True

    return state["counts"].get(provider, 0) < limit


def _get_provider_attempt_order(
    settings: Settings,
    *,
    preferred_provider: Literal["auto", "exa", "tavily", "brave"] = "auto",
) -> list[str]:
    with SEARCH_STATE_LOCK:
        state = _load_usage_state(settings)

    if preferred_provider != "auto":
        if not _provider_is_configured(settings, preferred_provider):
            raise RuntimeError(f"Search provider '{preferred_provider}' is not configured.")
        if not _is_provider_available(settings, preferred_provider, state):
            limit = _get_provider_limit(settings, preferred_provider)
            raise RuntimeError(
                f"Search provider '{preferred_provider}' reached its configured monthly limit ({limit})."
            )
        return [preferred_provider]

    available = [
        provider
        for provider in settings.search_provider_order
        if _is_provider_available(settings, provider, state)
    ]
    if not available:
        configured = [provider for provider in settings.search_provider_order if _provider_is_configured(settings, provider)]
        if not configured:
            raise RuntimeError(
                "No search providers are configured. Set at least one of EXA_API_KEY, TAVILY_API_KEY, or BRAVE_API_KEY."
            )
        raise RuntimeError(
            "All configured search providers reached their configured monthly limits. "
            "Increase the limits or wait for the next month."
        )

    last_provider = state.get("last_provider")
    if last_provider in available:
        start_index = (available.index(last_provider) + 1) % len(available)
        return available[start_index:] + available[:start_index]

    return available


def _record_provider_use(
    settings: Settings,
    provider: Literal["exa", "tavily", "brave"],
) -> dict[str, Any]:
    with SEARCH_STATE_LOCK:
        state = _load_usage_state(settings)
        state["month"] = _current_month_key()
        state["last_provider"] = provider
        state["counts"][provider] = int(state["counts"].get(provider, 0)) + 1
        _save_usage_state(settings, state)

        return {
            "month": state["month"],
            "provider_request_count_this_month": state["counts"][provider],
            "provider_monthly_limit": _get_provider_limit(settings, provider),
        }


def _normalize_string_list(values: list[str] | None) -> list[str] | None:
    if not values:
        return None

    normalized = [value.strip() for value in values if isinstance(value, str) and value.strip()]
    return normalized or None


def _map_tavily_search_depth(
    search_type: Literal["auto", "fast", "neural", "deep", "deep-reasoning", "instant"] | None,
) -> Literal["basic", "fast", "advanced", "ultra-fast"]:
    if search_type in {None, "auto"}:
        return "basic"
    if search_type == "instant":
        return "ultra-fast"
    if search_type == "fast":
        return "fast"
    return "advanced"


def _map_tavily_topic(
    category: Literal[
        "company",
        "research paper",
        "news",
        "personal site",
        "financial report",
        "people",
    ]
    | None,
) -> Literal["general", "news", "finance"]:
    if category == "news":
        return "news"
    if category == "financial report":
        return "finance"
    return "general"


def _append_domain_filters_to_query(
    query: str,
    include_domains: list[str] | None,
    exclude_domains: list[str] | None,
) -> str:
    parts = [query.strip()]
    for domain in include_domains or []:
        parts.append(f"site:{domain}")
    for domain in exclude_domains or []:
        parts.append(f"-site:{domain}")
    return " ".join(part for part in parts if part)


def _build_brave_freshness(
    start_published_date: str | None,
    end_published_date: str | None,
) -> str | None:
    if not start_published_date and not end_published_date:
        return None

    start = start_published_date or "1900-01-01"
    end = end_published_date or datetime.now(timezone.utc).date().isoformat()
    return f"{start}to{end}"


def _call_exa_search(
    query: str,
    settings: Settings,
    *,
    search_type: Literal["auto", "fast", "neural", "deep", "deep-reasoning", "instant"] | None = None,
    num_results: int | None = None,
    category: Literal[
        "company",
        "research paper",
        "news",
        "personal site",
        "financial report",
        "people",
    ]
    | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    start_published_date: str | None = None,
    end_published_date: str | None = None,
) -> dict[str, Any]:
    api_key = _require_value(settings.exa_api_key, "EXA_API_KEY")
    effective_search_type = search_type or settings.exa_search_type
    effective_num_results = settings.exa_num_results if num_results is None else max(1, min(num_results, 25))
    payload = {
        "query": query,
        "type": effective_search_type,
        "numResults": effective_num_results,
        "userLocation": settings.exa_user_location,
        "contents": {
            "highlights": {
                "maxCharacters": 1500,
            }
        },
    }
    normalized_include_domains = _normalize_string_list(include_domains)
    normalized_exclude_domains = _normalize_string_list(exclude_domains)
    if category:
        payload["category"] = category
    if normalized_include_domains:
        payload["includeDomains"] = normalized_include_domains
    if normalized_exclude_domains:
        payload["excludeDomains"] = normalized_exclude_domains
    if start_published_date:
        payload["startPublishedDate"] = start_published_date
    if end_published_date:
        payload["endPublishedDate"] = end_published_date
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }

    with httpx.Client(timeout=60) as client:
        response = client.post(settings.exa_api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

    compact_results = [_compact_exa_result(item) for item in result.get("results", [])]
    return {
        "provider": "exa",
        "query": query,
        "search_type": result.get("searchType", effective_search_type),
        "category": category,
        "num_results": effective_num_results,
        "include_domains": normalized_include_domains or [],
        "exclude_domains": normalized_exclude_domains or [],
        "start_published_date": start_published_date,
        "end_published_date": end_published_date,
        "request_id": result.get("requestId"),
        "results": compact_results,
        "related_suggestions": _build_related_suggestions(query, result.get("results", [])),
        "source": "Exa Search API",
    }


def _compact_tavily_result(result: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "title": result.get("title"),
        "url": result.get("url"),
        "summary": result.get("content"),
        "text": result.get("raw_content") or result.get("content"),
        "highlights": [],
        "score": result.get("score"),
    }
    if result.get("favicon"):
        compact["favicon"] = result.get("favicon")
    return compact


def _call_tavily_search(
    query: str,
    settings: Settings,
    *,
    search_type: Literal["auto", "fast", "neural", "deep", "deep-reasoning", "instant"] | None = None,
    num_results: int | None = None,
    category: Literal[
        "company",
        "research paper",
        "news",
        "personal site",
        "financial report",
        "people",
    ]
    | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    start_published_date: str | None = None,
    end_published_date: str | None = None,
) -> dict[str, Any]:
    api_key = _require_value(settings.tavily_api_key, "TAVILY_API_KEY")
    normalized_include_domains = _normalize_string_list(include_domains)
    normalized_exclude_domains = _normalize_string_list(exclude_domains)
    effective_num_results = settings.exa_num_results if num_results is None else max(1, min(num_results, 20))
    search_depth = _map_tavily_search_depth(search_type)
    topic = _map_tavily_topic(category)
    payload = {
        "query": query,
        "search_depth": search_depth,
        "max_results": effective_num_results,
        "topic": topic,
        "include_favicon": True,
        "include_usage": True,
    }
    if normalized_include_domains:
        payload["include_domains"] = normalized_include_domains
    if normalized_exclude_domains:
        payload["exclude_domains"] = normalized_exclude_domains
    if start_published_date:
        payload["start_date"] = start_published_date
    if end_published_date:
        payload["end_date"] = end_published_date

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    with httpx.Client(timeout=60) as client:
        response = client.post(settings.tavily_api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

    compact_results = [_compact_tavily_result(item) for item in result.get("results", [])]
    return {
        "provider": "tavily",
        "query": result.get("query", query),
        "search_type": search_depth,
        "category": topic,
        "num_results": effective_num_results,
        "include_domains": normalized_include_domains or [],
        "exclude_domains": normalized_exclude_domains or [],
        "start_published_date": start_published_date,
        "end_published_date": end_published_date,
        "request_id": result.get("request_id"),
        "usage": result.get("usage"),
        "answer": result.get("answer"),
        "results": compact_results,
        "related_suggestions": _build_related_suggestions(query, result.get("results", [])),
        "source": "Tavily Search API",
    }


def _compact_brave_result(result: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "title": result.get("title"),
        "url": result.get("url"),
        "published_date": result.get("page_age") or result.get("age"),
        "summary": result.get("description"),
        "highlights": result.get("extra_snippets") or [],
    }
    meta_url = result.get("meta_url") or {}
    if isinstance(meta_url, dict) and meta_url.get("favicon"):
        compact["favicon"] = meta_url.get("favicon")
    return compact


def _call_brave_search(
    query: str,
    settings: Settings,
    *,
    num_results: int | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    start_published_date: str | None = None,
    end_published_date: str | None = None,
) -> dict[str, Any]:
    api_key = _require_value(settings.brave_api_key, "BRAVE_API_KEY")
    normalized_include_domains = _normalize_string_list(include_domains)
    normalized_exclude_domains = _normalize_string_list(exclude_domains)
    effective_num_results = settings.exa_num_results if num_results is None else max(1, min(num_results, 20))
    effective_query = _append_domain_filters_to_query(
        query=query,
        include_domains=normalized_include_domains,
        exclude_domains=normalized_exclude_domains,
    )
    params = {
        "q": effective_query,
        "count": effective_num_results,
        "extra_snippets": "true",
        "country": settings.brave_country,
        "search_lang": settings.brave_search_lang,
        "ui_lang": settings.brave_ui_lang,
        "safesearch": settings.brave_safesearch,
    }
    freshness = _build_brave_freshness(
        start_published_date=start_published_date,
        end_published_date=end_published_date,
    )
    if freshness:
        params["freshness"] = freshness

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    with httpx.Client(timeout=60) as client:
        response = client.get(settings.brave_api_url, headers=headers, params=params)
        response.raise_for_status()
        result = response.json()

    web_results = ((result.get("web") or {}).get("results")) or []
    compact_results = [_compact_brave_result(item) for item in web_results]
    return {
        "provider": "brave",
        "query": ((result.get("query") or {}).get("original")) or query,
        "search_type": "web",
        "category": None,
        "num_results": effective_num_results,
        "include_domains": normalized_include_domains or [],
        "exclude_domains": normalized_exclude_domains or [],
        "start_published_date": start_published_date,
        "end_published_date": end_published_date,
        "more_results_available": bool((result.get("query") or {}).get("more_results_available")),
        "results": compact_results,
        "related_suggestions": _build_related_suggestions(query, web_results),
        "source": "Brave Search API",
    }


def _call_unified_search(
    query: str,
    settings: Settings,
    *,
    provider: Literal["auto", "exa", "tavily", "brave"] = "auto",
    search_type: Literal["auto", "fast", "neural", "deep", "deep-reasoning", "instant"] | None = None,
    num_results: int | None = None,
    category: Literal[
        "company",
        "research paper",
        "news",
        "personal site",
        "financial report",
        "people",
    ]
    | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    start_published_date: str | None = None,
    end_published_date: str | None = None,
) -> dict[str, Any]:
    attempt_order = _get_provider_attempt_order(settings, preferred_provider=provider)
    errors: list[str] = []

    for selected_provider in attempt_order:
        try:
            if selected_provider == "exa":
                result = _call_exa_search(
                    query=query,
                    settings=settings,
                    search_type=search_type,
                    num_results=num_results,
                    category=category,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                    start_published_date=start_published_date,
                    end_published_date=end_published_date,
                )
            elif selected_provider == "tavily":
                result = _call_tavily_search(
                    query=query,
                    settings=settings,
                    search_type=search_type,
                    num_results=num_results,
                    category=category,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                    start_published_date=start_published_date,
                    end_published_date=end_published_date,
                )
            else:
                result = _call_brave_search(
                    query=query,
                    settings=settings,
                    num_results=num_results,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                    start_published_date=start_published_date,
                    end_published_date=end_published_date,
                )

            usage_state = _record_provider_use(settings, selected_provider)
            result.update(usage_state)
            return result
        except Exception as exc:
            errors.append(f"{selected_provider}: {exc}")
            if provider != "auto":
                break

    raise RuntimeError(
        "All search providers failed. " + " | ".join(errors)
    )


def build_server(host: str, port: int, path: str) -> FastMCP:
    server = FastMCP(
        name="local-coding-plan",
        host=host,
        port=port,
        streamable_http_path=path,
    )

    @server.tool(
        name="web_search",
        description=(
            "Performs web searches through a unified provider wrapper over Exa, Tavily, and Brave. "
            "Automatically rotates providers in auto mode and reports which provider was used."
        ),
    )
    def web_search(
        query: str,
        provider: Literal["auto", "exa", "tavily", "brave"] = "auto",
        search_type: Literal["auto", "fast", "neural", "deep", "deep-reasoning", "instant"] | None = None,
        num_results: int | None = None,
        category: Literal[
            "company",
            "research paper",
            "news",
            "personal site",
            "financial report",
            "people",
        ]
        | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        start_published_date: str | None = None,
        end_published_date: str | None = None,
    ) -> str:
        settings = get_settings()
        result = _call_unified_search(
            query=query,
            settings=settings,
            provider=provider,
            search_type=search_type,
            num_results=num_results,
            category=category,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            start_published_date=start_published_date,
            end_published_date=end_published_date,
        )
        return json.dumps(result, ensure_ascii=False)

    @server.tool(
        name="understand_image",
        description=(
            "Performs image understanding and analysis, supporting HTTP/HTTPS URLs or local file paths."
        ),
    )
    def understand_image(prompt: str, image_url: str) -> str:
        settings = get_settings()
        image_payload, image_metadata = _resolve_image_payload(image_url=image_url, settings=settings)
        result = _call_local_vision_model(prompt=prompt, image_payload=image_payload, settings=settings)
        result.update(
            {
                "prompt": prompt,
                "image_url": image_url,
                "image_input": image_metadata,
            }
        )
        return json.dumps(result, ensure_ascii=False)

    return server


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Run a MiniMax-like MCP server backed by Exa search and a local vision LLM."
    )
    parser.add_argument("--host", default=settings.mcp_host)
    parser.add_argument("--port", type=int, default=settings.mcp_port)
    parser.add_argument("--path", default=settings.mcp_path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = build_server(host=args.host, port=args.port, path=args.path)
    print(f"Local MCP server listening on http://{args.host}:{args.port}{args.path}")
    print(f"Loaded env from {ACTIVE_ENV_PATH}")
    try:
        server.run(transport="streamable-http")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
