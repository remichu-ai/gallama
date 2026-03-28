from __future__ import annotations

from typing import Any

import anyio


def is_expected_disconnect_exception(exc: BaseException) -> bool:
    """Return True for request teardown exceptions that are safe to suppress."""
    return type(exc).__name__ in {"ClientDisconnect", "AssertionError"}


def format_exception_summary(exc: BaseException) -> str:
    message = str(exc).strip()
    exc_name = type(exc).__name__
    return f"{exc_name}: {message}" if message else exc_name


async def is_request_disconnected(request: Any, probe_timeout: float = 0.01) -> bool:
    """Safely probe HTTP disconnect state without tripping Starlette middleware assertions."""
    if request is None:
        return False

    if bool(getattr(request, "_is_disconnected", False)):
        return True

    receive = getattr(request, "receive", None)
    if not callable(receive):
        return False

    message: dict[str, Any] = {}

    try:
        with anyio.move_on_after(probe_timeout):
            message = await receive()
    except Exception as exc:
        if is_expected_disconnect_exception(exc):
            try:
                request._is_disconnected = True
            except Exception:
                pass
            return True
        raise

    if message.get("type") == "http.disconnect":
        try:
            request._is_disconnected = True
        except Exception:
            pass

    return bool(getattr(request, "_is_disconnected", False))
