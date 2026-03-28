from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from ..data_classes.server_dataclass import ModelInstanceInfo


VISION_CONTENT_TYPES = {"image", "image_url", "input_image"}


def request_requires_vision(payload: Any) -> bool:
    """Return True when the request body includes image content."""
    if isinstance(payload, list):
        return any(request_requires_vision(item) for item in payload)

    if not isinstance(payload, dict):
        return False

    payload_type = payload.get("type")
    if isinstance(payload_type, str) and payload_type in VISION_CONTENT_TYPES:
        return True

    image_url = payload.get("image_url")
    if isinstance(payload_type, str) and payload_type in {"image_url", "input_image"}:
        if isinstance(image_url, str) and image_url:
            return True
        if isinstance(image_url, dict) and any(
            image_url.get(key) for key in ("url", "data", "b64_json", "file_id")
        ):
            return True

    if isinstance(image_url, dict) and any(
        image_url.get(key) for key in ("url", "data", "b64_json", "file_id")
    ):
        return True

    source = payload.get("source")
    if isinstance(source, dict) and source.get("type") in {"base64", "url"}:
        media_type = str(source.get("media_type", "")).lower()
        if media_type.startswith("image/") or source.get("url") or source.get("data"):
            return True

    return any(request_requires_vision(value) for value in payload.values())


def instance_supports_vision(instance: "ModelInstanceInfo") -> bool:
    modalities = set(getattr(instance, "modalities", []) or [])
    return bool({"image", "video"} & modalities)


def prefer_vision_instances(
    instances: Iterable[ModelInstanceInfo],
    *,
    vision_required: bool,
) -> list[ModelInstanceInfo]:
    instance_list = list(instances)
    if not vision_required:
        return instance_list

    vision_instances = [instance for instance in instance_list if instance_supports_vision(instance)]
    return vision_instances or instance_list
