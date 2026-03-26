from collections.abc import Iterable, Mapping

from ..data_classes.data_class import (
    AnthropicModelListResponse,
    AnthropicModelObject,
    ModelObject,
    ModelObjectResponse,
)


def wants_anthropic_models_response(headers: Mapping[str, str]) -> bool:
    normalized_headers = {header.lower() for header in headers.keys()}
    return "anthropic-version" in normalized_headers or "anthropic-beta" in normalized_headers


def build_models_response(model_ids: Iterable[str], headers: Mapping[str, str]):
    model_ids = list(model_ids)

    if wants_anthropic_models_response(headers):
        data = [
            AnthropicModelObject(
                id=model_id,
                display_name=model_id,
            )
            for model_id in model_ids
        ]
        return AnthropicModelListResponse(
            data=data,
            first_id=model_ids[0] if model_ids else None,
            last_id=model_ids[-1] if model_ids else None,
            has_more=False,
        )

    return ModelObjectResponse(data=[ModelObject(id=model_id) for model_id in model_ids])
