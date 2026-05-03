import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from gallama.data_classes import ChatMLQuery, GenQueue
from gallama.logger import logger
from gallama.logger.logger import basic_log_extra


LEGACY_WARMUP_PROMPT = "Write a 500 words story on Llama"
DEFAULT_WARMUP_MAX_TOKENS = 64


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_warmup_mapping(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Warmup prompt file '{path}' must contain a mapping")
    return data


def resolve_warmup_prompt_config(
    warmup_prompt: Any,
    base_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    if warmup_prompt is None or warmup_prompt is False:
        return None
    if warmup_prompt is True:
        return None
    if not isinstance(warmup_prompt, dict):
        raise ValueError("'warmup_prompt' must be a mapping, boolean, or null")

    prompt_config = copy.deepcopy(warmup_prompt)
    prompt_path = prompt_config.pop("path", None)
    if prompt_path is None:
        return prompt_config

    path = Path(prompt_path).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()

    file_config = _load_warmup_mapping(path)
    return _deep_merge_dicts(file_config, prompt_config)


def build_warmup_query(
    *,
    model_name: str,
    warmup_prompt: Any,
    base_dir: Optional[Path] = None,
) -> Optional[ChatMLQuery]:
    resolved = resolve_warmup_prompt_config(warmup_prompt, base_dir=base_dir)
    if resolved is None:
        return None

    query_payload = copy.deepcopy(resolved)
    query_payload.setdefault("model", model_name)
    query_payload["stream"] = False
    if "max_tokens" not in query_payload and "max_completion_tokens" not in query_payload:
        query_payload["max_completion_tokens"] = DEFAULT_WARMUP_MAX_TOKENS

    return ChatMLQuery.model_validate(query_payload)


async def warmup_llm(
    *,
    model: Any,
    model_name: str,
    warmup_prompt: Any,
    gen_queue: GenQueue,
    base_dir: Optional[Path] = None,
) -> None:
    query = build_warmup_query(
        model_name=model_name,
        warmup_prompt=warmup_prompt,
        base_dir=base_dir,
    )

    if query is None:
        await model.chat_raw(
            prompt=LEGACY_WARMUP_PROMPT,
            max_tokens=50,
            gen_queue=gen_queue,
            quiet=True,
            request=None,
        )
        return

    await model.chat(
        query=query,
        prompt_eng=model.prompt_eng,
        gen_queue=gen_queue,
        request=None,
    )
    logger.info(f"LLM| {model_name} | warmup query executed", extra=basic_log_extra())
