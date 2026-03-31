import asyncio
import json
import time
import base64
import os
import re
import shlex
import socket
import subprocess
import threading
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import httpx
from PIL import Image
from fastapi import HTTPException, Request

from ..base import ModelInterface
from gallama.data_classes import (
    BaseMessage,
    GenEnd,
    GenQueue,
    GenQueueDynamic,
    GenStart,
    GenText,
    GenerationStats,
    ModelSpec,
    QueueContext,
    VideoFrame,
)
from gallama.backend.llm.prompt_engine.model_special_tag import MODEL_VISION_TOKEN
from gallama.logger.logger import logger
from gallama.utils.request_disconnect import is_request_disconnected
from gallama.utils.utils import get_image


class ModelLlamaCppServer(ModelInterface):
    def __init__(self, model_spec: ModelSpec):
        self.server_process: Optional[subprocess.Popen] = None
        self.server_port: Optional[int] = None
        self.server_host: str = "127.0.0.1"
        self.server_log_thread: Optional[threading.Thread] = None
        self.server_log_mode = "startup"
        self.server_log_lock = threading.Lock()
        self.server_log_path: Optional[Path] = None
        self.server_log_handle = None
        super().__init__(model_spec)
        self.model, self.tokenizer = self.load_model()

    @property
    def support_concurrency(self) -> bool:
        return True

    @property
    def support_format_enforcer(self) -> bool:
        return False

    def load_model(self):
        self.auto_start = self.backend_extra_args.get("auto_start", False)
        self.start_up_cmd = self.backend_extra_args.get("start_up_cmd")
        self.start_up_extra = self.backend_extra_args.get("start_up_extra")
        self.start_up_cwd = self.backend_extra_args.get("start_up_cwd")
        self.healthcheck_timeout = float(self.backend_extra_args.get("healthcheck_timeout", 60))
        self.healthcheck_poll_interval = float(self.backend_extra_args.get("healthcheck_poll_interval", 0.5))
        self.completion_path = self.backend_extra_args.get("completion_path", "/completion")
        self.tokenize_path = self.backend_extra_args.get("tokenize_path", "/tokenize")
        self.detokenize_path = self.backend_extra_args.get("detokenize_path", "/detokenize")
        self.health_path = self.backend_extra_args.get("health_path", "/health")
        self.timeout = self.backend_extra_args.get("timeout")
        self.cache_prompt = self.backend_extra_args.get("cache_prompt", True)
        self.use_server_tokenizer = self.backend_extra_args.get("use_server_tokenizer", True)
        self.require_server_healthcheck = self.backend_extra_args.get("require_server_healthcheck", True)
        self.multimodal_marker = self.backend_extra_args.get("multimodal_marker")
        self.max_seq_len = self.max_seq_len or self.backend_extra_args.get("max_seq_len")

        if not self.use_server_tokenizer:
            raise ValueError("llama_cpp_server currently supports only use_server_tokenizer=true")

        if not self.auto_start and not self.backend_extra_args.get("base_url"):
            raise ValueError("backend_extra_args.base_url is required for llama_cpp_server unless auto_start=true")

        self._configure_base_url()
        self.backend_extra_args["base_url"] = self.base_url

        if self.auto_start:
            self._ensure_server_started()
        elif self.require_server_healthcheck:
            self._probe_server()

        self.modalities.add("image")

        return {"base_url": self.base_url}, None

    def _configure_base_url(self):
        parsed_base_url = self.backend_extra_args.get("base_url")
        if parsed_base_url:
            parsed = urlparse(parsed_base_url)
            self.server_host = parsed.hostname or "127.0.0.1"
            self.server_port = parsed.port or self._allocate_free_port()
            scheme = parsed.scheme or "http"
            self.base_url = f"{scheme}://{self.server_host}:{self.server_port}"
            return

        self.server_host = self.backend_extra_args.get("host", "127.0.0.1")
        self.server_port = int(self.backend_extra_args.get("port") or self._allocate_free_port())
        self.base_url = f"http://{self.server_host}:{self.server_port}"

    @staticmethod
    def _allocate_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return sock.getsockname()[1]

    def _build_startup_command(self) -> List[str]:
        if not self.start_up_cmd:
            raise ValueError("backend_extra_args.start_up_cmd is required when auto_start=true")

        cmd = [self.start_up_cmd]

        mapped_args: Dict[str, Any] = {
            "model": self.model_id,
            "host": self.server_host,
            "port": self.server_port,
            "ctx-size": self.max_seq_len,
        }

        optional_map = {
            "threads": "threads",
            "threads-batch": "threads_batch",
            "n-gpu-layers": "n_gpu_layers",
            "parallel": "parallel",
            "jinja": "jinja",
            "flash-attn": "flash_attn",
            "cont-batching": "cont_batching",
            "batch-size": "batch_size",
            "ubatch-size": "ubatch_size",
            "tensor-split": "tensor_split",
        }
        for cli_key, config_key in optional_map.items():
            if config_key in self.backend_extra_args:
                mapped_args[cli_key] = self.backend_extra_args[config_key]

        structured_args = self.backend_extra_args.get("start_up_args", {})
        if structured_args:
            mapped_args.update(structured_args)

        for key, value in mapped_args.items():
            if value is None or value is False:
                continue

            flag = f"--{key}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
                continue

            cmd.extend([flag, str(value)])

        if self.start_up_extra:
            cmd.extend(shlex.split(self.start_up_extra))

        return cmd

    def _get_default_server_log_path(self) -> Path:
        safe_model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", self.model_name or "llama_cpp_server")
        return Path("./log") / f"{safe_model_name}_llama_cpp_server_{self.server_port}.log"

    def _start_log_pump(self):
        if self.server_process is None or self.server_process.stdout is None:
            return

        raw_log_path = self.backend_extra_args.get("start_up_log_file")
        self.server_log_path = Path(raw_log_path) if raw_log_path else self._get_default_server_log_path()
        self.server_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.server_log_handle = open(self.server_log_path, "a", encoding="utf-8")

        def pump():
            assert self.server_process is not None
            assert self.server_process.stdout is not None
            prefix = f"[llama.cpp:{self.model_name}:{self.server_port}] "

            for line in self.server_process.stdout:
                text = line.rstrip()
                if not text:
                    continue

                with self.server_log_lock:
                    mode = self.server_log_mode

                if mode == "startup":
                    logger.info(prefix + text)
                elif self.server_log_handle:
                    self.server_log_handle.write(text + "\n")
                    self.server_log_handle.flush()

        self.server_log_thread = threading.Thread(target=pump, daemon=True)
        self.server_log_thread.start()

    def _set_log_mode(self, mode: str):
        with self.server_log_lock:
            self.server_log_mode = mode

    def _spawn_server(self):
        cmd = self._build_startup_command()
        logger.info(f"Starting llama.cpp server: {' '.join(cmd)}")

        self.server_process = subprocess.Popen(
            cmd,
            cwd=self.start_up_cwd or None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=os.environ.copy(),
        )
        self._set_log_mode("startup")
        self._start_log_pump()

    def _wait_for_healthcheck(self):
        deadline = time.monotonic() + self.healthcheck_timeout
        last_error: Optional[Exception] = None

        while time.monotonic() < deadline:
            if self.server_process is not None and self.server_process.poll() is not None:
                raise RuntimeError(
                    f"llama.cpp server exited before becoming healthy with code {self.server_process.returncode}"
                )

            try:
                self._probe_server()
                self._set_log_mode("running")
                logger.info(f"llama.cpp server ready at {self.base_url}; logging to {self.server_log_path}")
                return
            except Exception as exc:
                last_error = exc
                time.sleep(self.healthcheck_poll_interval)

        raise RuntimeError(
            f"llama.cpp server did not become healthy within {self.healthcheck_timeout}s: {last_error}"
        )

    def _ensure_server_started(self):
        try:
            self._probe_server()
            logger.info(f"Using existing llama.cpp server at {self.base_url}")
            return
        except Exception:
            pass

        self._spawn_server()
        try:
            self._wait_for_healthcheck()
        except Exception:
            self.close()
            raise

    def _probe_server(self):
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(self._url(self.health_path), headers=self._headers())
                response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"Failed to reach llama.cpp server at {self.base_url}: {exc}") from exc

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}{path}"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self.backend_extra_args.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _rewrite_multimodal_prompt_string(
        self,
        prompt_text: str,
        multimodal_count: int,
        vision_token: Optional[str] = None,
    ) -> str:
        if not self.multimodal_marker or multimodal_count <= 0:
            return prompt_text

        rewritten = prompt_text
        replaced = 0
        candidate_markers: List[str] = []

        def _append_candidate(value: Optional[str]):
            if value and value not in candidate_markers:
                candidate_markers.append(value)

        _append_candidate(vision_token)

        ensure_vision_token = getattr(self.prompt_eng, "ensure_vision_token", None)
        if callable(ensure_vision_token):
            _append_candidate(ensure_vision_token())

        for model_vision_token in MODEL_VISION_TOKEN.values():
            _append_candidate(model_vision_token)

        for pattern in candidate_markers:
            occurrences = rewritten.count(pattern)
            if not occurrences:
                continue

            rewritten = rewritten.replace(pattern, self.multimodal_marker)
            replaced += occurrences

        if replaced == multimodal_count:
            return rewritten

        logger.warning(
            "Configured multimodal_marker=%r but replaced %s prompt marker(s) for %s multimodal input(s)",
            self.multimodal_marker,
            replaced,
            multimodal_count,
        )
        return rewritten

    @staticmethod
    async def _read_error_response_text(response: httpx.Response) -> str:
        try:
            await response.aread()
        except Exception as exc:
            return f"<failed to read error response body: {exc}>"

        try:
            return response.text
        except Exception:
            return response.content.decode("utf-8", errors="replace")

    def close(self):
        if self.server_process is not None and self.server_process.poll() is None:
            logger.info(f"Stopping managed llama.cpp server for {self.model_name}")
            self._set_log_mode("shutdown")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Managed llama.cpp server did not terminate gracefully, forcing kill")
                self.server_process.kill()
                self.server_process.wait(timeout=5)

        if self.server_log_thread is not None and self.server_log_thread.is_alive():
            self.server_log_thread.join(timeout=2)

        if self.server_log_handle is not None:
            self.server_log_handle.close()
            self.server_log_handle = None

        self.server_process = None

    async def _tokenize_async(self, text: str, add_special: bool = False, parse_special: bool = True) -> List[int]:
        payload = {
            "content": text,
            "add_special": add_special,
            "parse_special": parse_special,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self._url(self.tokenize_path), headers=self._headers(), json=payload)
            response.raise_for_status()
            data = response.json()

        tokens = data.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError(f"Unexpected /tokenize response from llama.cpp server: {data}")
        return tokens

    async def _detokenize_async(self, tokens: List[int]) -> str:
        payload = {"tokens": tokens}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self._url(self.detokenize_path), headers=self._headers(), json=payload)
            response.raise_for_status()
            data = response.json()
        return data.get("content", "")

    def generate_eos_tokens_id(self) -> List[int]:
        return []

    @staticmethod
    def _normalize_gen_queues(
        gen_queue: Union[GenQueue, GenQueueDynamic, QueueContext, List[Union[GenQueue, GenQueueDynamic, QueueContext]]]
    ) -> List[QueueContext]:
        gen_queue_list: List[QueueContext] = []

        if isinstance(gen_queue, QueueContext):
            gen_queue_list = [gen_queue]
        elif isinstance(gen_queue, GenQueueDynamic):
            gen_queue_list = [QueueContext.create(gen_queue=gen_queue, include_GenStats=True, include_GenEnd=True)]
        elif isinstance(gen_queue, GenQueue):
            gen_queue_list = [QueueContext.create(gen_queue=gen_queue, include_GenStats=True, include_GenEnd=True)]
        elif isinstance(gen_queue, list):
            for queue in gen_queue:
                if isinstance(queue, QueueContext):
                    gen_queue_list.append(queue)
                elif isinstance(queue, GenQueueDynamic):
                    gen_queue_list.append(
                        QueueContext.create(gen_queue=queue, include_GenStats=True, include_GenEnd=True)
                    )
                elif isinstance(queue, GenQueue):
                    gen_queue_list.append(
                        QueueContext.create(gen_queue=queue, include_GenStats=True, include_GenEnd=True)
                    )
                else:
                    raise TypeError("gen_queue list must contain only GenQueue, GenQueueDynamic or QueueContext objects")
        else:
            raise TypeError("gen_queue must be either a GenQueue, GenQueueDynamic, QueueContext or a list of them")

        return gen_queue_list

    def _load_image_from_ref(self, image_ref: str) -> Image.Image:
        if image_ref.startswith("file://"):
            return Image.open(image_ref[7:])
        return get_image(url=image_ref)

    def _image_ref_to_base64(self, image_ref: str) -> str:
        image = self._load_image_from_ref(image_ref)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def _pil_to_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _extract_multimodal_payloads(self, messages: Optional[List[BaseMessage]]) -> tuple[List[str], bool]:
        image_payloads: List[str] = []
        has_audio = False

        if not messages:
            return image_payloads, has_audio

        for message in messages:
            content = getattr(message, "content", None)
            if not isinstance(content, list):
                continue

            for chunk in content:
                chunk_type = getattr(chunk, "type", None)
                if chunk_type == "image_url":
                    image_url = getattr(chunk, "image_url", None)
                    if isinstance(image_url, str):
                        image_ref = image_url
                    else:
                        image_ref = image_url.url
                    image_payloads.append(self._image_ref_to_base64(image_ref))
                elif chunk_type == "image":
                    image_value = getattr(chunk, "image_url", None)
                    if isinstance(image_value, Image.Image):
                        image_payloads.append(self._pil_to_base64(image_value))
                    elif isinstance(image_value, str):
                        image_payloads.append(self._image_ref_to_base64(image_value))
                elif chunk_type == "audio":
                    has_audio = True

        return image_payloads, has_audio

    @staticmethod
    def _resolve_gen_type(gen_type: Union[str, GenStart, Any]) -> tuple[GenStart, str]:
        if isinstance(gen_type, str):
            return GenStart(gen_type=gen_type), gen_type

        resolved = getattr(gen_type, "gen_type", "text")
        if isinstance(resolved, str):
            text_type = resolved if resolved in {"text", "thinking", "tool"} else "text"
        else:
            text_type = getattr(resolved, "tag_type", "text")
            if text_type not in {"text", "thinking", "tool"}:
                text_type = "text"

        return gen_type, text_type

    def _build_completion_payload(
        self,
        prompt_tokens: List[int],
        prompt_text: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_words: Optional[List[str]],
        json_schema: Optional[dict],
        multimodal_data: Optional[List[str]] = None,
        vision_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "stream": True,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "cache_prompt": self.cache_prompt,
        }

        if multimodal_data:
            prompt_text = self._rewrite_multimodal_prompt_string(
                prompt_text,
                len(multimodal_data),
                vision_token=vision_token,
            )
            payload["prompt"] = {
                "prompt_string": prompt_text,
                "multimodal_data": multimodal_data,
            }
        else:
            payload["prompt"] = prompt_tokens

        if stop_words:
            payload["stop"] = stop_words

        if json_schema:
            payload["json_schema"] = json_schema

        passthrough_keys = (
            "min_p",
            "top_k",
            "repeat_penalty",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "samplers",
            "mirostat",
            "mirostat_tau",
            "mirostat_eta",
            "n_probs",
        )
        for key in passthrough_keys:
            if key in self.backend_extra_args:
                payload[key] = self.backend_extra_args[key]

        return payload

    @staticmethod
    def _iter_sse_json(line: str) -> Optional[Dict[str, Any]]:
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            return None

        if stripped.startswith("data:"):
            stripped = stripped[5:].strip()

        if stripped == "[DONE]":
            return None

        return json.loads(stripped)

    def _build_generation_stats(
        self,
        final_chunk: Dict[str, Any],
        input_token_count: int,
        output_token_count: int,
        stop_reason: str,
    ) -> GenerationStats:
        timings = final_chunk.get("timings") or {}

        prompt_n = final_chunk.get("tokens_evaluated", timings.get("prompt_n", input_token_count))
        predicted_n = final_chunk.get("tokens_predicted", timings.get("predicted_n", output_token_count))
        prompt_ms = timings.get("prompt_ms", 0.0)
        predicted_ms = timings.get("predicted_ms", 0.0)

        return GenerationStats(
            input_tokens_count=prompt_n,
            output_tokens_count=predicted_n,
            time_to_first_token=prompt_ms / 1000,
            time_generate=predicted_ms / 1000,
            cached_tokens=final_chunk.get("tokens_cached", 0),
            stop_reason=stop_reason,
        )

    @staticmethod
    def _map_stop_reason(final_chunk: Dict[str, Any], used_stop_word: bool) -> str:
        if final_chunk.get("truncated"):
            return "model_context_window_exceeded"

        stop_type = final_chunk.get("stop_type")
        if stop_type == "word" or used_stop_word:
            return "stop_sequence"
        if stop_type == "limit":
            return "max_tokens"
        if stop_type == "none":
            return "pause_turn"
        return "end_turn"

    async def generate(
        self,
        prompt: str,
        gen_queue: Union[GenQueue, QueueContext, List[QueueContext], GenQueueDynamic],
        request: Optional[Request] = None,
        gen_type: Union[str, GenStart] = "text",
        temperature: float = 0.01,
        top_p: float = 0.8,
        formatter=None,
        stop_words: Union[List[str], str] = None,
        prefix_strings: Optional[Union[str, List[str]]] = None,
        banned_strings: list[str] | None = None,
        max_tokens: int = None,
        quiet: bool = False,
        messages: List[BaseMessage] = None,
        video: List[VideoFrame] = None,
        stop_event: asyncio.Event = None,
        send_eos: bool = True,
        vision_token=None,
        json_schema: dict | None = None,
        return_stop_word: bool = True,
        **kwargs,
    ) -> str:
        del formatter, banned_strings

        if video:
            raise HTTPException(status_code=400, detail="llama_cpp_server does not support direct video input")

        if not quiet:
            logger.info("----------------------Prompt---------------\n" + prompt)
            logger.debug("----------------------temperature---------\n" + str(temperature))

        if json_schema and prefix_strings:
            logger.info("Ignoring prefix_strings because json_schema was provided")

        if json_schema is None and prefix_strings:
            if isinstance(prefix_strings, list):
                prefix_strings = prefix_strings[0] if prefix_strings else None
            if prefix_strings:
                prompt += prefix_strings

        multimodal_data, has_audio = self._extract_multimodal_payloads(messages)
        if has_audio:
            raise HTTPException(status_code=400, detail="llama_cpp_server does not support audio inputs yet")

        prompt_tokens = await self._tokenize_async(prompt, add_special=False, parse_special=True)
        if not multimodal_data:
            self.validate_token_length(len(prompt_tokens))

        stop_conditions: Optional[List[str]] = None
        if stop_words:
            stop_conditions = [stop_words] if isinstance(stop_words, str) else stop_words

        available_context = self.max_seq_len - len(prompt_tokens) if self.max_seq_len else 4096
        max_tokens_to_use = min(available_context, max_tokens or 4096, 4096)
        if max_tokens_to_use <= 0:
            raise HTTPException(status_code=400, detail="Prompt already consumes the available context window")

        gen_queue_list = self._normalize_gen_queues(gen_queue)
        gen_start, gen_text_type = self._resolve_gen_type(gen_type)
        for g_queue in gen_queue_list:
            await g_queue.get_queue().put(gen_start)

        payload = self._build_completion_payload(
            prompt_tokens=prompt_tokens,
            prompt_text=prompt,
            max_tokens=max_tokens_to_use,
            temperature=temperature,
            top_p=top_p,
            stop_words=stop_conditions,
            json_schema=json_schema,
            multimodal_data=multimodal_data or None,
            vision_token=vision_token,
        )

        full_completion = ""
        final_chunk: Dict[str, Any] = {}
        output_tokens_count = 0
        aborted = False
        request_start = time.monotonic()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    self._url(self.completion_path),
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    if response.is_error:
                        detail = await self._read_error_response_text(response)
                        logger.error(
                            f"llama.cpp server error {response.status_code} at {self._url(self.completion_path)} "
                            f"(multimodal={bool(multimodal_data)}, json_schema={bool(json_schema)}): {detail}"
                        )
                        raise HTTPException(status_code=response.status_code, detail=detail)

                    async for line in response.aiter_lines():
                        if stop_event and stop_event.is_set():
                            aborted = True
                            await response.aclose()
                            break

                        if request and await is_request_disconnected(request):
                            aborted = True
                            await response.aclose()
                            break

                        if not line:
                            continue

                        chunk = self._iter_sse_json(line)
                        if chunk is None:
                            continue

                        content = chunk.get("content", "")
                        stop = bool(chunk.get("stop"))
                        final_chunk = chunk if stop or chunk.get("timings") else final_chunk

                        if content:
                            full_completion += content
                            chunk_text = GenText(content=content, text_type=gen_text_type)
                            for g_queue in gen_queue_list:
                                await g_queue.get_queue().put(chunk_text)

                        if stop:
                            final_chunk = chunk
                            break
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail=f"Failed to reach llama.cpp server: {exc}") from exc

        stop_word_used = ""
        if return_stop_word:
            stop_word_used = final_chunk.get("stopping_word", "")

        if stop_word_used:
            full_completion += stop_word_used
            chunk_text = GenText(content=stop_word_used, text_type=gen_text_type)
            for g_queue in gen_queue_list:
                await g_queue.get_queue().put(chunk_text)

        if not final_chunk and aborted:
            final_chunk = {
                "timings": {
                    "prompt_ms": 0.0,
                    "predicted_ms": max((time.monotonic() - request_start) * 1000, 0.0),
                    "prompt_n": len(prompt_tokens),
                    "predicted_n": output_tokens_count,
                },
                "stop_type": "none",
            }

        if final_chunk:
            output_tokens_count = final_chunk.get("tokens_predicted") or final_chunk.get("timings", {}).get("predicted_n", 0)

        if output_tokens_count == 0 and full_completion:
            output_tokens_count = len(await self._tokenize_async(full_completion, add_special=False, parse_special=True))

        used_stop_word = bool(stop_word_used)
        stop_reason = self._map_stop_reason(final_chunk, used_stop_word) if final_chunk else "pause_turn"
        gen_stats = self._build_generation_stats(
            final_chunk=final_chunk,
            input_token_count=len(prompt_tokens),
            output_token_count=output_tokens_count,
            stop_reason=stop_reason,
        ) if final_chunk else GenerationStats(
            input_tokens_count=len(prompt_tokens),
            output_tokens_count=output_tokens_count,
            stop_reason=stop_reason,
        )

        if send_eos:
            for g_queue in gen_queue_list:
                if g_queue.include_GenStats:
                    await g_queue.get_queue().put(gen_stats)

            for g_queue in gen_queue_list:
                if g_queue.include_GenEnd:
                    await g_queue.get_queue().put(GenEnd())

        logger.debug("----------------------LLM Raw Response---------------\n" + full_completion)
        return full_completion
