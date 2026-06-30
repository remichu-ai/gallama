import os
import re
import time
from gallama.backend.llm.engine.base import (
    ModelInterface,
)
from typing import Optional, Dict, List, Union, get_args
import torch
import asyncio
from fastapi import Request                 # for type hint
from functools import lru_cache             # for image caching
import uuid                                 # use for generating id for api return

from gallama.logger.logger import basic_log_extra, logger
from gallama.backend.llm.json_schema_utils import normalize_json_schema_for_formatron
from gallama.utils.request_disconnect import (
    format_exception_summary,
    is_expected_disconnect_exception,
    is_request_disconnected,
)
from gallama.data_classes import (
    BaseMessage,
    ModelSpec,
    GenStart,
    GenEnd,
    GenQueue,
    GenText,
    GenerationStats,
    QueueContext,
    GenQueueDynamic,
    VideoFrame,
    TagDefinition,
    AnthropicStopReason
)
from gallama.utils.utils import get_image

try:
    from exllamav3 import (
        Model,
        Config,
        Cache,
        Tokenizer,
        CacheLayer_fp16,
        CacheLayer_quant,
        AsyncGenerator,
        AsyncJob,
        FormatronFilter,
    )
except ImportError:
    Model = None
    Config = None
    Cache = None
    Tokenizer = None
    AsyncGenerator = None

# import exllama v3 sampler
try:
     from exllamav3.generator.sampler import (
         CustomSampler,
         SS_Base,
         SS_Argmax,
         SS_Sample,
         SS_Temperature,
         SS_Normalize,
         SS_Sort,
         SS_TopK,
         SS_TopP,
         SS_NoOp,
     )
except ImportError:
    CustomSampler = None
    SS_Base = None
    SS_Argmax = None
    SS_Sample = None
    SS_Temperature = None
    SS_Normalize = None
    SS_Sort = None
    SS_TopK = None
    SS_TopP = None
    SS_NoOp = None

# format enforcement with formatron
from formatron.formatter import FormatterBuilder
from formatron.schemas import json_schema


def _is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_generator_kwargs(raw_kwargs: Dict | None) -> Dict:
    normalized = dict(raw_kwargs or {})

    int_keys = {
        "max_batch_size",
        "max_chunk_size",
        "max_q_size",
        "num_draft_tokens",
        "recurrent_cache_size",
        "recurrent_checkpoint_interval",
    }
    bool_keys = {
        "show_visualizer",
        "enable_defrag",
    }

    for key in int_keys:
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = int(value)

    for key in bool_keys:
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = _is_truthy(value)

    if normalized.get("max_batch_size") is None:
        normalized["max_batch_size"] = 4

    if normalized.get("max_chunk_size") is None:
        normalized["max_chunk_size"] = 2048

    return normalized


def _align_cache_size(cache_size: int | None, max_seq_len: int) -> int:
    base_size = cache_size or max_seq_len
    return ((max(base_size, max_seq_len) + 255) // 256) * 256


def _resolve_draft_max_history(
    draft_model_id: str | None,
    backend_extra_args: Dict | None,
    draft_model_component: str = "text",
) -> int:
    """Resolve recurrent cache history needed for speculative verification."""

    draft_extra = _normalize_generator_kwargs(backend_extra_args)
    configured_num_draft_tokens = draft_extra.get("num_draft_tokens")
    if configured_num_draft_tokens:
        return configured_num_draft_tokens

    if not draft_model_id:
        return 0

    draft_config = Config.from_directory(draft_model_id)
    if draft_model_component == "text":
        draft_model = Model.from_config(draft_config)
    else:
        draft_model = Model.from_config(draft_config, component=draft_model_component)
    return draft_model.caps.get("default_draft_size") or 4


def _is_insufficient_vram_error(exc: BaseException) -> bool:
    message = str(exc)
    return (
        "Insufficient VRAM in split for model and cache" in message
        or "CUDA out of memory" in message
        or "HIP out of memory" in message
    )


def _reset_cuda_memory_fraction():
    """Undo ExLlamaV3 autosplit's temporary per-process memory caps.

    ExLlamaV3 sets torch.cuda.set_per_process_memory_fraction while loading.
    If autosplit raises before its cleanup path, PyTorch keeps that cap for the
    process and later tiny allocations can fail despite many GiB being free.
    """

    for device_idx in range(torch.cuda.device_count()):
        try:
            torch.cuda.set_per_process_memory_fraction(1.0, device=device_idx)
        except Exception as exc:
            logger.debug(f"Failed to reset CUDA memory fraction for device {device_idx}: {exc}")


def _normalize_reserve_vram(raw_reserve_vram, num_devices: int) -> List[float]:
    if num_devices <= 0:
        return []

    if raw_reserve_vram is None:
        result = [0.4] * num_devices
        if num_devices > 0:
            result[0] = 0.8
        return result

    if isinstance(raw_reserve_vram, (int, float)):
        return [float(raw_reserve_vram)] * num_devices

    if isinstance(raw_reserve_vram, list):
        normalized = [float(value) for value in raw_reserve_vram]
        if len(normalized) < num_devices:
            normalized.extend([0.0] * (num_devices - len(normalized)))
        return normalized[:num_devices]

    raise ValueError("reserve_vram must be a float or list[float]")


def _auto_use_vram_for_existing_allocations(raw_reserve_vram, num_devices: int) -> List[float]:
    """Return per-device use limits that preserve auto reserve semantics after a model is already loaded.

    ExLlamaV3's reserve_per_device mode calls torch.cuda.set_per_process_memory_fraction
    based on currently free VRAM. That works for the first model in a fresh process, but
    if we load a draft model after the target model, the process already owns tens of GiB
    and reserve mode can set a cap below current PyTorch allocations. use_per_device is a
    total process cap, so total_vram - reserve_vram gives the intended remaining headroom.
    """

    reserves = _normalize_reserve_vram(raw_reserve_vram, num_devices)
    use_per_device = []
    gib = 1024 ** 3
    for device_idx in range(num_devices):
        reserve = reserves[device_idx] if device_idx < len(reserves) else 0.0
        if reserve < 0:
            use_per_device.append(0.0)
            continue
        total_gb = torch.cuda.get_device_properties(device_idx).total_memory / gib
        use_per_device.append(max(0.0, total_gb - reserve))
    return use_per_device


def _resolve_load_kwargs(gpus, reserve_vram, tensor_parallel: bool, num_devices: int) -> Dict:
    load_kwargs = {
        "progressbar": True,
        "tensor_p": tensor_parallel,
    }

    if isinstance(gpus, list):
        if reserve_vram is not None:
            raise ValueError(
                "ExLlamaV3 does not support `reserve_vram` together with an explicit `gpus` split. "
                "Use `gpus: auto` with `reserve_vram`, or remove `reserve_vram`."
            )
        load_kwargs["use_per_device"] = gpus
        return load_kwargs

    if isinstance(gpus, str) and gpus == "auto":
        return load_kwargs

    raise ValueError("Device map should be either 'auto' or a GPU split list")


def _gallama_config_dir() -> str:
    """The Gallama config directory (~/gallama or $GALLAMA_HOME_PATH)."""
    home = os.environ.get("GALLAMA_HOME_PATH")
    if home:
        return os.path.abspath(home)
    return os.path.join(os.path.expanduser("~"), "gallama")


def _resolve_topology_path(path: str) -> str:
    """Resolve a segment-topology / placement YAML path.

    Absolute paths and paths that resolve against the current working directory are used as-is.
    Bare relative paths are also looked up under the Gallama config directory (~/gallama), so a
    user can keep the topology/placement YAML next to model_config.yaml and reference it by name.
    """
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(_gallama_config_dir(), path)
    if os.path.exists(candidate):
        return candidate
    return path


def _materialize_segment_topology(inline: Dict) -> str:
    """Write an inline segment-topology mapping to a temp YAML file and return its path.

    Lets users embed the ExLlamaV3 segmented PP+TP topology directly in model_config.yaml under
    ``backend_extra_args.segment_topology`` instead of pointing ``segment_topology_yaml`` at an
    external file. The temp file is removed at process exit.
    """
    if not isinstance(inline, dict):
        raise ValueError("backend_extra_args.segment_topology must be a mapping with a 'segments' list")
    import atexit
    import tempfile
    import yaml

    fd, path = tempfile.mkstemp(prefix="gallama_segment_topology_", suffix=".yaml", text=True)
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(inline, f, sort_keys=False)
    atexit.register(lambda p=path: _cleanup_topology_temp_file(p))
    return path


def _cleanup_topology_temp_file(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


def _apply_exllamav3_tp_load_options(load_kwargs: Dict, backend_extra_args, tensor_parallel: bool):
    """Resolve tensor-parallel / segmented-topology load options from backend_extra_args into load_kwargs.

    Handles three ExLlamaV3 load customizations:
      * ``tp_backend`` and ``tp_options`` (passed straight through to model.load()).
      * a top-level ``placement_yaml`` convenience alias that is injected into ``tp_options.placement_yaml``
        (the manual TP placement overrides). In segmented mode the per-segment placement lives inside the
        topology YAML's segment ``options.placement_yaml`` and this alias is ignored.
      * the segmented PP+TP topology, specified either inline as ``segment_topology`` or as a file path via
        ``segment_topology_yaml`` (e.g. tensor-parallel across GPU pairs). Requires ``tensor_parallel``.
    """
    extra = backend_extra_args or {}

    tp_backend = extra.get("tp_backend")
    if tp_backend:
        load_kwargs["tp_backend"] = tp_backend

    raw_tp_options = extra.get("tp_options")
    if raw_tp_options is not None and not isinstance(raw_tp_options, dict):
        raise ValueError("backend_extra_args.tp_options must be a mapping")
    tp_options = dict(raw_tp_options or {})
    placement_yaml = extra.get("placement_yaml")
    if placement_yaml:
        if tp_options.get("placement_yaml") not in (None, placement_yaml):
            raise ValueError(
                "Conflicting placement_yaml set under both backend_extra_args.placement_yaml "
                "and backend_extra_args.tp_options.placement_yaml"
            )
        tp_options["placement_yaml"] = _resolve_topology_path(str(placement_yaml))
    if tp_options:
        load_kwargs["tp_options"] = tp_options

    segment_topology = extra.get("segment_topology")
    segment_topology_yaml = extra.get("segment_topology_yaml")
    if segment_topology is not None and segment_topology_yaml is not None:
        raise ValueError(
            "Specify either backend_extra_args.segment_topology (inline) or "
            "segment_topology_yaml (file path), not both"
        )
    if segment_topology is not None:
        segment_topology_yaml = _materialize_segment_topology(segment_topology)
    if segment_topology_yaml:
        if not tensor_parallel:
            raise ValueError(
                "segment_topology / segment_topology_yaml requires tensor parallel: set `tp: true` "
                "for this model in model_config.yaml"
            )
        load_kwargs["segment_topology_yaml"] = _resolve_topology_path(str(segment_topology_yaml))


def _resolve_vision_device(vision_device, num_devices: int):
    if vision_device is None:
        return None

    if isinstance(vision_device, str):
        vision_device = vision_device.strip().lower()
        if vision_device.startswith("cuda:"):
            vision_device = vision_device.split(":", 1)[1]
        vision_device = int(vision_device)

    if not isinstance(vision_device, int):
        raise ValueError("vision_device must be an integer GPU index or a cuda:<index> string")

    if vision_device < 0 or vision_device >= num_devices:
        raise ValueError(f"vision_device {vision_device} is out of range for {num_devices} CUDA devices")

    return torch.device(f"cuda:{vision_device}")


class ModelExllamaV3(ModelInterface):
    def __init__(self, model_spec:ModelSpec):
        super().__init__(model_spec)
        self.draft_model_component = self.backend_extra_args.get("draft_model_component", "text")
        if _is_truthy(self.backend_extra_args.get("mtp_draft")):
            if self.draft_model_id and os.path.abspath(self.draft_model_id) != os.path.abspath(self.model_id):
                raise ValueError("ExLlamaV3 MTP draft uses the main model directory; remove draft_model_id or set it to model_id")
            self.draft_model_id = self.model_id
            self.draft_model_component = "mtp"
        elif self.draft_model_component == "mtp" and not self.draft_model_id:
            self.draft_model_id = self.model_id
        self.model, self.tokenizer, self.cache, self.processor = self.load_model()

    @property
    def support_concurrency(self) -> bool:
        """
        whether this backend/ model support concurrent request
        """
        return True

    def load_model(self):
        """Load the model, tokenizer, cache, and optional processor."""

        # Resolve draft history before cache construction because ExLlamaV3's
        # recurrent cache must reserve rollback space up front.
        draft_num_tokens = _resolve_draft_max_history(
            self.draft_model_id,
            self.backend_extra_args,
            self.draft_model_component,
        )

        model, tokenizer, cache, processor = self.load_model_exllama(
            model_id=self.model_id,
            backend=self.backend,
            max_seq_len=self.max_seq_len,
            cache_size=self.cache_size,
            cache_quant=self.cache_quant,
            gpus=self.gpus,
            reserve_vram=self.reserve_vram,
            tensor_parallel=self.tensor_parallel,
            backend_extra_args=self.backend_extra_args,
            max_history=draft_num_tokens,
        )

        # load draft model
        if self.draft_model_id:
            # tokenizer and processor already set above
            draft_gpus = self.draft_gpus
            draft_reserve_vram = self.reserve_vram
            if draft_gpus == "auto":
                draft_gpus = _auto_use_vram_for_existing_allocations(
                    raw_reserve_vram=self.reserve_vram,
                    num_devices=torch.cuda.device_count(),
                )
                draft_reserve_vram = None
                logger.info(
                    "Resolved draft_gpus auto to post-target-load use limits: " + str(draft_gpus),
                    extra=basic_log_extra(),
                )

            try:
                self.draft_model, _, self.draft_cache, _ = self.load_model_exllama(
                    model_id=self.draft_model_id,
                    backend=self.backend,
                    max_seq_len=self.max_seq_len,  # draft model max_seq_len must be same as main model
                    cache_size=cache.max_num_tokens,
                    cache_quant=self.draft_cache_quant,
                    gpus=draft_gpus,
                    reserve_vram=draft_reserve_vram,
                    tensor_parallel=False,
                    backend_extra_args=self.backend_extra_args,
                    load_tokenizer=False,
                    load_processor=False,
                    max_history=draft_num_tokens,
                    model_component=self.draft_model_component,
                )
            except RuntimeError:
                _reset_cuda_memory_fraction()
                torch.cuda.empty_cache()
                raise

        self.eos_token_ids = self.generate_eos_tokens_id(tokenizer)

        return model, tokenizer, cache, processor

    def load_model_exllama(
        self,
        model_id,
        backend,
        cache_size,
        cache_quant,
        gpus,
        reserve_vram,
        max_seq_len=None,
        tensor_parallel=False,
        backend_extra_args=None,
        load_tokenizer=True,
        load_processor=True,
        max_history=0,
        model_component="text",
    ):
        """This function return the model and its tokenizer"""
        logger.info("Loading model: " + model_id, extra=basic_log_extra())
        if model_component != "text":
            logger.info("Loading model component: " + model_component, extra=basic_log_extra())

        config = Config.from_directory(model_id)
        if model_component == "text":
            model = Model.from_config(config)
        else:
            model = Model.from_config(config, component=model_component)
        tokenizer = Tokenizer.from_config(config) if load_tokenizer else None
        processor = None    # placeholder for visual processing tower

        if tensor_parallel and not model.caps.get("supports_tp", False):
            raise ValueError(
                f"Tensor parallel is not supported by the installed ExLlama V3 architecture "
                f"'{config.architecture}' for model '{model_id}'. Disable `tp`, use a different backend, "
                f"or install an ExLlama V3 build that supports this architecture."
            )


        # find the max sequence length
        if max_seq_len is not None:
            self.max_seq_len = max_seq_len
        else:
            # set the self.max_seq_len using model config file as it is None at the moment
            self.max_seq_len = config.config_dict.get("max_position_embeddings", 16384)

        # Normalize backend_extra_args to extract batch/chunk settings for Cache and model.load
        normalized_extra = _normalize_generator_kwargs(backend_extra_args)
        max_batch_size = normalized_extra.get("max_batch_size", 4)
        max_chunk_size = normalized_extra.get("max_chunk_size", 2048)

        # # a simple dict to help map cache quant
        cache_quant_dict = {
            "FP16": None,
            "Q4": {"k_bits": 4, "v_bits": 4},
            "Q6": {"k_bits": 6, "v_bits": 6},
            "Q8": {"k_bits": 8, "v_bits": 8},
        }

        # Align to 256, but ensure it is at least max_seq_len
        cache_size_to_use = _align_cache_size(cache_size, self.max_seq_len)

        # get the cache quantization to use
        cache_quant_to_use = cache_quant_dict.get(cache_quant, None)

        logger.info("max_seq_len: " + str(self.max_seq_len), extra=basic_log_extra())
        logger.info("cache_size: " + str(cache_size_to_use), extra=basic_log_extra())
        logger.info("Cache Quantization: " + str(cache_quant), extra=basic_log_extra())
        logger.info("gpus: " + str(gpus), extra=basic_log_extra())
        logger.info("reserve_vram: " + str(reserve_vram), extra=basic_log_extra())
        logger.info("max_batch_size (for load/cache): " + str(max_batch_size), extra=basic_log_extra())

        assert (isinstance(gpus, str) and gpus == "auto") or (isinstance(gpus, list)), \
            "Device map should be either 'auto', 'gpu' split"

        # create the layer if cache quant is needed
        cache_layer = None

        # TODO cache quant
        if cache_quant_to_use:
            cache_layer = CacheLayer_quant

        if cache_quant_to_use:
            logger.info("Using cache quant", extra=basic_log_extra())
            cache = Cache(
                model,
                max_num_tokens=cache_size_to_use,
                layer_type=cache_layer,
                max_batch_size=max_batch_size,
                max_history=max_history,
                **cache_quant_to_use
            )
        else:
            # FP16
            logger.info("Not using cache quant", extra=basic_log_extra())
            cache = Cache(
                model,
                max_num_tokens=cache_size_to_use,
                max_batch_size=max_batch_size,
                max_history=max_history,
            )
        # ExLlamaV3 supports either reserve_per_device or use_per_device, not both.
        load_kwargs = _resolve_load_kwargs(
            gpus=gpus,
            reserve_vram=reserve_vram,
            tensor_parallel=tensor_parallel,
            num_devices=torch.cuda.device_count(),
        )

        _apply_exllamav3_tp_load_options(load_kwargs, backend_extra_args, tensor_parallel)

        if load_kwargs.get("tp_backend"):
            logger.info("Tensor parallel backend: " + str(load_kwargs["tp_backend"]), extra=basic_log_extra())
        if load_kwargs.get("tp_options"):
            logger.info("Tensor parallel options: " + str(load_kwargs["tp_options"]), extra=basic_log_extra())
        if load_kwargs.get("segment_topology_yaml"):
            logger.info(
                "Segmented PP+TP topology: " + str(load_kwargs["segment_topology_yaml"]),
                extra=basic_log_extra(),
            )

        # Pass batch/chunk settings to model.load so the workspace is sized correctly
        load_kwargs["max_batch_size"] = max_batch_size
        load_kwargs["max_chunk_size"] = max_chunk_size

        # Load the vision tower before loading the text model/cache. ExLlamaV3's
        # upstream multimodal example does this so the subsequent text-model
        # autosplit sees the VRAM already occupied by vision weights. Loading
        # vision after a large text cache can fail even when the combined model
        # would fit with a better split.
        if load_processor:
            processor = self._load_vision_processor(config, backend_extra_args)

        try:
            model.load(
                **load_kwargs,
            )
        except RuntimeError:
            try:
                model.unload()
            finally:
                _reset_cuda_memory_fraction()
                torch.cuda.empty_cache()
            raise

        # if processor is not None, meaning at least image is supported
        if processor:
            self.modalities.add("image")
            vision_token = self.prompt_eng.ensure_vision_token()
            if vision_token is None:
                logger.warning("Vision tower loaded but no vision token could be resolved for prompt templating")

        # # check if video is supported
        # if processor and processor.video_preprocess_func:
        #     self.modalities.add("video")
        #
        # logger.info(f"Supported Modalities: {self.modalities}")

        return model, tokenizer, cache, processor


    @staticmethod
    def _load_vision_processor(config, backend_extra_args=None):
        """Load optional ExLlamaV3 vision component before the text model."""

        if "vision" not in config.model_classes:
            logger.info("No Vision Tower", extra=basic_log_extra())
            return None

        processor = Model.from_config(config, component="vision")

        vision_device = _resolve_vision_device(
            (backend_extra_args or {}).get("vision_device"),
            torch.cuda.device_count(),
        )
        if vision_device is not None:
            logger.info("Vision device: " + str(vision_device), extra=basic_log_extra())
            processor.load(device=vision_device)
        else:
            logger.info("Loading Vision Tower", extra=basic_log_extra())
            processor.load()

        return processor


    @property
    def video_token_by_backend(self) -> str:
        """ exllama use this specific token for video embedding"""
        return "{{VIDEO-PlaceHolderTokenHere}}"


    def generate_eos_tokens_id(self, tokenizer: Tokenizer = None) -> List[int]:
        """Generate the end-of-sequence token IDs."""
        tokenizer_to_use = tokenizer or self.tokenizer
        if not tokenizer_to_use:
            return []

        if not self.eos_token_str:
            # set eos token using variable from inside tokenizer if not manually set
            self.eos_token_str = [tokenizer_to_use.eos_token]

        return [tokenizer_to_use.single_id(token) for token in self.eos_token_str]


    # ********* from here is generation methods
    # helper function for job disconnection. Currently only exllama support this
    @staticmethod
    async def check_disconnection(request, job, gen_queue_list, stop_event=None):
        try:
            while True:
                if await is_request_disconnected(request):
                    logger.info("User disconnected")
                    if job:
                        await job.cancel()
                    chunk = GenEnd()
                    for g_queue in gen_queue_list:
                        try:
                            if not getattr(g_queue, "include_GenEnd", True):
                                continue

                            if hasattr(g_queue, "get_queue"):
                                await g_queue.get_queue().put(chunk)
                            else:
                                g_queue.put_nowait(chunk)
                        except Exception as e:
                            logger.debug(f"Skipping GenEnd queue cleanup after disconnect: {str(e)}")
                    break

                await asyncio.sleep(1)  # simple sleep, no wait_for wrapper

        except asyncio.CancelledError:
            logger.debug("Disconnection check was cancelled")
        except Exception as e:
            if stop_event and stop_event.is_set():
                logger.debug("Disconnection check exited after stop event")
            elif is_expected_disconnect_exception(e):
                logger.debug("Client disconnected while polling request state")
            else:
                logger.error(
                    f"Error in check_disconnection: {format_exception_summary(e)}",
                    exc_info=True,
                )
        finally:
            logger.debug("Exiting check_disconnection")

    class ExllamaV3Pipeline:
        """ class as wrapper for objects required for Exllama V2 text generation"""

        def __init__(
            self,
            cache: Cache,
            generator: AsyncGenerator,
        ):
            self.cache = cache
            self.generator = generator


    async def _get_pipeline_async(self):
        """
        as AsyncGenerator is async function, it can not be run in the class __init__
        this will be run in the first text generation to ensure that generator is initialized
        """

        generator_kwargs = _normalize_generator_kwargs(self.backend_extra_args)

        generator = AsyncGenerator(
            model=self.model,
            cache=self.cache,
            tokenizer=self.tokenizer,
            draft_model=self.draft_model,
            draft_cache=self.draft_cache,
            **generator_kwargs,
        )

        if self.draft_model is not None:
            if generator.generator.mtp_draft:
                mode = "mtp"
            elif generator.generator.dflash_draft:
                mode = "dflash"
            else:
                mode = "flash"
            logger.info(f"ExLlamaV3 speculative draft mode: {mode}", extra=basic_log_extra())

        return self.ExllamaV3Pipeline(
            cache=self.cache,
            generator=generator,
        )

    async def _ensure_pipeline_ready(self):
        """Create or replace the ExLlamaV3 async generator if its worker died."""
        if self.pipeline is None:
            self.pipeline = await self._get_pipeline_async()
            return

        iteration_task = getattr(self.pipeline.generator, "iteration_task", None)
        if iteration_task is not None and iteration_task.done():
            exc = None
            if not iteration_task.cancelled():
                try:
                    exc = iteration_task.exception()
                except Exception:
                    exc = None
            if exc:
                logger.error(f"ExLlamaV3 async generator stopped unexpectedly: {format_exception_summary(exc)}")
            else:
                logger.warning("ExLlamaV3 async generator is not running; restarting it")
            self.pipeline = await self._get_pipeline_async()

    async def _cancel_job_safely(self, job):
        if not job:
            return
        try:
            await job.cancel()
        except AssertionError:
            logger.debug("ExLlamaV3 job was already removed from async generator")
        except Exception as e:
            logger.debug(f"Error cancelling ExLlamaV3 job: {format_exception_summary(e)}")

    @staticmethod
    def _stop_event_is_set(stop_event: asyncio.Event | None) -> bool:
        return bool(stop_event and stop_event.is_set())

    @staticmethod
    def _put_gen_end(gen_queue_list):
        for g_queue in gen_queue_list:
            if g_queue.include_GenEnd:
                g_queue.put_nowait(GenEnd())

    @staticmethod
    def _get_exllama_gen_settings(
        temperature: float = 0.01,
        top_p: float = 0.8,
        top_k: Optional[int] = None,
        **kwargs,
    ):
        # settings
        samplers = [
            SS_Temperature(temperature),
            SS_TopP(top_p),
        ]
        if top_k is not None and SS_TopK is not None:
            samplers.append(SS_TopK(top_k))
        samplers.append(SS_Sample())
        settings = CustomSampler(samplers)

        # settings = ExLlamaV2Sampler.Settings()
        # settings.temperature = temperature
        # settings.min_temp = 0.15
        # settings.top_k = 50
        # settings.top_p = top_p
        # settings.min_p = 0.05
        # settings.token_repetition_penalty = 1.1
        # settings.token_frequency_penalty = 0.05
        # settings.token_repetition_range = 1024
        # # settings.token_repetition_decay: int = 0.98
        # settings.temperature_last = False

        return settings

    @staticmethod
    @lru_cache(1024)     # TODO set this dynamically
    def get_image_embedding_cached(processor, tokenizer, url):
        """
        function to return image embedding for exllama
        lru_cache to cache frequently used image
        """
        img = get_image(url=url)

        return processor.get_image_embeddings(
            tokenizer=tokenizer,
            image=img,
            text_alias=None,    # passing None will let the llm generate its own embedding
        )

    @staticmethod
    def get_video_embedding_cached(processor, model, tokenizer, video: List[VideoFrame]):
        """
        function to return image embedding for exllama
        lru_cache to cache frequently used image
        """

        return processor.get_video_embeddings(
            model=model,
            tokenizer=tokenizer,
            video=video,
            text_alias=None,    # passing None will let the llm generate its own embedding
        )

    def _generate_image_embeddings(self, prompt, vision_token: str, image_list):
        """Generate embeddings for images and update prompt"""
        # in prompt processing step, each image was substituted with the following token

        # Validate image token count matches number of images
        # logger.info(f"Prompt: {prompt}")
        # logger.info(f"Generating image embeddings for {len(image_list)} images")

        assert vision_token is not None, "vision token can not be None"

        token_count = prompt.count(vision_token)
        assert token_count == len(
            image_list), f"Image token mismatch: found {token_count} tokens, but got {len(image_list)} images."

        # Generate embeddings
        image_embeddings = [
            self.get_image_embedding_cached(
                processor=self.processor,
                tokenizer=self.tokenizer,
                url=url
            ) for url in image_list
        ]

        # Replace image tokens with embeddings
        for emb in image_embeddings:
            prompt = prompt.replace(vision_token, emb.text_alias, 1)

        return prompt, image_embeddings

    def _generate_video_embeddings(self, prompt, video: List[VideoFrame]):
        """Generate embeddings for video and update prompt"""
        # in prompt processing step, each image was substituted with the following token
        # TODO move this token to better place
        video_token = "{{VIDEO-PlaceHolderTokenHere}}"

        # Validate only 1 video
        assert prompt.count(video_token) == 1, "Video support currently limit to 1 token mismatch"

        # get the image from each of the VideoFrame object
        _video = [ f.image for f in video ]

        video_embeddings = self.get_video_embedding_cached(
            processor=self.processor,
            model=self.model,
            tokenizer=self.tokenizer,
            video=_video
        )

        # replace prompt token
        prompt = prompt.replace(video_token, video_embeddings.text_alias, 1)

        return prompt, video_embeddings

    def _process_vision_inputs(self, prompt, vision_token: str, messages: List[BaseMessage], video: List[VideoFrame] = None):
        """Handle image embedding and token replacement for vision inputs"""

        _prompt = prompt
        _vision_embeddings = []

        if not messages:
            return prompt, None

        image_list = []
        vision_required = False

        # Extract image URLs from messages
        for message in messages:
            if isinstance(message.content, list):
                image_urls = [
                    msg.image_url.url
                    for msg in message.content
                    if msg.type == "image_url"
                ]
                image_list.extend(image_urls)

        vision_required = True if len(image_list) >0 else False

        # Process image embeddings if vision is required
        if vision_required and self.processor:
            _prompt,_vision_embeddings = self._generate_image_embeddings(prompt, vision_token, image_list)

        # handle video input
        if video:
            _prompt, _video_embeddings = self._generate_video_embeddings(_prompt, video)
            _vision_embeddings.append(_video_embeddings)

        return _prompt, _vision_embeddings


    def _create_generation_filters(
        self,
        json_schema_dict: dict = None
    ) -> List:
        """Create filters for token generation"""
        filters = []

        if json_schema_dict is None:
            return []

        # Formatron cannot handle numeric bound metadata like minimum/maximum,
        # so normalize the schema into a compatible form before building filters.
        json_schema_dict = normalize_json_schema_for_formatron(json_schema_dict)

        if "$schema" not in json_schema_dict:
            json_schema_dict["$schema"] = "http://json-schema.org/draft-07/schema#"

        # 2. Add an $id key so the referencing registry can catalog it
        if "$id" not in json_schema_dict:
            json_schema_dict["$id"] = "http://gallama.local/chat-schema.json"

        schema = json_schema.create_schema(json_schema_dict)
        f = FormatterBuilder()
        f.append_line(f"{f.json(schema, capture_name='json')}")

        filters.append(
            FormatronFilter(self.tokenizer, eos_after_completed=True, formatter_builder=f),
        )

        return filters

    # noinspection PyTypeChecker
    @staticmethod
    def get_stop_reason(result: dict, use_stop_words: bool) -> AnthropicStopReason:
        exl_reason = result.get("eos_reason")
        reason = "end_turn"
        if use_stop_words:
            reason = "stop_sequence"
        elif exl_reason == "max_new_tokens":
            reason = "max_tokens"
        elif exl_reason in ["stop_string", "stop_token"]:
            reason = "end_turn"
        elif exl_reason == "end_filter":
            reason = "stop_sequence"

        assert reason in get_args(AnthropicStopReason), "Stop reason must be one of AnthropicStopReason"

        return reason

    async def generate(
        self,
        prompt: str,
        gen_queue: Union[GenQueue, GenQueueDynamic, List[GenQueueDynamic]],
        request: Optional[Request] = None,  # for disconnection check
        gen_type: Union[str, GenStart, TagDefinition] = "text",  # the generated result will be stored in this queue
        temperature: float = 0.01,
        top_p: float = 0.8,
        # formatter: Optional[FormatterBuilder] = None,
        stop_words: Union[List[str], str] = None,
        prefix_strings: Optional[Union[str, List[str]]] = None,
        banned_strings: list[str] | None = None,
        max_tokens: int = None,
        quiet=False,
        messages: List[BaseMessage] = None,  # query.message for multimodal
        video: List[VideoFrame] = None,
        stop_event: asyncio.Event = None,
        send_eos: bool = True,
        vision_token = None,
        json_schema = None,
        return_stop_word: bool = True,
        **kwargs,
    ) -> (str, GenerationStats):
        full_completion = ""

        try:
            top_k = kwargs.get("top_k")
            # Ensure that the generator is initialized and its worker is alive.
            await self._ensure_pipeline_ready()

            # Convert gen_queue to List[GenQueueDynamic] format to standardize downstream handling
            gen_queue_list = []
            if isinstance(gen_queue, GenQueueDynamic):
                gen_queue_list = [gen_queue]
            elif isinstance(gen_queue, GenQueue):
                # Wrap the GenQueue in a GenQueueDynamic
                gen_queue_list = [GenQueueDynamic(existing_queue=gen_queue, include_GenStats=True, include_GenEnd=True)]
            elif isinstance(gen_queue, list):
                # Ensure all items in the list are GenQueueDynamic objects
                for queue in gen_queue:
                    if isinstance(queue, GenQueueDynamic):
                        gen_queue_list.append(queue)
                    elif isinstance(queue, GenQueue):
                        # Wrap the GenQueue in a GenQueueDynamic
                        gen_queue_list.append(
                            GenQueueDynamic(existing_queue=queue, include_GenStats=True, include_GenEnd=True))
                    else:
                        raise TypeError("gen_queue list must contain only GenQueue or GenQueueDynamic objects")
            else:
                raise TypeError("gen_queue must be either a GenQueue, GenQueueDynamic, or a list of GenQueueDynamic")

            # Get generation settings
            settings = self._get_exllama_gen_settings(
                temperature,
                top_p=top_p,
                top_k=top_k,
            )

            # Vision support - get image embedding and construct the prompt with placeholder tokens for images
            prompt, image_embeddings = self._process_vision_inputs(prompt, vision_token, messages, video)
            if image_embeddings:
                inner_generator = self.pipeline.generator.generator
                if (
                    inner_generator.draft_model is not None
                    and not inner_generator.dflash_draft
                    and not inner_generator.mtp_draft
                ):
                    raise ValueError(
                        "ExLlamaV3 normal draft flash does not support multimodal embeddings; "
                        "disable draft or use a DFlash/MTP draft model."
                    )

            # Create filters for format enforcement
            # for now the filter is only for prefix string
            filters = None
            if json_schema:
                filters = self._create_generation_filters(json_schema)

            # add prefix string
            elif prefix_strings:
                if isinstance(prefix_strings, str):
                    prompt += prefix_strings
                elif isinstance(prefix_strings, list):
                    prefix_strings = prefix_strings[0]
                    prompt += prefix_strings

            # Convert prompt to token IDs
            if image_embeddings:
                input_ids = self.tokenizer.encode(
                    prompt,
                    encode_special_tokens=True,
                    embeddings=image_embeddings,
                )
            else:
                input_ids = self.tokenizer.encode(
                    prompt,
                    encode_special_tokens=True
                )

            self.validate_token_length(len(input_ids[0]))


            # Find stop conditions
            if stop_words:
                if isinstance(stop_words, str):
                    stop_words = [stop_words]

                if not self.eos_token_str:
                    raise Exception("EOS token not set in model_config")
                stop_conditions = self.eos_token_str + stop_words  # Concatenate the two lists
                logger.debug("stop_words: " + str(stop_conditions))
            else:
                stop_conditions = self.eos_token_str

            job_id = uuid.uuid4().hex

            # Calculate max tokens to use
            max_tokens_to_use = min(
                self.max_seq_len - len(input_ids[0]),
                max_tokens) \
                if max_tokens else self.max_seq_len - len(input_ids[0])

            logger.info("stop_conditions: " + str(stop_conditions))
            if not quiet:
                logger.info("----------------------Prompt---------------\n" + prompt)
                logger.debug("----------------------temperature---------\n" + str(temperature))

            # Prepare arguments for the job
            argument_list = {
                "generator": self.pipeline.generator,
                "input_ids": input_ids,
                "max_new_tokens": max_tokens_to_use,
                "sampler": settings,
                "stop_conditions": stop_conditions,
                "banned_strings": banned_strings,
                "decode_special_tokens": True,
                "filters": filters,
                "token_healing": False,
                "identifier": job_id,
            }

            # Add image embeddings if available
            if image_embeddings:
                argument_list["embeddings"] = image_embeddings

            # Create the job
            job = AsyncJob(**argument_list)

            generate_text = ""
            gen_stats = None
            eos = False
            wall_generation_start = None

            # Kick-start the generation and let downstream know the generation type
            if isinstance(gen_type, str):
                gen_type_str = gen_type
                gen_type = GenStart(gen_type=gen_type)
            elif isinstance(gen_type, GenStart) and isinstance(gen_type.gen_type, TagDefinition):
                gen_type_str = "text"
            else:
                gen_type_str = gen_type.gen_type  # Get the generation type in string format

            for g_queue in gen_queue_list:
                g_queue.put_nowait(gen_type)

            # Create a task to check for disconnection
            disconnect_check_task = None
            if request:
                disconnect_check_task = asyncio.create_task(self.check_disconnection(request, job, gen_queue_list, stop_event=stop_event))

            try:
                # send the prefix first
                if prefix_strings:
                    prefix_chunk = GenText(content=prefix_strings, text_type=gen_type_str)
                    for g_queue in gen_queue_list:
                        g_queue.put_nowait(prefix_chunk)

                # Start the generation
                async for result in job:
                    if eos or self._stop_event_is_set(stop_event):
                        await self._cancel_job_safely(job)
                        break

                    if not isinstance(result, dict):
                        logger.error(
                            f"BUG: AsyncJob yielded non-dict result! "
                            f"type={type(result).__name__}, "
                            f"repr={repr(result)[:500]}"
                        )
                        result = dict(result) if hasattr(result, '__iter__') else {"text": str(result)}

                    if result.get("stage") == "streaming" and wall_generation_start is None:
                        wall_generation_start = time.perf_counter()

                    chunk_text = result.get("text", "")
                    if chunk_text:
                        # logger.info(f"chunk_text: {chunk_text}")
                        chunk = GenText(content=chunk_text, text_type=gen_type_str)
                        for g_queue in gen_queue_list:
                            if chunk_text not in self.eos_token_str_set:  # Formatron returns EOS token
                                g_queue.put_nowait(chunk)

                    # Handle EOS signal
                    if result["eos"]:
                        eos = True
                        # logger.info(f"eos result {result}")
                        # If the stop word occurred is from the stop_words and not LLM result token -> include in result

                        stop_word_used = ""
                        if stop_words and result.get("held") and result.get("held").get("text"):
                            ending_string = result["held"]["text"].rstrip()

                            if ending_string:
                                # Find the stop word that was used to end the string
                                stop_word_used = self.get_stop_word(ending_string, stop_words)

                                if stop_word_used and return_stop_word:
                                    # If generation ended with one of the stop words
                                    # -> return that stop word as the last token
                                    chunk = GenText(content=stop_word_used, text_type=gen_type_str)
                                    for g_queue in gen_queue_list:
                                        g_queue.put_nowait(chunk)
                        use_stop_words = bool(stop_word_used)

                        # get stop reason
                        stop_reason = self.get_stop_reason(result, use_stop_words)

                        if send_eos:
                            measured_time_generate = None
                            if wall_generation_start is not None:
                                measured_time_generate = max(time.perf_counter() - wall_generation_start, 0)

                            # refer exllama generator.py for detail
                            gen_stats = GenerationStats(
                                input_tokens_count=result["prompt_tokens"],
                                output_tokens_count=result["new_tokens"],
                                time_to_first_token=result["time_prefill"],
                                time_generate=result["time_generate"],
                                measured_time_generate=measured_time_generate,
                                cached_pages=result["cached_pages"],
                                cached_tokens=result["cached_tokens"],
                                accepted_draft_tokens=result.get("accepted_draft_tokens"),
                                rejected_draft_tokens=result.get("rejected_draft_tokens"),
                                stop_reason=stop_reason,
                                stop_sequence=stop_word_used or None,
                            )

                            for g_queue in gen_queue_list:
                                if g_queue.include_GenStats:
                                    g_queue.put_nowait(gen_stats)

                            # Signal the end of generation
                            for g_queue in gen_queue_list:
                                if g_queue.include_GenEnd:
                                    g_queue.put_nowait(GenEnd())

                        full_completion = result["full_completion"] + stop_word_used

                        return full_completion

            except Exception as e:
                logger.error(e, exc_info=True)
                if stop_event:
                    stop_event.set()
                await self._cancel_job_safely(job)
                self.pipeline = None
                self._put_gen_end(gen_queue_list)
                raise
            finally:
                if disconnect_check_task:
                    disconnect_check_task.cancel()
                    try:
                        await asyncio.wait_for(disconnect_check_task, timeout=0.1)
                    except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                        pass
        except Exception as e:
            logger.error(e)
            raise e
