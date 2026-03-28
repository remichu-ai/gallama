import re
from gallama.backend.llm.engine.base import (
    ModelInterface,
)
from typing import Optional, Dict, List, Union, get_args
import torch
import asyncio
from fastapi import Request                 # for type hint
from functools import lru_cache             # for image caching
import uuid                                 # use for generating id for api return

from gallama.logger.logger import logger
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
from gallama.utils.utils import get_image, get_free_vram_gb

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
        FormatronFilter
    )
except ImportError:
    Model = None
    Config = None
    Cache = None
    Tokenizer = None
    AsyncGenerator = None
    Job = None

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

    if normalized.get("max_chunk_size") is None:
        normalized["max_chunk_size"] = 4096

    return normalized

class ModelExllamaV3(ModelInterface):
    def __init__(self, model_spec:ModelSpec):
        super().__init__(model_spec)
        self.model, self.tokenizer, self.cache, self.processor = self.load_model()

    @property
    def support_concurrency(self) -> bool:
        """
        whether this backend/ model support concurrent request
        """
        return True

    def load_model(self):
        """Load the model, tokenizer, cache, and optional processor."""

        model, tokenizer, cache, processor = self.load_model_exllama(
            model_id=self.model_id,
            backend=self.backend,
            max_seq_len=self.max_seq_len,
            cache_size=self.cache_size,
            cache_quant=self.cache_quant,
            gpus=self.gpus,
            reserve_vram=self._reserve_vram,
            tensor_parallel=self.tensor_parallel,
            backend_extra_args=self.backend_extra_args,
        )

        # # load draft model
        # if self.draft_model_id:
        #     # tokenizer and processor already set above
        #     self.draft_model, _, self.draft_cache, _ = self.load_model_exllama(
        #         model_id=self.draft_model_id,
        #         backend=self.backend,
        #         max_seq_len=self.max_seq_len,  # draft model max_seq_len must be same as main model
        #         cache_size=self.draft_cache_size,
        #         cache_quant=self.draft_cache_quant,
        #         gpus=self.draft_gpus,
        #         reserve_vram=self._reserve_vram,
        #     )

        self.eos_token_ids = self.generate_eos_tokens_id(tokenizer)

        return model, tokenizer, cache, processor


    @property
    def _reserve_vram(self):
        try:
            reserve_block_size = 1024 ** 2
            num_devices = torch.cuda.device_count()
            #reserved_vram = [192 * 1024**2] + [64 * 1024**2] * (num_devices - 1)
            #reserved_vram = [256 * 1024 ** 2] + [96 * 1024 ** 2] * (num_devices - 1)

            # GPU1 is the main GPU for my PC
            # The below is lower threshold than exllamav2 default setting
            reserve_per_gpu = [48 for _ in range(num_devices)]
            main_gpu = 0    # TODO pass it to front end
            reserve_per_gpu[main_gpu] = 96
            reserved_vram = [_reserve * reserve_block_size for _reserve in reserve_per_gpu]
            return reserved_vram
        except:
            # for non cuda env e.g. macbook
            return None

    def load_model_exllama(self, model_id, backend, cache_size, cache_quant, gpus, reserve_vram, max_seq_len=None, tensor_parallel=False, backend_extra_args=None):
        """This function return the model and its tokenizer"""
        logger.info("Loading model: " + model_id)

        config = Config.from_directory(model_id)
        model = Model.from_config(config)
        tokenizer = Tokenizer.from_config(config)
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

        # # a simple dict to help map cache quant
        cache_quant_dict = {
            "FP16": None,
            "Q4": {"k_bits": 4, "v_bits": 4},
            "Q6": {"k_bits": 6, "v_bits": 6},
            "Q8": {"k_bits": 8, "v_bits": 8},
        }

        # cache size needed to minimally max_seq_len size
        # Use provided cache_size or default to max_seq_len
        base_size = cache_size or self.max_seq_len

        # Align to 256, but ensure it is at least max_seq_len
        cache_size_to_use = (max(base_size, self.max_seq_len) // 256) * 256

        # get the cache quantization to use
        cache_quant_to_use = cache_quant_dict.get(cache_quant, None)

        logger.info("max_seq_len: " + str(self.max_seq_len))
        logger.info("cache_size: " + str(cache_size_to_use))
        logger.info("Cache Quantization: " + str(cache_quant))
        logger.info("gpus: " + str(gpus))

        assert (isinstance(gpus, str) and gpus == "auto") or (isinstance(gpus, list)), \
            "Device map should be either 'auto', 'gpu' split"

        # create the layer if cache quant is needed
        cache_layer = None

        # TODO cache quant
        if cache_quant_to_use:
            cache_layer = CacheLayer_quant

        if cache_quant_to_use:
            logger.info("Using cache quant")
            cache = Cache(
                model,
                max_num_tokens=cache_size_to_use,
                layer_type=cache_layer,
                **cache_quant_to_use
            )
        else:
            # FP16
            logger.info("Not using cache quant")
            cache = Cache(
                model,
                max_num_tokens=cache_size_to_use
            )
        # since exl3 only allow the reserve or use per device
        # TODO allow user to set reserve per device

        # find out free vramclear
        if gpus == "auto":
            free_vram = get_free_vram_gb()
            vram_reserve = 0.4
            free_vram = [max(0, f-vram_reserve) for f in free_vram]
            gpus = free_vram if free_vram else gpus

        load_kwargs = {
            "progressbar": True,
            "use_per_device": gpus if isinstance(gpus, list) else None,
            "tensor_p": tensor_parallel,
        }

        tp_backend = (backend_extra_args or {}).get("tp_backend")
        if tp_backend:
            logger.info("Tensor parallel backend: " + str(tp_backend))
            load_kwargs["tp_backend"] = tp_backend

        model.load(
            **load_kwargs,
        )

        # load vision processor if there is
        # if there is error, assume that the model doesnt have vision
        processor = None
        try:
            processor = Model.from_config(config, component = "vision")
            processor.load()
        except AssertionError:
            logger.info("No Vision Tower")
            processor = None

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
            **generator_kwargs,
        )

        return self.ExllamaV3Pipeline(
            cache=self.cache,
            generator=generator,
        )

    @staticmethod
    def _get_exllama_gen_settings(
        temperature: float = 0.01,
        top_p: float = 0.8,
        **kwargs,
    ):
        # settings
        settings = CustomSampler([
            SS_Temperature(temperature),
            SS_TopP(top_p),
            SS_Sample()
        ])

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
            # Ensure that the generator is initialized
            if self.pipeline is None:
                self.pipeline = await self._get_pipeline_async()

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
            settings = self._get_exllama_gen_settings(temperature, top_p=top_p)

            # Vision support - get image embedding and construct the prompt with placeholder tokens for images
            prompt, image_embeddings = self._process_vision_inputs(prompt, vision_token, messages, video)

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
                    encode_special_tokens=False
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
                "token_healing": True,
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
                    if eos or stop_event.is_set():
                        await job.cancel()
                        break

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

                        use_stop_words = False
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

                                    use_stop_words = True

                        # get stop reason
                        stop_reason = self.get_stop_reason(result, use_stop_words)

                        if send_eos:
                            # refer exllama generator.py for detail
                            gen_stats = GenerationStats(
                                input_tokens_count=result["prompt_tokens"],
                                output_tokens_count=result["new_tokens"],
                                time_to_first_token=result["time_prefill"],
                                time_generate=result["time_generate"],
                                cached_pages=result["cached_pages"],
                                cached_tokens=result["cached_tokens"],
                                stop_reason=stop_reason
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
                logger.error(e)
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
