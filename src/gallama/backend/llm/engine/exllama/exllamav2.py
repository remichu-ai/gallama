from gallama.backend.llm.engine.base import ModelInterface
from typing import Optional, Dict, List, Union
import torch
import asyncio
from fastapi import Request                 # for type hint
from importlib.metadata import version      # for checking of exllama version
from functools import lru_cache             # for image caching
import uuid                                 # use for generating id for api return

from gallama.logger.logger import logger
from gallama.data_classes import (
    ModelSpec,
    GenStart,
    GenEnd,
    GenQueue,
    GenText,
    GenerationStats,
    QueueContext,
)
from gallama.utils.utils import get_image


try:
    from exllamav2 import (
        ExLlamaV2,
        ExLlamaV2Tokenizer,
        ExLlamaV2Cache,
        ExLlamaV2Cache_Q4,
        ExLlamaV2Cache_Q6,
        ExLlamaV2Cache_Q8,
        ExLlamaV2Config,
    )
    from exllamav2.generator import (
        ExLlamaV2Sampler,
        ExLlamaV2DynamicGeneratorAsync,
        ExLlamaV2DynamicJobAsync,
    )

    from exllamav2.generator.filters import ExLlamaV2PrefixFilter

    if version('exllamav2') == '0.2.1' or version('exllamav2') == '0.2.2':
        raise "Please use version 0.2.3 onwards. There is some bug with v0.2.1 and 0.2.2 related with format enforcement"

except ImportError:
    ExLlamaV2 = None
    ExLlamaV2Tokenizer = None
    ExLlamaV2Cache = None
    ExLlamaV2Cache_Q4 = None
    ExLlamaV2Cache_Q6 = None
    ExLlamaV2Cache_Q8 = None
    ExLlamaV2Config = None
    ExLlamaV2Sampler = None
    ExLlamaV2DynamicGeneratorAsync = None
    ExLlamaV2DynamicJobAsync = None
    ExLlamaV2PrefixFilter = None


# tensor parallel from v0.1.9 onward
try:
    from exllamav2 import ExLlamaV2Cache_TP
except ImportError:
    # optional dependency
    ExLlamaV2Cache_TP = None

# vision support from v0.2.4 onward
try:
    from exllamav2 import ExLlamaV2VisionTower
except ImportError:
    ExLlamaV2VisionTower = None

# format enforcement with formatron
try:
    from formatron.formatter import FormatterBuilder
    from formatron.integrations.exllamav2 import create_formatter_filter
except ImportError:
    FormatterBuilder = None
    create_formatter_filter = None

# format enforcement with lmfe
from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
from lmformatenforcer.integrations.exllamav2 import (
    ExLlamaV2TokenEnforcerFilter,
    build_token_enforcer_tokenizer_data
)
from lmformatenforcer import JsonSchemaParser
try:
    # lm format enforcer does not work correctly without update with latest api from exllama
    # this wrapper class aim as stop gap solution and formatron is recommended instead
    from gallama.backend.llm.engine.exllama.inference_json_lmfe_wrapper import ExLlamaV2TokenEnforcerFilter as ExLlamaV2TokenEnforcerFilterTemp
except ImportError:
    TokenEnforcerTokenizerData = None
    ExLlamaV2TokenEnforcerFilter = None
    build_token_enforcer_tokenizer_data = None
    ExLlamaV2TokenEnforcerFilterTemp = None
    JsonSchemaParser = None


class ModelExllama(ModelInterface):
    def __init__(self, model_spec:ModelSpec):
        super().__init__(model_spec)
        self.model, self.tokenizer, self.cache, self.processor = self.load_model()


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
        )

        # load draft model
        if self.draft_model_id is not None:
            # tokenizer and processor already set above
            draft_model, _, draft_cache, _ = self.load_model_exllama(
                model_id=self.draft_model_id,
                backend=self.backend,
                max_seq_len=self.max_seq_len,  # draft model max_seq_len must be same as main model
                cache_size=self.draft_cache_size,
                cache_quant=self.draft_cache_quant,
                gpus=self.draft_gpus,
                reserve_vram=self._reserve_vram,
            )

        self.eos_token_ids = self.generate_eos_tokens_id()

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
            reserve_per_gpu = [32 for _ in range(num_devices)]
            main_gpu = 0    # TODO pass it to front end
            reserve_per_gpu[main_gpu] = 64
            reserved_vram = [_reserve * reserve_block_size for _reserve in reserve_per_gpu]
            return reserved_vram
        except:
            # for non cuda env e.g. macbook
            return None

    def load_model_exllama(self, model_id, backend, cache_size, cache_quant, gpus, reserve_vram, max_seq_len=None, tensor_parallel=False):
        """This function return the model and its tokenizer"""
        logger.info("Loading model: " + model_id)

        # initialize
        cache = None
        tokenizer = None

        config = ExLlamaV2Config(model_id)
        if max_seq_len is not None:
            config.max_seq_len = max_seq_len
        else:
            # set the self.max_seq_len using model config file as it is None at the moment
            max_seq_len = config.max_seq_len
            self.max_seq_len = config.max_seq_len

        model = ExLlamaV2(config)
        tokenizer = ExLlamaV2Tokenizer(config)

        # a simple dict to help map cache quant
        cache_quant_dict = {
            "FP16": ExLlamaV2Cache,
            "Q4": ExLlamaV2Cache_Q4,
            "Q6": ExLlamaV2Cache_Q6,
            "Q8": ExLlamaV2Cache_Q8,
        }

        # cache size need to minimally max_seq_len size
        cache_size_to_use = cache_size if cache_size else config.max_seq_len
        cache_size_to_use = (cache_size_to_use//256) * 256      # round to multiplier of 256 for paged attention
        # ensure cache_size is minimally max_seq_len
        cache_size_to_use = max(cache_size_to_use, max_seq_len)

        # get the cache quantization to use
        cache_quant_to_use = cache_quant_dict[cache_quant]

        logger.info("max_seq_len: " + str(self.max_seq_len))
        logger.info("cache_size: " + str(cache_size_to_use))
        logger.info("Cache Quantization: " + str(cache_quant))

        assert cache_quant_to_use is not None
        assert (isinstance(gpus, str) and gpus == "auto") or (isinstance(gpus, list)), \
            "Device map should be either 'auto', 'gpu' split"

        if not tensor_parallel:
            if isinstance(gpus, str) and gpus == "auto":
                cache = cache_quant_to_use(model, max_seq_len=cache_size_to_use, lazy=True)
                model.load_autosplit(cache, reserve_vram=reserve_vram, progress=True)
            elif isinstance(gpus, list):      # user specify the gpus split
                logger.info("Custom GPU Allocation in GB: " + str(gpus))
                model.load(gpu_split=gpus, progress=True)
                cache = cache_quant_to_use(model, max_seq_len=cache_size_to_use, lazy=not model.loaded)
        else:
            # tensor parallel mode
            logger.info("ExllamaV2 Tensor Parallel enabled")
            if ExLlamaV2Cache_TP:       # ensure that tensor parallel is available
                model.load_tp(progress=True, gpu_split = gpus if isinstance(gpus, list) else None)
                cache = ExLlamaV2Cache_TP(
                    model,
                    max_seq_len = cache_size_to_use,
                    base = cache_quant_to_use,
                )
            else:
                raise ValueError("ExllamaV2 was not installed with tensor parallel")

        # load vision processor
        processor = None
        if ExLlamaV2VisionTower:
            try:
                processor = ExLlamaV2VisionTower(config)
                processor.load(progress=True)
            except:
                processor = None

        return model, tokenizer, cache, processor



    def generate_eos_tokens_id(self) -> List[int]:
        """Generate the end-of-sequence token IDs."""
        if self.eos_token_str:
            # exllama
            if ExLlamaV2Tokenizer and isinstance(self.tokenizer, ExLlamaV2Tokenizer):
                eos_token_ids = [self.tokenizer.single_id(token) for token in self.eos_token_str]
                return eos_token_ids
        else:
            return []


    # ********* from here is generation methods
    # helper function for job disconnection. Currently only exllama support this
    @staticmethod
    async def check_disconnection(
        request: Request,
        job: ExLlamaV2DynamicJobAsync,
        gen_queue_list: Union[GenQueue, QueueContext, List[QueueContext]]
    ):
        """
        Helper function that handle stopping generation mid-stream
        """
        try:
            while True:
                if await request.is_disconnected():
                    logger.info("User disconnected")
                    await job.cancel()

                    # add GenEnd to signal the end of generation
                    chunk = GenEnd()
                    for g_queue in gen_queue_list:
                        try:
                            await g_queue.get_queue().put(chunk)
                        except Exception as e:
                            logger.error(f"Error putting GenEnd into queue: {str(e)}")

                    # break the while loop
                    break

                # Use asyncio.wait_for to implement a timeout
                try:
                    await asyncio.wait_for(asyncio.sleep(1), timeout=1.1)
                except asyncio.TimeoutError:
                    # This allows us to check for cancellation more frequently
                    pass

        except asyncio.CancelledError:
            logger.info("Disconnection check was cancelled")
        except Exception as e:
            logger.error(f"An error occurred in check_disconnection: {str(e)}")
        finally:
            logger.info("Exiting check_disconnection")


    class ExllamaV2Pipeline:
        """ class as wrapper for objects required for Exllama V2 text generation"""

        def __init__(
            self,
            cache: Union[ExLlamaV2Cache, ExLlamaV2Cache_Q4],
            generator: ExLlamaV2DynamicGeneratorAsync,
            lm_enforcer_tokenizer_data: TokenEnforcerTokenizerData,
        ):
            self.cache = cache
            self.generator = generator
            self.lm_enforcer_tokenizer_data = lm_enforcer_tokenizer_data

    async def _get_pipeline_async(self):
        """
        create generator for exllama and also build the tokenizer data for lmfe
        as ExLlamaV2DynamicGeneratorAsync is async function, it can not be run in the class __init__
        this will be run in the first text generation to ensure that generator is initialized
        """

        lm_enforcer_tokenizer_data = build_token_enforcer_tokenizer_data(self.tokenizer)

        generator = ExLlamaV2DynamicGeneratorAsync(
            model=self.model,
            cache=self.cache,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            draft_model=self.draft_model,
            draft_cache=self.draft_cache,
            num_draft_tokens=5,
        )

        return self.ExllamaV2Pipeline(
            cache=self.cache,
            generator=generator,
            lm_enforcer_tokenizer_data=lm_enforcer_tokenizer_data,
        )

    @staticmethod
    def _get_exllama_gen_settings(
        temperature: float = 0.01,
        top_p: float = 0.8,
        **kwargs,
    ):
        # settings
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = temperature
        settings.min_temp = 0.15
        settings.top_k = 50
        settings.top_p = top_p
        settings.min_p = 0.05
        settings.token_repetition_penalty = 1.1
        settings.token_frequency_penalty = 0.05
        settings.token_repetition_range = 1024
        # settings.token_repetition_decay: int = 0.98
        settings.temperature_last = False

        return settings

    @staticmethod
    def get_stop_word(text, stop_words) -> Union[str, None]:
        """ this function will match the stop word used given the text that llm ended generation with and a list of stop_words."""

        # sort the list by length to find the longest first
        sorted_stop_words = sorted(stop_words, key=len, reverse=True)

        text = text.lstrip()  # Remove trailing whitespace
        for stop_word in stop_words:
            if stop_word in text:
                return stop_word

        return None

    @staticmethod
    @lru_cache(128)     # TODO set this dynamically
    def get_image_embedding_cached(processor, model, tokenizer, url):
        """
        function to return image embedding for exllama
        lru_cache to cache frequently used image
        """
        img = get_image(url=url)

        return processor.get_image_embeddings(
            model=model,
            tokenizer=tokenizer,
            image=img,
            text_alias=None,    # passing None will let me llm generate its one embedding
        )

    def _generate_image_embeddings(self, prompt, image_list):
        """Generate embeddings for images and update prompt"""
        image_token = "{{IMG-PlaceHolderTokenHere}}"

        # Validate image token count matches number of images
        assert prompt.count(image_token) == len(image_list), "Image token mismatch"

        # Generate embeddings
        image_embeddings = [
            self.get_image_embedding_cached(
                processor=self.processor,
                model=self.model,
                tokenizer=self.tokenizer,
                url=url
            ) for url in image_list
        ]

        # Replace image tokens with embeddings
        for emb in image_embeddings:
            prompt = prompt.replace(image_token, emb.text_alias, 1)

        return prompt, image_embeddings

    def _process_vision_inputs(self, prompt, messages):
        """Handle image embedding and token replacement for vision inputs"""

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
                vision_required = bool(image_urls)

        # Process image embeddings if vision is required
        if vision_required and self.processor:
            return self._generate_image_embeddings(prompt, image_list)

        return prompt, None

    def _get_format_enforcer_filter(self, formatter):
        """Determine appropriate format enforcer filter"""
        if isinstance(formatter, (TokenEnforcerTokenizerData, JsonSchemaParser)):
            # Logic for LM format enforcer
            exllama_version = version('exllamav2')
            return [
                ExLlamaV2TokenEnforcerFilterTemp(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    character_level_parser=formatter,
                ) if exllama_version > '0.2.0' else
                ExLlamaV2TokenEnforcerFilter(
                    character_level_parser=formatter,
                    tokenizer_data=self.pipeline.lm_enforcer_tokenizer_data
                )
            ]
        elif FormatterBuilder and isinstance(formatter, FormatterBuilder):
            # Logic for Formatron
            return [create_formatter_filter(self.model, self.tokenizer, formatter)]

        raise ValueError("Unsupported formatter type")

    def _create_generation_filters(
        self,
        formatter: Optional[Union[TokenEnforcerTokenizerData, FormatterBuilder]] = None,
        prefix_strings: Optional[Union[str, List[str]]] = None
    ) -> List:
        """Create filters for token generation"""
        filters = []

        # Format enforcer filters
        if formatter:
            filters.extend(self._get_format_enforcer_filter(formatter))

        # Prefix filters
        if prefix_strings:
            filters.append(
                ExLlamaV2PrefixFilter(self.model, self.tokenizer, prefix_strings)
            )

        return filters

    async def generate(
        self,
        prompt: str,
        gen_queue: Union[GenQueue, QueueContext, List[QueueContext]],
        request: Optional[Request] = None,      # for disconnection check
        gen_type: Union[str, GenStart] = "text",  # the generated result will be store to this queue
        temperature: float = 0.01,
        top_p: float = 0.8,
        formatter: FormatterBuilder | TokenEnforcerTokenizerData = None,
        stop_words: Union[List[str], str] = None,
        prefix_strings: Optional[Union[str, List[str]]] = None,
        banned_strings: list[str] | None = None,
        max_tokens: int = None,
        quiet=False,
        messages: List = None,  # query.message for multimodal
        **kwargs,
    ) -> (str, GenerationStats):
        try:

            # ensure that generator is initialized
            if self.pipeline is None:
                self.pipeline = await self._get_pipeline_async()

            # convert gen_queue to List[QueueContext] format to standardize downstream handling
            gen_queue_list = None
            if isinstance(gen_queue, QueueContext):
                gen_queue_list = [gen_queue]
            elif isinstance(gen_queue, GenQueue):
                gen_queue_list = [QueueContext.create(gen_queue, include_GenEnd=True, include_GenStats=True)]
            elif isinstance(gen_queue, list):
                gen_queue_list = gen_queue
            else:
                raise Exception("gen_queue must be either a GenQueue, QueueContext or a list of QueueContext")

            if not quiet:
                logger.info("----------------------Prompt---------------\n" + prompt)
                logger.debug("----------------------temperature---------\n" + str(temperature))

            # for async generator, create it as part of the generate job

            # get generation setting
            settings = self._get_exllama_gen_settings(temperature, top_p=top_p)

            # vision support - get image embedding and construct the prompt with placeholder token for images
            prompt, image_embeddings = self._process_vision_inputs(prompt, messages)

            # convert prompt to token id
            if image_embeddings:
                input_ids = self.tokenizer.encode(
                    prompt,
                    encode_special_tokens=True,
                    embeddings = image_embeddings,
                )
            else:
                input_ids = self.tokenizer.encode(
                    prompt,
                )

            self.validate_token_length(len(input_ids[0]))

            # Create filters for format enforcement
            filters = self._create_generation_filters(formatter, prefix_strings)

            # find stop conditions
            if stop_words:
                if isinstance(stop_words, str):
                    stop_words = [stop_words]

                if not self.eos_token_str:
                    raise Exception("EOS token not set in model_config")
                stop_conditions = self.eos_token_str + stop_words  # concat the 2 list
                logger.debug("stop_words: " + str(stop_conditions))
            else:
                stop_conditions = self.eos_token_str

            job_id = uuid.uuid4().hex

            # logger.info("pending_jobs and active_jobs lists in the ExLlamaV2DynamicGenerator")
            # logger.info(f"job_id: {self.pipeline.generator.jobs}")
            max_tokens_to_use = min(
                self.max_seq_len - len(input_ids[0]),
                max_tokens, 4096) if max_tokens else min(self.max_seq_len - len(input_ids[0]), 4096)

            # for backward support for older version without vision
            argument_list = {
                "generator" : self.pipeline.generator,
                "input_ids" : input_ids,
                "max_new_tokens" : max_tokens_to_use,
                "gen_settings" : settings,
                "stop_conditions" : stop_conditions,  # self.eos_token_id if self.eos_token_id else None,
                "banned_strings" : banned_strings,
                "decode_special_tokens" : True,
                "filters" : filters,
                "token_healing" : True,
                "identifier" : job_id,
            }

            # add image embedding
            if image_embeddings:
                argument_list["embeddings"] = image_embeddings

            job = ExLlamaV2DynamicJobAsync(
                **argument_list
            )

            generate_text = ""
            gen_stats = None
            eos = False

            # kick-start the generation and let down stream know gen type
            if isinstance(gen_type, str):
                gen_type_str = gen_type
                gen_type = GenStart(gen_type=gen_type)
            else:
                gen_type_str = gen_type.gen_type  # get out the generation type in str format

            for g_queue in gen_queue_list:
                g_queue.get_queue().put_nowait(gen_type)

            # Create a task to check for disconnection
            disconnect_check_task = None
            if request:
                disconnect_check_task = asyncio.create_task(self.check_disconnection(request, job, gen_queue_list))

            try :
                # start the generation
                async for result in job:
                    if eos:
                        await job.cancel()
                        break

                    # print(result.get("text", ""))
                    # If we enqueue multiple jobs, an iteration might produce results for any (or all) of them. We could direct
                    # outputs to multiple clients here, using whatever dispatch mechanism, but in this example there will only be
                    # outputs pertaining to the single job started above, and it will all go straight to the console.
                    # assert result["job"] == job

                    # Prefilling/ingesting the prompt may happen over multiple iterations, during which the result will have
                    # a "stage" value of "prefill". We can ignore those results and only use the "streaming" results that will
                    # contain the actual output.
                    # if result["stage"] == "streaming":

                    # Depending on settings, the result dict can contain top-K probabilities, logits and more, but we'll just
                    # grab the output text stream.
                    # logger.info(f'{datetime.now()} {result.get("text", "")}')
                    chunk_text = result.get("text", "")
                    chunk = GenText(content=chunk_text, text_type=gen_type_str)
                    for g_queue in gen_queue_list:
                        if chunk_text not in self.eos_token_str_set:        # formatron return eos token
                            # generate_text += result.get("text", "")
                            g_queue.get_queue().put_nowait(chunk)

                    # logger.info(result.get("text", ""))
                    # logger.info(self.tokenizer.encode(result.get("text", "")))
                    # The "streaming" stage also emits the EOS signal when it occurs. If present, it will accompany a
                    # summary of the job. Print the last packet here to illustrate.
                    if result["eos"]:
                        eos = True

                        # if the stop word occurred is from the stop_words and not llm result token -> include in result
                        if stop_words and result.get("held") and result.get("held").get("text"):
                            ending_string = result["held"]["text"].rstrip()

                            if ending_string:
                                # find the stop word that was used to end string
                                stop_word_used = self.get_stop_word(ending_string, stop_words)

                                if stop_word_used:
                                    # if generation ended with one of the stopword
                                    # -> return that stop word as the last token
                                    chunk = GenText(content=stop_word_used, text_type=gen_type_str)
                                    for g_queue in gen_queue_list:
                                        g_queue.get_queue().put_nowait(chunk)
                                else:
                                    # ending token is llm eos token
                                    pass

                        gen_stats = GenerationStats(
                            input_tokens_count=result["prompt_tokens"],
                            output_tokens_count=result["new_tokens"],
                            time_to_first_token=result["time_prefill"],
                            time_generate=result["time_generate"],
                        )

                        for g_queue in gen_queue_list:
                            if g_queue.include_GenStats:
                                g_queue.get_queue().put_nowait(gen_stats)

                        # this to signal the end of generation
                        for g_queue in gen_queue_list:
                            if g_queue.include_GenEnd:
                                g_queue.get_queue().put_nowait(GenEnd())
            except Exception as e:
                logger.error(e)
            finally:
                if disconnect_check_task:
                    disconnect_check_task.cancel()
                    try:
                        await disconnect_check_task
                    except asyncio.CancelledError:
                        pass
        except Exception as e:
            logger.error(e)