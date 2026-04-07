from ..base import (
    ModelInterface,
)
from typing import Any, Optional, Dict, List, Union
import time                                 # for compute of generation time
import asyncio
from fastapi import Request                 # for type hint
import transformers
from gallama.utils.request_disconnect import (
    format_exception_summary,
    is_expected_disconnect_exception,
    is_request_disconnected,
)
from gallama.utils.utils import get_image
from functools import lru_cache

try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    from vllm.inputs import TextPrompt
    from vllm.multimodal.inputs import ImageItem, VideoItem, AudioItem
except ImportError:
    AsyncLLMEngine, AsyncEngineArgs, SamplingParams, GuidedDecodingParams = None, None, None, None
    TextPrompt, ImageItem, VideoItem, AudioItem = None, None, None, None

# format enforcement with formatron
try:
    from formatron.formatter import FormatterBuilder
    from formatron.integrations.transformers import create_formatter_logits_processor_list
except ImportError:
    FormatterBuilder = None
    create_formatter_logits_processor_list = None

# format enforcement with lmfe
try:
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )
except ImportError:
    TokenEnforcerTokenizerData = None
    build_transformers_prefix_allowed_tokens_fn = None

from ...format_enforcer import SGLangFormatter

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    process_mm_info = None


from gallama.utils import is_flash_attention_installed
from .....logger.logger import logger

import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)  # disable logging from sglang

# custom data classes
from .....data_classes import (
    BaseMessage,
    ModelSpec,
    GenStart,
    GenEnd,
    GenQueue,
    GenText,
    GenerationStats,
    QueueContext,
    GenQueueDynamic,
    VideoFrame
)


class ModelVLLM(ModelInterface):
    def __init__(self, model_spec:ModelSpec):
        super().__init__(model_spec)
        self.model, self.tokenizer, self.processor = self.load_model()

    @property
    def support_concurrency(self) -> bool:
        """
        whether this backend/ model support concurrent request
        """
        return True

    def load_model(self):
        """Load the model, tokenizer, cache, and optional processor."""
        # processor is for multimodal
        model, tokenizer, processor = self.load_model_vllm(
            model_id=self.model_id,
        )

        # load draft model
        if self.draft_model_id is not None:
            raise "Draft model currently not supported for vllm backend"

        self.eos_token_ids = self.generate_eos_tokens_id()

        return model, tokenizer, processor

    def load_model_vllm(
        self,
        model_id,
    ):
        """This function return the model and its tokenizer"""
        logger.info("Loading model: " + model_id)

        cache = None  # in case not a backend with separate cache like llama cpp
        tokenizer = None
        processor = None


        # arguments for model loading
        model_kargs = {
            'model': self.model_id,
        }


        # determine the class to use for loading
        if self.backend_extra_args.get('model_class_extra_kwargs'):

            model_extra_kwargs = self.backend_extra_args.get('model_class_extra_kwargs')
            if model_extra_kwargs:
                model_kargs.update(model_extra_kwargs)      # update any extra argument

        # set max_seq_len based on model    TODO: To find more reliable method
        self.max_seq_len = self.backend_extra_args.get('max_model_len') or 32768

        logger.info(model_kargs)
        engine_args = AsyncEngineArgs(**model_kargs)
        model = AsyncLLMEngine.from_engine_args(engine_args)

        # wrapped_tokenizer = AsyncEncodeWrapper(model)
        # fall back to transformers implementation for tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.model_id)

        return model, tokenizer, processor



    def generate_eos_tokens_id(self) -> List[int]:
        """Generate the end-of-sequence token IDs."""

        # currently the stop word for llama cpp work by string and not by token id
        # hence usage of this is not required
        return []


    # *************** generation method from here
    # helper function for job disconnection. Currently only exllama support this
    @staticmethod
    async def check_disconnection(
        model,
        request: Request,
        job_id: str,
        gen_queue_list: Union[GenQueueDynamic, GenQueue, QueueContext, List[QueueContext]],
        stop_event: asyncio.Event = None,
    ):
        """
        Helper function that handle stopping generation mid-stream
        """
        try:
            while True:
                if await is_request_disconnected(request):
                    logger.info("User disconnected")
                    await model.abort(job_id)

                    # add GenEnd to signal the end of generation
                    chunk = GenEnd()
                    for g_queue in gen_queue_list:
                        try:
                            if g_queue.include_GenEnd:
                                g_queue.put_nowait(chunk)
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
                raise
        finally:
            logger.debug("Exiting check_disconnection")

    @staticmethod
    @lru_cache(1024)  # TODO set this dynamically
    def get_image_cached(url):
        """
        function to return PIL.Image object
        with cache
        """
        img = get_image(url=url)
        return img

    async def generate(
        self,
        prompt: str,
        gen_queue: Union[GenQueue, QueueContext, List[QueueContext]],
        request: Optional[Request] = None,
        gen_type: Union[str, GenStart] = "text",    # the generated result will be store to this queue
        temperature: float = 0.01,
        top_p: float = 0.8,
        # TODO simplify the func argument for formatter
        formatter: Any = None,
        stop_words: Union[List[str], str] = None,
        prefix_strings: Optional[Union[str, List[str]]] = None,
        banned_strings: list[str] | None = None,
        max_tokens: int = None,
        quiet=False,
        messages: List[BaseMessage] = None,  # query.message for multimodal
        video: List[VideoFrame] = None,
        stop_event: asyncio.Event = None,
        **kwargs,
    ) -> (str, GenerationStats):

        if not quiet:
            logger.info("----------------------Prompt---------------\n" + prompt)
            logger.debug("----------------------temperature---------\n" + str(temperature))


        # make gen_queue to List[QueueContext] for standardize downstream handling
        gen_queue_list = None
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


        # vision support
        audios_input, image_inputs, video_inputs = None, None, None

        video_frame = []
        if video:
            video_frame = [frame.get_image() for frame in video]

        image_list = []
        if messages:
            messages_as_dicts = [message.model_dump() for message in messages]

            # find all the image
            for one_message in messages_as_dicts:
                if isinstance(one_message["content"], list):
                    for message in one_message["content"]:
                        if message.get("type") == "image_url":
                            # get the image into PIL.Image object
                            image_list.append(self.get_image_cached(message["image_url"]["url"]))


        input_ids = []

        # prefix is not an option in vllm and wont work well with format enforcer

        prefix_to_return = ""
        if not formatter and prefix_strings:
            if isinstance(prefix_strings, str):
                prefix_to_return = prefix_strings
            elif isinstance(prefix_strings, list):
                prefix_to_return = prefix_strings[0]

            prompt += prefix_to_return


        # self.validate_token_length(len(input_ids))

        # # format enforcer
        # logits_processor = None
        # prefix_allowed_tokens_fn = None
        # if formatter:
        #     if isinstance(formatter, FormatterBuilder):
        #         logits_processor = create_formatter_logits_processor_list(self.tokenizer, formatter)
        #     else:
        #         prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, formatter)

        # construct multimodal prompt
        multi_modal_data: Dict[str, Any] = {}
        if video and video_frame:
            multi_modal_data["video"] = video_frame

        if image_list:
            multi_modal_data["image"] = image_list

        multimodal_prompt = TextPrompt(
            prompt=prompt,
            multi_modal_data=multi_modal_data if multi_modal_data else None,
        )

        # find stop conditions
        if stop_words:
            if isinstance(stop_words, str):
                stop_words = [stop_words]

            if not self.eos_token_str:
                raise Exception("EOS token not set in model_config")

            stop_conditions = self.eos_token_str + stop_words  # concat the 2 list
            logger.debug("stop_words: " + str(stop_conditions))

        max_tokens_to_use = min(
            self.max_seq_len - len(input_ids),
            max_tokens, 4096) if max_tokens else min(self.max_seq_len - len(input_ids), 4096)

        # Get generation settings
        settings = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens_to_use,
            stop=stop_words,
            bad_words=banned_strings,
            guided_decoding = formatter if formatter else None
        )

        # format enforcement
        # if formatter:
        #     raise "tool not supported for SG lang yet"
            # settings.update(formatter.get_formatter_dict())

        # logger.info("FROG FROG" + str(settings))

        # Kick-start the generation and let downstream know the generation type
        if isinstance(gen_type, str):
            gen_type_str = gen_type
            gen_type = GenStart(gen_type=gen_type)
        else:
            gen_type_str = gen_type.gen_type  # Get the generation type in string format

        for g_queue in gen_queue_list:
            g_queue.put_nowait(gen_type)



        start_time = time.perf_counter()
        job_id = str(time.monotonic())
        job = self.model.generate(
            prompt=multimodal_prompt,
            sampling_params=settings,
            request_id=job_id
        )

        # Create a task to check for disconnection
        disconnect_check_task = None
        if request:
            disconnect_check_task = asyncio.create_task(
                self.check_disconnection(self.model, request, job_id, gen_queue_list, stop_event=stop_event))

        first_token_time = None

        # generate
        generate_text = ""
        previous_text = ""  # track the current text as vllm return who text and not just delta
        gen_stats = None
        eos = False

        try:
            # Start the generation
            async for result in job:
                # sglang will set job object to None if eos reached, hence need to break at the end of the loop
                if eos or stop_event.is_set():
                    await self.model.abort(job_id)
                    break

                # logger.info(result)
                previous_text = generate_text
                chunk_text = prefix_to_return + result.outputs[0].text
                chunk_text = chunk_text[len(previous_text):]

                if chunk_text:
                    if not first_token_time:
                        first_token_time = time.perf_counter()


                    # add the chunk to overall result
                    generate_text += chunk_text

                    # logger.info(f"chunk_text: {chunk_text}")
                    chunk = GenText(content=chunk_text, text_type=gen_type_str)
                    for g_queue in gen_queue_list:
                        if chunk_text not in self.eos_token_str_set:  # Formatron returns EOS token
                            g_queue.put_nowait(chunk)

                # Handle EOS signal
                if result.finished:
                    eos = True
                    meta_info = result.outputs[0]

                    logger.info(f"eos result {result}")
                    # If the stop word occurred is from the stop_words and not LLM result token -> include in result
                    if meta_info.finish_reason == "stop":

                        if meta_info.stop_reason and meta_info.stop_reason!=self.eos_token_str:
                            # vllm return the stopped word in stop_reason
                            chunk = GenText(content=meta_info.stop_reason, text_type=gen_type_str)
                            for g_queue in gen_queue_list:
                                g_queue.put_nowait(chunk)

                    gen_stats = GenerationStats(
                        input_tokens_count=len(result.prompt_token_ids),
                        output_tokens_count=len(meta_info.token_ids),
                        time_to_first_token=first_token_time - start_time,
                        time_generate=time.perf_counter() - start_time,
                    )

                    for g_queue in gen_queue_list:
                        if g_queue.include_GenStats:
                            g_queue.put_nowait(gen_stats)

                    # Signal the end of generation
                    for g_queue in gen_queue_list:
                        if g_queue.include_GenEnd:
                            g_queue.put_nowait(GenEnd())

                    # end the loop
                    break

        except Exception as e:
            logger.error(e)
        finally:
            pass
            if disconnect_check_task:
                disconnect_check_task.cancel()
                try:
                    await disconnect_check_task
                except:
                    pass

        logger.debug("----------------------LLM Raw Response---------------\n" + generate_text)
