from ..base import ModelInterface
from typing import Optional, Dict, List, Union
import time                                 # for compute of generation time
import asyncio
from fastapi import Request                 # for type hint
import os
from concurrent.futures import ThreadPoolExecutor
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput
)
import transformers

try:
    import sglang as sgl
    from sglang.utils import trim_overlap
except ImportError:
    sgl = None
    trim_overlap = None

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
from gallama.logger.logger import logger

# custom data classes
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
    VideoFrame
)

# class SGLEncodeSafe(sgl.Engine):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def encode_safe(
#         self,
#         prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
#         image_data: Optional[Union[List[str], str]] = None,
#     ) -> Dict:
#         """
#         The arguments of this function are the same as in
#         `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
#         """
#         obj = EmbeddingReqInput(text=prompt, image_data=image_data)
#         generator = self.tokenizer_manager.generate_request(obj, None)
#
#         try:
#             # Try to get the currently running event loop.
#             current_loop = asyncio.get_running_loop()
#             if current_loop.is_running():
#                 # A loop is already running, so create a new one for this call.
#                 new_loop = asyncio.new_event_loop()
#                 ret = new_loop.run_until_complete(generator.__anext__())
#                 new_loop.close()
#             else:
#                 ret = current_loop.run_until_complete(generator.__anext__())
#         except RuntimeError:
#             # No running loop was found, so get the default one.
#             loop = asyncio.get_event_loop()
#             ret = loop.run_until_complete(generator.__anext__())
#
#         return ret


class ModelSGLang(ModelInterface):
    def __init__(self, model_spec:ModelSpec):
        super().__init__(model_spec)
        self.model, self.tokenizer, self.processor = self.load_model()

    @property
    def video_token_by_backend(self) -> str:
        # TODO to use more generalized method than hardcoding Qwen Omni token
        video_token = "<|vision_bos|><|IMAGE|><|vision_eos|>"

        return video_token

    def load_model(self):
        """Load the model, tokenizer, cache, and optional processor."""
        # processor is for multimodal
        model, tokenizer, processor = self.load_model_sglang(
            model_id=self.model_id,
            gpus=self.gpus,
        )

        # load draft model
        if self.draft_model_id is not None:
            raise "Draft model currently not supported for llama cpp backend"

        self.eos_token_ids = self.generate_eos_tokens_id()

        return model, tokenizer, processor

    def load_model_sglang(
        self,
        model_id,
        gpus,
    ):
        """This function return the model and its tokenizer"""
        logger.info("Loading model: " + model_id)


        cache = None  # in case not a backend with separate cache like llama cpp
        tokenizer = None
        processor = None


        # arguments for model loading
        model_kargs = {
            'model_path': self.model_id,
            "grammar_backend": "xgrammar"
        }


        # determine the class to use for loading
        if self.backend_extra_args.get('model_class_extra_kwargs'):

            model_extra_kwargs = self.backend_extra_args.get('model_class_extra_kwargs')
            if model_extra_kwargs:
                model_kargs.update(model_extra_kwargs)      # update any extra argument

        # set max_seq_len based on model    TODO: To find more reliable method
        self.max_seq_len = self.backend_extra_args.get('max_seq_len') or 32768

        logger.info(model_kargs)
        model = sgl.Engine(**model_kargs)

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

    async def generate(
        self,
        prompt: str,
        gen_queue: Union[GenQueue, QueueContext, List[QueueContext]],
        request: Optional[Request] = None,
        gen_type: Union[str, GenStart] = "text",    # the generated result will be store to this queue
        temperature: float = 0.01,
        top_p: float = 0.8,
        formatter: FormatterBuilder | TokenEnforcerTokenizerData | SGLangFormatter = None,
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
            video_frame = [ frame.get_image() for frame in video]


        # if messages:
        #     messages_as_dicts = [message.model_dump() for message in messages]
        #
        #     # convert OpenAI to qwen format -> TODO find more generalized method
        #     # OpenAI format for image_url:
        #     # {
        #     #     "type": "image",
        #     #     "image": {
        #     #         "image_url": {
        #     #             "url": "url here"
        #     #         }
        #     #     }
        #     # }
        #     # convert to qwen2 VL format:
        #     # {
        #     #     "type": "image_url",
        #     #     "image_url": "url here"
        #     # }
        #     for one_message in messages_as_dicts:
        #         if isinstance(one_message["content"], list):
        #             for message in one_message["content"]:
        #                 if message.get("type") == "image_url":
        #                     message["type"] = "image"
        #                     message["image"] = message["image_url"]["url"]
        #                     message.pop("image_url", None)
        #
        #     # add in the message for video
        #     if video:
        #         messages_as_dicts.append({
        #             "type": "video",
        #             "video": video_frame,
        #             "use_audio_in_video": True if video_frame else False,
        #         })
        #
        #     audios_input, image_inputs, video_inputs = process_mm_info(messages_as_dicts)

        # convert prompt to token id
        # if image_inputs is None and video_inputs is None:
        #     input_ids = self.tokenizer(
        #         prompt,
        #     )
        # else:   # multimodal
        #     input_ids = self.processor(
        #         text=[prompt],
        #         audios=audios_input,
        #         images=image_inputs,
        #         videos=video_inputs,       # TODO currently Llama doesnt support videos, comment out for now.
        #         padding=True,
        #         add_special_tokens=False,
        #         return_tensors="pt",
        #     )

        input_ids = []

        # self.validate_token_length(len(input_ids))

        # # format enforcer
        # logits_processor = None
        # prefix_allowed_tokens_fn = None
        # if formatter:
        #     if isinstance(formatter, FormatterBuilder):
        #         logits_processor = create_formatter_logits_processor_list(self.tokenizer, formatter)
        #     else:
        #         prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, formatter)



        # find stop conditions
        stop_word_to_return = ""
        if stop_words:
            if isinstance(stop_words, str):
                stop_word_to_return = stop_words
                stop_words = [stop_words]

            elif isinstance(stop_words, list):
                stop_word_to_return = stop_words[0]

            if not self.eos_token_str:
                raise Exception("EOS token not set in model_config")
            stop_conditions = self.eos_token_str + stop_words  # concat the 2 list
            logger.debug("stop_words: " + str(stop_conditions))
        else:
            stop_conditions = self.eos_token_str

        max_tokens_to_use = min(
            self.max_seq_len - len(input_ids),
            max_tokens, 4096) if max_tokens else min(self.max_seq_len - len(input_ids), 4096)

        # Create a task to check for disconnection
        # pass

        # Get generation settings
        settings = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens_to_use,
            "stop": stop_words,
            # "skip_special_tokens": False,
        }

        # format enforcement
        if formatter:
            raise "tool not supported for SG lang yet"
            settings.update(formatter.get_formatter_dict())

        # logger.info("FROG FROG" + str(settings))

        # Kick-start the generation and let downstream know the generation type
        if isinstance(gen_type, str):
            gen_type_str = gen_type
            gen_type = GenStart(gen_type=gen_type)
        else:
            gen_type_str = gen_type.gen_type  # Get the generation type in string format

        for g_queue in gen_queue_list:
            g_queue.put_nowait(gen_type)

        # Create a task to check for disconnection
        # disconnect_check_task = None
        # if request:
        #     disconnect_check_task = asyncio.create_task(
        #         self.check_disconnection(request, job, gen_queue_list, stop_event=stop_event))

        start_time = time.time()
        job = await self.model.async_generate(
            prompt=prompt,
            sampling_params=settings,
            stream=True,
            image_data=video_frame if video_frame else None,
        )

        first_token_time = None

        # generate
        generate_text = ""
        gen_stats = None
        eos = False

        try:
            # Start the generation
            async for result in job:
                if eos or stop_event.is_set():
                    await job.cancel()
                    break

                # logger.info(result)
                chunk_text = result.get("text", "")

                # sg lang will have overlap in chunk and need to trim the overlap
                chunk_text = trim_overlap(generate_text, chunk_text)

                if chunk_text:
                    generate_text += chunk_text

                    if not first_token_time:
                        first_token_time = time.time()

                    # logger.info(f"chunk_text: {chunk_text}")
                    chunk = GenText(content=chunk_text, text_type=gen_type_str)
                    for g_queue in gen_queue_list:
                        if chunk_text not in self.eos_token_str_set:  # Formatron returns EOS token
                            g_queue.put_nowait(chunk)

                # Handle EOS signal
                if result.get("meta_info").get("finish_reason"):
                    eos = True

                    meta_info = result.get("meta_info")
                    finished_reason = meta_info.get("finish_reason")

                    # logger.info(f"eos result {result}")
                    # If the stop word occurred is from the stop_words and not LLM result token -> include in result
                    if finished_reason.get("type") == "stop":
                        ending_string = finished_reason.get("matched")

                        if ending_string:
                            # Find the stop word that was used to end the string
                            stop_word_used = self.get_stop_word(ending_string, stop_words)

                            if stop_word_used:
                                # If generation ended with one of the stop words
                                # -> return that stop word as the last token
                                chunk = GenText(content=stop_word_used, text_type=gen_type_str)
                                for g_queue in gen_queue_list:
                                    g_queue.put_nowait(chunk)

                    gen_stats = GenerationStats(
                        input_tokens_count=meta_info["prompt_tokens"],
                        output_tokens_count=meta_info["completion_tokens"],
                        time_to_first_token=first_token_time - start_time,
                        time_generate=meta_info["e2e_latency"],
                    )

                    for g_queue in gen_queue_list:
                        if g_queue.include_GenStats:
                            g_queue.put_nowait(gen_stats)

                    # Signal the end of generation
                    for g_queue in gen_queue_list:
                        if g_queue.include_GenEnd:
                            g_queue.put_nowait(GenEnd())

        except Exception as e:
            logger.error(e)
        finally:
            pass
            # if disconnect_check_task:
            #     disconnect_check_task.cancel()
            #     try:
            #         await disconnect_check_task
            #     except:
            #         pass

        logger.debug("----------------------LLM Raw Response---------------\n" + generate_text)