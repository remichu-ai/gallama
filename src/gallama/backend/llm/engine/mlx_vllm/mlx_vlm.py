from ..base import ModelInterface
from typing import Optional, Dict, List, Union
from fastapi import Request                 # for type hint
import time

# for async running of mlx
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from mlx_vlm import load, generate, stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    from mlx_vlm.video_generate import process_vision_info
except:
    load = None
    generate = None
    stream_generate = None
    apply_chat_template = None
    load_config = None
    process_vision_info = None


# custom data classes
from gallama.data_classes import (
    ModelSpec,
    GenStart,
    GenEnd,
    GenQueue,
    GenText,
    GenerationStats,
    QueueContext,
    GenQueueDynamic
)

from gallama.logger.logger import logger



class ModelMLXVLM(ModelInterface):
    def __init__(self, model_spec:ModelSpec):
        super().__init__(model_spec)

        self.model, self.tokenizer, self.processor = self.load_model()

    @property
    def support_tool(self) -> bool:
        """
        Currently no format enforcement
        """
        return False

    def load_model(self):
        """Load the model, tokenizer, cache, and optional processor."""

        model, processor = load(path_or_hf_repo=self.model_id)
        config = load_config(model_path=self.model_id)

        # set max_seq_len
        self.max_seq_len = config.get('max_position_embeddings')

        # load draft model
        if self.draft_model_id is not None:
            raise "Draft model currently not supported for mlx vllm backend"

        self.eos_token_ids = self.generate_eos_tokens_id()
        return model, processor.tokenizer, processor


    def generate_eos_tokens_id(self) -> List[int]:
        """Generate the end-of-sequence token IDs."""

        # currently the stop word for mlx vllm work by string and not by token id
        # hence usage of this is not required
        return []

    # generation method from here ----------------------
    def _run_generation(
        self,
        loop: asyncio.AbstractEventLoop,
        prompt,
        image_list: List[str],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        # stop,
        gen_queue_list: List[QueueContext] = None,
        top_p=0.8,
        # prefix_strings=None,
        # stop_word_to_return="",
        gen_type_str: str = "text",
        # logits_processor=None,              # for formatron format enforcement
        # prefix_allowed_tokens_fn=None       # for lmfe format enforcement
    ):

        # Run the llm's generate function in a separate thread
        generator = stream_generate(
            **{
                'model': self.model,
                'processor': self.processor,
                'prompt': prompt,
                'image': image_list,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'repetition_penalty': 1.1,
            }
        )

        generate_text = ""
        last_chunk = None
        for chunk in generator:
            last_chunk = chunk
            chunk_text = GenText(content=chunk.text, text_type=gen_type_str)
            generate_text += chunk_text.content
            for g_queue in gen_queue_list:
                future = asyncio.run_coroutine_threadsafe(
                    g_queue.put(chunk_text),
                    loop
                )
                future.result()


        return generate_text, last_chunk



    async def generate(
        self,
        prompt: str,
        gen_queue: Union[GenQueue, QueueContext, List[QueueContext]],
        request: Optional[Request] = None,
        gen_type: Union[str, GenStart] = "text",    # the generated result will be store to this queue
        temperature: float = 0.01,
        top_p: float = 0.8,
        formatter = None,   # not supported
        stop_words: Union[List[str], str] = None,
        prefix_strings: Optional[Union[str, List[str]]] = None,
        banned_strings: list[str] | None = None,
        max_tokens: int = None,
        quiet=False,
        messages: List = None,  # query.message for multimodal
        **kwargs,
    ) -> (str, GenerationStats):

        loop = asyncio.get_event_loop()

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
        image_inputs, video_inputs = None, None

        # for now support only image

        image_list = []
        if messages:
            # convert pydantic to basic dictionary
            messages_as_dicts = [message.dict() for message in messages]

            # convert OpenAI to qwen format -> TODO find more generalized method
            # OpenAI format for image_url:
            # {
            #     "type": "image",
            #     "image": {
            #         "image_url": {
            #             "url": "url here"
            #         }
            #     }
            # }
            # convert to qwen2 VL format:
            # {
            #     "type": "image_url",
            #     "image_url": "url here"
            # }
            for one_message in messages_as_dicts:
                if isinstance(one_message["content"], list):
                    for message in one_message["content"]:
                        if message.get("type") == "image_url":
                            image_list.append(message["image_url"]["url"])
                            # message["type"] = "image"
                            # message["image"] = message["image_url"]["url"]
                            # message.pop("image_url", None)

            # image_inputs, video_inputs = process_vision_info(messages_as_dicts)


        # # convert prompt to token id
        # if image_inputs is None and video_inputs is None:
        #     input_ids = self.tokenizer(prompt, return_tensors="pt")
        # else:   # multimodal
        #     input_ids = self.processor(
        #         text=[prompt],
        #         images=image_inputs,
        #         #videos=video_inputs,       # TODO currently Llama doesnt support videos, comment out for now.
        #         #padding=True,
        #         add_special_tokens=False,
        #         return_tensors="pt",
        #     )

        input_ids = self.tokenizer.encode(prompt)
        self.validate_token_length(len(input_ids))

        # # format enforcer
        # logits_processor = None
        # prefix_allowed_tokens_fn = None
        # if formatter:
        #     if isinstance(formatter, FormatterBuilder):
        #         logits_processor = create_formatter_logits_processor_list(self.tokenizer, formatter)
        #     else:
        #         prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, formatter)

        # manual end of string not supported yet -------
        # # find stop conditions
        # stop_word_to_return = ""
        # if stop_words:
        #     if isinstance(stop_words, str):
        #         stop_word_to_return = stop_words
        #         stop_words = [stop_words]
        #
        #     elif isinstance(stop_words, list):
        #         stop_word_to_return = stop_words[0]
        #
        #     if not self.eos_token_str:
        #         raise Exception("EOS token not set in model_config")
        #     stop_conditions = self.eos_token_str + stop_words  # concat the 2 list
        #     logger.debug("stop_words: " + str(stop_conditions))
        # else:
        #     stop_conditions = self.eos_token_str

        max_tokens_to_use = min(
            self.max_seq_len - len(input_ids),
            max_tokens, 4096) if max_tokens else min(self.max_seq_len - len(input_ids), 4096)

        # kickstart the generation and let down stream know gen type
        if isinstance(gen_type, str):
            gen_type_str = gen_type
            gen_type = GenStart(gen_type=gen_type)
        else:
            gen_type_str = gen_type.gen_type  # get out the generation type in str format

        for g_queue in gen_queue_list:
            g_queue.put_nowait(gen_type)

        # Create a task to check for disconnection
        # pass

        # generate

        start_time = time.time()

        with ThreadPoolExecutor() as pool:
            generate_text, last_chunk = await loop.run_in_executor(
                pool,
                self._run_generation,
                loop, # <-- Passing the main event loop
                prompt,
                image_list,
                max_tokens_to_use,
                temperature,
                gen_queue_list,
                top_p,
                gen_type_str,
            )

        duration = time.time() - start_time

        gen_stats = GenerationStats(
            input_tokens_count=last_chunk.prompt_tokens,
            output_tokens_count=last_chunk.generation_tokens,
            time_generate=duration,
        )
        for g_queue in gen_queue_list:
            if g_queue.include_GenStats:
                g_queue.put_nowait(gen_stats)

        # this to signal the end of generation
        for g_queue in gen_queue_list:
            if g_queue.include_GenEnd:
                g_queue.put_nowait(GenEnd())

        logger.debug("----------------------LLM Raw Response---------------\n" + generate_text)