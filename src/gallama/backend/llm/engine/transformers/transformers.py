from ..base import ModelInterface
from typing import Optional, Dict, List, Union
import transformers
import time                                 # for compute of generation time
import asyncio
from fastapi import Request                 # for type hint
from threading import Thread
from importlib import import_module

# custom model support
from qwen_vl_utils import process_vision_info

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


from .model_support.llama3_2_vision.text_streamer import CustomTextIteratorStreamer


from gallama.utils import is_flash_attention_installed
from gallama.logger.logger import logger

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

class ModelTransformers(ModelInterface):
    def __init__(self, model_spec:ModelSpec):
        super().__init__(model_spec)
        self.model, self.tokenizer, self.processor = self.load_model()


    def load_model(self):
        """Load the model, tokenizer, cache, and optional processor."""
        # processor is for multimodal
        model, tokenizer, processor = self.load_model_transformers(
            model_id=self.model_id,
            gpus=self.gpus,
        )

        # load draft model
        if self.draft_model_id is not None:
            raise "Draft model currently not supported for llama cpp backend"

        self.eos_token_ids = self.generate_eos_tokens_id()

        return model, tokenizer, processor

    def load_model_transformers(
        self,
        model_id,
        gpus,
    ):
        """This function return the model and its tokenizer"""
        logger.info("Loading model: " + model_id)

        cache = None  # in case not a backend with separate cache like llama cpp
        tokenizer = None
        processor = None

        # helper function for dynamic class loading
        def get_class(class_string):
            module_name, class_name = class_string.rsplit('.', 1)
            module = import_module(module_name)
            return getattr(module, class_name)

        # arguments for model loading
        model_kargs = {
            'pretrained_model_name_or_path': self.model_id,
            'torch_dtype' : "auto",
            'device_map': "auto",
        }

        tokenizer_args = {
            'pretrained_model_name_or_path': self.model_id,
        }

        # check if flash attention enabled
        flash_installed, flash_version = is_flash_attention_installed()
        if flash_installed:
            model_kargs["attn_implementation"]  = "flash_attention_2"


        # determine the class to use for loading
        logger.info("frog")
        logger.info(self.backend_extra_args)
        if self.backend_extra_args.get('model_class'):
            model_class = get_class(self.backend_extra_args['model_class'])

            model_extra_kwargs = self.backend_extra_args.get('model_class_extra_kwargs')
            if model_extra_kwargs:
                model_kargs.update(model_extra_kwargs)      # update any extra argument
        else:
            model_class = transformers.AutoModelForCausalLM

        if self.backend_extra_args.get('tokenizer_class'):
            tokenizer_class = get_class(self.backend_extra_args['tokenizer_class'])
        else:
            tokenizer_class = transformers.AutoTokenizer

        if self.backend_extra_args.get('processor_class'):
            processor_class = get_class(self.backend_extra_args['processor_class'])
        else:
            processor_class = None

        # currently speculative decoding not supported by model specific for LLama CPP python
        if isinstance(gpus, str) and gpus == "auto":
            logger.info(model_kargs)
            model = model_class.from_pretrained(**model_kargs)
            tokenizer = tokenizer_class.from_pretrained(**tokenizer_args)
            if processor_class:
                processor = processor_class.from_pretrained(**tokenizer_args)

        elif isinstance(gpus, list):
            raise "Specifying GPU for transformers is not supported"

        else:
            raise ValueError("Device map should be either 'auto', 'gpu' split")

        # set max_seq_len based on model    TODO: To find more reliable method
        try:
            self.max_seq_len = model.config.max_position_embeddings
        except:
            # for llama 3.2
            self.max_seq_len = model.config.text_config.max_position_embeddings

        return model, tokenizer, processor



    def generate_eos_tokens_id(self) -> List[int]:
        """Generate the end-of-sequence token IDs."""

        # currently the stop word for llama cpp work by string and not by token id
        # hence usage of this is not required
        return []

    # *************** generation method from here
    async def _run_generation(
        self,
        #prompt,
        input_ids,
        max_tokens,
        temperature,
        stop,
        gen_queue_list,
        top_p=0.8,
        prefix_strings=None,
        stop_word_to_return="",
        gen_type_str: str = "text",
        logits_processor=None,              # for formatron format enforcement
        prefix_allowed_tokens_fn=None       # for lmfe format enforcement
    ):
        # Tokenize the prompt
        input_ids = input_ids.to(self.model.device)

        # Create generation config
        generation_config = transformers.GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_strings=stop,
            repetition_penalty=1.1,
            do_sample=True,
        )

        # Create streamer
        streamer = CustomTextIteratorStreamer(
            tokenizer=self.processor if self.processor else self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Run the llm's generate function in a separate thread
        thread = Thread(
            target=self.model.generate,
            kwargs={
                **input_ids,
                'generation_config': generation_config,
                'streamer': streamer,
                'tokenizer': self.tokenizer,
                'logits_processor': logits_processor,
                'prefix_allowed_tokens_fn': prefix_allowed_tokens_fn,
            }
        )
        thread.start()

        generate_text = ""
        try:
            for chunk in streamer:
                chunk_text = GenText(content=chunk, text_type=gen_type_str)
                generate_text += chunk
                for g_queue in gen_queue_list:
                    await g_queue.get_queue().put(chunk_text)

                # Yield control back to the event loop
                await asyncio.sleep(0)
        finally:
            # Wait for the generation thread to finish
            thread.join()

        # Handle stop word if present
        if stop_word_to_return:
            chunk_text = GenText(content=stop_word_to_return, text_type=gen_type_str)
            generate_text += chunk_text.content
            for g_queue in gen_queue_list:
                await g_queue.get_queue().put(chunk_text)

        return generate_text


    async def generate(
        self,
        prompt: str,
        gen_queue: Union[GenQueue, QueueContext, List[QueueContext]],
        request: Optional[Request] = None,
        gen_type: Union[str, GenStart] = "text",    # the generated result will be store to this queue
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
        if messages:
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
                            message["type"] = "image"
                            message["image"] = message["image_url"]["url"]
                            message.pop("image_url", None)


            image_inputs, video_inputs = process_vision_info(messages_as_dicts)

        # convert prompt to token id
        if image_inputs is None and video_inputs is None:
            input_ids = self.tokenizer(prompt, return_tensors="pt")
        else:   # multimodal
            input_ids = self.processor(
                text=[prompt],
                images=image_inputs,
                #videos=video_inputs,       # TODO currently Llama doesnt support videos, comment out for now.
                #padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )

        self.validate_token_length(len(input_ids))

        # format enforcer
        logits_processor = None
        prefix_allowed_tokens_fn = None
        if formatter:
            if isinstance(formatter, FormatterBuilder):
                logits_processor = create_formatter_logits_processor_list(self.tokenizer, formatter)
            else:
                prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, formatter)

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
        generate_text = await self._run_generation(
            input_ids=input_ids,
            logits_processor=logits_processor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            max_tokens=max_tokens_to_use,
            temperature=temperature,
            stop=stop_words,
            gen_queue_list=gen_queue_list,
            top_p=top_p,
            prefix_strings=None,
            stop_word_to_return="",
            gen_type_str=gen_type_str,
        )

        duration = time.time() - start_time

        gen_stats = GenerationStats()
        for g_queue in gen_queue_list:
            if g_queue.include_GenStats:
                g_queue.put_nowait(gen_stats)

        # this to signal the end of generation
        for g_queue in gen_queue_list:
            if g_queue.include_GenEnd:
                g_queue.put_nowait(GenEnd())

        logger.debug("----------------------LLM Raw Response---------------\n" + generate_text)