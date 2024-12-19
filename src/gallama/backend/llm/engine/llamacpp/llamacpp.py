from gallama.backend.llm.engine.base import ModelInterface
from typing import Optional, Dict, List, Union
from fastapi import Request                 # for type hint
import time

# for async running of llama cpp
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from llama_cpp import Llama
    from llama_cpp import LogitsProcessorList
except:
    Llama = None
    LogitsProcessorList = None

# format enforcement with formatron
try:
    from formatron.formatter import FormatterBuilder
except ImportError:
    FormatterBuilder = None

# format enforcement with llama cpp
try:
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
    from lmformatenforcer.integrations.llamacpp import (
        build_llamacpp_logits_processor,
        build_token_enforcer_tokenizer_data as build_token_enforcer_tokenizer_data_llama_cpp
    )
except:
    # llama_cpp optional dependency
    build_llamacpp_logits_processor = None
    build_token_enforcer_tokenizer_data_llama_cpp = None
    LogitsProcessorList = None

# custom data classes
from gallama.data_classes import (
    ModelSpec,
    GenStart,
    GenEnd,
    GenQueue,
    GenText,
    GenerationStats,
    QueueContext,
)
from gallama.logger.logger import logger


class ModelLlamaCpp(ModelInterface):
    def __init__(self, model_spec:ModelSpec):
        super().__init__(model_spec)

        self.model, self.tokenizer = self.load_model()

        # initialize lmfe tokenizer data
        self.lm_enforcer_tokenizer_data = build_token_enforcer_tokenizer_data_llama_cpp(self.model)


    def load_model(self):
        """Load the model, tokenizer, cache, and optional processor."""

        model, tokenizer = self.load_model_llama_cpp(
            model_id=self.model_id,
            gpus=self.gpus,
        )

        # load draft model
        if self.draft_model_id is not None:
            raise "Draft model currently not supported for llama cpp backend"

        self.eos_token_ids = self.generate_eos_tokens_id()
        return model, tokenizer


    def load_model_llama_cpp(self, model_id, gpus):
        """This function return the model and its tokenizer"""
        logger.info("Loading model: " + model_id)

        # currently speculative decoding not supported by model specific for LLama CPP python
        if isinstance(gpus, str) and gpus == "auto":
            model = Llama(
                model_path=self.model_id,
                n_gpu_layers=-1,
                seed=1,
                n_ctx=self.max_seq_len if self.max_seq_len else 0,  # Uncomment to increase the context window
                flash_attn=True,
                offload_kqv=True,
                # draft_model=draf_model_id,
            )
        elif isinstance(gpus, list):  # user specify the gpus split
            model = Llama(
                model_path=self.model_id,
                n_gpu_layers=-1,
                seed=1,
                n_ctx=self.max_seq_len if self.max_seq_len else 0,  # Uncomment to increase the context window
                flash_attn=True,
                offload_kqv=True,
                tensor_split=gpus,
                # draf_model_id=draf_model_id,
            )
        else:
            raise ValueError("Device map should be either 'auto', 'gpu' split")

        # set max_seq_len based on model
        self.max_seq_len = model._model.n_ctx_train()
        tokenizer = model       # llama cpp doesnt have separate tokenizer object

        self.eos_token_ids = self.generate_eos_tokens_id()

        return model, tokenizer



    def generate_eos_tokens_id(self) -> List[int]:
        """Generate the end-of-sequence token IDs."""

        # currently the stop word for llama cpp work by string and not by token id
        # hence usage of this is not required
        return []


    # ************* method for generation from here
    def _run_generator_and_queue(self, prompt, logits_processor, max_tokens, temperature, stop, gen_queue_list,
                                 top_p=0.8, prefix_strings=None, stop_word_to_return="", gen_type_str: str="text"):

        loop = asyncio.get_event_loop()

        generator = self.model(
            prompt=prompt,
            #suffix=prefix_strings,
            logits_processor=logits_processor,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=0.05,
            stop=stop,
            repeat_penalty=1.1,
            stream=True,
        )

        generate_text = ""
        for chunk in generator:
            chunk_text = GenText(content=chunk['choices'][0]['text'], text_type=gen_type_str)
            generate_text += chunk_text.content
            for g_queue in gen_queue_list:
                # g_queue.get_queue().put_nowait(chunk_text)
                future = asyncio.run_coroutine_threadsafe(
                    g_queue.get_queue().put(chunk_text),
                    loop
                )
                future.result()

        # return stop_word if there is
        if stop_word_to_return:
            chunk_text = GenText(content=stop_word_to_return, text_type=gen_type_str)
            generate_text += chunk_text.content
            for g_queue in gen_queue_list:
                # g_queue.get_queue().put_nowait(chunk_text)
                future = asyncio.run_coroutine_threadsafe(
                    g_queue.get_queue().put(chunk_text),
                    loop
                )
                future.result()
        return generate_text

    async def generate(
        self,
        prompt: str,
        gen_queue: Union[GenQueue, QueueContext, List[QueueContext]],
        request: Optional[Request] = None,          # for disconnection check, however, no stopping feature for llama cpp
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

        loop = asyncio.get_event_loop()

        if not quiet:
            logger.info("----------------------Prompt---------------\n" + prompt)
            logger.debug("----------------------temperature---------\n" + str(temperature))

        # ensure that generator is initialized
        # if self.pipeline is None:
        #     self.pipeline = await self._get_pipeline_async()

        # make gen_queue to List[QueueContext] for standardize downstream handline
        gen_queue_list = None
        if isinstance(gen_queue, QueueContext):
            gen_queue_list = [gen_queue]
        elif isinstance(gen_queue, GenQueue):
            gen_queue_list = [QueueContext.create(gen_queue, include_GenEnd=True, include_GenStats=True)]
        elif isinstance(gen_queue, list):
            gen_queue_list = gen_queue
        else:
            raise Exception("gen_queue must be either a GenQueue, QueueContext or a list of QueueContext")

        # convert prompt to token id
        input_ids = self.tokenizer.tokenize(prompt.encode("utf-8"), add_bos=False)
        self.validate_token_length(len(input_ids))

        # format enforcer
        logits_processors = None
        if formatter:
            logits_processors = LogitsProcessorList([
                build_llamacpp_logits_processor(
                    llm=self.pipeline.lm_enforcer_tokenizer_data,
                    character_level_parser=formatter,
                )
            ])

        start_time = time.time()

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
            max_tokens, 4096) if max_tokens else min(self.max_seq_len - len(input_ids),
                                                     4096
                                                     )

        # kickstart the generation and let down stream know gen type
        # kick-start the generation and let down stream know gen type
        if isinstance(gen_type, str):
            gen_type_str = gen_type
            gen_type = GenStart(gen_type=gen_type)
        else:
            gen_type_str = gen_type.gen_type  # get out the generation type in str format

        for g_queue in gen_queue_list:
            # g_queue.get_queue().put_nowait(gen_type)
            future = asyncio.run_coroutine_threadsafe(
                g_queue.get_queue().put(gen_type),
                loop
            )
            future.result()

        # Create a task to check for disconnection
        # llama cpp python generator is not async, hence running fake async..
        # Run the synchronous generator in a separate thread
        with ThreadPoolExecutor() as pool:
            generate_text = await loop.run_in_executor(
                pool,
                self._run_generator_and_queue,
                prompt, logits_processors, max_tokens_to_use, temperature, stop_conditions, gen_queue_list,
                top_p, prefix_strings, stop_word_to_return, gen_type_str
            )

        duration = time.time() - start_time

        gen_stats = GenerationStats()
        for g_queue in gen_queue_list:
            if g_queue.include_GenStats:
                # g_queue.get_queue().put_nowait(gen_stats)
                future = asyncio.run_coroutine_threadsafe(
                    g_queue.get_queue().put(gen_stats),
                    loop
                )
                future.result()


        # this to signal the end of generation
        for g_queue in gen_queue_list:
            if g_queue.include_GenEnd:
                # g_queue.get_queue().put_nowait(GenEnd())
                future = asyncio.run_coroutine_threadsafe(
                    g_queue.get_queue().put(GenEnd()),
                    loop
                )
                future.result()
        logger.debug("----------------------LLM Raw Response---------------\n" + generate_text)