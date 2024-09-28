# generator.py
import time
import re
import asyncio
import uuid
import weakref
from typing import List, Union, Literal, Optional
from pydantic import BaseModel, Field
from fastapi import HTTPException, Request
from .model import Model
from gallama.data_classes.data_class import GenerationStats, GenEnd, GenText, GenQueue, ChatMLQuery, GenStart
from .tools import Tools, create_function_models_v2, create_function_models_formatron
from dataclasses import dataclass
from gallama.utils.utils import get_token_length
from gallama.logger.logger import logger
from .thinking_template import THINKING_TEMPLATE, Thinking
from gallama.api_response.chat_response import get_response_from_queue
from lmformatenforcer import JsonSchemaParser, RegexParser
from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from .format_enforcer import FormatEnforcer
from formatron.schemas.pydantic import ClassSchema

try:
    from formatron.formatter import FormatterBuilder
    from formatron.integrations.exllamav2 import create_formatter_filter

except:
    FormatterBuilder = None
    create_formatter_filter = None


try:
    from exllamav2 import (
        ExLlamaV2Cache,
        ExLlamaV2Cache_Q4,
    )
    from exllamav2.generator import (
        ExLlamaV2Sampler,
        ExLlamaV2DynamicGeneratorAsync,
        ExLlamaV2DynamicJobAsync,
    )
    from exllamav2.generator.filters import ExLlamaV2PrefixFilter
    from lmformatenforcer.integrations.exllamav2 import (
        ExLlamaV2TokenEnforcerFilter,         # TODO to uncomment this after LMFE take in the changes from turboderp
        build_token_enforcer_tokenizer_data
    )
    from .inference_json_lmfe_wrapper import ExLlamaV2TokenEnforcerFilter as ExLlamaV2TokenEnforcerFilterTemp # TODO to remove this after LMFE take in the changes from turboderp

    if version('exllamav2') == '0.2.1' or version('exllamav2') == '0.2.2':
        raise "Please use exllamav2 version 0.2.0 or 0.2.3 (not yet release). There is some bug with v0.2.1 and 0.2.2"

except:
    ExLlamaV2Cache = None
    ExLlamaV2Cache_Q4 = None
    ExLlamaV2Cache = None
    ExLlamaV2Sampler = None
    ExLlamaV2DynamicGeneratorAsync = None
    ExLlamaV2DynamicJobAsync = None
    ExLlamaV2TokenEnforcerFilter = None
    build_token_enforcer_tokenizer_data = None

try:
    from llama_cpp import LogitsProcessorList
    from lmformatenforcer.integrations.llamacpp import (
        build_llamacpp_logits_processor,
        build_token_enforcer_tokenizer_data as build_token_enforcer_tokenizer_data_llama_cpp
    )
except:
    # llama_cpp optional dependency
    build_llamacpp_logits_processor = None
    build_token_enforcer_tokenizer_data_llama_cpp = None
    LogitsProcessorList = None

assert ExLlamaV2Cache or LogitsProcessorList, "Please install ExllamaV2 or LLama CPP Python as backend"


TOOL_THINKING = THINKING_TEMPLATE["tool_necessity_evaluation"]
TOOL_FORCE_THINKING = THINKING_TEMPLATE["tool_forced_evaluation"]

@dataclass
class QueueContext:
    """ helper class for short live handling of multiple Queue"""
    gen_queue: 'weakref.ReferenceType[asyncio.Queue]'
    include_GenStats: bool
    include_GenEnd: bool

    @classmethod
    def create(cls, gen_queue: GenQueue, include_GenStats=True, include_GenEnd=True) -> 'QueueContext':
        return cls(
            gen_queue=weakref.ref(gen_queue),
            include_GenStats=include_GenStats,
            include_GenEnd=include_GenEnd)

    def get_queue(self) -> GenQueue | None:
        return self.gen_queue()


class ChatGenerator(Model):
    def __init__(
            self,
            llm_base: Model,
    ):
        # unpack all variables from llm_base
        # refer Model class for details of variable available
        self.__dict__.update(llm_base.__dict__)

        # placeholder
        self.pipeline = None

        # format enforcer
        self.formatter = FormatEnforcer()

    async def chat(self, query: ChatMLQuery, prompt_eng, gen_queue: GenQueue, request: Request):
        chat_method = self.chat_with_tool if query.tools or query.tool_choice != "none" else self.chat_no_tool
        return await chat_method(query=query, prompt_eng=prompt_eng, gen_queue=gen_queue, request=request)

    async def chat_raw(
        self,
        prompt: str,
        gen_queue: asyncio.Queue,
        request: Request,
        stream: bool = False,
        max_tokens: int = None,
        quiet=False,
    ):
        return await self.generate(prompt, max_tokens=max_tokens, gen_queue=gen_queue, quiet=quiet, request=request)

    def validate_token_length(self, token_length):
        # TODO to find max_seq_len for llama cpp from Model
        # if max_seq_len == None meaning there is no token length yet
        if self.max_seq_len and token_length > self.max_seq_len:
            raise HTTPException(status_code=400, detail=f"Token length exceeds max length of {self.max_seq_len}")

    # helper function
    async def check_disconnection(self, request: Request, job,
                                  gen_queue_list: Union[GenQueue, QueueContext, List[QueueContext]]):
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


    async def chat_no_tool(self, query: ChatMLQuery, prompt_eng, gen_queue, request: Request):

        prompt = prompt_eng.get_prompt(
            query,
            thinking_template=query.thinking_template,
        )

        formatter_prefix_regex = self.formatter.regex(
            query.regex_prefix_pattern, backend=self.backend, preference=query.guided_decoding_backend) if query.regex_prefix_pattern else None

        formatter_regex = self.formatter.regex(query.regex_pattern, backend=self.backend, preference=query.guided_decoding_backend) if query.regex_pattern else None

        token_length_prompt = get_token_length(self.tokenizer, prompt)
        self.validate_token_length(token_length_prompt)

        # think template prompting
        if query.thinking_template:
            try:
                thinking = Thinking(query.thinking_template)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"thinking_template is not valid XML string")

            # Not returning thinking
            thinking_queue = GenQueue()
            if query.return_thinking:
                # return thinking to front end
                queue_group = [
                    QueueContext.create(gen_queue=thinking_queue, include_GenEnd=True, include_GenStats=False),
                    QueueContext.create(gen_queue=gen_queue, include_GenEnd=False, include_GenStats=False)
                ]
            else:
                # not return thinking to front end
                queue_group = [
                    QueueContext.create(gen_queue=thinking_queue, include_GenEnd=True, include_GenStats=False),
                ]

            await self.generate(
                prompt,
                gen_type="thinking",
                gen_queue=queue_group,
                temperature=query.temperature,
                top_p=query.top_p,
                prefix_strings=f"<{thinking.root_tag}>",
                stop_words=thinking.root_key_stop_words,
                request=request,
            )
            thinking_response, _ = await get_response_from_queue(thinking_queue)

            # Get the new prompt with thinking response
            prompt = prompt_eng.get_prompt(
                query,
                thinking_template=query.thinking_template,
                thinking_response=thinking_response,
            )

        # 1st response if there is regex to match the regex pattern
        first_response = ""

        if query.regex_prefix_pattern:
            prefix_queue = GenQueue()
            queue_group = [
                QueueContext.create(gen_queue=prefix_queue, include_GenEnd=True, include_GenStats=False),
                QueueContext.create(gen_queue=gen_queue, include_GenEnd=False, include_GenStats=False)
            ]

            await self.generate(
                prompt,
                gen_queue=queue_group,
                temperature=query.temperature,
                top_p=query.top_p,
                formatter=formatter_prefix_regex,
                prefix_strings=query.prefix_strings,
                request=request,
                # stop_words=query.stop_words,
            )

            first_response, _ = await get_response_from_queue(prefix_queue)

            # append generated content to the full prompt
            prompt = prompt.strip() + first_response.strip()

        # Final generation to return to client
        # set prefix string
        prefix_strings = None if query.regex_prefix_pattern else query.prefix_strings

        # Handle Artifact mode: overwrite prefix_strings if in artifact mode
        stop_words_to_use = query.stop_words
        banned_strings = None
        if query.artifact and query.artifact == "Fast":
            prefix_strings = None
            manual_prefix_string = "<answer>\n "
            prompt += manual_prefix_string

            # ban XML comment format which could mess up the parsing of output
            banned_strings = ["<![CDATA[", "<!--"]

            # add the stopword for artifact tag to the answer
            if isinstance(stop_words_to_use, list):
                stop_words_to_use.append("</answer>")
            elif isinstance(stop_words_to_use, str):
                stop_words_to_use = [stop_words_to_use, "</answer>"]
            else:
                stop_words_to_use = "</answer>"

            # add the initial string as prefix_strings can not be used together with banned_strings
            chunk = GenText(content=manual_prefix_string)
            gen_queue.put_nowait(chunk)

        await self.generate(
            prompt=prompt,
            gen_queue=gen_queue,
            **{
                'temperature': query.temperature,
                'top_p': query.top_p,
                'formatter': formatter_regex,
                'stop_words': stop_words_to_use,
                'max_tokens': query.max_tokens,
                'prefix_strings': prefix_strings,  # already generated as part of the prefix string
                'banned_strings': banned_strings,
                'request': request,
            }
        )

    async def chat_with_tool(self, query: ChatMLQuery, prompt_eng, gen_queue, request: Request):
        # use_tool marker
        use_tool_bool = False  # this will be set to True if tool is used
        fall_back_bool = False  # this will decide if fallback generation with regex enforcement is required

        # tool class have the method to handle converting processing or tool requirement, schema and response
        tool_handler = Tools(
            prompt_eng=prompt_eng,
            tools=query.tools,
            tool_choice=query.tool_choice,
        )

        # substitute the actual list of tool to the thinking template
        tool_thinking_queue = GenQueue()

        tool_thinking_to_use = TOOL_THINKING
        if query.tool_choice != "auto":
            tool_thinking_to_use = TOOL_FORCE_THINKING

        tool_thinking_formatted = tool_thinking_to_use.xml.format_map(
            {"fstring_available_tools": tool_handler.tool_name_list}
        )

        # perform generation with tool thinking to evaluate if it is necessity to call a tool
        prompt = prompt_eng.get_prompt(
            query,
            pydantic_tool_dict=tool_handler.tool_dict,
            thinking_template=tool_thinking_formatted,
            answer_format_schema=False,
            # leading_prompt=leading_prompt,
        )

        await self.generate(
            prompt,
            gen_queue=tool_thinking_queue,
            temperature=query.temperature,
            top_p=query.top_p,
            stop_words=tool_thinking_to_use.root_key_stop_words,
            prefix_strings=f"<{tool_thinking_to_use.root_tag}>",
            request=request,
            # formatter=formatter_regex  # no longer enforce format
        )

        # evaluate tool usage necessity
        tool_thinking_response, _ = await get_response_from_queue(tool_thinking_queue)

        # see if llm able to generate the xml format correctly
        if query.tool_choice != "auto":
            use_tool_bool = True
        else:
            try:
                # parse the xml object
                tool_thinking_response_dict = Thinking.parse_xml_to_dict(tool_thinking_response)
                tool_decision = tool_thinking_response_dict[tool_thinking_to_use.root_tag]["final_decision"][
                    "is_tool_needed"]  # TODO implement a less hardcoding way

                if tool_decision.lower().strip() == "yes":
                    use_tool_bool = True
                elif tool_decision.lower().strip() == "no":
                    use_tool_bool = False
                else:
                    # the format was not enforce, to perform fall back check
                    fall_back_bool = True

            except Exception as e:
                logger.error(f"XML parsing failed: {e}")
                # Fallback: check for the presence of Yes/No in the raw XML
                yes_pattern = r'<is_tool_needed>\s*Yes\s*</is_tool_needed>'
                no_pattern = r'<is_tool_needed>\s*No\s*</is_tool_needed>'

                if re.search(yes_pattern, tool_thinking_response, re.IGNORECASE):
                    use_tool_bool = True
                elif re.search(no_pattern, tool_thinking_response, re.IGNORECASE):
                    use_tool_bool = False
                else:
                    fall_back_bool = True

        logger.info(tool_thinking_response)

        # Fall back plan
        fall_back_prompt = ""
        if fall_back_bool:
            logger.info("Tool Analysis fallback")
            # generate fall back response with regex enforcement:
            fall_back_prompt = ("\n Fill in the blank in below sentence with either 'needed' or 'not needed'"
                                "In summary, tool calling is {blank}\n"
                                "Answer: blank=")
            prompt += tool_thinking_response + fall_back_prompt

            # perform generation with tool thinking to evaluate if it is necessity
            tool_thinking_queue_fallback = GenQueue()

            formatter_regex = self.formatter.regex('(needed|not needed)', backend=self.backend, preference=query.guided_decoding_backend)

            await self.generate(
                prompt,
                gen_queue=tool_thinking_queue_fallback,
                temperature=query.temperature,
                top_p=query.top_p,
                request=request,
                # prefix_strings="n",
                # stop_words=TOOL_THINKING.root_key_stop_words,
                formatter=formatter_regex  # no longer enforce format
            )

            # evaluate tool usage necessity
            tool_thinking_decision_fallback, _ = await get_response_from_queue(tool_thinking_queue_fallback)

            # decide if tool call is required
            if "not needed" in tool_thinking_decision_fallback.lower():
                use_tool_bool = False
            elif "needed" in tool_thinking_decision_fallback.lower():
                use_tool_bool = True

        # USE TOOL
        if use_tool_bool:
            # create the pydantic schema to enforce generation
            tool_combined_pydantic_lmfe = create_function_models_v2(tool_handler.tool_dict)

            class ToolCalling_LMFE(ClassSchema):
                """ The format to call one or multiple tools """
                functions_calling: List[Union[tuple(tool_combined_pydantic_lmfe)]] = []

            # create the pydantic schema to enforce generation for formatron which use ClassSchema
            tool_combined_pydantic_formatron = create_function_models_formatron(tool_handler.tool_dict_formatron)
            class ToolCalling_formatron(ClassSchema):
                """ The format to call one or multiple tools """
                functions_calling: List[Union[tuple(tool_combined_pydantic_formatron)]] = []

            formatter_json = self.formatter.json(
                pydantic_model_lmfe=ToolCalling_LMFE,
                pydantic_model_formatron=ToolCalling_formatron,
                backend=self.backend,
                preference = query.guided_decoding_backend
            )

            # Experiment feature, formulate function calling as python programming. Which is more natural than a random Json output as part of conversation
            tool_as_code_prompt = """
def run_function(arg_dict):
    function_calls = arg_dict["functions_calling"]

    if function_calls == []:
        print("No function/tool calling needed")
        return

    for call in function_calls:
        function_name = call["name"]
        arguments = call["arguments"]
        globals()[function_name](**arguments)

# Perform function calling if need to
arg_dict = """

            # get final prompt
            prompt = prompt_eng.get_prompt(
                query,
                thinking_template=TOOL_THINKING.xml,
                thinking_response=tool_thinking_response,
                pydantic_tool_dict=tool_handler.tool_dict,
                answer_format_schema=True,
                leading_prompt=(
                    f"{fall_back_prompt}\n"
                    'Now i will convert my answer above into "functions_calling" format by continuing this continue this code.\n'
                    f"{tool_as_code_prompt}"
                ),
            )
            logger.info(prompt)

            # generate
            await self.generate(
                prompt,
                gen_queue=gen_queue,
                gen_type=GenStart(gen_type="tool"),
                temperature=query.temperature,
                top_p=query.top_p,
                # stop_words=TOOL_THINKING.root_key_stop_words,
                prefix_strings=['{\n "functions_calling": ['],
                formatter=formatter_json,
                max_tokens=query.max_tokens,
                request=request,
            )

        # NOT USE TOOL
        if not use_tool_bool:
            prompt = prompt_eng.get_prompt(query)

            if query.tool_choice == "auto":
                # Normal generation
                await self.generate(
                    prompt,
                    gen_queue=gen_queue,
                    gen_type=GenStart(gen_type="text"),
                    temperature=query.temperature,
                    prefix_strings=query.prefix_strings,
                    max_tokens=query.max_tokens,
                    request=request,
                )
            else:
                # tool choice is forced -> return empty tool calling
                gen_queue.put_nowait(GenStart(gen_type="tool"))
                gen_queue.put_nowait(GenText(content='{"functions_calling":[]}'))
                gen_queue.put_nowait(GenerationStats())
                gen_queue.put_nowait(GenEnd())

    class ExllamaV2Pipeline:
        """ class to hold objects required for Exllama V2 text generation"""

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
        # if not self.cache:
        #     logger.info("Custom Cache size: " + str(self.cache_size))
        #     self.cache = ExLlamaV2Cache_Q4(self.model, max_seq_len=self.cache_size, lazy=not self.model.loaded)

        # Test VRAM allocation with a full-length forward pass
        # input_ids = torch.zeros((1, self.max_seq_len), dtype=torch.long)
        # model.forward(input_ids, cache=cache, preprocess_only=True)

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

    def _get_exllama_gen_settings(
            self,
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
        settings.token_repetition_range: int = 1024
        # settings.token_repetition_decay: int = 0.98
        settings.temperature_last = False

        return settings

    @staticmethod
    def get_stop_word(text, stop_words) -> Union[str, None]:
        """ this function will match the stop word used given the text that model ended generation with and a list of stop_words."""

        # sort the list by length to find the longest first
        sorted_stop_words = sorted(stop_words, key=len, reverse=True)

        text = text.lstrip()  # Remove trailing whitespace
        for stop_word in stop_words:
            if stop_word in text:
                return stop_word

        return None

    async def generate(
        self,
        prompt: str,
        gen_queue: Union[GenQueue, QueueContext, List[QueueContext]],
        request: Optional[Request] = None,
        # the generated result will be store to this queue
        gen_type: Union[str, GenStart] = "text",
        temperature: float = 0.01,
        top_p: float = 0.8,
        formatter: FormatterBuilder | TokenEnforcerTokenizerData = None,
        stop_words: Union[List[str], str] = None,
        prefix_strings: Optional[Union[str, List[str]]] = None,
        banned_strings: list[str] | None = None,
        max_tokens: int = None,
        quiet=False,
        **kwargs,
    ) -> (str, GenerationStats):
        try:
            # ensure that generator is initialized
            if self.pipeline is None:
                self.pipeline = await self._get_pipeline_async()

            # make gen_queue to List[QueueContext] for standardize downstream handline
            gen_queue_list = None
            if isinstance(gen_queue, QueueContext):
                gen_queue_list = [gen_queue]
            elif isinstance(gen_queue, GenQueue):
                gen_queue_list = [QueueContext.create(gen_queue, include_GenEnd=True, include_GenStats=True)]
            elif isinstance(gen_queue, list):
                gen_queue_list = gen_queue
                # TODO add validation
                # if any(not isinstance(g_queue, QueueContext) for g_queue in gen_queue_list):
                #     raise Exception("gen_queue must be either a GenQueue, QueueContext or a list of QueueContext")
            else:
                raise Exception("gen_queue must be either a GenQueue, QueueContext or a list of QueueContext")

            if not quiet:
                logger.info("----------------------Prompt---------------\n" + prompt)
                logger.debug("----------------------temperature---------\n" + str(temperature))

            # for async generator, create it as part of the generate job

            # get generation setting
            settings = self._get_exllama_gen_settings(temperature, top_p=top_p)

            # convert prompt to token id
            input_ids = self.tokenizer.encode(prompt)
            self.validate_token_length(len(input_ids[0]))

            # format enforcer
            filters = []
            if formatter:
                if isinstance(formatter, TokenEnforcerTokenizerData) or isinstance(formatter, JsonSchemaParser):  # lm format enforcer
                    exllamva_version = version('exllamav2')
                    if exllamva_version<='0.2.0':
                        filters = [ExLlamaV2TokenEnforcerFilter(
                            character_level_parser=formatter,
                            tokenizer_data=self.pipeline.lm_enforcer_tokenizer_data
                        )]
                    else:
                        filters = [ExLlamaV2TokenEnforcerFilterTemp(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            character_level_parser=formatter,
                        )]
                elif FormatterBuilder and isinstance(formatter, FormatterBuilder):  # formatron
                    filters = [create_formatter_filter(self.model, self.tokenizer, formatter)]
                else:
                    raise "Format enforcer is not correctly initialized"

            if prefix_strings:
                assert isinstance(prefix_strings, str) or (isinstance(prefix_strings, list) and len(prefix_strings) > 0)
                filters.append(ExLlamaV2PrefixFilter(self.model, self.tokenizer, prefix_strings))

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

            job = ExLlamaV2DynamicJobAsync(
                generator=self.pipeline.generator,
                input_ids=input_ids,
                max_new_tokens=max_tokens_to_use,
                gen_settings=settings,
                stop_conditions=stop_conditions,  # self.eos_token_id if self.eos_token_id else None,
                banned_strings=banned_strings,
                decode_special_tokens=True,
                filters=filters,
                token_healing=True,
                identifier=job_id,
            )

            # break the pipeline if it takes longer than 1s for 1 iteration
            # async def run_job():

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

                        # if the stop word occurred is from the stop_words and not model result token -> include in result
                        if stop_words and result.get("held") and result.get("held").get("text"):
                            ending_string = result["held"]["text"].rstrip()

                            if ending_string:
                                # find the stop word that was used to end string
                                stop_word_used = self.get_stop_word(ending_string, stop_words)

                                if stop_word_used:
                                    # end_string is custom token -> return
                                    chunk = GenText(content=stop_word_used, text_type=gen_type_str)
                                    for g_queue in gen_queue_list:
                                        g_queue.get_queue().put_nowait(chunk)
                                else:
                                    # ending token is model eos token
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


class ChatGeneratorLlamaCpp(ChatGenerator):
    def __init__(
            self,
            llm_base: Model,
    ):
        # unpack all variables from llm_base
        # refer Model class for details of variable available
        self.__dict__.update(llm_base.__dict__)
        self.pipeline = self._get_pipeline()
        # format enforcer
        self.formatter = FormatEnforcer()

    class LLamaCppPipeline:
        """ class to hold objects required for Exllama V2 text generation"""

        def __init__(
                self,
                generator,
                lm_enforcer_tokenizer_data: TokenEnforcerTokenizerData,
        ):
            self.generator = generator
            self.lm_enforcer_tokenizer_data = lm_enforcer_tokenizer_data

    def _get_pipeline(self):
        lm_enforcer_tokenizer_data = build_token_enforcer_tokenizer_data_llama_cpp(self.model)

        return self.LLamaCppPipeline(
            generator=self.model,
            lm_enforcer_tokenizer_data=lm_enforcer_tokenizer_data,
        )

    async def _async_generator(self, sync_generator):
        for item in sync_generator:
            yield item

    def _run_generator_and_queue(self, prompt, logits_processor, max_tokens, temperature, stop, gen_queue_list,
                                 top_p=0.8, prefix_strings=None, stop_word_to_return="", gen_type_str: str="text"):
        generator = self.pipeline.generator(
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
                g_queue.get_queue().put_nowait(chunk_text)

        # return stop_word if there is
        if stop_word_to_return:
            chunk_text = GenText(content=stop_word_to_return, text_type=gen_type_str)
            generate_text += chunk_text.content
            for g_queue in gen_queue_list:
                g_queue.get_queue().put_nowait(chunk_text)

        return generate_text

    # llama cpp
    async def generate(
        self,
        prompt: str,
        gen_queue: Union[GenQueue, QueueContext, List[QueueContext]],
        request: Request,
        # the generated result will be store to this queue
        gen_type: Union[str, GenStart] = "text",
        temperature: float = 0.01,
        top_p: float = 0.8,
        formatter: FormatterBuilder | TokenEnforcerTokenizerData = None,
        stop_words: Union[List[str], str] = None,
        prefix_strings: Optional[Union[str, List[str]]] = None,
        banned_strings: list[str] | None = None,
        max_tokens: int = None,
        quiet=False,
        **kwargs,
    ) -> (str, GenerationStats):

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
            # TODO add validation
            # if any(not isinstance(g_queue, QueueContext) for g_queue in gen_queue_list):
            #     raise Exception("gen_queue must be either a GenQueue, QueueContext or a list of QueueContext")
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
            g_queue.get_queue().put_nowait(gen_type)

        # Create a task to check for disconnection

        # llama cpp python generator is not async, hence running fake async..
        # Run the synchronous generator in a separate thread
        loop = asyncio.get_event_loop()
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
                g_queue.get_queue().put_nowait(gen_stats)

        # this to signal the end of generation
        for g_queue in gen_queue_list:
            if g_queue.include_GenEnd:
                g_queue.get_queue().put_nowait(GenEnd())

        logger.debug("----------------------LLM Raw Response---------------\n" + generate_text)
