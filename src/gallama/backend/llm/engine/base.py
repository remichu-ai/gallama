from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Literal, Callable, Tuple
import asyncio
import re       # for text processing of the thinking
from fastapi import HTTPException, Request
import textwrap

# logger
from gallama.logger import logger

# format enforcement
from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
from formatron.formatter import FormatterBuilder
from formatron.schemas.pydantic import ClassSchema
from gallama.backend.llm.format_enforcer import FormatEnforcer

# thinking
from gallama.backend.llm.thinking_template import THINKING_TEMPLATE, Thinking

# function calling
from gallama.backend.llm.tools import Tools, create_function_models_v2, create_function_models_formatron

from gallama.utils.utils import get_token_length
from gallama.api_response.chat_response import get_response_from_queue   # helper function to collect result from queue
from gallama.data_classes import (
    ModelSpec,
    ChatMLQuery,
    GenEnd,
    GenText,
    GenQueue,
    GenStart,
    GenerationStats,
    QueueContext,
    GenQueueDynamic,
    VideoFrame,
)
from ....config.config_manager import ConfigManager
# handle prompting
from gallama.backend.llm.prompt_engine import PromptEngine
from dataclasses import dataclass


config_manager = ConfigManager()

@dataclass
class ToolCallV2:
    gen_dynamic_queue: List[GenQueueDynamic]
    stop_event: asyncio.Event
    generate_kwargs: dict

class ModelInterface(ABC):
    @abstractmethod
    def __init__(self, model_spec:ModelSpec):
        # initialization share the same code to keep the frontend consistent
        # if a backend does not support the value, please set it in the __init__ or load_model()

        # load prompt engine
        self.prompt_eng = PromptEngine(prompt_format=model_spec.prompt_template)

        # model_spec capture cli argument
        # model_config is from yml file
        self.model_id = model_spec.model_id
        # self.model_name = model_spec.model_name or model_config["model_name"]
        self.model_name = model_spec.model_name
        # self.max_seq_len = model_spec.max_seq_len or model_config.get("max_seq_len", None)
        self.max_seq_len = model_spec.max_seq_len
        if self.max_seq_len is not None:
            self.max_seq_len = (self.max_seq_len//256) * 256     # for paged attention

        # self.gpus = model_spec.gpus or model_config.get("gpus") or "auto"
        self.gpus = model_spec.gpus or "auto"
        # self.cache_size = model_spec.cache_size or model_config.get("cache_size") or self.max_seq_len   # default to max_seq_len if not set
        self.cache_size = model_spec.cache_size or self.max_seq_len   # default to max_seq_len if not set
        if self.cache_size is not None:
            self.cache_size = (self.cache_size//256) * 256     # for paged attention
            if self.max_seq_len is not None:
                # cache size must be greater or equal to max_seq_len
                self.cache_size = max(self.cache_size, self.max_seq_len)

        # self.cache_quant = model_spec.cache_quant or model_config.get("cache_quant") or "Q4"
        self.cache_quant = model_spec.cache_quant or "Q6"       # default to cache quant 6
        # self.backend = model_spec.backend or model_config["backend"] or "exllama"
        self.backend = model_spec.backend   # default should be set as exllama if not defined
        # self.tensor_parallel = model_spec.tensor_parallel or model_config.get("tensor_parallel", False)
        self.tensor_parallel = model_spec.tensor_parallel or False      # tensor parallel is False unless explicitly

        # transformers specific arguments
        # self.backend_extra_args = model_spec.get("backend_extra_args") or {}
        self.backend_extra_args = model_spec.backend_extra_args or {}


        # handle draft model
        draft_model_config = {}
        if model_spec.draft_model_id:
            draft_model_config = config_manager.get_model_config(model_spec.draft_model_name)
            if not draft_model_config:
                raise HTTPException(f"Model config for '{model_spec.draft_model_name}' not exist")


        # draft model is via cli only
        self.draft_model_id = draft_model_config.get("model_id")
        self.draft_model_name = model_spec.draft_model_name or None
        self.draft_gpus = model_spec.draft_gpus or draft_model_config.get("draft_gpus") or "auto"
        self.draft_cache_size = self.cache_size   # set to the same as main model
        self.draft_cache_quant = model_spec.draft_cache_quant or draft_model_config.get("cache_quant") or "Q4"
        # assert (self.draft_model_id is None) == (self.draft_model_name is None)

        # get the eos_token_str by merging the default config with anything set by user
        self.eos_token_str = list(set(model_spec.eos_token_list + self.prompt_eng.eos_token_list))
        self.eos_token_str_set = set(self.eos_token_str)    # set for some more efficient operation

        # load_model method in each subclass should set the following parameters:
        self.model = None
        self.tokenizer = None
        self.cache = None
        self.draft_model = None
        self.draft_cache = None
        self.processor = None       # processor is for processing of vision input
        self.eos_token_ids = None

        # placeholder
        # pipeline is a wrapper for model so that generation can be more standardized
        # this is because each backend expose different interface
        self.pipeline = None

        # format enforcer
        self.formatter = FormatEnforcer()

        # standard tool template
        self.TOOL_THINKING = THINKING_TEMPLATE["tool_necessity_evaluation"]
        self.TOOL_FORCE_THINKING = THINKING_TEMPLATE["tool_forced_evaluation"]


    ## *************** the following method must be implemented by each backend ********
    @abstractmethod
    def load_model(self):
        """Load the model, tokenizer, cache, and optional processor."""
        pass

    @abstractmethod
    def generate_eos_tokens_id(self) -> List[int]:
        """Generate the end-of-sequence token IDs."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        gen_queue: Union[GenQueue, QueueContext, List[QueueContext]],
        request: Optional[Request] = None,
        gen_type: Union[str, GenStart] = "text", # the generated result will be store to this queue
        temperature: float = 0.01,
        top_p: float = 0.8,
        formatter: FormatterBuilder | TokenEnforcerTokenizerData = None,
        stop_words: Union[List[str], str] = None,
        prefix_strings: Optional[Union[str, List[str]]] = None,
        banned_strings: list[str] | None = None,
        max_tokens: int = None,
        quiet: bool = False,
        messages: List = None,  # query.message is used for multimodal
        video: List[VideoFrame] = None,
        stop_event: asyncio.Event = None,
        **kwargs,
    ) -> (str, GenerationStats):
        pass


    ## ******************************************************************************
    async def chat(
        self,
        query: ChatMLQuery,
        prompt_eng: PromptEngine,
        gen_queue: GenQueue,
        request: Request,
        stop_event: asyncio.Event = asyncio.Event()
    ):
        """
        This function will route the request to chat with tool or without tool accordingly
        """
        if query.tools or query.tool_choice != "none":
            chat_method = self.chat_with_tool_v2    # TODO
        else:
            chat_method = self.chat_no_tool
        try:
            await chat_method(
                query=query,
                prompt_eng=prompt_eng,
                gen_queue=gen_queue,
                request=request,
                stop_event=stop_event,
            )
        except Exception as e:
            if stop_event.is_set():
                logger.info("Stop event is set, aborting chat")
            else:
                logger.error("Error while generating response: " + str(e))
                raise Exception("Error while generating chat response: " + str(e))
        return True

    async def chat_raw(
        self,
        prompt: str,
        gen_queue: asyncio.Queue,
        request: Request,
        # stream: bool = False,
        max_tokens: int = None,
        quiet=False,    # to disable any logging, mostly used for initialization cache prefill
        stop_event: asyncio.Event = asyncio.Event()
    ):
        """
        This function handle chat with input as a string prompt.
        This is mostly used for internal generation of the engine where the input is only a string
        """
        return await self.generate(
            prompt,
            max_tokens=max_tokens,
            gen_queue=gen_queue,
            quiet=quiet,
            request=request,
            stop_event=stop_event,
        )

    def validate_token_length(self, token_length):
        """
        validate that token_length is within max sequence length
        """
        # if max_seq_len == None meaning there is no token length yet
        if self.max_seq_len and token_length > self.max_seq_len:
            raise HTTPException(status_code=400, detail=f"Token length exceeds max length of {self.max_seq_len}")


    async def chat_no_tool(
        self,
        query: ChatMLQuery,
        prompt_eng: PromptEngine,
        gen_queue, request: Request,
        stop_event: asyncio.Event = None
    ):

        prompt = prompt_eng.get_prompt(
            query,
            thinking_template=query.thinking_template,
            backend=self.backend
        )

        formatter_prefix_regex = self.formatter.regex(
            query.regex_prefix_pattern,
            backend=self.backend,
            preference=query.guided_decoding_backend
        ) if query.regex_prefix_pattern else None

        formatter_regex = self.formatter.regex(
            query.regex_pattern,
            backend=self.backend,
            preference=query.guided_decoding_backend
        ) if query.regex_pattern else None

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
                messages=query.messages,
                gen_type="thinking",
                gen_queue=queue_group,
                temperature=query.temperature,
                top_p=query.top_p,
                prefix_strings=f"<{thinking.root_tag}>",
                stop_words=thinking.root_key_stop_words,
                request=request,
                stop_event=stop_event,
                video=query.video,
            )
            thinking_response, _ = await get_response_from_queue(thinking_queue)

            # Get the new prompt with thinking response
            prompt = prompt_eng.get_prompt(
                query,
                thinking_template=query.thinking_template,
                thinking_response=thinking_response,
                backend=self.backend
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
                messages=query.messages,
                gen_queue=queue_group,
                temperature=query.temperature,
                top_p=query.top_p,
                formatter=formatter_prefix_regex,
                prefix_strings=query.prefix_strings,
                request=request,
                # stop_words=query.stop_words,
                stop_event=stop_event,
                video=query.video,
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
            messages=query.messages,
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
                'stop_event': stop_event,
                'video': query.video,
            }
        )

    async def chat_with_tool(
        self,
        query: ChatMLQuery,
        prompt_eng,
        gen_queue,
        request: Request,
        stop_event: asyncio.Event = None
    ):
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

        tool_thinking_to_use = self.TOOL_THINKING
        if query.tool_choice != "auto":
            tool_thinking_to_use = self.TOOL_FORCE_THINKING

        tool_thinking_formatted = tool_thinking_to_use.xml.format_map(
            {"fstring_available_tools": tool_handler.tool_name_list}
        )

        # perform generation with tool thinking to evaluate if it is necessity to call a tool
        prompt = prompt_eng.get_prompt(
            query,
            pydantic_tool_dict=tool_handler.tool_dict,
            thinking_template=tool_thinking_formatted,
            answer_format_schema=False,
            backend=self.backend
            # leading_prompt=leading_prompt,
        )

        await self.generate(
            prompt,
            messages=query.messages,
            gen_queue=tool_thinking_queue,
            temperature=query.temperature,
            top_p=query.top_p,
            stop_words=tool_thinking_to_use.root_key_stop_words,
            prefix_strings=f"<{tool_thinking_to_use.root_tag}>",
            request=request,
            # formatter=formatter_regex  # no longer enforce format
            stop_event=stop_event,
            video=query.video,
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
                messages=query.messages,
                gen_queue=tool_thinking_queue_fallback,
                temperature=query.temperature,
                top_p=query.top_p,
                request=request,
                # prefix_strings="n",
                # stop_words=TOOL_THINKING.root_key_stop_words,
                formatter=formatter_regex,  # no longer enforce format
                stop_event = stop_event,
                video=query.video,
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
                thinking_template=self.TOOL_THINKING.xml,
                thinking_response=tool_thinking_response,
                pydantic_tool_dict=tool_handler.tool_dict,
                answer_format_schema=True,
                leading_prompt=(
                    f"{fall_back_prompt}\n"
                    'Now i will convert my answer above into "functions_calling" format by continuing this continue this code.\n'
                    f"{tool_as_code_prompt}"
                ),
                backend=self.backend,
            )
            logger.info(prompt)

            # generate
            await self.generate(
                prompt,
                messages=query.messages,
                gen_queue=gen_queue,
                gen_type=GenStart(gen_type="tool"),
                temperature=query.temperature,
                top_p=query.top_p,
                # stop_words=TOOL_THINKING.root_key_stop_words,
                prefix_strings=['{\n "functions_calling": ['],
                formatter=formatter_json,
                max_tokens=query.max_tokens,
                request=request,
                stop_event=stop_event,
                video=query.video,
            )

        # NOT USE TOOL
        if not use_tool_bool:
            prompt = prompt_eng.get_prompt(
                query,
                backend=self.backend,
            )

            if query.tool_choice == "auto":
                # Normal generation
                await self.generate(
                    prompt,
                    messages=query.messages,
                    gen_queue=gen_queue,
                    gen_type=GenStart(gen_type="text"),
                    temperature=query.temperature,
                    prefix_strings=query.prefix_strings,
                    max_tokens=query.max_tokens,
                    request=request,
                    stop_event=stop_event,
                    video=query.video,
                )
            else:
                # tool choice is forced -> return empty tool calling
                gen_queue.put_nowait(GenStart(gen_type="tool"))
                gen_queue.put_nowait(GenText(content='{"functions_calling":[]}'))
                gen_queue.put_nowait(GenerationStats())
                gen_queue.put_nowait(GenEnd())


    async def _tool_calling_task(
        self,
        prompt_eng: PromptEngine,
        query:ChatMLQuery,
        request: Request,
        tool_handler: Tools,
        tool_as_code_prompt: str
    ) -> ToolCallV2:

        # queue for generating output
        gen_queue_dynamic = [GenQueueDynamic()]

        # create the pydantic schema to enforce generation
        tool_combined_pydantic_lmfe = create_function_models_v2(tool_handler.tool_dict)

        class ToolCalling_LMFE(ClassSchema):
            """ The format to call one or multiple tools """
            functions_calling: List[Union[tuple(tool_combined_pydantic_lmfe)]] = []

        # create the pydantic schema to enforce generation for formatron which use ClassSchema
        tool_combined_pydantic_formatron = create_function_models_formatron(
            tool_handler.tool_dict_formatron)

        class ToolCalling_formatron(ClassSchema):
            """ The format to call one or multiple tools """
            functions_calling: List[Union[tuple(tool_combined_pydantic_formatron)]] = []

        formatter_json = self.formatter.json(
            pydantic_model_lmfe=ToolCalling_LMFE,
            pydantic_model_formatron=ToolCalling_formatron,
            backend=self.backend,
            preference=query.guided_decoding_backend
        )

        # shar prompt with tool decision for better KV
        tool_answer_as_code = """
    
# Perform function calling if need to by continuing this code. Return {"functions_calling": []} if function calling is not needed.
response(to="function", arg_dict="""

        if query.tool_instruction_position == "prefix":
            _prefix_prompt = tool_as_code_prompt
            _leading_prompt = tool_answer_as_code
        else:
            _prefix_prompt = ""
            _leading_prompt = tool_as_code_prompt + tool_answer_as_code

        prompt = prompt_eng.get_prompt(
            query,
            pydantic_tool_dict=tool_handler.tool_dict,
            # pydantic_tool_code=tool_handler.tool_def_as_code,
            answer_format_schema=True,
            backend=self.backend,
            prefix_prompt=_prefix_prompt,
            leading_prompt=_leading_prompt,
        )

        stop_event = asyncio.Event()

        generate_kwargs = {
            "prompt": prompt,
            "messages": query.messages,
            "gen_queue": gen_queue_dynamic,
            "gen_type": GenStart(gen_type="tool"),
            "temperature": query.temperature,
            "top_p": query.top_p,
            "prefix_strings": ['{\n "functions_calling": ['],
            "formatter": formatter_json,
            "max_tokens": query.max_tokens,
            "request": request,
            "stop_event": stop_event,
            "video": query.video
        }

        return ToolCallV2(
            gen_dynamic_queue=gen_queue_dynamic,
            stop_event=stop_event,
            generate_kwargs=generate_kwargs
        )

    async def _tool_decision_task(
        self,
        prompt_eng: PromptEngine,
        query: ChatMLQuery,
        request: Request,
        tool_handler: Tools,
        tool_as_code_prompt: str
    ) -> Tuple[ToolCallV2, Callable[[str], bool]]:
        # queue for generating output
        gen_queue_dynamic = [GenQueueDynamic()]

        # create prompt
        # reuse tool call to share the kv cache as possible
        _tool_answer_prefix = "to="
        _tool_thinking_length_guide = ""
        if query.tool_call_thinking:
            _tool_answer_prefix = "thinking="
            _tool_thinking_length_guide = "# briefly think if tool thinking is needed. Never redo a tool with same arguments."

        tool_decision_answer_as_code_prompt = f"""
{_tool_thinking_length_guide}
response(""" + _tool_answer_prefix

        def tool_decision_check_fn(tool_decision_answer: str) -> bool:
            """ return True if tool usage is true"""
            if tool_decision_answer.lower().endswith('"function"') or tool_decision_answer.lower().endswith("'function'"):
                logger.info(f"Tool decision: {tool_decision_answer}")
                return True
            else:
                return False

        def tool_decision_check_with_thinking_fn(tool_decision_answer: str) -> bool:
            """ return True if tool usage is true"""
            if tool_decision_answer.lower().endswith('"function"') or tool_decision_answer.lower().endswith("'function'"):
                logger.info(f"Tool decision: internal_thinking: {tool_decision_answer}")
                return True
            else:
                return False

        if query.tool_instruction_position=="prefix":
            _prefix_prompt = tool_as_code_prompt
            _postfix_prompt = tool_decision_answer_as_code_prompt
        else:
            _prefix_prompt = ""
            _postfix_prompt = tool_as_code_prompt + tool_decision_answer_as_code_prompt

        prompt = prompt_eng.get_prompt(
            query,
            pydantic_tool_dict=tool_handler.tool_dict,
            # pydantic_tool_code=tool_handler.tool_def_as_code,
            answer_format_schema=True,
            backend=self.backend,
            prefix_prompt=_prefix_prompt,
            leading_prompt=_postfix_prompt
        )

        formatter_regex = self.formatter.regex(
            regex_pattern=r'"user"|"function"',
            backend=self.backend,
            preference=query.guided_decoding_backend
        )

        stop_event = asyncio.Event()

        generate_kwargs = {
            "prompt": prompt,
            "messages": query.messages,
            "gen_queue": gen_queue_dynamic,
            "gen_type": GenStart(gen_type="tool"),
            "temperature": query.temperature,
            "top_p": query.top_p,
            "prefix_strings": '"',
            "formatter": formatter_regex if not query.tool_call_thinking else None,
            "max_tokens": query.tool_call_thinking_token,
            "request": request,
            "stop_event": stop_event,
            "stop_words": ['"user"', '"function"', "'user'", "'function'"],
            "video": query.video
        }

        tool_decision_check_fn_to_use = tool_decision_check_fn if query.tool_call_thinking else tool_decision_check_with_thinking_fn

        return ToolCallV2(
            gen_dynamic_queue=gen_queue_dynamic,
            stop_event=stop_event,
            generate_kwargs=generate_kwargs
        ), tool_decision_check_fn_to_use


    async def _no_tool_task(
        self,
        prompt_eng: PromptEngine,
        query:ChatMLQuery,
        request: Request
    ) -> ToolCallV2:

        gen_queue_dynamic = [GenQueueDynamic()]

        prompt = prompt_eng.get_prompt(
            query,
            backend=self.backend,
        )

        stop_event = asyncio.Event()

        generate_kwargs = {
            "prompt": prompt,
            "messages": query.messages,
            "gen_queue": gen_queue_dynamic,
            "gen_type": GenStart(gen_type="text"),
            "temperature": query.temperature,
            "top_p": query.top_p,
            "prefix_strings": query.prefix_strings,
            "max_tokens": query.max_tokens,
            "request": request,
            "stop_event": stop_event,
            "video": query.video,
        }

        return ToolCallV2(
            gen_dynamic_queue=gen_queue_dynamic,
            stop_event=stop_event,
            generate_kwargs=generate_kwargs
        )



    async def chat_with_tool_v2(
        self,
        query: ChatMLQuery,
        prompt_eng: PromptEngine,
        gen_queue: GenQueueDynamic,
        request: Request,
        stop_event: asyncio.Event = None
    ):
        """
        handle tool calling llm generation
        main purpose is to handle auto model of generation by generate both tool and non tool usage at the same time
        dynamically swap to the best answer by LLM
        This will reduce the wait time from generating sequentially
        """
        # TODO implement non async version

        # tool class have the method to handle converting processing or tool requirement, schema and response
        tool_handler = Tools(
            prompt_eng=prompt_eng,
            tools=query.tools,
            tool_choice=query.tool_choice,
        )

        # create prompt
        _tool_thinking_fn_header = ""
        _tool_answer_prefix = "to="
        if query.tool_call_thinking:
            _tool_thinking_fn_header = 'internal_thinking: str="",'
            _tool_answer_prefix = "thinking="

        tool_as_code_prompt = """
Function/ Tool calling Instruction:
# Use Function calling if:
- It is needed to answer user question.
- Be conservative and only use function calling if it is necessary and suitable.

# Reply directly to user if:
- It is uncertain if user wants function calling.
- Function calling is not related to user's question.
- Clarification is needed. 
- Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.

When a tool/ function is requested, it's result will be run by the system and be returned with reference to the tool call if available
e.g. Result of tool call reference id <tool_call_id>.
```python
def run_function(arg_dict):
    function_calls = arg_dict["functions_calling"]

    if function_calls == []:
        print("No function/tool calling needed")
        return

    for call in function_calls:
        function_name = call["name"]
        arguments = call["arguments"]
        globals()[function_name](**arguments)
        
allow_functions = [""" + tool_handler.tool_name_list + """]

def response(""" + _tool_thinking_fn_header + """to: Literal["user","function"], arg_dict: Optional[List[]]=[]):
   if to=="user":
       # NO function calling needed
       break  # exit for a normal answer
   elif to=="function" and arg_dict!=[]:
       run_function(arg_dict)
   else:
      raise Exception("Either response to user without function calling or function calling is required.")
```
Example:
# No function calling needed:
response(thinking="because... so function calling is not required ", to="user")
my answer is...

# Function calling needed:
response(thinking="because... so function calling is required", to="function", arg_dict={
  "functions_calling": [{
      "name": "function_name",
      "arguments": {"argument_name": "value"}
    }
]}) 
# answer to user is prohibited if using function calling

# Example of follow-up answer after function calling result provided by user
User: what is 2^3?
Assistant:                                                                                                                                                             
power_number_tool({number: 2, power: 3})
8
Assistant: 2^3 is 8
---
IMPORTANT: 'Request for tool call with reference id' and 'Result of tool call reference id' are auto populated by the system for reference
            DO NOT output Request for tool call with reference id by yourself
End of Function Calling Instruction
---
"""


        tool_call = await self._tool_calling_task(
            prompt_eng=prompt_eng,
            query=query,
            request=request,
            tool_handler=tool_handler,
            tool_as_code_prompt=tool_as_code_prompt
        )

        tool_decision, tool_decision_check_fn = await self._tool_decision_task(
            prompt_eng=prompt_eng,
            query=query,
            request=request,
            tool_handler=tool_handler,
            tool_as_code_prompt=tool_as_code_prompt
        )


        no_tool = await self._no_tool_task(
            prompt_eng=prompt_eng,
            query=query,
            request=request
        )

        tasks = []  # to track the task running

        tool_decision_task = None
        tool_task = None
        no_tool_task = None


        # ensure result is out for decision
        tool_decision_outcome: Literal["user", "function"] = "function"

        # tool task will always be run. It is assumed that if this function is called, tool generation is required
        tool_task = asyncio.create_task(self.generate(**tool_call.generate_kwargs))
        tasks.append(tool_task)

        # if mode is auto, run the tool usage decision generation
        if query.tool_choice == "auto":
            tool_decision_task = asyncio.create_task(self.generate(**tool_decision.generate_kwargs))
            tasks.append(tool_decision_task)

            no_tool_task = asyncio.create_task(self.generate(**no_tool.generate_kwargs))
            tasks.append(no_tool_task)

            tool_decision_outcome, _ = await get_response_from_queue(tool_decision.gen_dynamic_queue)


        if tool_decision_check_fn(tool_decision_outcome):
            logger.info("Tool auto decision: function")

            _has_tool = True

            # check first if tool managed to generate
            if tool_call.gen_dynamic_queue[0].qsize() < 3:
                await asyncio.sleep(0.5)    # wait for 0.5 more second

                if tool_call.gen_dynamic_queue[0].qsize() < 3:
                    _has_tool = False

            if _has_tool:
                if no_tool:
                    no_tool.stop_event.set()

                # swap queue for output to function calling
                gen_queue.swap(tool_call.gen_dynamic_queue[0])
            else:
                # tool not managed to generate, fall back to standard reply
                tool_call.stop_event.set()

                # swap queue non function calling
                gen_queue.swap(no_tool.gen_dynamic_queue[0])

        else:
            logger.info("Tool auto decision: user")
            if tool_call:
                tool_call.stop_event.set()

            # swap queue non function calling
            gen_queue.swap(no_tool.gen_dynamic_queue[0])


        # Wait for the remaining tasks to complete (if needed)
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in one of the tasks: {e}")

