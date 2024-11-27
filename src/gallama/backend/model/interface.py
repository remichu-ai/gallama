from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional
from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
from formatron.formatter import FormatterBuilder
import asyncio
from fastapi import HTTPException, Request

from ...logger.logger import logger
from ...utils.utils import get_token_length
from ...api_response.chat_response import get_response_from_queue
from ...data_classes import (
    ModelParser,
    ChatMLQuery,
    GenEnd,
    GenText,
    GenQueue,
    GenStart,
    GenerationStats,
    QueueContext
)
from ..format_enforcer import FormatEnforcer
from ..prompt_engine import PromptEngine

class ModelInterface(ABC):
    @abstractmethod
    def __init__(self,
        model_spec:ModelParser,
        model_config: Dict,
        draft_model_config: Dict = None,
        eos_token_list_from_prompt_template: List[str] = None
    ):
        # initialization share the same code to keep the frontend consistent
        # if a backend does not support the value, please set it in the __init__ or load_model()

        # model_spec capture cli argument
        # model_config is from yml file
        self.model_id = model_config["model_id"]
        self.model_name = model_spec.model_name or model_config["model_name"]
        self.max_seq_len = model_spec.max_seq_len or model_config.get("max_seq_len", None)
        if self.max_seq_len is not None:
            self.max_seq_len = (self.max_seq_len//256) * 256     # for paged attention

        self.gpus = model_spec.gpus or model_config.get("gpus") or "auto"
        self.cache_size = model_spec.cache_size or model_config.get("cache_size") or self.max_seq_len   # default to max_seq_len if not set
        if self.cache_size is not None:
            self.cache_size = (self.cache_size//256) * 256     # for paged attention
            if self.max_seq_len is not None:
                # cache size must be greater or equal to max_seq_len
                self.cache_size = max(self.cache_size, self.max_seq_len)

        self.cache_quant = model_spec.cache_quant or model_config.get("cache_quant") or "Q4"
        self.backend = model_spec.backend or model_config["backend"] or "exllama"
        self.tensor_parallel = model_spec.tensor_parallel or model_config.get("tensor_parallel", False)

        # transformers specific arguments
        self.backend_extra_args = model_config.get("backend_extra_args") or {}

        # draft model is via cli only
        self.draft_model_id = draft_model_config.get("model_id")
        self.draft_model_name = model_spec.draft_model_name or None
        self.draft_gpus = model_spec.gpus or draft_model_config.get("gpus") or "auto"
        self.draft_cache_size = self.cache_size   # set to the same as main model
        self.draft_cache_quant = model_spec.draft_cache_quant or draft_model_config.get("cache_quant") or "Q4"
        assert (self.draft_model_id is None) == (self.draft_model_name is None)

        # get the eos_token_str by merging the default config with anything set by user
        self.eos_token_str = list(set(model_config.get("eos_token_list", []) + eos_token_list_from_prompt_template))
        self.eos_token_str_set = set(self.eos_token_str)    # set for some more efficient operation

        # load_model method in each subclass should set the following parameters:
        self.model = None
        self.tokenizer = None
        self.cache = None
        self.draft_model = None
        self.draft_cache = None
        self.processor = None       # processor is for processing of vision input
        self.eos_token_ids = None
        self.load_model()

        # placeholder
        # pipeline is a wrapper for model so that generation can be more standardized
        # this is because each backend expose different interface
        self.pipeline = None

        # format enforcer
        self.formatter = FormatEnforcer()

    ## *************** the following method must be implemented by each backend ********
    @abstractmethod
    def load_model(self):
        """Load the model, tokenizer, cache, and optional processor."""
        pass

    @abstractmethod
    def generate_eos_tokens_id(self) -> List[int]:
        """Generate the end-of-sequence token IDs."""
        pass

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
        quiet: bool = False,
        messages: List = None,  # query.message for multimodal
        **kwargs,
    ) -> (str, GenerationStats):
        pass


    ## *******************************************************************

    @abstractmethod
    async def chat(self, query: ChatMLQuery, prompt_eng: PromptEngine, gen_queue: GenQueue, request: Request):
        """
        This function will route the request to chat with tool or without tool accordingly
        """
        if query.tools or query.tool_choice != "none":
            chat_method = self.chat_with_tool
        else:
            chat_method = self.chat_no_tool
        return await chat_method(
            query=query,
            prompt_eng=prompt_eng,
            gen_queue=gen_queue,
            request=request
        )

    @abstractmethod
    async def chat_raw(
        self,
        prompt: str,
        gen_queue: asyncio.Queue,
        request: Request,
        # stream: bool = False,
        max_tokens: int = None,
        quiet=False,    # to disable any logging, mostly used for initialization cache prefill
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
            request=request
        )

    def validate_token_length(self, token_length):
        """
        validate that token_length is within max sequence length
        """
        # if max_seq_len == None meaning there is no token length yet
        if self.max_seq_len and token_length > self.max_seq_len:
            raise HTTPException(status_code=400, detail=f"Token length exceeds max length of {self.max_seq_len}")


    async def chat_no_tool(self, query: ChatMLQuery, prompt_eng: PromptEngine, gen_queue, request: Request):

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
            )
            thinking_response, _ = await get_response_from_queue(thinking_queue)

            # Get the new prompt with thinking response
            prompt = prompt_eng.get_prompt(
                query,
                thinking_template=query.thinking_template,
                thinking_response=thinking_response,
                exllama_vision_token=(self.backend=="exllama")
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
            exllama_vision_token=(self.backend == "exllama")
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
                exllama_vision_token=(self.backend=="exllama"),
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
            )

        # NOT USE TOOL
        if not use_tool_bool:
            prompt = prompt_eng.get_prompt(
                query,
                exllama_vision_token=(self.backend == "exllama"),
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
                )
            else:
                # tool choice is forced -> return empty tool calling
                gen_queue.put_nowait(GenStart(gen_type="tool"))
                gen_queue.put_nowait(GenText(content='{"functions_calling":[]}'))
                gen_queue.put_nowait(GenerationStats())
                gen_queue.put_nowait(GenEnd())