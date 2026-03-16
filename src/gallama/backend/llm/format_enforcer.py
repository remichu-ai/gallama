import json
from typing import Literal, Dict, Union
from formatron.schemas.pydantic import ClassSchema
from lmformatenforcer import JsonSchemaParser, RegexParser
from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
from gallama.logger.logger import logger
from pydantic import BaseModel
from gallama.backend.llm.tools import Tools


try:
    from formatron.formatter import FormatterBuilder

except ImportError:
    FormatterBuilder = None


try:
    from vllm.sampling_params import GuidedDecodingParams
except ImportError:
    GuidedDecodingParams = None


class SGLangFormatter:
    def __init__(self, regex_pattern: str = None, json_schema: str = None):
        self.regex_pattern = regex_pattern if regex_pattern else None
        self.json_schema = json.dumps(json_schema) if json_schema else None

    def get_formatter_dict(self) -> Dict[str, str]:
        enforcement_dict = {}

        if self.regex_pattern:
            enforcement_dict["regex"] = self.regex_pattern

        if self.json_schema:
            enforcement_dict["json_schema"] = self.json_schema

        return enforcement_dict

FilterEngine = Literal["formatron", "lm-format-enforcer", "sglang_formatter"]

FilterEngineOption = Union[FilterEngine, Literal["auto"]]

class FormatEnforcer:
    """ this class will help to create filter for generation enforcement"""

    def __init__(self):
        pass

    @staticmethod
    def get_default_engine(
        backend: Literal["exllama", "llama_cpp", "llama_cpp_server", "transformers", "sglang", "vllm"] = "exllama",
        preference: FilterEngineOption = "auto",
    ) -> FilterEngine:

        """ this function will select the format enforcer engine to use if not selected by user"""

        if preference != "auto":
            logger.info(f"guided encoding preference: {preference}")

        # formatron does not support llama cpp at the moment
        if backend in ["llama_cpp", "llama_cpp_server"]:
            return "lm_enforcer"
        elif backend in ["exllamav3"]:
            return "formatron"
        elif backend in ["exllama", "transformers", "exllamav3"]:
            # use formatron if it is available if it is exllama
            if preference == "auto":
                if FormatterBuilder:
                    return "formatron"
                else:
                    return "lm_enforcer"
            else:
                if preference == "formatron" and FormatterBuilder:
                    return "formatron"
                elif preference == "lm-format-enforcer":
                    return "lm_enforcer"
                else:
                    raise "Invalid backend"
        elif backend == "sglang":
            return "sglang_formatter"
        elif backend == "vllm":
            return "auto"   # vllm require setting guided decoding upon server start
        else:
            raise "Invalid backend"


    def regex(
        self,
        regex_pattern: str,
        filter_engine: FilterEngine = None,
        backend: Literal["exllama", "llama_cpp", "llama_cpp_server", "transformers", "sglang", "vllm"] = "exllama",
        preference: FilterEngineOption = "auto",
    ) -> FormatterBuilder | TokenEnforcerTokenizerData | SGLangFormatter | GuidedDecodingParams:

        # set the filter engine to use
        if not filter_engine:
            filter_engine = FormatEnforcer.get_default_engine(
                backend=backend,
                preference=preference
            )  # if engine is specified, use it

        if backend != "vllm":
            # create filter if engine is lm_enforcer
            if filter_engine == "lm_enforcer":
                return RegexParser(regex_pattern)

            # create filter if engine is formatron
            elif filter_engine == "formatron":
                f = FormatterBuilder()
                _regex = f.regex(regex_pattern, capture_name='regex')
                f.append_line(f"{_regex}")
                return f

            # for sglang, return the regex pattern itself
            elif filter_engine == "sglang_formatter":
                return SGLangFormatter(regex_pattern=regex_pattern)

            else:
                raise "Invalid backend"

        else:
            return GuidedDecodingParams(
                regex=regex_pattern,
            )


    def json(
        self,
        pydantic_model_lmfe: BaseModel,
        pydantic_model_formatron: ClassSchema,
        filter_engine: FilterEngine = None,
        backend: Literal["llama_cpp", "llama_cpp_server", "exllama", "transformers", "sglang", "vllm"] = "exllama",
        preference: FilterEngineOption = "auto",
    ) -> JsonSchemaParser | FormatterBuilder | SGLangFormatter | GuidedDecodingParams:
        """ this function will return the filters for format enforcer to generate json output based on Pyantic model"""

        # set the filter engine to use
        if not filter_engine:
            filter_engine = FormatEnforcer.get_default_engine(backend=backend, preference=preference)  # if engine is specified, use it

        # create filter if engine is lm_enforcer
        if backend != "vllm":
            if filter_engine == "lm_enforcer":  # TODO currently formatron and nested pydantic model is having issue
                json_schema = Tools.replace_refs_with_definitions_v2(pydantic_model_lmfe.model_json_schema())
                return JsonSchemaParser(json_schema)

            # create filter if engine is formatron
            elif filter_engine == "formatron":
                f = FormatterBuilder()
                f.append_line(f"{f.json(pydantic_model_formatron, capture_name='json')}")
                return f

            # for sglang, return the regex pattern itself
            elif filter_engine == "sglang_formatter":
                return SGLangFormatter(json_schema=pydantic_model_lmfe.model_json_schema())
            else:
                raise "Invalid backend"
        else:
            logger.info(f"guided encoding filter_engine: {filter_engine}")
            json_schema = Tools.replace_refs_with_definitions_v2(pydantic_model_lmfe.model_json_schema())

            return GuidedDecodingParams(
                json=pydantic_model_lmfe.model_json_schema(),
            )
