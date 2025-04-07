import json
from typing import Literal, Dict
from formatron.schemas.pydantic import ClassSchema
from lmformatenforcer import JsonSchemaParser, RegexParser
from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
from gallama.logger.logger import logger
from pydantic import BaseModel
from gallama.backend.llm.tools import Tools


try:
    from formatron.formatter import FormatterBuilder
    from formatron.integrations.exllamav2 import create_formatter_filter

except:
    FormatterBuilder = None
    create_formatter_filter = None


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


class FormatEnforcer:
    """ this class will help to create filter for generation enforcement"""

    def __init__(self):
        pass

    @staticmethod
    def get_default_engine(
        backend: Literal["exllama", "llama_cpp", "transformers", "sglang"] = "exllama",
        preference: Literal["auto", "formatron", "lm-format-enforcer", "sglang_formatter"] = "auto",
    ) -> Literal["formatron", "lm_enforcer", "sglang_formatter"]:

        """ this function will select the format enforcer engine to use if not selected by user"""

        if preference != "auto":
            logger.info(f"guided encoding preference: {preference}")

        # formatron doesnt support llama cpp at the moment
        if backend == "llama_cpp":
            return "lm_enforcer"
        elif backend == "exllama" or backend=="transformers":
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
        else:
            raise "Invalid backend"



    def regex(
        self,
        regex_pattern: str,
        filter_engine: Literal["formatron", "lm_enforcer", "sglang_formatter"] = None,
        backend: Literal["exllama", "llama_cpp", "transformers", "sglang"] = "exllama",
        preference: Literal["auto", "formatron", "lm-format-enforcer", "sglang_formatter"] = "auto",
    ) -> FormatterBuilder | TokenEnforcerTokenizerData | SGLangFormatter:

        # set the filter engine to use
        if not filter_engine:
            filter_engine = FormatEnforcer.get_default_engine(backend=backend, preference=preference)  # if engine is specified, use it

        # create filter if engine is lm_enforcer
        if filter_engine == "lm_enforcer":
            return RegexParser(regex_pattern)

        # create filter if engine is formatron
        if filter_engine == "formatron":
            f = FormatterBuilder()
            _regex = f.regex(regex_pattern, capture_name='regex')
            f.append_line(f"{_regex}")
            return f

        # for sglang, return the regex pattern it self
        if filter_engine == "sglang_formatter":
            return SGLangFormatter(regex_pattern=regex_pattern)


    def json(
        self,
        pydantic_model_lmfe: BaseModel,
        pydantic_model_formatron: ClassSchema,
        filter_engine: Literal["formatron", "lm_enforcer", "sglang_formatter"] = None,
        backend: Literal["llama_cpp", "exllama", "transformers", "sglang"] = "exllama",
        preference: Literal["auto", "formatron", "lm-format-enforcer", "sglang_formatter"] = "auto",
    ) -> FormatterBuilder | TokenEnforcerTokenizerData | SGLangFormatter:
        """ this function will return the filters for format enforcer to generate json output based on Pyantic model"""

        # set the filter engine to use
        if not filter_engine:
            filter_engine = FormatEnforcer.get_default_engine(backend=backend, preference=preference)  # if engine is specified, use it

        assert filter_engine == "lm_enforcer" or filter_engine == "formatron" or filter_engine == "sglang_formatter"

        # create filter if engine is lm_enforcer
        # if filter_engine == "lm_enforcer" or filter_engine == "formatron":  # TODO currently formatron and nested pydantic model is having issue
        if filter_engine == "lm_enforcer":  # TODO currently formatron and nested pydantic model is having issue
            json_schema = Tools.replace_refs_with_definitions_v2(pydantic_model_lmfe.model_json_schema())
            return JsonSchemaParser(json_schema)

        # create filter if engine is formatron
        elif filter_engine == "formatron":
            f = FormatterBuilder()
            f.append_line(f"{f.json(pydantic_model_formatron, capture_name='json')}")
            return f

        # for sglang, return the regex pattern itself
        if filter_engine == "sglang_formatter":
            return SGLangFormatter(json_schema=pydantic_model_lmfe.model_json_schema())