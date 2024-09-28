from logging import raiseExceptions
from typing import List, Union, Literal, Optional
from formatron.schemas.pydantic import ClassSchema
from lmformatenforcer import JsonSchemaParser, RegexParser
from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
from gallama.logger.logger import logger
from pydantic import BaseModel
from .tools import Tools


# experimental support for formatron
try:
    from formatron.formatter import FormatterBuilder
    from formatron.integrations.exllamav2 import create_formatter_filter

except:
    FormatterBuilder = None
    create_formatter_filter = None

class FormatEnforcer:
    """ this class will help to create filter for generation enforcement"""

    def __init__(self):
        pass

    @staticmethod
    def get_default_engine(
        backend:str = "exllama",
        preference: Literal["auto", "formatron", "lm-format-enforcer"] = "auto",
    ) -> Literal["formatron", "lm_enforcer"]:
        """ this function will select the format enforcer engine to use if not selected by user"""

        if preference != "auto":
            logger.info(f"guided encoding preference: {preference}")

        # formatron doesnt support llama cpp at the moment
        if backend == "llama_cpp":
            return "lm_enforcer"
        elif backend == "exllama":
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
        else:
            raise "Invalid backend"



    def regex(
        self,
        regex_pattern: str,
        filter_engine: Literal[
        "formatron", "lm_enforcer"] = None,
        backend: str = "exllama",
        preference: Literal["auto", "formatron", "lm-format-enforcer"] = "auto",
    ) -> FormatterBuilder | TokenEnforcerTokenizerData:

        logger.info(backend)
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

    def json(
        self,
        pydantic_model_lmfe: BaseModel,
        pydantic_model_formatron: ClassSchema,
        filter_engine: Literal["formatron", "lm_enforcer"] = None,
        backend: Literal["llama_cpp", "exllama"] = "exllama",
        preference: Literal["auto", "formatron", "lm-format-enforcer"] = "auto",
    ) -> FormatterBuilder | TokenEnforcerTokenizerData:
        """ this function will return the filters for format enforcer to generate json output based on Pyantic model"""

        # set the filter engine to use
        if not filter_engine:
            filter_engine = FormatEnforcer.get_default_engine(backend=backend, preference=preference)  # if engine is specified, use it

        assert filter_engine == "lm_enforcer" or filter_engine == "formatron"

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
