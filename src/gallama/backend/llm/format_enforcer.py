import json
from typing import Any, Dict, Literal, Optional, Union
from gallama.logger.logger import logger
from pydantic import BaseModel
from gallama.backend.llm.tools import Tools

try:
    from formatron.schemas.pydantic import ClassSchema
except ImportError:
    ClassSchema = None

try:
    from lmformatenforcer import JsonSchemaParser, RegexParser
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
except ImportError:
    JsonSchemaParser = None
    RegexParser = None
    TokenEnforcerTokenizerData = None


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
    def is_formatron_available() -> bool:
        return FormatterBuilder is not None and ClassSchema is not None

    @staticmethod
    def is_lmfe_available() -> bool:
        return (
            JsonSchemaParser is not None
            and RegexParser is not None
            and TokenEnforcerTokenizerData is not None
        )

    @staticmethod
    def _dependency_error(backend: str, engine: FilterEngine) -> RuntimeError:
        extra = {
            "formatron": {
                "exllama": "exl2",
                "exllamav3": "exl3",
                "transformers": "transformers-backend",
            }.get(backend, "guided-decoding-all"),
            "lm-format-enforcer": {
                "exllama": "exl2",
                "llama_cpp": "llama-cpp",
                "llama_cpp_server": "llama-cpp",
                "ik_llama": "guided-decoding-all",
            }.get(backend, "guided-decoding-all"),
        }.get(engine, "guided-decoding-all")
        return RuntimeError(
            f"Guided decoding engine '{engine}' for backend '{backend}' is not installed. "
            f"Install the backend extra 'gallama[{extra}]' or a superset extra."
        )

    @staticmethod
    def _require_engine(backend: str, engine: FilterEngine) -> None:
        if engine == "formatron" and not FormatEnforcer.is_formatron_available():
            raise FormatEnforcer._dependency_error(backend, engine)
        if engine == "lm-format-enforcer" and not FormatEnforcer.is_lmfe_available():
            raise FormatEnforcer._dependency_error(backend, engine)

    @staticmethod
    def get_default_engine(
        backend: Literal["exllama", "exllamav3", "llama_cpp", "llama_cpp_server", "ik_llama", "transformers", "sglang", "vllm"] = "exllama",
        preference: FilterEngineOption = "auto",
    ) -> FilterEngine:

        """ this function will select the format enforcer engine to use if not selected by user"""

        if preference != "auto":
            logger.info(f"guided encoding preference: {preference}")

        # formatron does not support llama cpp at the moment
        if backend in ["llama_cpp", "llama_cpp_server", "ik_llama"]:
            engine = "lm-format-enforcer"
        elif backend in ["exllamav3"]:
            engine = "formatron"
        elif backend in ["exllama", "transformers"]:
            # use formatron if it is available if it is exllama
            if preference == "auto":
                if FormatEnforcer.is_formatron_available():
                    engine = "formatron"
                elif FormatEnforcer.is_lmfe_available():
                    engine = "lm-format-enforcer"
                else:
                    raise RuntimeError(
                        f"Backend '{backend}' requires a guided decoding engine, but neither "
                        "'formatron' nor 'lm-format-enforcer' is installed."
                    )
            else:
                if preference == "formatron":
                    engine = "formatron"
                elif preference == "lm-format-enforcer":
                    engine = "lm-format-enforcer"
                else:
                    raise ValueError("Invalid backend")
        elif backend == "sglang":
            return "sglang_formatter"
        elif backend == "vllm":
            return "auto"   # vllm require setting guided decoding upon server start
        else:
            raise ValueError("Invalid backend")

        if preference != "auto" and backend == "exllamav3" and preference != "formatron":
            raise RuntimeError("Backend 'exllamav3' only supports the 'formatron' guided decoding engine.")

        FormatEnforcer._require_engine(backend, engine)
        return engine


    def regex(
        self,
        regex_pattern: str,
        filter_engine: FilterEngine = None,
        backend: Literal["exllama", "exllamav3", "llama_cpp", "llama_cpp_server", "ik_llama", "transformers", "sglang", "vllm"] = "exllama",
        preference: FilterEngineOption = "auto",
    ) -> Any:

        # set the filter engine to use
        if not filter_engine:
            filter_engine = FormatEnforcer.get_default_engine(
                backend=backend,
                preference=preference
            )  # if engine is specified, use it

        if backend != "vllm":
            # create filter if engine is lm_enforcer
            if filter_engine == "lm-format-enforcer":
                self._require_engine(backend, filter_engine)
                return RegexParser(regex_pattern)

            # create filter if engine is formatron
            elif filter_engine == "formatron":
                self._require_engine(backend, filter_engine)
                f = FormatterBuilder()
                _regex = f.regex(regex_pattern, capture_name='regex')
                f.append_line(f"{_regex}")
                return f

            # for sglang, return the regex pattern itself
            elif filter_engine == "sglang_formatter":
                return SGLangFormatter(regex_pattern=regex_pattern)

            else:
                raise ValueError("Invalid backend")

        else:
            if GuidedDecodingParams is None:
                raise RuntimeError(
                    "Backend 'vllm' guided decoding requires vllm to be installed."
                )
            return GuidedDecodingParams(
                regex=regex_pattern,
            )


    def json(
        self,
        pydantic_model_lmfe: BaseModel,
        pydantic_model_formatron: Optional[Any] = None,
        filter_engine: FilterEngine = None,
        backend: Literal["llama_cpp", "llama_cpp_server", "ik_llama", "exllama", "exllamav3", "transformers", "sglang", "vllm"] = "exllama",
        preference: FilterEngineOption = "auto",
    ) -> Any:
        """ this function will return the filters for format enforcer to generate json output based on Pyantic model"""

        # set the filter engine to use
        if not filter_engine:
            filter_engine = FormatEnforcer.get_default_engine(backend=backend, preference=preference)  # if engine is specified, use it

        # create filter if engine is lm_enforcer
        if backend != "vllm":
            if filter_engine == "lm-format-enforcer":  # TODO currently formatron and nested pydantic model is having issue
                self._require_engine(backend, filter_engine)
                json_schema = Tools.replace_refs_with_definitions_v2(pydantic_model_lmfe.model_json_schema())
                return JsonSchemaParser(json_schema)

            # create filter if engine is formatron
            elif filter_engine == "formatron":
                self._require_engine(backend, filter_engine)
                if pydantic_model_formatron is None:
                    raise RuntimeError("Formatron model schema is required when using the 'formatron' engine.")
                f = FormatterBuilder()
                f.append_line(f"{f.json(pydantic_model_formatron, capture_name='json')}")
                return f

            # for sglang, return the regex pattern itself
            elif filter_engine == "sglang_formatter":
                return SGLangFormatter(json_schema=pydantic_model_lmfe.model_json_schema())
            else:
                raise ValueError("Invalid backend")
        else:
            logger.info(f"guided encoding filter_engine: {filter_engine}")
            if GuidedDecodingParams is None:
                raise RuntimeError(
                    "Backend 'vllm' guided decoding requires vllm to be installed."
                )

            return GuidedDecodingParams(
                json=pydantic_model_lmfe.model_json_schema(),
            )
