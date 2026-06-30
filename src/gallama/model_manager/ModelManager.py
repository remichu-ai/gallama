from ..data_classes import ModelSpec
from typing import Dict, Any, Literal, Optional, Tuple
from ..logger import logger
from ..logger.logger import basic_log_extra
from ..config.config_manager import ConfigManager



class ModelManager:
    def __init__(self):
        self.llm_dict: Dict[str, Any] = {}               # dict to all llm models process object
        self.llm_dict_non_strict: Dict[str, Any] = {}
        self.llm_aliases: Dict[str, str] = {}
        self.tts_dict: Dict[str, Any] = {}
        self.tts_dict_non_strict: Dict[str, Any] = {}
        self.tts_aliases: Dict[str, str] = {}
        self.stt_dict: Dict[str, Any] = {}
        self.stt_dict_non_strict: Dict[str, Any] = {}
        self.stt_aliases: Dict[str, str] = {}
        self.embedding_dict: Dict[str, Any] = {}
        self.embedding_dict_non_strict: Dict[str, Any] = {}
        self.embedding_aliases: Dict[str, str] = {}
        self.reranker_dict: Dict[str, Any] = {}
        self.reranker_dict_non_strict: Dict[str, Any] = {}
        self.reranker_aliases: Dict[str, str] = {}
        self.config_manager = ConfigManager()
        self.model_ready = False

    def close_all_models(self):
        seen = set()
        all_model_dicts = (
            self.llm_dict,
            self.tts_dict,
            self.stt_dict,
            self.embedding_dict,
            self.reranker_dict,
        )

        for model_dict in all_model_dicts:
            for model in model_dict.values():
                if id(model) in seen:
                    continue
                seen.add(id(model))

                if hasattr(model, "close"):
                    try:
                        model.close()
                    except Exception as exc:
                        logger.error(f"Failed to close model resource cleanly: {exc}")

    def _get_model_dicts(
        self,
        _type: Literal["llm", "tts", "stt", "embedding", "reranker"]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
        if _type == "llm":
            return self.llm_dict, self.llm_dict_non_strict, self.llm_aliases
        elif _type == "tts":
            return self.tts_dict, self.tts_dict_non_strict, self.tts_aliases
        elif _type == "stt":
            return self.stt_dict, self.stt_dict_non_strict, self.stt_aliases
        elif _type == "embedding":
            return self.embedding_dict, self.embedding_dict_non_strict, self.embedding_aliases
        elif _type == "reranker":
            return self.reranker_dict, self.reranker_dict_non_strict, self.reranker_aliases
        else:
            raise ValueError(f"Invalid model type: {_type}")

    def get_model(self, model_name: str, _type: Literal["llm", "tts", "stt", "embedding", "reranker"]) -> Optional[Any]:
        strict_dict, non_strict_dict, aliases = self._get_model_dicts(_type)

        # Check if the model exists in the strict dictionary
        if model_name in strict_dict:
            return strict_dict[model_name]

        canonical_name = aliases.get(model_name)
        if canonical_name in strict_dict:
            return strict_dict[canonical_name]

        # If not, check if there are any models in the non-strict dictionary
        if non_strict_dict:
            # Return the first model in the non-strict dictionary
            return next(iter(non_strict_dict.values()))

        # If no model is found, return None
        return None

    def list_model_ids(self) -> list[str]:
        model_ids = []
        seen = set()

        for strict_dict, _, aliases in (
            self._get_model_dicts("llm"),
            self._get_model_dicts("stt"),
            self._get_model_dicts("tts"),
            self._get_model_dicts("embedding"),
            self._get_model_dicts("reranker"),
        ):
            for model_name in strict_dict.keys():
                if model_name not in seen:
                    model_ids.append(model_name)
                    seen.add(model_name)
            for alias in aliases.keys():
                if alias not in seen:
                    model_ids.append(alias)
                    seen.add(alias)

        return model_ids

    def _update_model(self, model_name: str, model_spec: ModelSpec, model_object: Any):
        strict_dict, non_strict_dict, aliases = self._get_model_dicts(model_spec.model_type)
        strict_dict[model_name] = model_object
        if not model_spec.strict:
            non_strict_dict[model_name] = model_object

        for alias in model_spec.aliases:
            if alias == model_name:
                continue
            existing_model_name = aliases.get(alias)
            if existing_model_name and existing_model_name != model_name:
                raise ValueError(f"Alias '{alias}' is already assigned to model '{existing_model_name}'")
            if alias in strict_dict and alias != model_name:
                raise ValueError(f"Alias '{alias}' conflicts with loaded model '{alias}'")
            aliases[alias] = model_name


    def load_model(self, model_spec: ModelSpec):
        """
        model_spec is model specification coming from cli
        it might not have all the properties required for the model to be loaded
        the config_manager below contain all the models properties
        """
        model_name = model_spec.model_name
        if not model_name:
            raise Exception("model_name is required when loading a model from CLI")

        # get the config from the yml, if available
        model_config = self.config_manager.get_effective_model_config(model_name) or {}
        if model_config:
            model_config = model_config.copy()
            model_config.update({"model_name": model_name})
        else:
            logger.info(
                f"Model config for '{model_name}' not found in ~/gallama/model_config.yaml, using CLI arguments only",
                extra=basic_log_extra(),
            )

        # handle draft model
        if model_spec.draft_model_name and not model_spec.draft_model_id:
            draft_model_config = self.config_manager.get_model_config(model_spec.draft_model_name)
            if not draft_model_config:
                raise Exception(
                    f"Draft model config for '{model_spec.draft_model_name}' not exist in ~/gallama/model_config.yaml"
                )
            model_config.update({
                "draft_model_id": model_spec.draft_model_id or draft_model_config["model_id"],
                "draft_model_name": model_spec.draft_model_name or draft_model_config["model_name"],
                "draft_gpus": model_spec.draft_gpus or draft_model_config["gpus"],
                "draft_cache_quant": model_spec.draft_cache_quant or draft_model_config["cache_quant"],
            })

        if model_config:
            _default_model_spec = ModelSpec.from_dict(model_config)
            # Merge configurations that user pass in with default setting of the model
            model_spec = ModelSpec.from_merged_config(model_spec, _default_model_spec.model_dump())
            logger.info(f"Resolved model_spec from config: {model_spec}", extra=basic_log_extra())

        if not model_spec.model_id:
            raise Exception(f"model_id is required for '{model_name}' when it is not fully defined in model_config.yaml")

        if not model_spec.backend:
            raise Exception(f"backend is required for '{model_name}' when it is not defined in model_config.yaml")

        if not model_spec.model_type:
            model_spec.model_type = ModelSpec.get_model_type_from_backend(model_spec.backend)


        # load the model with config from the model_spec and yml. model_spec comes from cli
        logger.info(f"model_spec.backend: {model_spec.backend}", extra=basic_log_extra())
        if model_spec.backend in ["exllama", "llama_cpp", "llama_cpp_server", "ik_llama", "transformers", "mlx_vlm", "sglang", "exllamav3", "vllm"]:  # llm loading
            if model_spec.backend == "exllama":
                from gallama.backend.llm import ModelExllama as ModelClass
            elif model_spec.backend == "llama_cpp":
                from gallama.backend.llm import ModelLlamaCpp as ModelClass
            elif model_spec.backend == "llama_cpp_server":
                from gallama.backend.llm import ModelLlamaCppServer as ModelClass
            elif model_spec.backend == "ik_llama":
                from gallama.backend.llm import ModelIKLlama as ModelClass
            elif model_spec.backend == "transformers":
                from gallama.backend.llm import ModelTransformers as ModelClass
            elif model_spec.backend == "mlx_vlm":
                from gallama.backend.llm import ModelMLXVLM as ModelClass
            elif model_spec.backend == "sglang":
                from gallama.backend.llm import ModelSGLang as ModelClass
            elif model_spec.backend == "exllamav3":
                from gallama.backend.llm import ModelExllamaV3 as ModelClass
            elif model_spec.backend == "vllm":
                from gallama.backend.llm import ModelVLLM as ModelClass
            else:
                raise Exception(f"Unknown backend: {model_spec.backend}")

            if model_spec.draft_model_name:
                draft_model_config = self.config_manager.get_model_config(model_spec.draft_model_name)
                if not draft_model_config:
                    raise Exception(
                        f"Model config for '{model_spec.draft_model_name}' not exist in ~/gallama/model_config.yaml")
            else:
                draft_model_config = {}

            llm = ModelClass(model_spec=model_spec)

            # update dict
            self._update_model(
                model_name=model_name,
                model_spec=model_spec,
                model_object=llm
            )
        elif model_spec.backend == "embedding":  # embedding model
            from gallama.backend.embedding.embedding import EmbeddingModel

            self._update_model(
                model_name=model_name,
                model_spec=model_spec,
                model_object=EmbeddingModel(model_spec=model_spec)
            )

        elif model_spec.backend == "reranker":  # reranker model
            from gallama.backend.reranker.reranker import RerankerModel

            self._update_model(
                model_name=model_name,
                model_spec=model_spec,
                model_object=RerankerModel(model_spec=model_spec)
            )


        elif model_spec.backend == "faster_whisper":  # embedding model
            from gallama.backend.stt import ASRProcessor, ASRFasterWhisper

            stt_base = ASRFasterWhisper(model_spec=model_spec)

            stt = ASRProcessor(asr=stt_base)

            # update dict
            self._update_model(
                model_name=model_name,
                model_spec=model_spec,
                model_object=stt
            )

        elif model_spec.backend == "mlx_whisper":  # embedding model
            from gallama.backend.stt import ASRProcessor, ASRMLXWhisper

            stt_base = ASRMLXWhisper(model_spec=model_spec)

            stt = ASRProcessor(asr=stt_base)

            # update dict
            self._update_model(
                model_name=model_name,
                model_spec=model_spec,
                model_object=stt
            )

        elif model_spec.backend == "nemo_asr":
            from gallama.backend.stt import ASRProcessor, ASRNeMo

            stt_base = ASRNeMo(model_spec=model_spec)
            stt = ASRProcessor(asr=stt_base)

            self._update_model(
                model_name=model_name,
                model_spec=model_spec,
                model_object=stt
            )

        elif model_spec.backend == "kokoro":  # embedding model
            from gallama.backend.tts import TTSKokoro
            tts = TTSKokoro(model_spec=model_spec)

            # update dict
            self._update_model(
                model_name=model_name,
                model_spec=model_spec,
                model_object=tts
            )

        else:
            raise Exception(f"Unknown backend: {model_spec.backend}")



        logger.info("Loaded: " + model_name, extra=basic_log_extra())
