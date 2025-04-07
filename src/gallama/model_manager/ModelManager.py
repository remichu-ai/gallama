from ..data_classes import ModelSpec
from typing import Dict, Any, Literal, Optional
from ..logger import logger
from ..config.config_manager import ConfigManager



class ModelManager:
    def __init__(self):
        self.llm_dict: Dict[str, Any] = {}               # dict to all llm models process object
        self.llm_dict_non_strict: Dict[str, Any] = {}
        self.tts_dict: Dict[str, Any] = {}
        self.tts_dict_non_strict: Dict[str, Any] = {}
        self.stt_dict: Dict[str, Any] = {}
        self.stt_dict_non_strict: Dict[str, Any] = {}
        self.embedding_dict: Dict[str, Any] = {}
        self.embedding_dict_non_strict: Dict[str, Any] = {}
        self.config_manager = ConfigManager()
        self.model_ready = False

    def get_model(self, model_name: str, _type: Literal["llm", "tts", "stt", "embedding"]) -> Optional[Any]:
        # Determine which dictionaries to use based on the type
        if _type == "llm":
            strict_dict = self.llm_dict
            non_strict_dict = self.llm_dict_non_strict
        elif _type == "tts":
            strict_dict = self.tts_dict
            non_strict_dict = self.tts_dict_non_strict
        elif _type == "stt":
            strict_dict = self.stt_dict
            non_strict_dict = self.stt_dict_non_strict
        elif _type == "embedding":
            strict_dict = self.embedding_dict
            non_strict_dict = self.embedding_dict_non_strict
        else:
            raise ValueError(f"Invalid model type: {_type}")

        # Check if the model exists in the strict dictionary
        if model_name in strict_dict:
            return strict_dict[model_name]

        # If not, check if there are any models in the non-strict dictionary
        if non_strict_dict:
            # Return the first model in the non-strict dictionary
            return next(iter(non_strict_dict.values()))

        # If no model is found, return None
        return None

    def _update_model(self, model_name: str, model_spec: ModelSpec, model_object: Any):
        if model_spec.model_type == "llm":
            self.llm_dict[model_name] = model_object
            if not model_spec.strict:
                self.llm_dict_non_strict[model_name] = model_object
        elif model_spec.model_type == "tts":
            self.tts_dict[model_name] = model_object
            if not model_spec.strict:
                self.tts_dict_non_strict[model_name] = model_object
        elif model_spec.model_type == "stt":
            self.stt_dict[model_name] = model_object
            if not model_spec.strict:
                self.stt_dict_non_strict[model_name] = model_object
        elif model_spec.model_type == "embedding":
            self.embedding_dict[model_name] = model_object
            if not model_spec.strict:
                self.embedding_dict_non_strict[model_name] = model_object


    def load_model(self, model_spec: ModelSpec):
        """
        model_spec is model specification coming from cli
        it might not have all the properties required for the model to be loaded
        the config_manager below contain all the models properties
        """
        # global config_manager, llm_dict, stt_dict, tts_dict

        # get the config from the yml
        model_name = model_spec.model_name
        model_config = self.config_manager.get_model_config(model_name)
        if not model_config:
            raise Exception(f"Model config for '{model_name}' not exist in ~/gallama/model_config.yaml")

        model_config.update({"model_name": model_name})

        # handle draft model
        if model_spec.draft_model_name and not model_spec.draft_model_id:
            draft_model_config = self.config_manager.get_model_config(model_spec.draft_model_name)
            model_config.update({
                "draft_model_id": model_spec.draft_model_id or draft_model_config["model_id"],
                "draft_model_name": model_spec.draft_model_name or draft_model_config["model_name"],
                "draft_gpus": model_spec.draft_gpus or draft_model_config["gpus"],
                "draft_cache_quant": model_spec.draft_cache_quant or draft_model_config["cache_quant"],
            })

        _default_model_spec = ModelSpec.from_dict(model_config)

        # Merge configurations that user pass in with default setting of the model
        model_spec = ModelSpec.from_merged_config(model_spec, _default_model_spec.model_dump())


        # load the model with config from the model_spec and yml. model_spec comes from cli
        if model_spec.backend in ["exllama", "llama_cpp", "transformers", "mlx_vlm", "sglang", "exllamav3"]:  # llm loading
            if model_spec.backend == "exllama":
                from gallama.backend.llm import ModelExllama as ModelClass
            elif model_spec.backend == "llama_cpp":
                from gallama.backend.llm import ModelLlamaCpp as ModelClass
            elif model_spec.backend == "transformers":
                from gallama.backend.llm import ModelTransformers as ModelClass
            elif model_spec.backend == "mlx_vlm":
                from gallama.backend.llm import ModelMLXVLM as ModelClass
            elif model_spec.backend == "sglang":
                from gallama.backend.llm import ModelSGLang as ModelClass
            elif model_spec.backend == "exllamav3":
                from gallama.backend.llm import ModelExllamaV3 as ModelClass
            else:
                raise Exception(f"Unknown backend: {model_spec.backend}")

            if model_spec.draft_model_id:
                draft_model_config = self.config_manager.get_model_config(model_spec.draft_model_name)
                if not draft_model_config:
                    raise Exception(
                        f"Model config for '{model_spec.draft_model_name}' not exist in ~/gallama/model_config.yaml")
            else:
                draft_model_config = {}

            logger.info(f"model_spec: {model_spec}")
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

        elif model_spec.backend == "gpt_sovits":  # embedding model
            from gallama.backend.tts import TTS_GPT_SoVITS
            tts = TTS_GPT_SoVITS(model_spec=model_spec)

            # update dict
            self._update_model(
                model_name=model_name,
                model_spec=model_spec,
                model_object=tts
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



        logger.info("Loaded: " + model_name)