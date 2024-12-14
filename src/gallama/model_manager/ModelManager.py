from ..data_classes import ModelSpec
from typing import Dict, Any
from ..logger import logger
from ..config.config_manager import ConfigManager



class ModelManager:
    def __init__(self):
        self.llm_dict: Dict[str, Any] = {}               # dict to all llm models process object
        self.tts_dict: Dict[str, Any] = {}
        self.stt_dict: Dict[str, Any] = {}
        self.embedding_dict: Dict[str, Any] = {}
        self.config_manager = ConfigManager()
        self.model_ready = False

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

        # Merge configurations that user pass in with default setting of the model
        model_spec = ModelSpec.from_merged_config(model_spec, model_config)

        if not model_config:
            raise Exception(f"Model config for '{model_name}' not exist in ~/gallama/model_config.yaml")

        # load the model with config from the model_spec and yml. model_spec comes from cli
        if model_spec.backend in ["exllama", "llama_cpp", "transformers"]:  # llm loading
            if model_spec.backend == "exllama":
                from gallama.backend.llm import ModelExllama as ModelClass
            elif model_spec.backend == "llama_cpp":
                from gallama.backend.llm import ModelLlamaCpp as ModelClass
            elif model_spec.backend == "transformers":
                from gallama.backend.llm import ModelTransformers as ModelClass
            else:
                raise Exception(f"Unknown backend: {model_spec.backend}")

            logger.info("TODO")
            if model_spec.draft_model_id:
                draft_model_config = self.config_manager.get_model_config(model_spec.draft_model_name)
                if not draft_model_config:
                    raise Exception(
                        f"Model config for '{model_spec.draft_model_name}' not exist in ~/gallama/model_config.yaml")
            else:
                draft_model_config = {}

            llm = ModelClass(model_spec=model_spec)

            # update dict
            self.llm_dict[model_name] = llm
        elif model_spec.backend == "embedding":  # embedding model
            from gallama.backend.embedding.embedding import EmbeddingModel

            self.embedding_dict[model_name] = EmbeddingModel(model_spec=model_spec)


        elif model_spec.backend == "faster_whisper":  # embedding model
            from gallama.backend.stt import ASRProcessor, ASRFasterWhisper

            stt_base = ASRFasterWhisper(model_spec=model_spec)

            stt = ASRProcessor(asr=stt_base)

            # update dict
            self.stt_dict[model_name] = stt

        elif model_spec.backend == "gpt_sovits":  # embedding model
            from gallama.backend.tts import TTS_GPT_SoVITS
            tts = TTS_GPT_SoVITS(model_spec=model_spec)

            # update dict
            self.tts_dict[model_name] = tts

        else:
            raise Exception(f"Unknown backend: {model_spec.backend}")



        logger.info("Loaded: " + model_name)