from .interface import ModelInterface
from typing import Tuple, Optional, Any, Dict, List
from gallama.data_classes.data_class import ModelParser
from gallama.logger.logger import logger
import transformers
from importlib import import_module


class ModelTransformers(ModelInterface):
    def __init__(self,
        model_spec:ModelParser,
        model_config: Dict,
        draft_model_config: Dict = None,
        eos_token_list_from_prompt_template: List[str] = None
    ):
        super().__init__(model_spec, model_config, draft_model_config, eos_token_list_from_prompt_template)


    def load_model(self):
        """Load the model, tokenizer, cache, and optional processor."""
        # processor is for multimodal
        self.model, self.tokenizer, self.cache, self.processor = self.load_model_transformers(
            model_id=self.model_id,
            gpus=self.gpus,
        )

        # load draft model
        if self.draft_model_id is not None:
            raise "Draft model currently not supported for llama cpp backend"

        self.eos_token_ids = self.generate_eos_tokens_id()


    def load_model_transformers(
        self,
        model_id,
        gpus,
    ):
        """This function return the model and its tokenizer"""
        logger.info("Loading model: " + model_id)

        cache = None  # in case not a backend with separate cache like llama cpp
        tokenizer = None
        processor = None

        # helper function for dynamic class loading
        def get_class(class_string):
            module_name, class_name = class_string.rsplit('.', 1)
            module = import_module(module_name)
            return getattr(module, class_name)

        # arguments for model loading
        model_kargs = {
            'pretrained_model_name_or_path': self.model_id,
            'torch_dtype' : "auto",
            'device_map': "auto",
        }

        tokenizer_args = {
            'pretrained_model_name_or_path': self.model_id,
        }

        # check if flash attention enabled
        flash_installed, flash_version = self.is_flash_attention_installed()
        if flash_installed:
            model_kargs["attn_implementation"]  = "flash_attention_2"


        # determine the class to use for loading
        if self.backend_extra_args.get('model_class'):
            model_class = get_class(self.backend_extra_args['model_class'])

            model_extra_kwargs = self.backend_extra_args.get('model_class_extra_kwargs')
            if model_extra_kwargs:
                model_kargs.update(model_extra_kwargs)      # update any extra argument
        else:
            model_class = transformers.AutoModelForCausalLM

        if self.backend_extra_args.get('tokenizer_class'):
            tokenizer_class = get_class(self.backend_extra_args['tokenizer_class'])
        else:
            tokenizer_class = transformers.AutoTokenizer

        if self.backend_extra_args.get('processor_class'):
            processor_class = get_class(self.backend_extra_args['processor_class'])
        else:
            processor_class = None

        # currently speculative decoding not supported by model specific for LLama CPP python
        if isinstance(gpus, str) and gpus == "auto":
            logger.info(model_kargs)
            model = model_class.from_pretrained(**model_kargs)
            tokenizer = tokenizer_class.from_pretrained(**tokenizer_args)
            if processor_class:
                processor = processor_class.from_pretrained(**tokenizer_args)

        elif isinstance(gpus, list):
            raise "Specifying GPU for transformers is not supported"

        else:
            raise ValueError("Device map should be either 'auto', 'gpu' split")

        # set max_seq_len based on model    TODO: To find more reliable method
        try:
            self.max_seq_len = model.config.max_position_embeddings
        except:
            # for llama 3.2
            self.max_seq_len = model.config.text_config.max_position_embeddings


        return model, tokenizer, cache, processor



    def generate_eos_tokens_id(self) -> List[int]:
        """Generate the end-of-sequence token IDs."""

        # currently the stop word for llama cpp work by string and not by token id
        # hence usage of this is not required
        return []
