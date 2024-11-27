from .interface import ModelInterface
from typing import Tuple, Optional, Any, Dict, List
from gallama.data_classes.data_class import ModelParser
from gallama.logger.logger import logger

try:
    from llama_cpp import Llama
except:
    # optional dependency
    Llama = None


class ModelLlamaCpp(ModelInterface):
    def __init__(self,
        model_spec:ModelParser,
        model_config: Dict,
        draft_model_config: Dict = None,
        eos_token_list_from_prompt_template: List[str] = None
    ):
        super().__init__(model_spec, model_config, draft_model_config, eos_token_list_from_prompt_template)


    def load_model(self):
        """Load the model, tokenizer, cache, and optional processor."""
        model, tokenizer, cache, draft_model, draft_cache, processor = None, None, None, None, None, None

        model, tokenizer, cache = self.load_model_llama_cpp(
            model_id=self.model_id,
            gpus=self.gpus,
        )

        # load draft model
        if self.draft_model_id is not None:
            raise "Draft model currently not supported for llama cpp backend"

        self.eos_token_ids = self.generate_eos_tokens_id()


    def load_model_llama_cpp(self, model_id, gpus):
        """This function return the model and its tokenizer"""
        logger.info("Loading model: " + model_id)

        cache = None  # in case not a backend with separate cache like llama cpp
        tokenizer = None

        # currently speculative decoding not supported by model specific for LLama CPP python
        if isinstance(gpus, str) and gpus == "auto":
            model = Llama(
                model_path=self.model_id,
                n_gpu_layers=-1,
                seed=1,
                n_ctx=self.max_seq_len if self.max_seq_len else 0,  # Uncomment to increase the context window
                flash_attn=True,
                offload_kqv=True,
                # draft_model=draf_model_id,
            )
        elif isinstance(gpus, list):  # user specify the gpus split
            model = Llama(
                model_path=self.model_id,
                n_gpu_layers=-1,
                seed=1,
                n_ctx=self.max_seq_len if self.max_seq_len else 0,  # Uncomment to increase the context window
                flash_attn=True,
                offload_kqv=True,
                tensor_split=gpus,
                # draf_model_id=draf_model_id,
            )
        else:
            raise ValueError("Device map should be either 'auto', 'gpu' split")

        # set max_seq_len based on model
        self.max_seq_len = model._model.n_ctx_train()
        tokenizer = model       # llama cpp doesnt have separate tokenizer object

        self.eos_token_ids = self.generate_eos_tokens_id()

        return model, tokenizer, cache



    def generate_eos_tokens_id(self) -> List[int]:
        """Generate the end-of-sequence token IDs."""

        # currently the stop word for llama cpp work by string and not by token id
        # hence usage of this is not required
        return []
