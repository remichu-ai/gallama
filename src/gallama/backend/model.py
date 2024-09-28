# import transformers
import torch
from typing import List, Dict
from gallama.logger.logger import logger
from gallama.data_classes.data_class import ModelParser

try:
    from exllamav2 import (
        ExLlamaV2,
        ExLlamaV2Tokenizer,
        ExLlamaV2Cache,
        ExLlamaV2Cache_Q4,
        ExLlamaV2Cache_Q6,
        ExLlamaV2Cache_Q8,
        ExLlamaV2Config
    )
except:
    ExLlamaV2 = None
    ExLlamaV2Tokenizer = None
    ExLlamaV2Cache = None
    ExLlamaV2Cache_Q4 = None
    ExLlamaV2Cache_Q6 = None
    ExLlamaV2Cache_Q8 = None
    ExLlamaV2Config = None

try:
    from llama_cpp import Llama
except:
    # optional dependency
    Llama = None

# experimental feature: tensor parallel
try:
    from exllamav2 import ExLlamaV2Cache_TP
except:
    # optional dependency
    ExLlamaV2Cache_TP = None

assert ExLlamaV2 or Llama, "Please install ExllamaV2 or LLama CPP Python as backend"


class Model:
    """A model class that contain the llm and tokenizer"""
    def __init__(self,
        model_spec:ModelParser,
        model_config: Dict,
        draft_model_config: Dict = {},
        eos_token_list_from_prompt_template: List[str] = []
    ):
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


        # draft model is via cli only
        self.draft_model_id = draft_model_config.get("model_id")
        self.draft_model_name = model_spec.draft_model_name or None
        self.draft_gpus = model_spec.gpus or draft_model_config.get("gpus") or "auto"
        self.draft_cache_size = self.cache_size   # set to the same as main model
        self.draft_cache_quant = model_spec.draft_cache_quant or draft_model_config.get("cache_quant") or "Q4"
        assert (self.draft_model_id is None) == (self.draft_model_name is None)

        # load model and tokenizer; cache is for exllamav2
        self.model, self.tokenizer, self.cache, self.draft_model, self.draft_cache = self.load_model()

        # TODO, to auto detect
        # get the eos_token_str by merging the default config with anything set by user
        self.eos_token_str = list(set(model_config.get("eos_token_list", []) + eos_token_list_from_prompt_template))
        self.eos_token_str_set = set(self.eos_token_str)    # set for some more efficient operation
        self.eos_token_ids = self.generate_eos_tokens_id()


    def load_model(self):
        model, tokenizer, cache, draft_model, draft_cache = None, None, None, None, None

        if self.backend=="exllama":
            model, tokenizer, cache = self.load_model_exllama(
                model_id=self.model_id,
                backend=self.backend,
                max_seq_len=self.max_seq_len,
                cache_size=self.cache_size,
                cache_quant=self.cache_quant,
                gpus=self.gpus,
                reserve_vram=self._reserve_vram,
                tensor_parallel=self.tensor_parallel,
            )

            # load model and tokenizer; cache is for exllamav2
            if self.draft_model_id is not None:
                draft_model, _, draft_cache = self.load_model_exllama(
                    model_id=self.draft_model_id,
                    backend=self.backend,
                    max_seq_len=self.max_seq_len,   # draft model max_seq_len must be same as main model
                    cache_size=self.draft_cache_size,
                    cache_quant=self.draft_cache_quant,
                    gpus=self.draft_gpus,
                    reserve_vram=self._reserve_vram,
                )

        elif self.backend=="llama_cpp":
            model, tokenizer, cache = self.load_model_llama_cpp(
                model_id=self.model_id,
                backend=self.backend,
                max_seq_len=self.max_seq_len,
                cache_size=self.cache_size,
                cache_quant=self.cache_quant,
                gpus=self.gpus,
                reserve_vram=self._reserve_vram,
                tensor_parallel=self.tensor_parallel,
                draf_model_id=self.draft_model_id
            )

        return model, tokenizer, cache, draft_model, draft_cache


    def load_model_exllama(self, model_id, backend, cache_size, cache_quant, gpus, reserve_vram, max_seq_len=None, tensor_parallel=False):
        """This function return the model and its tokenizer"""
        logger.info("Loading model: " + model_id)

        cache = None    # in case not a backend with separate cache like llama cpp
        tokenizer = None

        config = ExLlamaV2Config(model_id)
        if max_seq_len is not None:
            config.max_seq_len = max_seq_len
        else:
            # set the self.max_seq_len using model config file as it is None at the moment
            max_seq_len = config.max_seq_len
            self.max_seq_len = config.max_seq_len

        model = ExLlamaV2(config)
        tokenizer = ExLlamaV2Tokenizer(config)

        # a simple dict to help map cache quant
        cache_quant_dict = {
            "FP16": ExLlamaV2Cache,
            "Q4": ExLlamaV2Cache_Q4,
            "Q6": ExLlamaV2Cache_Q6,
            "Q8": ExLlamaV2Cache_Q8,
        }

        # cache size need to minimally max_seq_len size
        cache_size_to_use = cache_size if cache_size else config.max_seq_len
        cache_size_to_use = (cache_size_to_use//256) * 256      # round to multiplier of 256 for paged attention
        # ensure cache_size is minimally max_seq_len
        cache_size_to_use = max(cache_size_to_use, max_seq_len)

        # get the cache quantization to use
        cache_quant_to_use = cache_quant_dict[cache_quant]

        logger.info("max_seq_len: " + str(self.max_seq_len))
        logger.info("cache_size: " + str(cache_size_to_use))
        logger.info("Cache Quantization: " + str(cache_quant))

        assert cache_quant_to_use is not None
        assert (isinstance(gpus, str) and gpus == "auto") or (isinstance(gpus, list)), \
            "Device map should be either 'auto', 'gpu' split"

        if not tensor_parallel:
            if isinstance(gpus, str) and gpus == "auto":
                cache = cache_quant_to_use(model, max_seq_len=cache_size_to_use, lazy=True)
                model.load_autosplit(cache, reserve_vram=reserve_vram, progress=True)
            elif isinstance(gpus, list):      # user specify the gpus split
                logger.info("Custom GPU Allocation in GB: " + str(gpus))
                model.load(gpu_split=gpus, progress=True)
                cache = cache_quant_to_use(model, max_seq_len=cache_size_to_use, lazy=not model.loaded)
        else:
            # tensor parallel mode
            logger.info("ExllamaV2 Tensor Parallel enabled")
            if ExLlamaV2Cache_TP:       # ensure that tensor parallel is available
                model.load_tp(progress=True, gpu_split = gpus if isinstance(gpus, list) else None)
                cache = ExLlamaV2Cache_TP(
                    model,
                    max_seq_len = cache_size_to_use,
                    base = cache_quant_to_use,
                )
            else:
                raise ValueError("ExllamaV2 was not installed with tensor parallel")


        return model, tokenizer, cache


    def load_model_llama_cpp(self, model_id, backend, cache_size, cache_quant, gpus, reserve_vram, max_seq_len=None,
                           tensor_parallel=False, draf_model_id=None):
        """This function return the model and its tokenizer"""
        logger.info("Loading model: " + model_id)

        cache = None  # in case not a backend with separate cache like llama cpp
        tokenizer = None

        # TODO to add equivalent support for all exllama option
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
        tokenizer = model

        return model, tokenizer, cache


    @property
    def _reserve_vram(self):
        try:
            reserve_block_size = 1024 ** 2
            num_devices = torch.cuda.device_count()
            #reserved_vram = [192 * 1024**2] + [64 * 1024**2] * (num_devices - 1)
            #reserved_vram = [256 * 1024 ** 2] + [96 * 1024 ** 2] * (num_devices - 1)

            # GPU1 is the main GPU for my PC
            # The below is lower threshold than exllamav2 default setting
            reserve_per_gpu = [32 for _ in range(num_devices)]
            main_gpu = 0    # TODO pass it to front end
            reserve_per_gpu[main_gpu] = 64
            reserved_vram = [_reserve * reserve_block_size for _reserve in reserve_per_gpu]
            return reserved_vram
        except:
            # for non cuda env e.g. macbook
            return None


    def generate_eos_tokens_id(self):
        if self.eos_token_str:
            # exllama
            if ExLlamaV2Tokenizer and isinstance(self.tokenizer, ExLlamaV2Tokenizer):
                eos_token_ids = [self.tokenizer.single_id(token) for token in self.eos_token_str]
                return eos_token_ids
            elif self.backend in ["llama_cpp"]:
                return []   # no tokenizer for llama cpp
            else:
                # transformer:
                eos_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in self.eos_token_str]
                return eos_token_ids
        else:
            return []
