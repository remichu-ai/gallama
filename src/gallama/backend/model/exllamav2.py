from .interface import ModelInterface
from typing import Tuple, Optional, Any, Dict, List
from gallama.data_classes.data_class import ModelParser
from gallama.logger.logger import logger
import torch

try:
    from exllamav2 import (
        ExLlamaV2,
        ExLlamaV2Tokenizer,
        ExLlamaV2Cache,
        ExLlamaV2Cache_Q4,
        ExLlamaV2Cache_Q6,
        ExLlamaV2Cache_Q8,
        ExLlamaV2Config,
    )
except:
    ExLlamaV2 = None
    ExLlamaV2Tokenizer = None
    ExLlamaV2Cache = None
    ExLlamaV2Cache_Q4 = None
    ExLlamaV2Cache_Q6 = None
    ExLlamaV2Cache_Q8 = None
    ExLlamaV2Config = None


# tensor parallel from v0.1.9 onward
try:
    from exllamav2 import ExLlamaV2Cache_TP
except:
    # optional dependency
    ExLlamaV2Cache_TP = None

# vision support from v0.2.4 onward
try:
    from exllamav2 import ExLlamaV2VisionTower
except:
    ExLlamaV2VisionTower = None



class ModelExllama(ModelInterface):
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

        self.model, self.tokenizer, self.cache, self.processor = self.load_model_exllama(
            model_id=self.model_id,
            backend=self.backend,
            max_seq_len=self.max_seq_len,
            cache_size=self.cache_size,
            cache_quant=self.cache_quant,
            gpus=self.gpus,
            reserve_vram=self._reserve_vram,
            tensor_parallel=self.tensor_parallel,
        )

        # load draft model
        if self.draft_model_id is not None:
            # tokenizer and processor already set above
            draft_model, _, draft_cache, _ = self.load_model_exllama(
                model_id=self.draft_model_id,
                backend=self.backend,
                max_seq_len=self.max_seq_len,  # draft model max_seq_len must be same as main model
                cache_size=self.draft_cache_size,
                cache_quant=self.draft_cache_quant,
                gpus=self.draft_gpus,
                reserve_vram=self._reserve_vram,
            )

        self.eos_token_ids = self.generate_eos_tokens_id()


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

        # load vision processor
        processor = None
        if ExLlamaV2VisionTower:
            try:
                processor = ExLlamaV2VisionTower(config)
                processor.load(progress=True)
            except:
                processor = None

        return model, tokenizer, cache, processor



    def generate_eos_tokens_id(self) -> List[int]:
        """Generate the end-of-sequence token IDs."""
        if self.eos_token_str:
            # exllama
            if ExLlamaV2Tokenizer and isinstance(self.tokenizer, ExLlamaV2Tokenizer):
                eos_token_ids = [self.tokenizer.single_id(token) for token in self.eos_token_str]
                return eos_token_ids
        else:
            return []


    # helper function
    async def check_disconnection(
        self,
        request: Request,
        job,
        gen_queue_list: Union[GenQueue, QueueContext, List[QueueContext]]
    ):
        """
        Helper function that handle stopping generation mid stream
        """
        try:
            while True:
                if await request.is_disconnected():
                    logger.info("User disconnected")
                    await job.cancel()

                    # add GenEnd to signal the end of generation
                    chunk = GenEnd()
                    for g_queue in gen_queue_list:
                        try:
                            await g_queue.get_queue().put(chunk)
                        except Exception as e:
                            logger.error(f"Error putting GenEnd into queue: {str(e)}")

                    # break the while loop
                    break

                # Use asyncio.wait_for to implement a timeout
                try:
                    await asyncio.wait_for(asyncio.sleep(1), timeout=1.1)
                except asyncio.TimeoutError:
                    # This allows us to check for cancellation more frequently
                    pass

        except asyncio.CancelledError:
            logger.info("Disconnection check was cancelled")
        except Exception as e:
            logger.error(f"An error occurred in check_disconnection: {str(e)}")
        finally:
            logger.info("Exiting check_disconnection")
