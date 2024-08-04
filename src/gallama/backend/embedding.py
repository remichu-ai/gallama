from gallama.data_classes.data_class import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingObject,
)
from gallama.utils.utils import floats_to_base64
from typing import Dict
import logging
from infinity_emb import EngineArgs, AsyncEmbeddingEngine
from gallama.data_classes.data_class import ModelParser
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)      # disable logging from infinity


class EmbeddingModel:
    def __init__(self, model_id: str, model_name: str, model_spec: ModelParser, model_config: Dict):
        self.model_id = model_id
        self.model_name = model_name
        self.gpus = model_spec.gpus or model_config.get("gpus") or "auto"
        self.model = self.load_embedding_model()

    def get_visible_gpu_indices(self) -> str:
        """
        Generate a string of GPU indices based on allocated GPUs.
        If no GPUs are specified, return all available GPU indices.

        Returns:
            str: A comma-separated string of GPU indices with allocated VRAM,
                 or all available GPU indices if none are specified.
        """
        if self.gpus is None or self.gpus == "auto":
            import torch
            return ','.join(str(i) for i in range(torch.cuda.device_count()))

        if all(vram == 0 for vram in self.gpus):
            return ""  # No GPUs allocated

        visible_devices = [str(i) for i, vram in enumerate(self.gpus) if vram > 0]
        return ','.join(visible_devices)

    def load_embedding_model(self):
        # load model
        emb_model = AsyncEmbeddingEngine.from_args(
            EngineArgs(
                model_name_or_path=self.model_id,
                engine="torch",
                embedding_dtype="float32",
                dtype="auto"
            )
        )

        return emb_model           # TODO support multiple model

    async def text_embeddings(
        self,
        query: EmbeddingRequest,
    ) -> EmbeddingResponse:

        # initialize the input
        input_texts = []
        if isinstance(query.input, str):
            input_texts = [query.input]
        elif isinstance(query.input, list):
            input_texts = query.input
        else:
            raise Exception("Data not supported")

        async with self.model:
            embeddings, usage = await self.model.embed(sentences=input_texts)

        # create the return embedding data
        emb_response_list = []
        for idx, text_emb in enumerate(embeddings):
            emb_response_list.append(
                EmbeddingObject(
                    index=idx,
                    embedding=text_emb if query.encoding_format == "float" else floats_to_base64(text_emb),
                )
            )

        # return the response
        return EmbeddingResponse(
            model=self.model_name,
            usage=EmbeddingResponse.Usage(
                prompt_tokens=usage,
                total_tokens=usage,
            ),
            data=emb_response_list,
        )
