from .data_class import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingObject,
)
from .utils import floats_to_base64
# import asyncio
import logging
from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)      # disable logging from infinity
from .logger import logger




class EmbeddingModel:
    def __init__(self, model_id='', model_name=None, max_seq_len=None):
        self.model_id = model_id
        self.model_name = model_name
        self.model = self.load_embedding_model(model_id)

    @staticmethod
    def load_embedding_model(model_id):
        emb_model = AsyncEmbeddingEngine.from_args(
            EngineArgs(
                model_name_or_path=model_id,
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
