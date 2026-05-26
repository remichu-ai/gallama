import asyncio

import torch

from ...data_classes.data_class import ModelSpec
from ...logger import logger


CrossEncoder = None


def get_cross_encoder_class():
    global CrossEncoder
    if CrossEncoder is not None:
        return CrossEncoder

    try:
        from sentence_transformers import CrossEncoder as CrossEncoderClass
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. Install Gallama with `gallama[embedding]` "
            "or install sentence-transformers to use the reranker backend."
        ) from exc

    CrossEncoder = CrossEncoderClass
    return CrossEncoder


class RerankerModel:
    def __init__(self, model_spec: ModelSpec):
        self.model_id = model_spec.model_id
        self.model_name = model_spec.model_name or model_spec.model_id
        self.backend_extra_args = model_spec.backend_extra_args or {}
        self.model = self.load_reranker_model()
        self._encode_lock = asyncio.Lock()

    def load_reranker_model(self):
        init_kwargs = dict(self.backend_extra_args.get("cross_encoder_kwargs", {}))
        init_kwargs.setdefault("trust_remote_code", True)
        logger.info(f"Loading reranker model with sentence-transformers CrossEncoder: {self.model_id}")
        return get_cross_encoder_class()(self.model_id, **init_kwargs)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[tuple[int, float]]:
        """Rerank documents against a query.

        Qwen3-VL-Reranker-2B uses CrossEncoder.predict() which passes
        (query, document) pairs through the full model with cross-attention,
        producing a relevance score per pair.

        Returns:
            list of (original_index, relevance_score) sorted by score descending.
            Scores are sigmoid-normalized to the 0..1 range.
        """
        if not documents:
            return []

        async with self._encode_lock:
            pairs = [(query, doc) for doc in documents]
            scores = await asyncio.to_thread(
                self.model.predict,
                pairs,
                activation_fn=torch.nn.Sigmoid(),
                prompt="Retrieve text relevant to the user's query.",
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        # scores is a numpy array of sigmoid-normalized floats
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        if top_n is not None:
            indexed = indexed[:top_n]
        return [(idx, float(score)) for idx, score in indexed]
