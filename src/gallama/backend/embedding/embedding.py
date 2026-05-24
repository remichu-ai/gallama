import asyncio
import base64
import io
from typing import Any, List, Optional, Union

from PIL import Image

from ...data_classes.data_class import (
    EmbeddingContentImage,
    EmbeddingContentPart,
    EmbeddingContentText,
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelSpec,
    MultimodalEmbeddingInput,
)
from ...logger import logger
from ...utils.utils import floats_to_base64

SentenceTransformer = None


SENTENCE_TRANSFORMER_INIT_KEYS = {
    "device",
    "prompts",
    "default_prompt_name",
    "similarity_fn_name",
    "cache_folder",
    "trust_remote_code",
    "revision",
    "local_files_only",
    "token",
    "use_auth_token",
    "truncate_dim",
    "model_kwargs",
    "tokenizer_kwargs",
    "config_kwargs",
    "backend",
}

SENTENCE_TRANSFORMER_ENCODE_KEYS = {
    "batch_size",
    "prompt_name",
    "prompt",
    "precision",
    "normalize_embeddings",
    "truncate_dim",
    "chunk_size",
}


def get_sentence_transformer_class():
    global SentenceTransformer
    if SentenceTransformer is not None:
        return SentenceTransformer

    try:
        from sentence_transformers import SentenceTransformer as SentenceTransformerClass
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. Install Gallama with `gallama[embedding]` "
            "or install sentence-transformers to use the embedding backend."
        ) from exc

    SentenceTransformer = SentenceTransformerClass
    return SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_spec: ModelSpec):
        self.model_id = model_spec.model_id
        self.model_name = model_spec.model_name or model_spec.model_id
        self.backend_extra_args = model_spec.backend_extra_args or {}
        self.model = self.load_embedding_model()
        self._encode_lock = asyncio.Lock()
        self._is_multimodal: Optional[bool] = None  # cached

    def load_embedding_model(self):
        init_kwargs = dict(self.backend_extra_args.get("sentence_transformer_kwargs", {}))
        for key in SENTENCE_TRANSFORMER_INIT_KEYS:
            if key in self.backend_extra_args and key not in init_kwargs:
                init_kwargs[key] = self.backend_extra_args[key]

        init_kwargs.setdefault("trust_remote_code", True)
        logger.info(f"Loading embedding model with sentence-transformers: {self.model_id}")
        return get_sentence_transformer_class()(self.model_id, **init_kwargs)

    @property
    def is_multimodal(self) -> bool:
        """Check if the loaded model supports image modality."""
        if self._is_multimodal is None:
            try:
                self._is_multimodal = self.model.supports(["image"])
            except Exception:
                self._is_multimodal = False
        return self._is_multimodal

    def _get_encode_kwargs(self, query: EmbeddingRequest) -> dict[str, Any]:
        encode_kwargs = dict(self.backend_extra_args.get("encode_kwargs", {}))
        for key in SENTENCE_TRANSFORMER_ENCODE_KEYS:
            if key in self.backend_extra_args and key not in encode_kwargs:
                encode_kwargs[key] = self.backend_extra_args[key]

        encode_kwargs.setdefault("batch_size", 32)
        encode_kwargs.setdefault("convert_to_numpy", True)
        encode_kwargs.setdefault("show_progress_bar", False)

        if query.dimension is not None:
            encode_kwargs["truncate_dim"] = query.dimension

        # Pass task hint if provided (supports asymmetric encode)
        if query.task is not None:
            encode_kwargs["task"] = query.task

        return encode_kwargs

    # ── Multimodal input conversion helpers ───────────────────────────

    @staticmethod
    def _data_uri_to_pil(data_uri: str) -> Image.Image:
        """Decode a base64 data URI to a PIL Image."""
        if "," not in data_uri:
            raise ValueError(f"Invalid data URI (missing comma): {data_uri[:80]}...")
        header, b64 = data_uri.split(",", 1)
        if "base64" not in header:
            raise ValueError(f"Data URI is not base64-encoded: {header}")
        img_bytes = base64.b64decode(b64)
        return Image.open(io.BytesIO(img_bytes))

    @staticmethod
    def _is_data_uri(value: str) -> bool:
        """Check if a string looks like a base64 data URI."""
        return isinstance(value, str) and value.startswith("data:")

    @classmethod
    def _resolve_image_value(cls, image: str) -> Union[str, Image.Image]:
        """Convert an image string (URL, path, or data URI) to st-compatible value.

        - URLs / file paths: returned as-is (sentence-transformers handles them)
        - data URIs: decoded to PIL Image
        """
        if cls._is_data_uri(image):
            return cls._data_uri_to_pil(image)
        return image

    @classmethod
    def _convert_multimodal_input(cls, item) -> Union[str, dict, Image.Image]:
        """Convert a single API multimodal input item to sentence-transformers format.

        Returns:
        - ``str`` for plain text
        - ``dict`` like ``{"text": "...", "image": PIL.Image or str}`` for multimodal
        - ``PIL.Image`` for image-only inputs
        """
        # 1. Plain string → pass through
        if isinstance(item, str):
            return item

        # 2. MultimodalEmbeddingInput dict: {"text": "...", "image": "..."}
        if isinstance(item, dict) and "type" not in item:
            text = item.get("text")
            image = item.get("image")
            result: dict = {}
            if text:
                result["text"] = text
            if image:
                result["image"] = cls._resolve_image_value(image)
            if not result:
                raise ValueError("Multimodal input must have at least 'text' or 'image'")
            # If only one key, return the value directly
            if len(result) == 1 and "text" in result:
                return result["text"]
            if len(result) == 1 and "image" in result:
                return result["image"]
            return result

        # 3. OpenAI content-parts list: [{"type": "text", ...}, {"type": "image_url", ...}]
        if isinstance(item, list):
            text_parts: List[str] = []
            image_parts: list = []
            for part in item:
                if isinstance(part, dict):
                    part_type = part.get("type", "")
                elif hasattr(part, "type"):
                    part_type = part.type
                else:
                    raise ValueError(f"Unknown content part: {part}")

                if part_type == "text":
                    text = part.get("text", "") if isinstance(part, dict) else part.text
                    text_parts.append(text)
                elif part_type == "image_url":
                    url = part.get("image_url", {}).get("url", "") if isinstance(part, dict) else part.image_url.url
                    image_parts.append(cls._resolve_image_value(url))
                else:
                    raise ValueError(f"Unknown content part type: {part_type}")

            if image_parts and text_parts:
                return {"text": " ".join(text_parts), "image": image_parts[0]}
            if image_parts:
                return image_parts[0]
            if text_parts:
                return " ".join(text_parts)
            raise ValueError("Empty content parts list")

        raise ValueError(f"Unsupported embedding input type: {type(item)}")

    def _prepare_inputs(self, query: EmbeddingRequest) -> List[Union[str, dict, Image.Image]]:
        """Convert EmbeddingRequest.input to sentence-transformers-compatible inputs.

        For text-only models: strings (backward-compatible).
        For multimodal models: mixed list of strings, PIL Images, and dicts.
        """
        raw_input = query.input

        # Single string
        if isinstance(raw_input, str):
            return [raw_input]

        # Token-ID lists → decode to strings
        if isinstance(raw_input, list) and raw_input and isinstance(raw_input[0], list):
            first = raw_input[0]
            if all(isinstance(t, int) for t in first):
                tokenizer = getattr(self.model, "tokenizer", None)
                if tokenizer is None:
                    raise ValueError("Token-array embedding input requires a tokenizer on the embedding model.")
                return [
                    tokenizer.decode(tokens, skip_special_tokens=True)
                    for tokens in raw_input
                ]

        # List of mixed items
        if isinstance(raw_input, list):
            # All strings: fast path
            if all(isinstance(item, str) for item in raw_input):
                return raw_input

            # Multimodal: convert each item
            if self.is_multimodal:
                return [self._convert_multimodal_input(item) for item in raw_input]

            # Not multimodal — reject non-string inputs
            for item in raw_input:
                if not isinstance(item, str):
                    raise ValueError(
                        f"Embedding model '{self.model_id}' does not support multimodal inputs. "
                        f"Got input type: {type(item).__name__}"
                    )

        raise ValueError(
            "Embedding input must be a string, a list of strings, a list of token-id lists, "
            "or (for multimodal models) a list of multimodal inputs."
        )

    def _count_tokens(self, inputs: list) -> int:
        """Count tokens, handling mixed text/image inputs."""
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None:
            try:
                texts_only = [t for t in inputs if isinstance(t, str)]
                if texts_only:
                    encoded = tokenizer(texts_only, add_special_tokens=False)
                    input_ids = encoded["input_ids"]
                    return sum(len(token_ids) for token_ids in input_ids)
            except Exception:
                logger.debug("Failed to count embedding tokens with model tokenizer", exc_info=True)

        return sum(len(text.split()) for text in inputs if isinstance(text, str))

    def _encode(self, inputs: list, encode_kwargs: dict[str, Any]) -> List[List[float]]:
        """Encode inputs via sentence-transformers. Handles mixed text/image natively."""
        embeddings = self.model.encode(inputs, **encode_kwargs)
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        if embeddings and all(isinstance(value, (float, int)) for value in embeddings):
            embeddings = [embeddings]

        return [
            [float(value) for value in embedding]
            for embedding in embeddings
        ]

    async def text_embeddings(
        self,
        query: EmbeddingRequest,
    ) -> EmbeddingResponse:
        inputs = self._prepare_inputs(query)
        encode_kwargs = self._get_encode_kwargs(query)

        async with self._encode_lock:
            embeddings = await asyncio.to_thread(self._encode, inputs, encode_kwargs)

        usage = self._count_tokens(inputs)
        use_base64 = query.encoding_format == "base64"
        emb_response_list = [
            EmbeddingObject(
                index=idx,
                embedding=floats_to_base64(emb) if use_base64 else emb,
            )
            for idx, emb in enumerate(embeddings)
        ]

        return EmbeddingResponse(
            model=self.model_name,
            usage=EmbeddingResponse.Usage(
                prompt_tokens=usage,
                total_tokens=usage,
            ),
            data=emb_response_list,
        )
