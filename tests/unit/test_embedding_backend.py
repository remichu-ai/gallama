"""Unit tests for gallama embedding backend (Phase 26.2 / 26.6)."""

import asyncio
import base64
import struct

from gallama.backend.embedding import embedding as embedding_module
from gallama.data_classes.data_class import EmbeddingRequest, ModelSpec


class FakeTokenizer:
    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[idx for idx, _ in enumerate(text.split())] for text in texts]}

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(token) for token in tokens)


class FakeSentenceTransformer:
    init_calls = []

    def __init__(self, model_name_or_path, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.kwargs = kwargs
        self.tokenizer = FakeTokenizer()
        self.encode_calls = []
        FakeSentenceTransformer.init_calls.append((model_name_or_path, kwargs))

    def encode(self, sentences, **kwargs):
        self.encode_calls.append((sentences, kwargs))
        truncate_dim = kwargs.get("truncate_dim")
        embeddings = []
        for index, _sentence in enumerate(sentences):
            vector = [float(index + 1), float(index + 2), float(index + 3)]
            embeddings.append(vector[:truncate_dim] if truncate_dim else vector)
        return embeddings


def make_model(monkeypatch, backend_extra_args=None):
    FakeSentenceTransformer.init_calls = []
    monkeypatch.setattr(embedding_module, "SentenceTransformer", FakeSentenceTransformer)
    return embedding_module.EmbeddingModel(
        ModelSpec(
            model_name="fake-embed",
            model_id="fake/model",
            backend="embedding",
            backend_extra_args=backend_extra_args or {},
        )
    )


def test_embedding_backend_uses_sentence_transformer_kwargs(monkeypatch):
    make_model(
        monkeypatch,
        {
            "device": "cpu",
            "trust_remote_code": False,
            "sentence_transformer_kwargs": {"revision": "main"},
        },
    )

    assert FakeSentenceTransformer.init_calls == [
        (
            "fake/model",
            {
                "revision": "main",
                "device": "cpu",
                "trust_remote_code": False,
            },
        )
    ]


def test_text_embeddings_returns_openai_compatible_float_response(monkeypatch):
    model = make_model(monkeypatch, {"batch_size": 4, "normalize_embeddings": True})
    query = EmbeddingRequest(
        input=["hello world", "again"],
        model="fake-embed",
        encoding_format="float",
        dimension=2,
    )

    response = asyncio.run(model.text_embeddings(query))

    assert response.model == "fake-embed"
    assert response.usage.prompt_tokens == 3
    assert response.usage.total_tokens == 3
    assert [item.index for item in response.data] == [0, 1]
    assert [item.embedding for item in response.data] == [[1.0, 2.0], [2.0, 3.0]]
    assert model.model.encode_calls == [
        (
            ["hello world", "again"],
            {
                "batch_size": 4,
                "normalize_embeddings": True,
                "convert_to_numpy": True,
                "show_progress_bar": False,
                "truncate_dim": 2,
            },
        )
    ]


def test_text_embeddings_supports_base64_and_token_arrays(monkeypatch):
    model = make_model(monkeypatch)
    query = EmbeddingRequest(
        input=[[10, 20]],
        model="fake-embed",
        encoding_format="base64",
    )

    response = asyncio.run(model.text_embeddings(query))

    expected = base64.b64encode(struct.pack("fff", 1.0, 2.0, 3.0)).decode("utf-8")
    assert response.data[0].embedding == expected
    assert model.model.encode_calls[0][0] == ["10 20"]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 26.6 — Batch embedding ordering (200+ texts)
# ═══════════════════════════════════════════════════════════════════════════════

def test_batch_embedding_200_texts_preserves_order(monkeypatch):
    """Embedding 200+ texts preserves input order: text[i] → vector[i]."""
    model = make_model(monkeypatch)
    input_count = 200
    texts = [f"document number {i}" for i in range(input_count)]

    query = EmbeddingRequest(
        input=texts,
        model="fake-embed",
        encoding_format="float",
    )

    response = asyncio.run(model.text_embeddings(query))

    assert len(response.data) == input_count, (
        f"Expected {input_count} embeddings, got {len(response.data)}"
    )

    # Verify indices match 0..199 and each vector corresponds to its index
    indices = [item.index for item in response.data]
    assert indices == list(range(input_count)), (
        f"Indices not sequential 0..{input_count - 1}"
    )

    # Each vector should match the FakeSentenceTransformer pattern:
    # text[i] → [i+1.0, i+2.0, i+3.0]
    for item in response.data:
        i = item.index
        expected_vec = [float(i + 1), float(i + 2), float(i + 3)]
        assert item.embedding == expected_vec, (
            f"Vector at index {i} should be {expected_vec}, got {item.embedding}"
        )


def test_batch_embedding_single_text_preserves_order(monkeypatch):
    """Single text embedding returns correct index 0 vector."""
    model = make_model(monkeypatch)
    query = EmbeddingRequest(
        input=["only one"],
        model="fake-embed",
        encoding_format="float",
    )

    response = asyncio.run(model.text_embeddings(query))

    assert len(response.data) == 1
    assert response.data[0].index == 0
    assert response.data[0].embedding == [1.0, 2.0, 3.0]
