"""Unit tests for gallama reranker backend (Phase 26.1 / 26.6)."""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from gallama.backend.reranker import reranker as reranker_module
from gallama.data_classes.data_class import ModelSpec, RerankRequest
from gallama.model_manager.ModelManager import ModelManager
from gallama.routes.rerank import rerank as rerank_endpoint


class FakeCrossEncoder:
    """Mock sentence-transformers CrossEncoder that returns predictable scores."""

    init_calls = []

    def __init__(self, model_name_or_path, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.kwargs = kwargs
        self.predict_calls = []
        FakeCrossEncoder.init_calls.append((model_name_or_path, kwargs))

    def predict(self, pairs, activation_fn=None, prompt=None, show_progress_bar=False, convert_to_numpy=True):
        """Return scores that are proportional to document length (longer = more relevant for testing)."""
        self.predict_calls.append({
            "pairs": pairs,
            "activation_fn": type(activation_fn).__name__ if activation_fn else None,
            "prompt": prompt,
        })
        scores = np.array([float(len(doc)) / 10.0 for _, doc in pairs], dtype=np.float32)
        if activation_fn is not None:
            scores = activation_fn(torch.from_numpy(scores)).numpy()
        return scores


def make_model(monkeypatch, backend_extra_args=None):
    FakeCrossEncoder.init_calls = []
    monkeypatch.setattr(reranker_module, "CrossEncoder", FakeCrossEncoder)
    return reranker_module.RerankerModel(
        ModelSpec(
            model_name="fake-rerank",
            model_id="fake/reranker",
            backend="reranker",
            backend_extra_args=backend_extra_args or {},
        )
    )


def test_reranker_empty_documents_returns_empty(monkeypatch):
    model = make_model(monkeypatch)
    results = asyncio.run(model.rerank("query", []))
    assert results == []


def test_reranker_single_document(monkeypatch):
    model = make_model(monkeypatch)
    results = asyncio.run(model.rerank("query", ["a document"]))
    assert len(results) == 1
    assert results[0][0] == 0  # original index
    assert 0.0 <= results[0][1] <= 1.0  # sigmoid-normalized score


def test_reranker_multiple_documents_descending_order(monkeypatch):
    model = make_model(monkeypatch)
    documents = [
        "short",
        "this is a medium length document here",
        "this is a very very long document that should score highest",
    ]
    results = asyncio.run(model.rerank("query", documents))
    assert len(results) == 3
    # Longer docs get higher base scores in our fake, so they should sort descending
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), f"Expected descending scores, got {scores}"
    # Verify original indices are preserved
    indices = [idx for idx, _ in results]
    assert set(indices) == {0, 1, 2}


def test_reranker_top_n_caps_results(monkeypatch):
    model = make_model(monkeypatch)
    documents = ["a", "b", "c", "d", "e"]
    results = asyncio.run(model.rerank("query", documents, top_n=3))
    assert len(results) == 3


def test_reranker_uses_sigmoid_activation(monkeypatch):
    model = make_model(monkeypatch)
    asyncio.run(model.rerank("query", ["test doc"]))
    assert len(model.model.predict_calls) == 1
    call = model.model.predict_calls[0]
    assert call["activation_fn"] == "Sigmoid"
    assert call["prompt"] == "Retrieve text relevant to the user's query."


def test_reranker_scores_in_0_to_1_range(monkeypatch):
    model = make_model(monkeypatch)
    documents = ["a", "longer document", "the longest document in the batch"]
    results = asyncio.run(model.rerank("query", documents))
    for idx, score in results:
        assert 0.0 <= score <= 1.0, f"Score {score} for idx {idx} is out of [0, 1] range"


def test_reranker_backend_init_kwargs_trust_remote_code_default(monkeypatch):
    make_model(monkeypatch)
    assert FakeCrossEncoder.init_calls == [
        ("fake/reranker", {"trust_remote_code": True})
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 26.6 — Reranker route integration test (return_documents)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_rerank_route_return_documents_includes_text(monkeypatch):
    """POST /v1/rerank with return_documents=True includes document text in results."""
    # Create a fake reranker model that the route can call
    fake_model = make_model(monkeypatch)

    # Build a ModelManager with the fake reranker already registered
    mm = ModelManager()
    model_spec = ModelSpec(
        model_name="fake-rerank",
        model_id="fake/reranker",
        backend="reranker",
        model_type="reranker",
    )
    mm._update_model("fake-rerank", model_spec, fake_model)

    # Mock get_model_manager to return our prepared ModelManager
    monkeypatch.setattr(
        "gallama.routes.rerank.get_model_manager",
        lambda: mm,
    )

    request = RerankRequest(
        model="fake-rerank",
        query="What food?",
        documents=["Meo loves phở", "Server runs Ubuntu", "Meo ordered rice"],
        top_n=2,
        return_documents=True,
    )

    response = await rerank_endpoint(request)

    assert response.model == "fake-rerank"
    assert len(response.results) == 2

    # Verify returned documents match the original input
    for result in response.results:
        assert result.document is not None
        assert result.document["text"] == request.documents[result.index]

    # Scores are sigmoid-normalized 0..1
    for result in response.results:
        assert 0.0 <= result.relevance_score <= 1.0


@pytest.mark.asyncio
async def test_rerank_route_return_documents_false_no_text(monkeypatch):
    """POST /v1/rerank with return_documents=False omits document text."""
    fake_model = make_model(monkeypatch)

    mm = ModelManager()
    model_spec = ModelSpec(
        model_name="fake-rerank",
        model_id="fake/reranker",
        backend="reranker",
        model_type="reranker",
    )
    mm._update_model("fake-rerank", model_spec, fake_model)

    monkeypatch.setattr(
        "gallama.routes.rerank.get_model_manager",
        lambda: mm,
    )

    request = RerankRequest(
        model="fake-rerank",
        query="test",
        documents=["doc1", "doc2"],
        top_n=1,
        return_documents=False,
    )

    response = await rerank_endpoint(request)

    assert response.model == "fake-rerank"
    assert len(response.results) == 1
    assert response.results[0].document is None


@pytest.mark.asyncio
async def test_rerank_route_model_not_loaded_returns_404(monkeypatch):
    """POST /v1/rerank with an unloaded model returns 404."""
    from fastapi import HTTPException

    mm = ModelManager()  # empty, no models loaded
    monkeypatch.setattr(
        "gallama.routes.rerank.get_model_manager",
        lambda: mm,
    )

    request = RerankRequest(
        model="nonexistent-rerank",
        query="test",
        documents=["doc1"],
    )

    with pytest.raises(HTTPException) as exc_info:
        await rerank_endpoint(request)

    assert exc_info.value.status_code == 404
    assert "not loaded" in exc_info.value.detail.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 26.6 — ModelManager.get_model(_type="reranker")
# ═══════════════════════════════════════════════════════════════════════════════

def test_model_manager_get_model_reranker_strict(monkeypatch):
    """ModelManager.get_model with _type='reranker' resolves from reranker_dict."""
    fake_model = make_model(monkeypatch)
    model_name = "Qwen3-VL-Reranker-2B"

    mm = ModelManager()
    model_spec = ModelSpec(
        model_name=model_name,
        model_id="fake/reranker",
        backend="reranker",
        model_type="reranker",
        strict=True,
    )
    mm._update_model(model_name, model_spec, fake_model)

    # strict lookup by exact name
    result = mm.get_model(model_name, _type="reranker")
    assert result is fake_model

    # wrong name with strict=True → None (non_strict dict is empty)
    result_wrong = mm.get_model("other-model", _type="reranker")
    assert result_wrong is None


def test_model_manager_get_model_reranker_non_strict_fallback(monkeypatch):
    """Non-strict reranker model allows lookup by any name."""
    fake_model = make_model(monkeypatch)
    model_name = "Qwen3-VL-Reranker-2B"

    mm = ModelManager()
    model_spec = ModelSpec(
        model_name=model_name,
        model_id="fake/reranker",
        backend="reranker",
        model_type="reranker",
        strict=False,
    )
    mm._update_model(model_name, model_spec, fake_model)

    # Exact name match works
    assert mm.get_model(model_name, _type="reranker") is fake_model

    # Non-strict: any name returns the first non-strict model
    result_fallback = mm.get_model("any-other-name", _type="reranker")
    assert result_fallback is fake_model


def test_model_manager_get_model_reranker_not_loaded():
    """Empty reranker_dict returns None."""
    mm = ModelManager()
    result = mm.get_model("any-model", _type="reranker")
    assert result is None


def test_model_manager_get_model_reranker_close_all_includes_reranker(monkeypatch):
    """close_all_models() iterates reranker_dict and calls close()."""
    fake_model = make_model(monkeypatch)
    fake_model.close = MagicMock()

    mm = ModelManager()
    model_spec = ModelSpec(
        model_name="fake-rerank",
        model_id="fake/reranker",
        backend="reranker",
        model_type="reranker",
    )
    mm._update_model("fake-rerank", model_spec, fake_model)

    mm.close_all_models()
    fake_model.close.assert_called_once()
