from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace

import dashscope
import pytest

from pkb_agent.agent.errors import EmbeddingError
from pkb_agent.rag.embeddings import EmbeddingProvider


async def test_embedding_provider_calls_dashscope_sdk_and_restores_input_order(monkeypatch):
    calls: list[dict[str, object]] = []

    def call(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        texts = kwargs["input"]
        assert isinstance(texts, list)
        return SimpleNamespace(
            status_code=HTTPStatus.OK,
            output={
                "embeddings": [
                    {"text_index": index, "embedding": [float(index)] * 3}
                    for index in reversed(range(len(texts)))
                ]
            },
        )

    monkeypatch.setattr(dashscope.TextEmbedding, "call", staticmethod(call))
    provider = EmbeddingProvider(
        api_key="test-key",
        model="qwen3.7-text-embedding",
        dimensions=3,
        batch_size=2,
        timeout=17,
    )

    embeddings = await provider.embed(["one", "two", "three"])

    assert embeddings == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    assert calls == [
        {
            "model": "qwen3.7-text-embedding",
            "input": ["one", "two"],
            "api_key": "test-key",
            "dimension": 3,
            "request_timeout": 17,
        },
        {
            "model": "qwen3.7-text-embedding",
            "input": ["three"],
            "api_key": "test-key",
            "dimension": 3,
            "request_timeout": 17,
        },
    ]


async def test_embedding_provider_exposes_non_retryable_provider_error_detail(monkeypatch):
    calls = 0

    def call(**kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            status_code=HTTPStatus.BAD_REQUEST,
            code="InvalidParameter",
            message="Input list may contain at most 20 items.",
            output=None,
        )

    monkeypatch.setattr(dashscope.TextEmbedding, "call", staticmethod(call))
    provider = EmbeddingProvider(
        api_key="test-key",
        model="qwen3.7-text-embedding",
        dimensions=1536,
        batch_size=64,
    )

    with pytest.raises(EmbeddingError, match="batch_items=64") as exc_info:
        await provider.embed(["chunk"] * 64)

    assert calls == 1
    assert "at most 20 items" in str(exc_info.value)
