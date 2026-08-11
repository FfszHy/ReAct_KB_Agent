from __future__ import annotations

import httpx
import pytest

from pkb_agent.agent.errors import EmbeddingError
from pkb_agent.rag.embeddings import EmbeddingProvider


async def test_embedding_provider_exposes_non_retryable_provider_error_detail():
    class Client:
        def __init__(self) -> None:
            self.calls = 0

        async def post(self, url: str, **kwargs):
            self.calls += 1
            request = httpx.Request("POST", url, json=kwargs["json"])
            return httpx.Response(
                400,
                json={"error": {"message": "Input list may contain at most 10 items."}},
                request=request,
            )

    client = Client()
    provider = EmbeddingProvider(
        api_key="test-key",
        base_url="https://embeddings.example/v1",
        model="text-embedding-v4",
        dimensions=1536,
        batch_size=64,
    )
    provider._client = client  # type: ignore[assignment]

    with pytest.raises(EmbeddingError, match="batch_items=64") as exc_info:
        await provider.embed(["chunk"] * 64)

    assert client.calls == 1
    assert "at most 10 items" in str(exc_info.value)
