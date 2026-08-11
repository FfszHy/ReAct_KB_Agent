from __future__ import annotations

from unittest.mock import AsyncMock, Mock

from pkb_agent.rag.ingestion import IngestionPipeline


async def test_ingest_resumes_existing_document_missing_embeddings():
    documents = Mock()
    documents.get_by_hash.return_value = {"id": "doc-1", "chunk_count": 1}
    chunks = Mock()
    chunks.list_by_document.return_value = [{"id": "chunk-1", "content": "resume me"}]
    chunks.get_embeddings_by_chunk_ids.return_value = {}
    embedder = Mock(model="qwen3.7-text-embedding", dimensions=3)
    embedder.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    pipeline = IngestionPipeline(documents, chunks, embedder)

    document = await pipeline.ingest_text("resume me", title="Resume", user_id="eval")

    assert document["id"] == "doc-1"
    documents.create.assert_not_called()
    embedder.embed.assert_awaited_once_with(["resume me"])
    chunks.create_embeddings.assert_called_once_with(
        [
            {
                "chunk_id": "chunk-1",
                "embedding": [0.1, 0.2, 0.3],
                "model": "qwen3.7-text-embedding",
                "dimensions": 3,
            }
        ]
    )
