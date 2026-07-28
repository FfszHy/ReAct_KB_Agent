from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

from pkb_agent.agent.verification import AnswerVerifier, Evidence, extract_evidence


def _kb_evidence() -> Evidence:
    return Evidence(
        citation_id="kb:chunk-1",
        source_type="kb_chunk",
        source_id="chunk-1",
        title="Architecture decision",
        locator="doc-1",
        excerpt="Use SSO.",
        metadata={"document_id": "doc-1", "chunk_index": 0},
    )


def test_answer_verifier_materializes_only_current_run_evidence():
    verifier = AnswerVerifier()
    response = {
        "status": "grounded",
        "answer": "The document says to use SSO, so centralizing authentication is appropriate.",
        "claims": [
            {"text": "The document says to use SSO.", "kind": "fact", "citations": ["kb:chunk-1"]},
            {
                "text": "Centralizing authentication is appropriate.",
                "kind": "inference",
                "citations": ["kb:chunk-1"],
            },
        ],
        "citations": [{"id": "kb:chunk-1"}],
    }

    result = verifier.validate(json.dumps(response), {"kb:chunk-1": _kb_evidence()})

    assert result.valid is True
    assert result.payload is not None
    assert result.payload.claims[0].kind == "fact"
    assert result.payload.claims[1].kind == "inference"
    assert result.payload.to_dict()["citations"][0]["source_id"] == "chunk-1"


def test_answer_verifier_rejects_fabricated_and_undeclared_citations():
    verifier = AnswerVerifier()
    response = {
        "answer": "Unsupported claim.",
        "claims": [
            {"text": "Unsupported claim.", "kind": "fact", "citations": ["kb:invented"]}
        ],
        "citations": [{"id": "kb:invented"}],
    }

    result = verifier.validate(json.dumps(response), {"kb:chunk-1": _kb_evidence()})

    assert result.valid is False
    assert "was not retrieved in this run" in result.errors[0]


def test_web_evidence_records_fetch_provenance_trust_and_expiry():
    settings = SimpleNamespace(
        web_evidence_ttl_hours=24,
        web_trusted_domains=["archives.example"],
    )
    records = extract_evidence(
        "web_fetch",
        {
            "url": "https://archives.example/report",
            "final_url": "https://archives.example/report",
            "title": "Official report",
            "text": "A source excerpt.",
            "fetched_at": "2020-01-01T00:00:00+00:00",
            "status_code": 200,
            "content_type": "text/html",
            "truncated": False,
        },
        settings=settings,
        observed_at=datetime(2020, 1, 1, tzinfo=UTC),
    )

    assert len(records) == 1
    evidence = records[0]
    assert evidence.citation_id.startswith("web:")
    assert evidence.metadata is not None
    assert evidence.metadata["domain"] == "archives.example"
    assert evidence.metadata["trust_level"] == "high"
    assert evidence.metadata["fetched_at"] == "2020-01-01T00:00:00+00:00"
    assert evidence.metadata["expiration_state"] == "expired"
    assert len(evidence.metadata["content_sha256"]) == 64

    response = {
        "answer": "Old report.",
        "claims": [{"text": "Old report.", "kind": "fact", "citations": [evidence.citation_id]}],
        "citations": [{"id": evidence.citation_id}],
    }
    verdict = AnswerVerifier().validate(json.dumps(response), {evidence.citation_id: evidence})
    assert verdict.valid is False
    assert "is expired" in verdict.errors[0]


def test_web_search_results_are_not_citation_eligible_until_fetched():
    records = extract_evidence(
        "web_search",
        {"results": [{"url": "https://example.com", "snippet": "not enough"}]},
        settings=SimpleNamespace(),
    )

    assert records == []
