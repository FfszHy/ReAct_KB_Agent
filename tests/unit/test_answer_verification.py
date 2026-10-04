from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from pkb_agent.agent.verification import AnswerVerifier, Evidence, extract_evidence


def test_explicit_answer_character_limits_do_not_confuse_domain_numbers():
    from pkb_agent.agent.verification import answer_char_limit

    assert answer_char_limit("回答控制在 250 字以内") == 250
    assert answer_char_limit("区分事实与推断, 控制在400字以内") == 400
    assert answer_char_limit("Use at most 120 characters.") == 120
    assert answer_char_limit("10000 requests per second, Kubernetes 1.31") is None
    assert answer_char_limit("最多 200 字。正文100字以内。") == 100
    assert answer_char_limit("请解释 Pydantic 如何验证用户名最多 20 字符, 并给出示例。") is None
    assert answer_char_limit("请解释数据库表最多 100 字段时的设计取舍。") is None
    assert answer_char_limit("回答控制在100001字以内") is None
    assert answer_char_limit("用户名 at most 20 characters 如何验证?") is None
    assert answer_char_limit("请解释如何把用户名长度控制在20字符以内。") is None


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
        "answer": "The document says to use SSO. Centralizing authentication is appropriate.",
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


def _validate_claim_spans(answer: str, claim_texts: list[str]):
    response = {
        "status": "grounded",
        "answer": answer,
        "claims": [
            {"text": text, "kind": "fact", "citations": ["kb:chunk-1"]}
            for text in claim_texts
        ],
        "citations": [{"id": "kb:chunk-1"}],
    }
    return AnswerVerifier().validate(json.dumps(response), {"kb:chunk-1": _kb_evidence()})


@pytest.mark.parametrize(
    "answer,claims",
    [
        ("First fact.\n\nSecond fact.", ["First fact.", "Second fact."]),
        ("First fact. \t\n\u3000Second fact.", ["First fact.", "Second fact."]),
        ("第一条事实。第二条事实。", ["第一条事实。", "第二条事实。"]),
        ("Use  SSO.\nKeep the  internal spaces.", ["Use  SSO.\nKeep the  internal spaces."]),
        ("  First fact.\nSecond fact.\n  ", ["  First fact. ", "\nSecond fact.\n"]),
    ],
)
def test_grounded_answer_accepts_exact_ordered_spans_and_inter_claim_whitespace(answer, claims):
    result = _validate_claim_spans(answer, claims)

    assert result.valid
    assert result.payload.answer == answer.strip()
    assert [claim.text for claim in result.payload.claims] == [text.strip() for text in claims]


@pytest.mark.parametrize(
    "answer,claims",
    [
        (
            "只有以卷挂载的 ConfigMap 才会自动更新。",
            ["以卷挂载的 ConfigMap 会最终更新。"],
        ),
        (
            "include 按字段值筛选输出。",
            ["include 按字段名筛选输出。"],
        ),
        ("Summary: First fact.", ["First fact."]),
        ("Possibly the first fact.", ["The first fact."]),
        ("Second fact. First fact.", ["First fact.", "Second fact."]),
        ("First fact.", ["First fact.", "Second fact."]),
        ("First fact, Second fact.", ["First fact.", "Second fact."]),
        ("First fact. An extra conclusion.", ["First fact."]),
        ("First fact. However, second fact.", ["First fact.", "Second fact."]),
        ("Use SSO.", ["Use  SSO."]),
        ("Use  SSO.", ["Use SSO."]),
    ],
)
def test_grounded_answer_rejects_drift_reordering_missing_spans_and_uncovered_prose(answer, claims):
    result = _validate_claim_spans(answer, claims)

    assert result.valid is False
    assert "exact ordered spans" in result.errors[0]
    assert "joining every claim text verbatim in claims order" in result.errors[0]
    assert "character" in result.errors[0]


def test_uncovered_tail_has_a_specific_repair_error():
    result = _validate_claim_spans("Use SSO. Extra assertion.", ["Use SSO."])

    assert result.valid is False
    assert "uncovered answer text starts at character 9" in result.errors[0]


def test_exact_span_check_does_not_claim_to_prove_source_entailment():
    # The fixture source says to use SSO. Identical answer/claim text can still
    # be unsupported by that source; semantic review must catch that separately.
    unsupported = "The document requires password-only authentication."

    result = _validate_claim_spans(unsupported, [unsupported])

    assert result.valid


def test_insufficient_evidence_response_does_not_require_claim_spans():
    response = {
        "status": "insufficient_evidence",
        "answer": "No measurements were retrieved to answer this question.",
        "claims": [],
        "citations": [],
    }

    result = AnswerVerifier().validate(json.dumps(response), {})

    assert result.valid


@pytest.mark.parametrize("malformed", [[], ["grounded"], {}, {"value": "grounded"}])
def test_malformed_status_returns_invalid_instead_of_raising(malformed):
    response = {"status": malformed, "answer": "Use SSO.", "claims": [], "citations": []}

    result = AnswerVerifier().validate(json.dumps(response), {})

    assert result.valid is False
    assert result.errors == ("status must be 'grounded' or 'insufficient_evidence'",)


@pytest.mark.parametrize("malformed", [[], ["fact"], {}, {"value": "fact"}])
def test_malformed_claim_kind_returns_invalid_instead_of_raising(malformed):
    response = {
        "status": "grounded",
        "answer": "Use SSO.",
        "claims": [{"text": "Use SSO.", "kind": malformed, "citations": ["kb:chunk-1"]}],
        "citations": [{"id": "kb:chunk-1"}],
    }

    result = AnswerVerifier().validate(json.dumps(response), {"kb:chunk-1": _kb_evidence()})

    assert result.valid is False
    assert result.errors == ("claims[0].kind must be 'fact' or 'inference'",)


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


def test_document_catalog_page_is_a_single_citation_eligible_evidence_record():
    records = extract_evidence(
        "rag_list_documents",
        {
            "items": [
                {
                    "document_id": "doc-1",
                    "title": "产品需求说明书",
                    "source_uri": "file://requirements.pdf",
                    "source_type": "file",
                    "chunk_count": 8,
                    "char_count": 2400,
                }
            ]
        },
        settings=SimpleNamespace(),
    )

    assert len(records) == 1
    evidence = records[0]
    assert evidence.citation_id.startswith("kbcatalog:0:")
    assert evidence.source_type == "kb_catalog"
    assert evidence.title == "知识库资料目录 - 第 1 至 1 份"
    assert evidence.metadata == {"offset": 0, "count": 1, "next_offset": None}

    response = {
        "status": "grounded",
        "answer": "知识库目录中记录了一份名为《产品需求说明书》的资料。",
        "claims": [
            {
                "text": "知识库目录中记录了一份名为《产品需求说明书》的资料。",
                "kind": "fact",
                "citations": [evidence.citation_id],
            }
        ],
        "citations": [{"id": evidence.citation_id}],
    }
    assert AnswerVerifier().validate(json.dumps(response), {evidence.citation_id: evidence}).valid


def test_large_catalog_uses_one_evidence_record_per_page():
    first_page = {
        "offset": 0,
        "next_offset": 20,
        "items": [{"document_id": f"doc-{index}", "title": f"资料 {index}"} for index in range(20)],
    }
    second_page = {
        "offset": 20,
        "next_offset": None,
        "items": [{"document_id": f"doc-{index}", "title": f"资料 {index}"} for index in range(20, 40)],
    }
    evidence = {
        record.citation_id: record
        for page in (first_page, second_page)
        for record in extract_evidence("rag_list_documents", page, settings=SimpleNamespace())
    }

    first_id, second_id = evidence
    assert first_id.startswith("kbcatalog:0:")
    assert second_id.startswith("kbcatalog:20:")
    response = {
        "status": "grounded",
        "answer": "目录第 1 页列出 20 份资料。\n目录第 2 页列出另外 20 份资料。",
        "claims": [
            {
                "text": "目录第 1 页列出 20 份资料。",
                "kind": "fact",
                "citations": [first_id],
            },
            {
                "text": "目录第 2 页列出另外 20 份资料。",
                "kind": "fact",
                "citations": [second_id],
            },
        ],
        "citations": [{"id": first_id}, {"id": second_id}],
    }
    assert AnswerVerifier().validate(json.dumps(response), evidence).valid


def test_catalog_evidence_identity_includes_contents_and_cursor_not_just_offset():
    items = [{"document_id": f"doc-{index}", "title": f"Title {index}"} for index in range(20)]
    pages = [
        {"offset": 20, "next_offset": 40, "items": items},
        {"offset": 20, "next_offset": 30, "items": items[:10]},
        {"offset": 20, "next_offset": None, "items": items[:10]},
        {"offset": 20, "next_offset": 30, "items": [{**row, "title": "Changed"} for row in items[:10]]},
    ]
    records = [
        extract_evidence("rag_list_documents", page, settings=SimpleNamespace())[0]
        for page in pages
    ]
    ledger = {record.citation_id: record for record in records}

    assert len(ledger) == 4
    assert records[0].metadata["count"] == 20
    assert records[1].metadata["count"] == 10
    assert records[0].review_text != records[1].review_text
    repeated = extract_evidence("rag_list_documents", pages[0], settings=SimpleNamespace())[0]
    assert repeated.citation_id == records[0].citation_id
