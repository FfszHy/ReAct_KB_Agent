from __future__ import annotations

from dataclasses import replace

import pytest

from pkb_agent.agent.verification import Evidence, merge_evidence


def _source(**overrides) -> Evidence:
    return replace(
        Evidence(
            citation_id="kb:c1",
            source_type="kb_chunk",
            source_id="c1",
            title="Source title",
            locator="eval://source",
            excerpt="The complete source passage.",
            review_text="The complete source passage, including its conditions and exceptions.",
            metadata={"document_id": "d1", "chunk_index": 1, "score": 0.8},
        ),
        **overrides,
    )


def test_sparse_reread_keeps_known_title_uri_and_more_complete_text():
    previous = _source()
    sparse = _source(
        title=None,
        locator="d1",
        excerpt="Short preview.",
        review_text="Short preview.",
        metadata={"document_id": "d1", "chunk_index": 1},
    )

    merged = merge_evidence(previous, sparse)

    assert merged.title == previous.title
    assert merged.locator == previous.locator
    assert merged.excerpt == previous.excerpt
    assert merged.review_text == previous.review_text
    assert merged.metadata == previous.metadata
    assert sparse.title is None


def test_full_read_can_enrich_a_search_preview_and_update_present_metadata():
    previous = _source(title=None, locator="d1", excerpt="Short.", review_text="Short.")
    current = _source(metadata={"document_id": "d1", "chunk_index": 1, "score": 0.9})

    assert merge_evidence(previous, current) == current


def test_missing_metadata_stays_missing_without_an_earlier_source():
    source = _source(title=None, locator="d1")
    assert merge_evidence(None, source) is source


def test_evidence_from_different_chunks_or_documents_is_not_merged():
    previous = _source()
    for source in (
        _source(citation_id="kb:c2", source_id="c2", title=None),
        _source(title=None, metadata={"document_id": "d2"}),
    ):
        assert merge_evidence(previous, source) is source


def test_web_snapshot_does_not_inherit_metadata_or_text_from_an_earlier_fetch():
    previous = _source(source_type="web_page")
    current = replace(previous, title=None, excerpt="New body.", review_text="New body.", metadata={})

    assert merge_evidence(previous, current) is current


@pytest.mark.parametrize(
    "field,old_value,new_value",
    [
        ("upstream_sha256", "a" * 64, "b" * 64),
        ("upstream_revision", "v1.0", "v2.0"),
    ],
)
def test_changed_source_version_preserves_entire_current_snapshot(field, old_value, new_value):
    previous = _source(metadata={"document_id": "d1", field: old_value, "score": 0.8})
    current = _source(
        title=None,
        locator="d1",
        excerpt="New.",
        review_text="New.",
        metadata={"document_id": "d1", field: new_value},
    )

    merged = merge_evidence(previous, current)

    assert merged is current
    assert merged.title is None
    assert merged.locator == "d1"
    assert merged.excerpt == "New."
    assert merged.review_text == "New."
    assert "score" not in merged.metadata


@pytest.mark.parametrize(
    "old_provenance,new_provenance",
    [
        ({"upstream_revision": "v1.0"}, {}),
        ({}, {"upstream_revision": "v1.0"}),
        ({"upstream_sha256": "a" * 64}, {"upstream_revision": "v1.0"}),
        ({"upstream_revision": "v1.0"}, {"upstream_sha256": "a" * 64}),
        ({"upstream_revision": "v1.0"}, {"upstream_revision": None}),
        ({"upstream_revision": "v1.0"}, {"upstream_revision": ""}),
        ({"upstream_revision": "v1.0"}, {"upstream_revision": "v1.0"}),
        ({"upstream_sha256": "a" * 64}, {"upstream_sha256": "A" * 64}),
    ],
)
def test_missing_or_equivalent_source_version_does_not_block_enrichment(old_provenance, new_provenance):
    previous = _source(metadata={"document_id": "d1", **old_provenance})
    current = _source(
        title=None, locator="d1", excerpt="Short.", review_text="Short.",
        metadata={"document_id": "d1", **new_provenance},
    )

    merged = merge_evidence(previous, current)

    assert merged.title == previous.title
    assert merged.locator == previous.locator
    assert merged.review_text == previous.review_text
