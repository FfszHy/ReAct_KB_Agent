"""Machine-verifiable final-answer contracts and per-run evidence ledger.

The model is allowed to *select* evidence, but it is never allowed to invent
the evidence metadata that is shown to a user.  This module turns successful
retrieval observations into an in-memory ledger and validates the model's
final JSON against that ledger.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from pkb_agent.app.settings import Settings


_CLAIM_KINDS = frozenset({"fact", "inference"})
_ANSWER_STATUSES = frozenset({"grounded", "insufficient_evidence"})
_MAX_ANSWER_CHARS = 20_000
_MAX_CLAIMS = 100
_MAX_CITATIONS = 100
_EXCERPT_CHARS = 500


@dataclass(frozen=True)
class Evidence:
    """A citation-eligible source observed during the current agent run."""

    citation_id: str
    source_type: str
    source_id: str
    title: str | None = None
    locator: str | None = None
    excerpt: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical, user-facing citation object."""
        return {
            "id": self.citation_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "title": self.title,
            "locator": self.locator,
            "excerpt": self.excerpt,
            "metadata": dict(self.metadata or {}),
        }

    def to_prompt_dict(self) -> dict[str, Any]:
        """Return the small source descriptor the model needs to cite it."""
        item: dict[str, Any] = {
            "id": self.citation_id,
            "source_type": self.source_type,
            "title": self.title,
            "locator": self.locator,
        }
        if self.source_type == "web_page":
            metadata = self.metadata or {}
            item["web"] = {
                "domain": metadata.get("domain"),
                "trust_level": metadata.get("trust_level"),
                "expiration_state": metadata.get("expiration_state"),
            }
        return item


@dataclass(frozen=True)
class Claim:
    """A model statement labelled as source fact or model inference."""

    text: str
    kind: str
    citation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind,
            "citations": list(self.citation_ids),
        }


@dataclass(frozen=True)
class VerifiedAnswer:
    """A parsed answer whose citation IDs have passed ledger validation."""

    answer: str
    claims: tuple[Claim, ...]
    citations: tuple[Evidence, ...]
    status: str = "grounded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "answer": self.answer,
            "claims": [claim.to_dict() for claim in self.claims],
            "citations": [citation.to_dict() for citation in self.citations],
        }


@dataclass(frozen=True)
class ValidationResult:
    """Successful parsed answer or deterministic validation failures."""

    payload: VerifiedAnswer | None = None
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.payload is not None and not self.errors


class AnswerVerifier:
    """Validate the final model JSON against evidence retrieved in this run."""

    def validate(
        self,
        content: str | None,
        evidence_by_id: Mapping[str, Evidence],
    ) -> ValidationResult:
        if not isinstance(content, str) or not content.strip():
            return _invalid("final answer is empty; expected a JSON object")
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            return _invalid(f"final answer is not valid JSON: {exc.msg}")
        if not isinstance(raw, Mapping):
            return _invalid("final answer must be a JSON object")

        allowed_root = {"status", "answer", "claims", "citations"}
        unknown_root = sorted(set(raw) - allowed_root)
        if unknown_root:
            return _invalid(f"final answer contains unsupported fields: {', '.join(unknown_root)}")

        missing = [name for name in ("answer", "claims", "citations") if name not in raw]
        if missing:
            return _invalid(f"final answer is missing required fields: {', '.join(missing)}")

        answer = raw.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return _invalid("answer must be a non-empty string")
        answer = answer.strip()
        if len(answer) > _MAX_ANSWER_CHARS:
            return _invalid(f"answer exceeds {_MAX_ANSWER_CHARS} characters")

        status = raw.get("status", "grounded")
        if status not in _ANSWER_STATUSES:
            return _invalid("status must be 'grounded' or 'insufficient_evidence'")

        citations_raw = raw.get("citations")
        citations, citation_errors = self._validate_citations(citations_raw, evidence_by_id)
        if citation_errors:
            return ValidationResult(errors=tuple(citation_errors))

        claims_raw = raw.get("claims")
        claims, claim_errors = self._validate_claims(claims_raw, citations)
        if claim_errors:
            return ValidationResult(errors=tuple(claim_errors))

        if status == "grounded":
            if not claims:
                return _invalid("grounded answers require at least one cited claim")
            if not citations:
                return _invalid("grounded answers require at least one citation")
        else:
            if claims or citations:
                return _invalid(
                    "insufficient_evidence answers must not contain claims or citations; "
                    "state the refusal in answer instead"
                )

        used_citations = {citation_id for claim in claims for citation_id in claim.citation_ids}
        unused = [citation.citation_id for citation in citations if citation.citation_id not in used_citations]
        if unused:
            return _invalid(f"citations are not attached to a claim: {', '.join(unused)}")

        return ValidationResult(
            payload=VerifiedAnswer(
                answer=answer,
                claims=tuple(claims),
                citations=tuple(citations),
                status=str(status),
            )
        )

    def _validate_citations(
        self,
        raw: Any,
        evidence_by_id: Mapping[str, Evidence],
    ) -> tuple[list[Evidence], list[str]]:
        if not isinstance(raw, list):
            return [], ["citations must be an array"]
        if len(raw) > _MAX_CITATIONS:
            return [], [f"citations exceeds {_MAX_CITATIONS} items"]

        citations: list[Evidence] = []
        errors: list[str] = []
        seen: set[str] = set()
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                errors.append(f"citations[{index}] must be an object with only an id")
                continue
            if set(item) != {"id"}:
                errors.append(f"citations[{index}] must contain only the id field")
                continue
            citation_id = item.get("id")
            if not isinstance(citation_id, str) or not citation_id.strip():
                errors.append(f"citations[{index}].id must be a non-empty string")
                continue
            citation_id = citation_id.strip()
            if citation_id in seen:
                errors.append(f"citations[{index}].id duplicates {citation_id}")
                continue
            seen.add(citation_id)
            evidence = evidence_by_id.get(citation_id)
            if evidence is None:
                errors.append(
                    f"citations[{index}].id {citation_id!r} was not retrieved in this run"
                )
                continue
            if evidence.source_type == "web_page" and (evidence.metadata or {}).get("expired"):
                errors.append(f"citations[{index}].id {citation_id!r} is expired")
                continue
            citations.append(evidence)
        return citations, errors

    def _validate_claims(
        self,
        raw: Any,
        citations: list[Evidence],
    ) -> tuple[list[Claim], list[str]]:
        if not isinstance(raw, list):
            return [], ["claims must be an array"]
        if len(raw) > _MAX_CLAIMS:
            return [], [f"claims exceeds {_MAX_CLAIMS} items"]

        declared_ids = {citation.citation_id for citation in citations}
        claims: list[Claim] = []
        errors: list[str] = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                errors.append(f"claims[{index}] must be an object")
                continue
            if set(item) != {"text", "kind", "citations"}:
                errors.append(
                    f"claims[{index}] must contain exactly text, kind, and citations fields"
                )
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"claims[{index}].text must be a non-empty string")
                continue
            kind = item.get("kind")
            if kind not in _CLAIM_KINDS:
                errors.append(f"claims[{index}].kind must be 'fact' or 'inference'")
                continue
            citation_ids = item.get("citations")
            if not isinstance(citation_ids, list) or not citation_ids:
                errors.append(f"claims[{index}].citations must be a non-empty array")
                continue

            normalised_ids: list[str] = []
            seen: set[str] = set()
            for ref_index, citation_id in enumerate(citation_ids):
                if not isinstance(citation_id, str) or not citation_id.strip():
                    errors.append(
                        f"claims[{index}].citations[{ref_index}] must be a non-empty string"
                    )
                    continue
                citation_id = citation_id.strip()
                if citation_id in seen:
                    errors.append(
                        f"claims[{index}].citations[{ref_index}] duplicates {citation_id}"
                    )
                    continue
                seen.add(citation_id)
                if citation_id not in declared_ids:
                    errors.append(
                        f"claims[{index}] references undeclared citation {citation_id!r}"
                    )
                    continue
                normalised_ids.append(citation_id)
            if not normalised_ids:
                continue
            claims.append(
                Claim(text=text.strip(), kind=str(kind), citation_ids=tuple(normalised_ids))
            )
        return claims, errors


def extract_evidence(
    tool_name: str,
    result: Any,
    *,
    settings: Settings,
    observed_at: datetime | None = None,
) -> list[Evidence]:
    """Build citation-eligible records from a successful tool result.

    Search result snippets are not treated as web-page evidence: a page must be
    fetched through ``web_fetch`` before it can be cited.  KB search previews
    and reads represent chunks already retrieved in this run and are eligible.
    """
    if not isinstance(result, Mapping):
        return []
    now = _as_utc(observed_at or datetime.now(UTC))
    if tool_name == "rag_search":
        rows = result.get("results")
        if not isinstance(rows, list):
            return []
        search_records: list[Evidence] = []
        for row in rows:
            record = _kb_evidence(row)
            if record is not None:
                search_records.append(record)
        return search_records
    if tool_name == "rag_read":
        if result.get("kind") == "chunk":
            record = _kb_evidence(result)
            return [record] if record is not None else []
        if result.get("kind") == "document":
            document_id = _string_or_none(result.get("document_id"))
            rows = result.get("chunks")
            if not isinstance(rows, list):
                return []
            document_records: list[Evidence] = []
            for row in rows:
                item = dict(row) if isinstance(row, Mapping) else {}
                if document_id and not item.get("document_id"):
                    item["document_id"] = document_id
                record = _kb_evidence(item)
                if record is not None:
                    document_records.append(record)
            return document_records
        return []
    if tool_name == "web_fetch":
        record = _web_evidence(result, settings=settings, observed_at=now)
        return [record] if record is not None else []
    return []


def refusal_payload(message: str = "无法基于本轮检索到的证据提供可验证回答。") -> VerifiedAnswer:
    """Return the only answer shape allowed without supporting evidence."""
    return VerifiedAnswer(answer=message, claims=(), citations=(), status="insufficient_evidence")


def _kb_evidence(row: Any) -> Evidence | None:
    if not isinstance(row, Mapping):
        return None
    chunk_id = _string_or_none(row.get("chunk_id") or row.get("id"))
    if chunk_id is None:
        return None
    excerpt = _string_or_none(row.get("content") or row.get("content_preview"))
    return Evidence(
        citation_id=f"kb:{chunk_id}",
        source_type="kb_chunk",
        source_id=chunk_id,
        title=_string_or_none(row.get("doc_title") or row.get("title")),
        locator=_string_or_none(row.get("source_uri") or row.get("document_id")),
        excerpt=_truncate_excerpt(excerpt),
        metadata={
            key: row[key]
            for key in ("document_id", "chunk_index", "score", "vector_score", "fts_score")
            if row.get(key) is not None
        },
    )


def _web_evidence(
    row: Mapping[str, Any],
    *,
    settings: Settings,
    observed_at: datetime,
) -> Evidence | None:
    locator = _string_or_none(row.get("final_url") or row.get("url"))
    if locator is None:
        return None
    fetched_at = _parse_datetime(row.get("fetched_at")) or observed_at
    ttl_hours = _non_negative_int(getattr(settings, "web_evidence_ttl_hours", 168), default=168)
    expires_at = fetched_at + timedelta(hours=ttl_hours)
    now = datetime.now(UTC)
    expired = now >= expires_at
    parsed = urlparse(locator)
    domain = (parsed.hostname or "").lower()
    text = _string_or_none(row.get("text"))
    citation_key = f"{locator}|{fetched_at.isoformat()}"
    citation_id = f"web:{hashlib.sha256(citation_key.encode('utf-8')).hexdigest()[:24]}"
    metadata = {
        "fetched_at": fetched_at.isoformat(),
        "domain": domain,
        "trust_level": _web_trust_level(domain, parsed.scheme, settings),
        "expires_at": expires_at.isoformat(),
        "expired": expired,
        "expiration_state": "expired" if expired else "fresh",
        "content_sha256": hashlib.sha256((text or "").encode("utf-8")).hexdigest(),
        "content_type": _string_or_none(row.get("content_type")),
        "http_status": row.get("status_code"),
        "truncated": bool(row.get("truncated", False)),
    }
    return Evidence(
        citation_id=citation_id,
        source_type="web_page",
        source_id=locator,
        title=_string_or_none(row.get("title")),
        locator=locator,
        excerpt=_truncate_excerpt(text),
        metadata=metadata,
    )


def _web_trust_level(domain: str, scheme: str, settings: Settings) -> str:
    configured = getattr(settings, "web_trusted_domains", ()) or ()
    if isinstance(configured, str):
        configured = [configured]
    for candidate in configured:
        trusted = str(candidate).strip().lower().lstrip(".")
        if trusted and (domain == trusted or domain.endswith(f".{trusted}")):
            return "high"
    if scheme.lower() == "https":
        return "medium"
    return "low"


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _truncate_excerpt(value: str | None) -> str | None:
    if value is None or len(value) <= _EXCERPT_CHARS:
        return value
    return value[:_EXCERPT_CHARS].rstrip() + "…"


def _non_negative_int(value: Any, *, default: int) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return default


def _invalid(*errors: str) -> ValidationResult:
    return ValidationResult(errors=tuple(errors))
