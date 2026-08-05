"""Versioned evaluation-set schema and strict JSONL loading.

The annotations use document keys rather than database UUIDs.  The corpus
ingester assigns every source an ``eval://<corpus>/<document-key>`` URI, so an
evaluation remains stable across re-ingestion and Supabase projects.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DatasetError(ValueError):
    """An invalid or internally inconsistent benchmark annotation."""


@dataclass(frozen=True)
class RelevantDocument:
    """A relevant source document and its graded relevance (1 or 2)."""

    key: str
    grade: int = 1

    @classmethod
    def from_dict(cls, value: Any, *, context: str) -> RelevantDocument:
        if not isinstance(value, dict):
            raise DatasetError(f"{context}: relevant document must be an object")
        key = _required_text(value.get("key"), f"{context}.key")
        try:
            grade = int(value.get("grade", 1))
        except (TypeError, ValueError) as exc:
            raise DatasetError(f"{context}.grade must be an integer") from exc
        if grade not in (1, 2):
            raise DatasetError(f"{context}.grade must be 1 or 2")
        return cls(key=key, grade=grade)


@dataclass(frozen=True)
class KeyFact:
    """One auditable fact that a correct answer should cover."""

    id: str
    text: str
    source_documents: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Any, *, context: str) -> KeyFact:
        if not isinstance(value, dict):
            raise DatasetError(f"{context}: key fact must be an object")
        fact_id = _required_text(value.get("id"), f"{context}.id")
        text = _required_text(value.get("text"), f"{context}.text")
        source_documents = _text_list(value.get("source_documents"), f"{context}.source_documents")
        if not source_documents:
            raise DatasetError(f"{context}.source_documents must not be empty")
        return cls(id=fact_id, text=text, source_documents=tuple(source_documents))


@dataclass(frozen=True)
class AgentExpectation:
    """A tolerant agent policy oracle.

    Exact tool sequences are intentionally avoided: several valid ReAct plans
    can solve the same task.  The labels describe required capabilities,
    prohibited tools, and the expected terminal behavior instead.
    """

    required_any_tools: tuple[str, ...] = ()
    required_all_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    expected_outcome: str = "answer"  # answer | refuse
    permission_outcome: str | None = None  # denied | confirmation_required

    @classmethod
    def from_dict(cls, value: Any, *, context: str) -> AgentExpectation | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise DatasetError(f"{context}: agent expectation must be an object")
        outcome = str(value.get("expected_outcome", "answer")).strip()
        if outcome not in {"answer", "refuse"}:
            raise DatasetError(f"{context}.expected_outcome must be 'answer' or 'refuse'")
        permission = value.get("permission_outcome")
        if permission is not None:
            permission = str(permission).strip()
            if permission not in {"denied", "confirmation_required"}:
                raise DatasetError(
                    f"{context}.permission_outcome must be 'denied' or 'confirmation_required'"
                )
        return cls(
            required_any_tools=tuple(_text_list(value.get("required_any_tools", []), context)),
            required_all_tools=tuple(_text_list(value.get("required_all_tools", []), context)),
            forbidden_tools=tuple(_text_list(value.get("forbidden_tools", []), context)),
            expected_outcome=outcome,
            permission_outcome=permission,
        )


@dataclass(frozen=True)
class EvalCase:
    """A question plus retrieval, answer and agent ground truth."""

    id: str
    question: str
    answerable: bool
    relevant_documents: tuple[RelevantDocument, ...] = ()
    key_facts: tuple[KeyFact, ...] = ()
    expected_answer: str | None = None
    agent: AgentExpectation | None = None
    tags: tuple[str, ...] = ()
    split: str = ""

    @classmethod
    def from_dict(cls, value: Any, *, context: str, split: str) -> EvalCase:
        if not isinstance(value, dict):
            raise DatasetError(f"{context}: each JSONL row must be an object")
        case_id = _required_text(value.get("id"), f"{context}.id")
        question = _required_text(value.get("question"), f"{context}.question")
        answerable = value.get("answerable")
        if not isinstance(answerable, bool):
            raise DatasetError(f"{context}.answerable must be true or false")
        raw_relevance = value.get("relevant_documents", [])
        if not isinstance(raw_relevance, list):
            raise DatasetError(f"{context}.relevant_documents must be an array")
        relevant = tuple(
            RelevantDocument.from_dict(item, context=f"{context}.relevant_documents[{index}]")
            for index, item in enumerate(raw_relevance)
        )
        _ensure_unique((item.key for item in relevant), f"{context}.relevant_documents")
        raw_facts = value.get("key_facts", [])
        if not isinstance(raw_facts, list):
            raise DatasetError(f"{context}.key_facts must be an array")
        facts = tuple(
            KeyFact.from_dict(item, context=f"{context}.key_facts[{index}]")
            for index, item in enumerate(raw_facts)
        )
        _ensure_unique((item.id for item in facts), f"{context}.key_facts")
        if answerable and not relevant:
            raise DatasetError(f"{context}: answerable cases need relevant_documents")
        if not answerable and (relevant or facts):
            raise DatasetError(f"{context}: unanswerable cases cannot have evidence labels")
        expected_answer = value.get("expected_answer")
        if expected_answer is not None:
            expected_answer = _required_text(expected_answer, f"{context}.expected_answer")
        return cls(
            id=case_id,
            question=question,
            answerable=answerable,
            relevant_documents=relevant,
            key_facts=facts,
            expected_answer=expected_answer,
            agent=AgentExpectation.from_dict(value.get("agent"), context=f"{context}.agent"),
            tags=tuple(_text_list(value.get("tags", []), f"{context}.tags")),
            split=split,
        )

    @property
    def relevance_by_document(self) -> dict[str, int]:
        return {item.key: item.grade for item in self.relevant_documents}


@dataclass(frozen=True)
class BenchmarkMeta:
    id: str
    title: str
    version: str
    corpus_manifest: str
    default_top_k: int = 6
    language: str = "en"
    description: str = ""

    @classmethod
    def load(cls, path: Path) -> BenchmarkMeta:
        raw = _load_json(path)
        if not isinstance(raw, dict):
            raise DatasetError(f"{path}: benchmark metadata must be an object")
        try:
            top_k = int(raw.get("default_top_k", 6))
        except (TypeError, ValueError) as exc:
            raise DatasetError(f"{path}: default_top_k must be an integer") from exc
        if top_k < 1:
            raise DatasetError(f"{path}: default_top_k must be positive")
        return cls(
            id=_required_text(raw.get("id"), f"{path}.id"),
            title=_required_text(raw.get("title"), f"{path}.title"),
            version=_required_text(raw.get("version"), f"{path}.version"),
            corpus_manifest=_required_text(raw.get("corpus_manifest"), f"{path}.corpus_manifest"),
            default_top_k=top_k,
            language=str(raw.get("language", "en")).strip() or "en",
            description=str(raw.get("description", "")).strip(),
        )


@dataclass(frozen=True)
class EvalDataset:
    """All loaded benchmark splits under one immutable directory."""

    root: Path
    meta: BenchmarkMeta
    cases: tuple[EvalCase, ...]

    @classmethod
    def load(cls, root: str | Path, *, splits: Iterable[str] = ("dev", "test")) -> EvalDataset:
        root_path = Path(root).resolve()
        meta = BenchmarkMeta.load(root_path / "benchmark.json")
        requested = tuple(dict.fromkeys(str(split).strip() for split in splits if str(split).strip()))
        if not requested:
            raise DatasetError("at least one split is required")
        cases: list[EvalCase] = []
        seen_ids: set[str] = set()
        for split in requested:
            path = root_path / f"questions.{split}.jsonl"
            for line_no, raw in _load_jsonl(path):
                case = EvalCase.from_dict(raw, context=f"{path}:{line_no}", split=split)
                if case.id in seen_ids:
                    raise DatasetError(f"duplicate case id across requested splits: {case.id}")
                seen_ids.add(case.id)
                cases.append(case)
        return cls(root=root_path, meta=meta, cases=tuple(cases))

    def for_split(self, split: str) -> tuple[EvalCase, ...]:
        return tuple(case for case in self.cases if case.split == split)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text("utf-8"))
    except FileNotFoundError as exc:
        raise DatasetError(f"benchmark file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(f"invalid JSON in {path}: {exc.msg}") from exc


def _load_jsonl(path: Path) -> Iterable[tuple[int, Any]]:
    try:
        lines = path.read_text("utf-8").splitlines()
    except FileNotFoundError as exc:
        raise DatasetError(f"benchmark split not found: {path}") from exc
    for line_no, line in enumerate(lines, start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            yield line_no, json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"invalid JSONL in {path}:{line_no}: {exc.msg}") from exc


def _required_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{context} must be a non-empty string")
    return value.strip()


def _text_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list):
        raise DatasetError(f"{context} must be an array of strings")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_required_text(item, f"{context}[{index}]"))
    _ensure_unique(result, context)
    return result


def _ensure_unique(values: Iterable[str], context: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise DatasetError(f"{context} contains duplicate values: {', '.join(sorted(duplicates))}")
