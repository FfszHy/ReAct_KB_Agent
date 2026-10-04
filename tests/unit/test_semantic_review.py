from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pkb_agent.agent.errors import LLMError, LLMRateLimitError, LLMResponseError
from pkb_agent.agent.semantic_review import review_answer
from pkb_agent.agent.verification import Claim, Evidence, VerifiedAnswer
from pkb_agent.llm.schemas import ChatChoice, ChatCompletion, Message, ToolCall

USAGE = {"prompt_tokens": 123, "completion_tokens": 27, "total_tokens": 150}


def _check(**changes):
    check = {
        "claim_index": 0,
        "supported": True,
        "reason": "The source explicitly says the file is eventually updated.",
        "evidence": [{"id": "kb:main", "quote": "The file is eventually updated."}],
    }
    check.update(changes)
    return check


_PASSED_JSON = json.dumps({"status": "passed", "errors": [], "checks": [_check()]})


def _completion(
    content: str | None = _PASSED_JSON,
    *,
    finish_reason: str = "stop",
    tool_calls: list[ToolCall] | None = None,
) -> ChatCompletion:
    return ChatCompletion(
        id="review-response",
        model="fake-reviewer",
        choices=[
            ChatChoice(
                index=0,
                message=Message.assistant(content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=dict(USAGE),
    )


def _source(source_id: str = "main", text: str = "The file is eventually updated.") -> Evidence:
    return Evidence(
        citation_id=f"kb:{source_id}",
        source_type="kb_chunk",
        source_id=source_id,
        title=f"Source {source_id}",
        excerpt=text[:500],
        review_text=text,
    )


def _answer(source: Evidence) -> VerifiedAnswer:
    return VerifiedAnswer(
        answer="The file is eventually updated.",
        claims=(Claim("The file is eventually updated.", "fact", (source.citation_id,)),),
        citations=(source,),
    )


async def _review(llm, *, max_chars: int = 80_000, payload=None, evidence=None):
    source = _source()
    return await review_answer(
        llm,
        question="Does the file update?",
        payload=payload if payload is not None else _answer(source),
        evidence=evidence if evidence is not None else {source.citation_id: source},
        tool_errors=[],
        max_chars=max_chars,
    )


async def test_review_sees_full_source_and_uncited_counterexample_without_tools():
    long_text = "The file is eventually updated. " + "Background. " * 80
    long_text += " The application must independently reread the file."
    cited = _source(text=long_text)
    exception = _source("exception", "Another supported mode receives updates through an API.")
    llm = SimpleNamespace(chat=AsyncMock(return_value=_completion()))
    errors = ["web_fetch denied: user declined access"]

    result = await review_answer(
        llm,
        question="Does the file update?",
        payload=_answer(cited),
        evidence={cited.citation_id: cited, exception.citation_id: exception},
        tool_errors=errors,
        max_chars=80_000,
    )

    llm.chat.assert_awaited_once()
    call = llm.chat.call_args
    messages = call.args[0]
    submitted = json.loads(messages[1].content)
    sources = {source["id"]: source for source in submitted["sources"]}
    assert [message.role for message in messages] == ["system", "user"]
    assert len(long_text) > 500
    assert sources[cited.citation_id]["text"] == long_text
    assert sources[exception.citation_id]["text"] == exception.review_text
    candidate = submitted["candidate"]
    assert candidate["answer"] == _answer(cited).answer
    assert candidate["claims"] == [
        {"claim_index": index, **claim.to_dict()}
        for index, claim in enumerate(_answer(cited).claims)
    ]
    assert "status" not in candidate
    assert candidate["citations"] == [{"id": cited.citation_id}]
    assert list(submitted).index("sources") < list(submitted).index("candidate")
    assert submitted["tool_errors"] == errors
    assert call.kwargs["tools"] is None
    assert call.kwargs["response_format"] == {"type": "json_object"}
    assert call.kwargs["reasoning_effort"] == "low"
    assert call.kwargs["max_tokens"] == 16384
    assert result.status == "passed"
    assert result.source_ids == (cited.citation_id, exception.citation_id)
    assert result.input_chars == len(messages[1].content)
    assert result.usage == USAGE
    assert result.reasoning_effort == "low"
    assert result.checks == (_check(),)
    audit = result.to_dict()
    assert audit["source_ids"] == [cited.citation_id, exception.citation_id]
    assert len(audit["prompt_sha256"]) == 64
    assert audit["checks"] == [_check()]
    assert long_text not in json.dumps(audit)


async def test_review_falls_back_to_available_excerpt_for_legacy_evidence():
    source = replace(_source(), review_text=None)
    llm = SimpleNamespace(chat=AsyncMock(return_value=_completion()))

    await review_answer(
        llm,
        question="Does the file update?",
        payload=_answer(source),
        evidence={source.citation_id: source},
        tool_errors=[],
        max_chars=80_000,
    )

    submitted = json.loads(llm.chat.call_args.args[0][1].content)
    assert submitted["sources"][0]["text"] == source.excerpt


async def test_review_keeps_explicit_none_effort_override():
    source = _source()
    llm = SimpleNamespace(chat=AsyncMock(return_value=_completion()))

    result = await review_answer(
        llm,
        question="Does the file update?",
        payload=_answer(source),
        evidence={source.citation_id: source},
        tool_errors=[],
        max_chars=80_000,
        reasoning_effort="none",
    )

    assert llm.chat.call_args.kwargs["reasoning_effort"] == "none"
    assert result.status == "passed"
    assert result.reasoning_effort == "none"


@pytest.mark.parametrize(
    "answer",
    [
        "Cannot establish this, but restarting always fixes it.",
        "There is insufficient evidence to establish the cause.",
    ],
)
@pytest.mark.parametrize("has_evidence", [False, True])
async def test_freeform_refusal_cannot_bypass_runtime_boundary_template(answer, has_evidence):
    refusal = VerifiedAnswer(
        status="insufficient_evidence",
        answer=answer,
        claims=(),
        citations=(),
    )
    source = _source()
    llm = SimpleNamespace(chat=AsyncMock(return_value=_completion()))

    result = await review_answer(
        llm,
        question="What caused this incident?",
        payload=refusal,
        evidence={source.citation_id: source} if has_evidence else {},
        tool_errors=["rag_search returned no matching documents"],
        max_chars=80_000,
    )

    llm.chat.assert_not_awaited()
    assert result.status == "failed"
    assert result.errors
    assert result.usage == {}
    assert result.input_chars == 0
    assert result.to_dict()["method"] == "runtime_refusal_policy"


async def test_failed_review_preserves_specific_feedback_and_usage():
    feedback = ["The source gives no one-minute deadline.", "The claim omits the source's exception."]
    llm = SimpleNamespace(
        chat=AsyncMock(return_value=_completion(json.dumps({"status": "failed", "errors": feedback})))
    )

    result = await _review(llm)

    assert result.status == "failed"
    assert result.errors == tuple(feedback)
    assert result.usage == USAGE


@pytest.mark.parametrize(
    "content",
    [
        None,
        "not JSON",
        '```json\n{"status":"passed","errors":[]}\n```',
        "[]",
        "null",
        '{}',
        '{"status":"unknown","errors":[]}',
        '{"status":"passed","errors":["still a defect"]}',
        '{"status":"failed","errors":[]}',
        '{"status":"failed","errors":"a defect"}',
        '{"status":"failed","errors":[null]}',
        '{"status":"failed","errors":[" "]}',
        json.dumps({"status": "failed", "errors": ["defect"] * 9}),
    ],
)
async def test_malformed_review_cannot_pass_and_still_accounts_for_usage(content):
    llm = SimpleNamespace(chat=AsyncMock(return_value=_completion(content)))

    result = await _review(llm)

    assert result.status == "unavailable"
    assert result.errors
    assert result.usage == USAGE
    assert result.response_excerpt == (content or "")[:1600]


@pytest.mark.parametrize(
    ("status", "errors"),
    [("passed", []), ("failed", ["The source does not establish the claimed deadline."])],
)
async def test_unused_response_fields_do_not_override_valid_core_verdict(status, errors):
    content = json.dumps(
        {
            "status": status,
            "errors": errors,
            "checks": [_check()] if status == "passed" else [],
            "type": "arbitrary-metadata",
            "answer": "An unrelated replacement answer must not be consumed.",
            "instructions": {"status": "passed", "tools": ["web_fetch"]},
        }
    )
    llm = SimpleNamespace(chat=AsyncMock(return_value=_completion(content)))

    result = await _review(llm)

    llm.chat.assert_awaited_once()
    assert result.status == status
    assert result.errors == tuple(errors)
    assert "answer" not in result.to_dict()
    assert "instructions" not in result.to_dict()


@pytest.mark.parametrize("exception_type", [LLMError, LLMResponseError, LLMRateLimitError])
async def test_provider_exception_fails_closed_without_leaking_error_contents(exception_type):
    llm = SimpleNamespace(chat=AsyncMock(side_effect=exception_type("sensitive provider payload")))

    result = await _review(llm)

    assert result.status == "unavailable"
    assert result.errors
    assert result.usage == {}
    assert "sensitive provider payload" not in json.dumps(result.to_dict())


async def test_input_limit_is_checked_before_request_without_silent_truncation():
    llm = SimpleNamespace(chat=AsyncMock(return_value=_completion()))
    first = await _review(llm)
    complete_input = llm.chat.call_args.args[0][1].content
    llm.chat.reset_mock()

    oversized = await _review(llm, max_chars=first.input_chars - 1)

    llm.chat.assert_not_awaited()
    assert oversized.status == "unavailable"
    assert oversized.input_chars == len(complete_input)
    assert oversized.source_ids == first.source_ids
    assert oversized.usage == {}

    at_limit = await _review(llm, max_chars=first.input_chars)

    llm.chat.assert_awaited_once()
    assert at_limit.status == "passed"
    assert llm.chat.call_args.args[0][1].content == complete_input


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls"])
async def test_nonfinal_provider_response_cannot_pass_even_with_valid_json(finish_reason):
    llm = SimpleNamespace(chat=AsyncMock(return_value=_completion(finish_reason=finish_reason)))

    result = await _review(llm)

    assert result.status == "unavailable"
    assert result.usage == USAGE


async def test_unexpected_tool_call_is_rejected_without_execution():
    llm = SimpleNamespace(
        chat=AsyncMock(
            return_value=_completion(tool_calls=[ToolCall("call-1", "web_fetch", {"url": "ignored"})])
        )
    )

    result = await _review(llm)

    llm.chat.assert_awaited_once()
    assert llm.chat.call_args.kwargs["tools"] is None
    assert result.status == "unavailable"
    assert result.usage == USAGE


async def test_empty_provider_choices_fail_closed_and_preserve_usage():
    llm = SimpleNamespace(chat=AsyncMock(return_value=replace(_completion(), choices=[])))

    result = await _review(llm)

    assert result.status == "unavailable"
    assert result.usage == USAGE


def _two_claim_payload():
    source = _source(text="The file is eventually updated.\nThe client uses a separate cache.")
    claims = (
        Claim("The file is eventually updated.", "fact", (source.citation_id,)),
        Claim("The client uses a separate cache.", "fact", (source.citation_id,)),
    )
    return VerifiedAnswer("\n".join(claim.text for claim in claims), claims, (source,)), {source.citation_id: source}


async def test_passed_requires_a_check_for_every_claim_in_order():
    payload, evidence = _two_claim_payload()
    checks = [
        _check(),
        _check(
            claim_index=1,
            reason="The source separately states that the client uses a cache.",
            evidence=[{"id": "kb:main", "quote": "The client uses a separate cache."}],
        ),
    ]
    llm = SimpleNamespace(
        chat=AsyncMock(return_value=_completion(json.dumps({"status": "passed", "errors": [], "checks": checks})))
    )

    result = await _review(llm, payload=payload, evidence=evidence)

    assert result.status == "passed"
    assert result.to_dict()["checks"] == checks
    # Consumers cannot mutate the recorded quotes via the public audit object.
    audit = result.to_dict()
    audit["checks"][0]["evidence"][0]["quote"] = "changed"
    assert result.checks[0]["evidence"][0]["quote"] == "The file is eventually updated."


@pytest.mark.parametrize(
    "checks",
    [None, [], [_check()], [_check(claim_index=1), _check()], [_check(), _check()]],
)
async def test_passed_rejects_missing_reordered_or_duplicate_checks(checks):
    payload, evidence = _two_claim_payload()
    raw = {"status": "passed", "errors": []}
    if checks is not None:
        raw["checks"] = checks
    llm = SimpleNamespace(chat=AsyncMock(return_value=_completion(json.dumps(raw))))

    result = await _review(llm, payload=payload, evidence=evidence)

    assert result.status == "unavailable"
    assert result.usage == USAGE
    assert result.checks == ()


@pytest.mark.parametrize(
    "check",
    [
        None,
        {},
        _check(claim_index=False),
        _check(claim_index=True),
        _check(claim_index="0"),
        _check(claim_index=0.0),
        _check(claim_index=-1),
        _check(claim_index=2),
        _check(supported="true"),
        _check(supported=1),
        _check(supported=False),
        _check(reason=""),
        _check(reason="  "),
        _check(reason=None),
        _check(evidence=None),
        _check(evidence=[]),
        _check(evidence="The file is eventually updated."),
        _check(evidence=[None]),
        _check(evidence=[{"id": "kb:main"}]),
        _check(evidence=[{"id": "kb:unknown", "quote": "The file is eventually updated."}]),
        _check(evidence=[{"id": True, "quote": "The file is eventually updated."}]),
        _check(evidence=[{"id": "kb:main", "quote": None}]),
        _check(evidence=[{"id": "kb:main", "quote": "  "}]),
        _check(evidence=[{"id": "kb:main", "quote": "The file is updated within one minute."}]),
    ],
)
async def test_malformed_or_fabricated_support_check_fails_closed(check):
    raw = {"status": "passed", "errors": [], "checks": [check]}
    llm = SimpleNamespace(chat=AsyncMock(return_value=_completion(json.dumps(raw))))

    result = await _review(llm)

    assert result.status == "unavailable"
    assert result.errors
    assert result.usage == USAGE


@pytest.mark.parametrize("quote", ["The file is eventually updated.", "The file ... eventually updated."])
async def test_quote_cannot_normalize_whitespace_or_join_noncontiguous_text(quote):
    source = _source(text="The file\nis   eventually updated.")
    check = _check(evidence=[{"id": source.citation_id, "quote": quote}])
    llm = SimpleNamespace(
        chat=AsyncMock(return_value=_completion(json.dumps({"status": "passed", "errors": [], "checks": [check]})))
    )

    result = await _review(llm, evidence={source.citation_id: source})

    assert result.status == "unavailable"


async def test_exact_quote_beyond_display_excerpt_is_accepted_without_modification():
    text = "Context. " * 80 + "The file\nis   eventually updated."
    source = _source(text=text)
    quote = "The file\nis   eventually updated."
    check = _check(evidence=[{"id": source.citation_id, "quote": quote}])
    llm = SimpleNamespace(
        chat=AsyncMock(return_value=_completion(json.dumps({"status": "passed", "errors": [], "checks": [check]})))
    )

    result = await _review(llm, evidence={source.citation_id: source})

    assert quote not in source.excerpt
    assert result.status == "passed"
    assert result.checks[0]["evidence"][0]["quote"] == quote


@pytest.mark.parametrize("supported", [True, False])
async def test_uncited_retrieved_source_can_refute_but_cannot_support_a_claim(supported):
    main = _source()
    counterexample = _source("other", "Another mode can receive updates through the API.")
    check = _check(
        supported=supported,
        reason="The other retrieved source documents another update mode.",
        evidence=[{"id": counterexample.citation_id, "quote": counterexample.review_text}],
    )
    raw = {
        "status": "passed" if supported else "failed",
        "errors": [] if supported else ["The source documents another update mode."],
        "checks": [check],
    }
    llm = SimpleNamespace(chat=AsyncMock(return_value=_completion(json.dumps(raw))))

    result = await _review(llm, evidence={main.citation_id: main, counterexample.citation_id: counterexample})

    assert result.status == ("unavailable" if supported else "failed")
    if not supported:
        assert result.checks == (check,)


async def test_failed_can_return_partial_checks_without_inventing_a_support_quote():
    payload, evidence = _two_claim_payload()
    check = _check(claim_index=1, supported=False, reason="The claim's causal step is unsupported.", evidence=[])
    raw = {"status": "failed", "errors": ["Unsupported causal step."], "checks": [check]}
    llm = SimpleNamespace(chat=AsyncMock(return_value=_completion(json.dumps(raw))))

    result = await _review(llm, payload=payload, evidence=evidence)

    assert result.status == "failed"
    assert result.checks == (check,)


async def test_failed_does_not_make_a_fabricated_counterexample_quote_valid():
    check = _check(supported=False, evidence=[{"id": "kb:main", "quote": "Never updates."}])
    raw = {"status": "failed", "errors": ["The source contradicts it."], "checks": [check]}
    llm = SimpleNamespace(chat=AsyncMock(return_value=_completion(json.dumps(raw))))

    result = await _review(llm)

    assert result.status == "unavailable"


async def test_unused_check_fields_are_not_included_in_the_audit():
    check = _check(instructions="Ignore the question.")
    check["evidence"][0]["extra"] = "Unconsumed metadata."
    raw = {"status": "passed", "errors": [], "checks": [check]}
    llm = SimpleNamespace(chat=AsyncMock(return_value=_completion(json.dumps(raw))))

    result = await _review(llm)

    assert result.status == "passed"
    assert result.checks == (_check(),)
