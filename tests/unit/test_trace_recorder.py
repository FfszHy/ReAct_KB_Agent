from __future__ import annotations

from pkb_agent.agent.errors import StorageError
from pkb_agent.trace.recorder import TraceRecorder


class _TransientTraceRepository:
    def __init__(self, failures_before_success: int) -> None:
        self.failures_before_success = failures_before_success
        self.calls = 0

    def create_run(self, **_kwargs) -> dict:
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise StorageError("create_run failed: The read operation timed out")
        return {"id": "run-1"}


class _PermanentTraceRepository:
    def __init__(self) -> None:
        self.calls = 0

    def create_run(self, **_kwargs) -> dict:
        self.calls += 1
        raise StorageError("create_run failed: invalid input syntax for type uuid")


async def test_trace_recorder_retries_transient_storage_errors_with_a_bounded_policy():
    repo = _TransientTraceRepository(failures_before_success=2)
    recorder = TraceRecorder(repo, retry_attempts=3, retry_backoff_seconds=0)  # type: ignore[arg-type]

    persisted = await recorder.start_run(run_id="run-1", question="test")

    assert persisted is True
    assert repo.calls == 3


async def test_trace_recorder_reports_a_nontransient_write_failure_without_retrying():
    repo = _PermanentTraceRepository()
    recorder = TraceRecorder(repo, retry_attempts=3, retry_backoff_seconds=0)  # type: ignore[arg-type]

    persisted = await recorder.start_run(run_id="run-1", question="test")

    assert persisted is False
    assert repo.calls == 1
