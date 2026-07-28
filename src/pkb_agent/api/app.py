"""FastAPI + SSE façade for the PKB visual workbench."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pkb_agent.api.schemas import ApprovalDecisionRequest, StartRunRequest
from pkb_agent.api.sessions import RunSession, WorkbenchStore
from pkb_agent.app.settings import Settings, get_settings
from pkb_agent.rag.embeddings import EmbeddingProvider
from pkb_agent.rag.ingestion import IngestionPipeline
from pkb_agent.storage.repositories.chunks import ChunksRepository
from pkb_agent.storage.repositories.documents import DocumentsRepository
from pkb_agent.storage.repositories.traces import TracesRepository
from pkb_agent.storage.supabase_client import SupabaseClient

_MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an API application without assembling the Agent Runtime eagerly."""
    runtime_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = WorkbenchStore(runtime_settings)
        app.state.workbench = store
        yield
        await store.close()

    app = FastAPI(
        title="PKB-Agent Workbench API",
        version="0.1.0",
        description="SSE workbench boundary around the permissioned PKB-Agent runtime.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.api_allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "runtime": "python", "stream": "sse"}

    @app.get("/api/knowledge-base")
    async def knowledge_base(user_id: str = "default") -> dict[str, Any]:
        documents = await _list_documents(runtime_settings, user_id=user_id, limit=100)
        return {
            "id": user_id,
            "name": "默认知识库" if user_id == "default" else f"知识库 · {user_id}",
            "document_count": len(documents),
            "chunk_count": sum(int(item.get("chunk_count") or 0) for item in documents),
            "status": "ready",
        }

    @app.get("/api/documents")
    async def list_documents(
        user_id: str = "default", limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        documents = await _list_documents(
            runtime_settings,
            user_id=user_id,
            limit=min(max(limit, 1), 100),
            offset=max(offset, 0),
        )
        return {"items": documents, "count": len(documents)}

    @app.post("/api/documents/upload", status_code=status.HTTP_201_CREATED)
    async def upload_document(
        file: UploadFile = File(...),  # noqa: B008 - FastAPI request contract
        user_id: str = Form("default"),
        title: str | None = Form(None),
    ) -> dict[str, Any]:
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="上传文件为空")
        if len(payload) > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="单个文件不能超过 25 MB")
        try:
            runtime_settings.require_supabase()
            runtime_settings.require_embedding()
            with _supabase_client(runtime_settings) as supabase:
                documents_repo = DocumentsRepository(supabase)
                chunks_repo = ChunksRepository(supabase)
                embedder = EmbeddingProvider.from_settings(runtime_settings)
                pipeline = IngestionPipeline.from_settings(
                    runtime_settings,
                    documents_repo,
                    chunks_repo,
                    embedder,
                )
                try:
                    document = await pipeline.ingest_upload(
                        payload,
                        filename=file.filename,
                        title=title or None,
                        user_id=user_id,
                        meta={
                            "source_status": "ready",
                            "content_type": file.content_type or "application/octet-stream",
                        },
                    )
                finally:
                    await embedder.close()
        except HTTPException:
            raise
        except Exception as exc:
            raise _service_error(exc) from exc
        return {"document": _serialize_document(document)}

    @app.get("/api/documents/{document_id}/chunks")
    async def document_chunks(document_id: str) -> dict[str, Any]:
        try:
            with _chunks_repo(runtime_settings) as repo:
                rows = await asyncio.to_thread(repo.list_by_document, document_id)
        except Exception as exc:
            raise _service_error(exc) from exc
        return {"items": rows, "count": len(rows)}

    @app.get("/api/chunks/{chunk_id}")
    async def get_chunk(chunk_id: str) -> dict[str, Any]:
        try:
            with _chunks_repo(runtime_settings) as repo:
                chunk = await asyncio.to_thread(repo.get_chunk_with_doc, chunk_id)
        except Exception as exc:
            raise _service_error(exc) from exc
        if not chunk:
            raise HTTPException(status_code=404, detail="未找到原始 chunk")
        return {"chunk": chunk}

    @app.post("/api/runs", status_code=status.HTTP_202_ACCEPTED)
    async def start_run(payload: StartRunRequest, request: Request) -> dict[str, Any]:
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="问题不能为空")
        store = _store(request)
        session = await store.start_run(question=question, user_id=payload.user_id)
        return {
            **session.public_view(),
            "events_url": f"/api/runs/{session.run_id}/events",
        }

    @app.get("/api/runs")
    async def list_runs(user_id: str = "default", limit: int = 25) -> dict[str, Any]:
        try:
            with _traces_repo(runtime_settings) as repo:
                rows = await asyncio.to_thread(
                    repo.list_runs, user_id, min(max(limit, 1), 100), 0
                )
        except Exception as exc:
            raise _service_error(exc) from exc
        return {"items": [_serialize_run(row) for row in rows], "count": len(rows)}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str, request: Request) -> dict[str, Any]:
        session = await _store(request).get(run_id)
        if session is not None:
            return session.public_view()
        try:
            with _traces_repo(runtime_settings) as repo:
                run = await asyncio.to_thread(repo.get_run, run_id)
                if not run:
                    raise HTTPException(status_code=404, detail="未找到运行记录")
                steps, tool_calls = await asyncio.gather(
                    asyncio.to_thread(repo.list_steps, run_id),
                    asyncio.to_thread(repo.list_tool_calls, run_id),
                )
        except HTTPException:
            raise
        except Exception as exc:
            raise _service_error(exc) from exc
        return {
            "run_id": run_id,
            "status": run.get("status"),
            "snapshot": _serialize_historical_run(run, steps, tool_calls),
        }

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request) -> StreamingResponse:
        session = await _store(request).get(run_id)
        if session is None:
            raise HTTPException(status_code=404, detail="该运行不在当前工作台进程中")
        return StreamingResponse(
            _event_stream(request, session),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/runs/{run_id}/approvals/{approval_id}")
    async def decide_approval(
        run_id: str,
        approval_id: str,
        payload: ApprovalDecisionRequest,
        request: Request,
    ) -> dict[str, Any]:
        accepted = await _store(request).approve(
            run_id=run_id,
            approval_id=approval_id,
            approved=payload.approved,
        )
        if not accepted:
            raise HTTPException(status_code=404, detail="审批已处理、超时或不属于本次运行")
        return {"approval_id": approval_id, "approved": payload.approved, "accepted": True}

    @app.get("/api/metrics")
    async def metrics(request: Request, user_id: str = "default") -> dict[str, Any]:
        """Return a compact rolling success/cost view plus local live snapshots."""
        rows: list[dict[str, Any]] = []
        try:
            with _traces_repo(runtime_settings) as repo:
                rows = await asyncio.to_thread(repo.list_runs, user_id, 100, 0)
        except Exception:
            # The live workbench remains usable while a storage outage is being
            # diagnosed; surface the currently held runs below.
            rows = []

        persisted = [_serialize_run(row) for row in rows]
        successes = [item for item in persisted if item["metrics"]["run_succeeded"]]
        estimated_cost = sum(_usage_cost(item["metrics"]["usage"]) for item in persisted)
        live = await _store(request).recent_snapshots(user_id=user_id)
        return {
            "run_count": len(persisted),
            "success_rate": round(len(successes) / len(persisted) * 100, 1) if persisted else None,
            "estimated_cost": round(estimated_cost, 8),
            "cost_currency": runtime_settings.observability_currency.upper(),
            "live_runs": live,
        }

    return app


def _store(request: Request) -> WorkbenchStore:
    return request.app.state.workbench


async def _list_documents(
    settings: Settings, *, user_id: str, limit: int, offset: int = 0
) -> list[dict[str, Any]]:
    try:
        with _documents_repo(settings) as repo:
            rows = await asyncio.to_thread(repo.list, user_id, limit, offset)
    except Exception as exc:
        raise _service_error(exc) from exc
    return [_serialize_document(row) for row in rows]


@contextmanager
def _supabase_client(settings: Settings) -> Iterator[SupabaseClient]:
    settings.require_supabase()
    client = SupabaseClient.from_settings(settings)
    try:
        yield client
    finally:
        client.close()


@contextmanager
def _documents_repo(settings: Settings) -> Iterator[DocumentsRepository]:
    with _supabase_client(settings) as client:
        yield DocumentsRepository(client)


@contextmanager
def _chunks_repo(settings: Settings) -> Iterator[ChunksRepository]:
    with _supabase_client(settings) as client:
        yield ChunksRepository(client)


@contextmanager
def _traces_repo(settings: Settings) -> Iterator[TracesRepository]:
    with _supabase_client(settings) as client:
        yield TracesRepository(client)


def _serialize_document(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    stale_days = _stale_days(meta)
    source_status = "expired" if stale_days is not None else str(meta.get("source_status") or "ready")
    return {
        **row,
        "version": f"v{str(row.get('content_hash') or 'pending')[:8]}",
        "source_status": source_status,
        "source_label": _source_label(row, meta, stale_days=stale_days),
    }


def _source_label(row: dict[str, Any], meta: dict[str, Any], *, stale_days: int | None) -> str:
    if stale_days is not None:
        return f"该网页已过期 {stale_days} 天"
    if source := meta.get("freshness_label"):
        return str(source)
    source_type = str(row.get("source_type") or "file")
    if source_type == "url":
        return "网页来源"
    if source_type == "text":
        return "手动资料"
    return "已上传"


def _stale_days(meta: dict[str, Any]) -> int | None:
    raw = meta.get("expires_at")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        expires_at = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None
    elapsed = (datetime.now(UTC) - expires_at).total_seconds()
    if elapsed < 0:
        return None
    return max(1, int(elapsed // 86_400))


def _serialize_run(row: dict[str, Any]) -> dict[str, Any]:
    usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
    verification = row.get("verification") if isinstance(row.get("verification"), dict) else {}
    status_value = str(row.get("status") or "unknown")
    run_succeeded = status_value == "finished" and verification.get("status") == "verified"
    return {
        "run_id": row.get("id"),
        "question": row.get("question"),
        "status": status_value,
        "created_at": row.get("created_at"),
        "step_count": row.get("step_count", 0),
        "metrics": {
            "duration_ms": usage.get("duration_ms"),
            "tool_duration_ms": usage.get("tool_duration_ms"),
            "tool_call_count": usage.get("tool_call_count") or 0,
            "tool_success_rate": usage.get("tool_success_rate"),
            "run_succeeded": run_succeeded,
            "run_success_rate": 100.0 if run_succeeded else 0.0,
            "usage": usage,
        },
    }


def _usage_cost(usage: dict[str, Any]) -> float:
    """Read new generic cost records and pre-workbench USD records safely."""
    try:
        return float(usage.get("estimated_cost", usage.get("estimated_cost_usd", 0)) or 0)
    except (TypeError, ValueError):
        return 0.0


def _serialize_historical_run(
    run: dict[str, Any], steps: list[dict[str, Any]], tool_calls: list[dict[str, Any]]
) -> dict[str, Any]:
    payload = _serialize_run(run)
    payload.update(
        {
            "final_answer": run.get("final_answer"),
            "answer": run.get("answer_payload"),
            "verification": run.get("verification") or {},
            "error": run.get("error"),
            "steps": steps,
            "tool_calls": tool_calls,
        }
    )
    return payload


async def _event_stream(request: Request, session: RunSession) -> AsyncIterator[str]:
    queue = await session.subscribe()
    yield "retry: 2000\n\n"
    try:
        while True:
            if await request.is_disconnected():
                return
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15)
            except TimeoutError:
                yield ": keep-alive\n\n"
                continue
            encoded = json.dumps(event, ensure_ascii=False, default=str)
            yield f"event: message\ndata: {encoded}\n\n"
            if event.get("type") == "run_finished":
                return
    finally:
        await session.unsubscribe(queue)


def _service_error(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"知识库服务暂不可用: {type(exc).__name__}",
    )
