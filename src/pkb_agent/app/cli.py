"""Typer CLI for PKB-Agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from pkb_agent.agent.runtime import AgentRuntime
from pkb_agent.app.settings import get_settings

app = typer.Typer(
    name="pkb-agent",
    help="Supabase-backed ReAct personal knowledge base agent.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
eval_app = typer.Typer(
    help="Run reproducible retrieval, grounded-answer and agent evaluations.",
    no_args_is_help=True,
)
app.add_typer(eval_app, name="eval")


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------
@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask the agent."),
    user_id: str = typer.Option("default", "--user", "-u", help="User id scope."),
    json_output: bool = typer.Option(False, "--json", help="Emit final state as JSON."),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Override max steps."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve ask-permission tools."),
) -> None:
    """Ask the agent a question (ReAct loop)."""
    settings = get_settings()
    if max_steps is not None:
        settings.agent_max_steps = max_steps

    confirm = _make_confirm(yes)

    def on_event(event: dict[str, Any]) -> None:
        _render_event(event)

    async def _run() -> None:
        rt = AgentRuntime.build(settings, user_id=user_id, confirm=confirm)
        async with rt:
            state = await rt.run(question, on_event=on_event)
            if json_output:
                console.print_json(json.dumps(state.to_dict(), ensure_ascii=False))
            else:
                if state.answer_payload:
                    _render_verified_answer(state.answer_payload, state.verification)
                elif state.final_answer:
                    console.print(
                        Panel(
                            state.final_answer,
                            title="Answer",
                            border_style="green",
                        )
                    )
                if state.status.value != "finished":
                    console.print(f"[yellow]status:[/yellow] {state.status.value} ({state.error})")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
        raise typer.Exit(130) from None


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------
@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Recurse directories."),
    user_id: str = typer.Option("default", "--user", "-u"),
    title: str | None = typer.Option(None, "--title", help="Override document title."),
) -> None:
    """Ingest a file or directory into the knowledge base."""
    if not path.exists():
        console.print(f"[red]not found:[/red] {path}")
        raise typer.Exit(1)

    files = _collect_files(path, recursive)
    if not files:
        console.print("[yellow]no text files found[/yellow]")
        raise typer.Exit(0)

    async def _run() -> None:
        from pkb_agent.rag.embeddings import EmbeddingProvider
        from pkb_agent.rag.ingestion import IngestionPipeline
        from pkb_agent.storage.repositories.chunks import ChunksRepository
        from pkb_agent.storage.repositories.documents import DocumentsRepository
        from pkb_agent.storage.supabase_client import SupabaseClient

        settings = get_settings()
        settings.require_supabase()
        settings.require_embedding()
        sb = SupabaseClient.from_settings(settings)
        docs_repo = DocumentsRepository(sb)
        chunks_repo = ChunksRepository(sb)
        embedder = EmbeddingProvider.from_settings(settings)
        pipeline = IngestionPipeline.from_settings(settings, docs_repo, chunks_repo, embedder)
        try:
            for f in files:
                console.print(f"[cyan]ingesting[/cyan] {f}")
                try:
                    doc = await pipeline.ingest_file(
                        f,
                        title=title if len(files) == 1 else None,
                        user_id=user_id,
                    )
                    console.print(
                        f"  [green]ok[/green] id={doc.get('id')} "
                        f"chunks={doc.get('chunk_count')} title={doc.get('title')!r}"
                    )
                except Exception as e:
                    console.print(f"  [red]failed:[/red] {type(e).__name__}: {e}")
        finally:
            await embedder.close()
            sb.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
@app.command()
def doctor() -> None:
    """Check configuration and connectivity."""
    settings = get_settings()
    _check("DeepSeek API key", bool(settings.deepseek_api_key))
    _check("Supabase URL", bool(settings.supabase_url))
    _check("Supabase service role key", bool(settings.supabase_service_role_key))
    _check("Embedding API key", bool(settings.embedding_api_key))
    _check(
        f"Embedding model/dimensions ({settings.embedding_model}/{settings.embedding_dimensions})",
        True,
    )
    _check(
        f"Web search provider ({settings.web_search_provider})",
        bool(settings.web_search_api_key),
    )

    async def _probe() -> None:
        try:
            from pkb_agent.storage.supabase_client import SupabaseClient

            sb = SupabaseClient.from_settings(settings)
            try:
                sb.client.table("documents").select("id").limit(1).execute()
                _check("Supabase connectivity", True)
            finally:
                sb.close()
        except Exception as e:
            _check(f"Supabase connectivity ({e})", False)

    try:
        asyncio.run(_probe())
    except Exception as e:
        _check(f"Supabase connectivity ({e})", False)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
@eval_app.command("validate")
def eval_validate(
    dataset: Path = typer.Argument(
        Path("data/evals/fastapi-0.115"), help="Versioned benchmark directory."
    ),
) -> None:
    """Validate annotations and ensure every relevance label exists in the corpus manifest."""
    from pkb_agent.evaluation.corpus import CorpusManifest
    from pkb_agent.evaluation.dataset import EvalDataset

    benchmark = EvalDataset.load(dataset)
    manifest = CorpusManifest.load(benchmark.root / benchmark.meta.corpus_manifest)
    _validate_eval_document_keys(benchmark, manifest)
    split_counts = {split: len(benchmark.for_split(split)) for split in ("dev", "test")}
    answerable = sum(case.answerable for case in benchmark.cases)
    console.print(
        "[green]valid[/green] "
        f"{benchmark.meta.id}@{benchmark.meta.version} · "
        f"{len(benchmark.cases)} cases ({answerable} answerable, "
        f"dev={split_counts['dev']}, test={split_counts['test']}) · "
        f"{len(manifest.documents)} corpus documents pinned to {manifest.revision}"
    )


@eval_app.command("ingest-corpus")
def eval_ingest_corpus(
    dataset: Path = typer.Argument(
        Path("data/evals/fastapi-0.115"), help="Versioned benchmark directory."
    ),
    user_id: str = typer.Option("eval-fastapi-0.115", "--user", help="Dedicated KB user scope."),
    timeout_seconds: float = typer.Option(60.0, "--timeout", min=1.0, help="Per-source fetch timeout."),
    attempts: int = typer.Option(
        3, "--attempts", min=1, max=10, help="Maximum fetch attempts per source."
    ),
    concurrency: int = typer.Option(
        3, "--concurrency", min=1, max=10, help="Maximum concurrent corpus downloads."
    ),
    trust_env: bool = typer.Option(
        False,
        "--trust-env",
        help="Use HTTP(S)_PROXY and related environment settings for corpus downloads.",
    ),
    lock_file: Path | None = typer.Option(
        None, "--lock-file", help="Where to write the exact fetched-source hashes."
    ),
) -> None:
    """Fetch the pinned benchmark corpus and ingest it under stable eval:// URIs."""
    from pkb_agent.evaluation.corpus import (
        CorpusManifest,
        ingest_corpus,
        make_corpus_lock,
        write_json,
    )
    from pkb_agent.evaluation.dataset import EvalDataset

    benchmark = EvalDataset.load(dataset)
    manifest = CorpusManifest.load(benchmark.root / benchmark.meta.corpus_manifest)
    _validate_eval_document_keys(benchmark, manifest)
    settings = get_settings()

    async def _run() -> tuple[dict[str, Any], ...]:
        return await ingest_corpus(
            manifest,
            settings,
            user_id=user_id,
            timeout_seconds=timeout_seconds,
            trust_env=trust_env,
            max_attempts=attempts,
            max_concurrency=concurrency,
        )

    records = asyncio.run(_run())
    output = lock_file or benchmark.root / "corpus.lock.json"
    write_json(output, make_corpus_lock(manifest, records))
    console.print(
        f"[green]ingested[/green] {len(records)} documents into user={user_id!r}; lock: {output}"
    )


@eval_app.command("run")
def eval_run(
    dataset: Path = typer.Argument(
        Path("data/evals/fastapi-0.115"), help="Versioned benchmark directory."
    ),
    split: str = typer.Option("test", "--split", help="dev, test, or all."),
    strategies: str = typer.Option(
        "vector,hybrid,rrf,rewrite_rrf,rewrite_hybrid",
        "--strategies",
        help=(
            "Comma-separated retrieval ablations: vector, hybrid, rrf, rewrite_rrf, "
            "rewrite_hybrid (fts is diagnostic)."
        ),
    ),
    top_k: int | None = typer.Option(None, "--top-k", min=1, help="Override benchmark K."),
    candidate_multiplier: int | None = typer.Option(
        None,
        "--candidate-multiplier",
        min=1,
        help="Chunks retrieved per scored document before document-level deduplication.",
    ),
    repetitions: int = typer.Option(1, "--repetitions", min=1, max=10, help="Repeat each case."),
    user_id: str = typer.Option("eval-fastapi-0.115", "--user", help="KB user scope."),
    with_agent: bool = typer.Option(
        False, "--with-agent", help="Also run expensive end-to-end agent cases."
    ),
    agent_profile: str = typer.Option(
        "full",
        "--agent-profile",
        help="Agent eval profile: full, kb_only, permission, or web_approved.",
    ),
    web_allow_host: list[str] = typer.Option(
        [],
        "--web-allow-host",
        help="Exact host approved for web_approved fetches; repeat the option for multiple hosts.",
    ),
    offset: int = typer.Option(0, "--offset", min=0, help="Skip this many selected-split cases."),
    limit: int = typer.Option(0, "--limit", min=0, help="Limit cases for a smoke run (0 = all)."),
    output_root: Path = typer.Option(
        Path("artifacts/evals"), "--output-root", help="Parent directory for timestamped run artifacts."
    ),
) -> None:
    """Run a fair retrieval ablation, emit raw records, audit template and static report."""
    from pkb_agent.evaluation.agent_profiles import (
        get_agent_evaluation_profile,
        make_profile_confirmation,
        select_agent_cases,
    )
    from pkb_agent.evaluation.corpus import CorpusManifest, write_json
    from pkb_agent.evaluation.dataset import EvalDataset
    from pkb_agent.evaluation.metrics import summarize_records
    from pkb_agent.evaluation.report import write_report
    from pkb_agent.evaluation.runner import (
        RetrieverStrategy,
        make_audit_template,
        run_agent_evaluation,
        run_retrieval_evaluation,
        write_records,
    )
    from pkb_agent.llm.deepseek_client import DeepSeekClient
    from pkb_agent.prompts import PromptComposer, PromptRegistry
    from pkb_agent.prompts.query_rewriter import QueryPlanner
    from pkb_agent.rag.embeddings import EmbeddingProvider
    from pkb_agent.rag.retriever import Retriever
    from pkb_agent.storage.repositories.chunks import ChunksRepository
    from pkb_agent.storage.supabase_client import SupabaseClient

    split_names = _parse_eval_splits(split)
    benchmark = EvalDataset.load(dataset, splits=split_names)
    manifest = CorpusManifest.load(benchmark.root / benchmark.meta.corpus_manifest)
    _validate_eval_document_keys(benchmark, manifest)
    remaining_cases = benchmark.cases[offset:]
    selected_cases = list(remaining_cases[:limit] if limit else remaining_cases)
    if not selected_cases:
        raise typer.BadParameter("offset and limit selected zero cases")
    strategy_names = [name.strip() for name in strategies.split(",") if name.strip()]
    allowed = {"vector", "fts", "hybrid", "rrf", "rewrite_rrf", "rewrite_hybrid"}
    unknown = sorted(set(strategy_names) - allowed)
    if unknown:
        raise typer.BadParameter(f"unknown strategies: {', '.join(unknown)}", param_hint="--strategies")
    if not strategy_names and not with_agent:
        raise typer.BadParameter("select at least one retrieval strategy or --with-agent")
    try:
        profile = get_agent_evaluation_profile(agent_profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--agent-profile") from exc
    if not with_agent and (profile.name != "full" or web_allow_host):
        raise typer.BadParameter("--agent-profile and --web-allow-host require --with-agent")
    if web_allow_host and not profile.requires_web_allowlist:
        raise typer.BadParameter(
            "--web-allow-host is only valid with --agent-profile web_approved",
            param_hint="--web-allow-host",
        )
    agent_cases = select_agent_cases(selected_cases, profile) if with_agent else ()
    if with_agent and not agent_cases:
        tag = profile.required_tag or "selected profile"
        raise typer.BadParameter(f"agent profile {profile.name!r} selected zero cases for tag {tag!r}")
    try:
        agent_confirmation = make_profile_confirmation(profile, web_allow_host)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--web-allow-host") from exc
    k = top_k or benchmark.meta.default_top_k
    settings = get_settings()
    candidate_multiplier = candidate_multiplier or settings.evaluation_document_candidate_multiplier
    rewrite_strategies = {"rewrite_rrf", "rewrite_hybrid"}
    if rewrite_strategies.intersection(strategy_names) or with_agent:
        settings.require_llm()
    settings.require_supabase()
    settings.require_embedding()
    retrieval_case_runs = len(selected_cases) * len(strategy_names) * repetitions
    agent_case_runs = len(agent_cases) * repetitions if with_agent else 0
    total_case_runs = retrieval_case_runs + agent_case_runs
    rewrite_case_runs = len(selected_cases) * repetitions * len(
        rewrite_strategies.intersection(strategy_names)
    )
    start_details = (
        f"{len(selected_cases)} cases · {len(strategy_names)} retrieval strategies · "
        f"{repetitions} repetition(s) = {retrieval_case_runs} retrieval case-runs"
    )
    if with_agent:
        tools = ",".join(profile.allowed_tools or ("all",))
        start_details += (
            f" + {agent_case_runs} agent case-runs "
            f"({profile.name}; {len(agent_cases)} cases; tools={tools})"
        )
    if rewrite_case_runs:
        start_details += f" · {rewrite_case_runs} Query Rewrite calls"
    console.print(f"[cyan]starting evaluation[/cyan] {start_details}")

    async def _run() -> list[dict[str, Any]]:
        all_records: list[dict[str, Any]] = []
        completed_case_runs = 0
        execution_failed_case_runs = 0
        non_interactive_interval = max(1, max(len(selected_cases), len(agent_cases)) // 10)
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            "•",
            TimeElapsedColumn(),
            "• ETA",
            TimeRemainingColumn(),
            "• execution errors: {task.fields[execution_errors]}",
            console=console,
            disable=not console.is_terminal,
        )

        def make_progress_callback(label: str):
            def on_progress(completed: int, total: int, record: Mapping[str, Any]) -> None:
                nonlocal completed_case_runs, execution_failed_case_runs
                completed_case_runs += 1
                execution_success = record.get("execution_success", record.get("success"))
                if execution_success is False or bool(record.get("error")):
                    execution_failed_case_runs += 1
                description = f"{label} · {completed}/{total} · {record['case_id']}"
                progress.update(
                    progress_task,
                    completed=completed_case_runs,
                    description=description,
                    execution_errors=execution_failed_case_runs,
                )
                if not console.is_terminal and (
                    completed == total or completed % non_interactive_interval == 0
                ):
                    console.print(
                        f"[cyan]progress[/cyan] {description} · overall "
                        f"{completed_case_runs}/{total_case_runs} · "
                        f"execution_errors={execution_failed_case_runs}"
                    )

            return on_progress

        client = SupabaseClient.from_settings(settings)
        embedder = EmbeddingProvider.from_settings(settings)
        llm: DeepSeekClient | None = None
        with progress:
            progress_task = progress.add_task(
                "preparing evaluation",
                total=total_case_runs,
                execution_errors=0,
            )
            try:
                retriever = Retriever.from_settings(settings, embedder, ChunksRepository(client))
                if rewrite_strategies.intersection(strategy_names):
                    llm = DeepSeekClient.from_settings(settings)
                    composer = PromptComposer(PromptRegistry.from_settings(settings))
                else:
                    composer = None
                for iteration in range(1, repetitions + 1):
                    for strategy_name in strategy_names:
                        label = f"retrieval {iteration}/{repetitions} · {strategy_name}"
                        progress.update(progress_task, description=f"{label} · starting")
                        planner = (
                            QueryPlanner(
                                llm,
                                composer,
                                enabled=True,
                                max_queries=settings.prompts_query_rewrite_max_queries,
                            )
                            if strategy_name in rewrite_strategies and llm is not None and composer is not None
                            else None
                        )
                        strategy = RetrieverStrategy(
                            strategy_name,
                            retriever,
                            user_id=user_id,
                            planner=planner,
                            settings=settings,
                            candidate_multiplier=candidate_multiplier,
                        )
                        rows = await run_retrieval_evaluation(
                            selected_cases,
                            strategy,
                            top_k=k,
                            on_progress=make_progress_callback(label),
                        )
                        for row in rows:
                            row["iteration"] = iteration
                        all_records.extend(rows)
                        progress.console.print(
                            f"[green]completed[/green] {label} · {len(rows)} cases"
                        )
            finally:
                if llm is not None:
                    await llm.close()
                await embedder.close()
                client.close()

            if with_agent:
                agent_strategy_name = _agent_strategy_name(
                    profile.name,
                    rewrite_enabled=settings.prompts_query_rewrite_enabled,
                )
                # Evaluation must not inherit mutable permission rows from a
                # long-lived workbench. The checked-in YAML policy and the
                # selected profile are the experiment contract.
                agent_settings = settings.model_copy(deep=True)
                agent_settings.permissions_db_overrides_enabled = False
                for iteration in range(1, repetitions + 1):
                    label = f"agent {iteration}/{repetitions} · {agent_strategy_name}"
                    progress.update(progress_task, description=f"{label} · starting")
                    runtime = AgentRuntime.build(
                        agent_settings,
                        user_id=user_id,
                        confirm=agent_confirmation,
                    )
                    if profile.allowed_tools is not None:
                        runtime.restrict_tools(profile.allowed_tools)
                    async with runtime:
                        rows = await run_agent_evaluation(
                            agent_cases,
                            runtime,
                            strategy_name=agent_strategy_name,
                            on_progress=make_progress_callback(label),
                        )
                    for row in rows:
                        row["iteration"] = iteration
                        row["agent_profile"] = profile.name
                        row["agent_allowed_tools"] = list(profile.allowed_tools or ())
                    all_records.extend(rows)
                    progress.console.print(f"[green]completed[/green] {label} · {len(rows)} cases")
        return all_records

    records = asyncio.run(_run())
    run_name = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / benchmark.meta.id / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    record_path = write_records(output_dir / "records.jsonl", records)
    audit_template = make_audit_template(benchmark.cases, records)
    if audit_template:
        write_records(output_dir / "audit.template.jsonl", audit_template)
    summary = summarize_records(benchmark, records, top_k=k)
    report_paths = write_report(output_dir, summary)
    write_json(
        output_dir / "run.json",
        {
            "schema_version": 1,
            "benchmark_id": benchmark.meta.id,
            "benchmark_version": benchmark.meta.version,
            "corpus_revision": manifest.revision,
            "case_count": len(selected_cases),
            "agent_case_count": len(agent_cases),
            "agent_profile": profile.name if with_agent else None,
            "agent_allowed_tools": list(profile.allowed_tools or ()) if with_agent else [],
            "agent_web_allow_hosts": web_allow_host if with_agent else [],
            "agent_permissions_db_overrides_enabled": False if with_agent else None,
            "offset": offset,
            "splits": split_names,
            "strategies": strategy_names,
            "with_agent": with_agent,
            "repetitions": repetitions,
            "top_k": k,
            "candidate_multiplier": candidate_multiplier,
            "embedding_model": settings.embedding_model,
            "llm_model": (
                settings.deepseek_model
                if (rewrite_strategies.intersection(strategy_names) or with_agent)
                else None
            ),
            "config_sha256": _sha256_file(Path("config/default.yaml")),
            "benchmark_sha256": _sha256_file(benchmark.root / "benchmark.json"),
            "records": record_path.name,
        },
    )
    console.print(f"[green]complete[/green] {len(records)} records → {output_dir}")
    console.print(f"report: {report_paths['html']}")
    if audit_template:
        console.print(f"human audit template: {output_dir / 'audit.template.jsonl'}")


@eval_app.command("report")
def eval_report(
    dataset: Path = typer.Argument(
        Path("data/evals/fastapi-0.115"), help="Versioned benchmark directory."
    ),
    records: list[Path] = typer.Argument(..., help="One or more raw records.jsonl files."),
    audits: Path | None = typer.Option(None, "--audits", help="Completed audit JSONL to merge by case_id."),
    output_dir: Path = typer.Option(Path("artifacts/evals/report"), "--output-dir"),
    top_k: int | None = typer.Option(None, "--top-k", min=1),
) -> None:
    """Re-score an existing run after reviewers complete semantic audit labels."""
    from pkb_agent.evaluation.dataset import EvalDataset
    from pkb_agent.evaluation.metrics import summarize_records
    from pkb_agent.evaluation.report import write_report
    from pkb_agent.evaluation.runner import attach_audits, read_records, write_records

    benchmark = EvalDataset.load(dataset)
    run_records = [record for path in records for record in read_records(path)]
    if audits is not None:
        run_records = attach_audits(run_records, read_records(audits))
    write_records(output_dir / "records.jsonl", run_records)
    report_paths = write_report(
        output_dir,
        summarize_records(benchmark, run_records, top_k=top_k or benchmark.meta.default_top_k),
    )
    console.print(f"[green]report written[/green] {report_paths['html']}")


@eval_app.command("audit-template")
def eval_audit_template(
    dataset: Path = typer.Argument(Path("data/evals/fastapi-0.115")),
    records: Path = typer.Argument(..., help="Raw records.jsonl containing agent records."),
    output: Path = typer.Option(Path("audit.template.jsonl"), "--output"),
) -> None:
    """Generate a blank semantic citation/fact review sheet for an agent run."""
    from pkb_agent.evaluation.dataset import EvalDataset
    from pkb_agent.evaluation.runner import make_audit_template, read_records, write_records

    benchmark = EvalDataset.load(dataset)
    template = make_audit_template(benchmark.cases, read_records(records))
    write_records(output, template)
    console.print(f"[green]wrote[/green] {len(template)} audit rows to {output}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _parse_eval_splits(value: str) -> tuple[str, ...]:
    normalized = value.strip().casefold()
    if normalized == "all":
        return ("dev", "test")
    splits = tuple(dict.fromkeys(part.strip().casefold() for part in value.split(",") if part.strip()))
    if not splits or any(part not in {"dev", "test"} for part in splits):
        raise typer.BadParameter("split must be dev, test, all, or a comma-separated dev,test")
    return splits


def _agent_strategy_name(profile_name: str, *, rewrite_enabled: bool) -> str:
    """Keep profile-specific Agent artifacts distinct in reports and JSONL."""
    if profile_name == "full":
        return "agent_rewrite_hybrid" if rewrite_enabled else "agent_hybrid"
    if profile_name == "kb_only":
        return "agent_kb_only_rewrite_hybrid" if rewrite_enabled else "agent_kb_only_hybrid"
    return f"agent_{profile_name}"


def _validate_eval_document_keys(benchmark, manifest) -> None:
    """Fail early when an annotation points at a source absent from its corpus."""
    known = {document.key for document in manifest.documents}
    missing: list[str] = []
    for case in benchmark.cases:
        labels = [item.key for item in case.relevant_documents]
        labels.extend(key for fact in case.key_facts for key in fact.source_documents)
        for key in labels:
            if key not in known:
                missing.append(f"{case.id}:{key}")
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " …" if len(missing) > 8 else ""
        raise typer.BadParameter(f"annotation refers to unknown corpus document(s): {preview}{suffix}")


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _make_confirm(yes: bool):
    def confirm(tool_name: str, args: dict[str, Any]) -> bool:
        if yes:
            return True
        args_str = json.dumps(args, ensure_ascii=False)
        if len(args_str) > 300:
            args_str = args_str[:300] + "…"
        return typer.confirm(f"Allow tool [bold]{tool_name}[/bold] with args {args_str}?", default=False)

    return confirm


def _render_event(event: dict[str, Any]) -> None:
    etype = event.get("type")
    if etype == "start":
        console.print(Panel(event.get("question", ""), title="Question", border_style="cyan"))
    elif etype == "tool_call":
        args = event.get("args", {})
        args_str = json.dumps(args, ensure_ascii=False)
        if len(args_str) > 200:
            args_str = args_str[:200] + "…"
        console.print(f"[bold blue]→ step {event.get('step')}[/bold blue] {event.get('tool')}({args_str})")
    elif etype == "tool_result":
        ok = event.get("ok")
        color = "green" if ok else "red"
        obs = str(event.get("observation", ""))
        if len(obs) > 600:
            obs = obs[:600] + "…"
        console.print(
            f"  [{color}]{'ok' if ok else 'error'}[/{color}] "
            f"({event.get('duration_ms')}ms) {obs}",
            style=color if not ok else None,
        )
    elif etype == "answer":
        # answer is rendered by the caller (panel); nothing here.
        pass
    elif etype == "answer_verification_failed":
        errors = event.get("errors") or []
        first_error = str(errors[0]) if errors else "invalid final answer"
        console.print(
            "[yellow]answer verification failed; requesting a repair "
            f"({event.get('retry')}/{event.get('retry_limit')}):[/yellow] {first_error}"
        )
    elif etype == "max_steps":
        console.print(f"[yellow]max steps reached ({event.get('steps')})[/yellow]")
    elif etype == "error":
        console.print(f"[red]error:[/red] {event.get('error')}")


def _render_verified_answer(payload: dict[str, Any], verification: dict[str, Any]) -> None:
    """Render source facts and model inferences as visibly different blocks."""
    answer = str(payload.get("answer") or "")
    status = str(payload.get("status") or "grounded")
    border = "green" if status == "grounded" else "yellow"
    title = "Answer" if status == "grounded" else "Evidence boundary"
    console.print(Panel(Text(answer), title=title, border_style=border))

    claims = payload.get("claims")
    if not isinstance(claims, list):
        claims = []
    facts = [claim for claim in claims if isinstance(claim, dict) and claim.get("kind") == "fact"]
    inferences = [
        claim for claim in claims if isinstance(claim, dict) and claim.get("kind") == "inference"
    ]
    if facts:
        console.print(
            Panel(
                _claims_text(facts),
                title="原文事实 / Source facts",
                border_style="cyan",
            )
        )
    if inferences:
        console.print(
            Panel(
                _claims_text(inferences),
                title="模型推断 / Model inferences",
                border_style="magenta",
            )
        )

    citations = payload.get("citations")
    if isinstance(citations, list) and citations:
        console.print(
            Panel(
                _citations_text(citations),
                title="Verified citations",
                border_style="blue",
            )
        )
    if verification.get("status") == "refused":
        console.print(
            f"[yellow]verification:[/yellow] refused ({verification.get('reason', 'unknown')})"
        )


def _claims_text(claims: list[dict[str, Any]]) -> Text:
    text = Text()
    for index, claim in enumerate(claims, start=1):
        if index > 1:
            text.append("\n")
        text.append(f"{index}. {claim.get('text', '')}\n")
        citation_ids = claim.get("citations") or []
        text.append(f"   Evidence: {', '.join(str(item) for item in citation_ids)}", style="dim")
    return text


def _citations_text(citations: list[Any]) -> Text:
    text = Text()
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        if text.plain:
            text.append("\n")
        title = str(citation.get("title") or citation.get("source_id") or "(untitled)")
        text.append(f"{citation.get('id', '')}  {title}\n")
        text.append(str(citation.get("locator") or ""), style="dim")
        metadata = citation.get("metadata")
        if isinstance(metadata, dict) and citation.get("source_type") == "web_page":
            text.append(
                "\n"
                f"domain={metadata.get('domain')}  trust={metadata.get('trust_level')}  "
                f"freshness={metadata.get('expiration_state')}  fetched={metadata.get('fetched_at')}",
                style="dim",
            )
    return text


def _collect_files(path: Path, recursive: bool) -> list[Path]:
    text_suffixes = {".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".csv", ".html", ".org", ".pdf"}
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    glob_fn = path.rglob if recursive else path.glob
    return sorted(p for p in glob_fn("*") if p.is_file() and p.suffix.lower() in text_suffixes)


def _check(label: str, ok: bool) -> None:
    mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
    console.print(f"{mark} {label}")
    if not ok:
        _check.failed = True  # type: ignore[attr-defined]


_check.failed = False  # type: ignore[attr-defined]


if __name__ == "__main__":
    app()
