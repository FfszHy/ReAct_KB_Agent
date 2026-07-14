"""Typer CLI for PKB-Agent."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel

from pkb_agent.agent.runtime import AgentRuntime
from pkb_agent.app.settings import get_settings

app = typer.Typer(
    name="pkb-agent",
    help="Supabase-backed ReAct personal knowledge base agent.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


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
                if state.final_answer:
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
    _check("Embedding base URL", bool(settings.embedding_api_base_url))
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
            sb.client.table("documents").select("id").limit(1).execute()
            _check("Supabase connectivity", True)
        except Exception as e:
            _check(f"Supabase connectivity ({e})", False)

    try:
        asyncio.run(_probe())
    except Exception as e:
        _check(f"Supabase connectivity ({e})", False)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
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
    elif etype == "max_steps":
        console.print(f"[yellow]max steps reached ({event.get('steps')})[/yellow]")
    elif etype == "error":
        console.print(f"[red]error:[/red] {event.get('error')}")


def _collect_files(path: Path, recursive: bool) -> list[Path]:
    text_suffixes = {".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".csv", ".html", ".org"}
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
