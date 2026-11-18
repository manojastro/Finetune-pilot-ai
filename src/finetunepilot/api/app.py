"""FastAPI app: start agent sessions and browse sessions, runs and reports.

    GET  /                     run-history UI (single HTML page, no build step)
    GET  /health
    POST /sessions             start an agent session in the background
    GET  /sessions             list sessions (newest first)
    GET  /sessions/{id}        one session: runs, metrics and the report markdown
    GET  /runs/{run_id}        one run log (runs/<run_id>.json)

Real (non dry-run) sessions call the billed Anthropic API and use the GPU, so they are refused
unless the server was started with FINETUNEPILOT_ALLOW_REAL=1. Posting a session counts as
approving all of its runs (there is no terminal to ask), so the agent runs with auto_approve.

Run with:  uvicorn finetunepilot.api.app:app --port 8000
"""

from __future__ import annotations

import os
import re
import traceback
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from finetunepilot import __version__
from finetunepilot.agent.graph import SessionStore, new_session_id, run_session
from finetunepilot.agent.llm import load_llm_settings
from finetunepilot.agent.nodes import AgentDeps
from finetunepilot.agent.state import AgentSettings
from finetunepilot.runlog import RunLog, read_run

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,200}$")


class ApiSettings(BaseModel):
    runs_dir: str = "runs"
    reports_dir: str = "reports"
    sessions_dir: str = "runs/sessions"
    output_root: str = "outputs/agent"
    agent_config: str = "configs/agent.yaml"
    allow_real: bool = False  # real sessions call a billed API: off unless explicitly enabled

    @classmethod
    def from_env(cls) -> ApiSettings:
        env = os.environ.get
        return cls(
            runs_dir=env("FINETUNEPILOT_RUNS_DIR", "runs"),
            reports_dir=env("FINETUNEPILOT_REPORTS_DIR", "reports"),
            sessions_dir=env("FINETUNEPILOT_SESSIONS_DIR", "runs/sessions"),
            agent_config=env("FINETUNEPILOT_AGENT_CONFIG", "configs/agent.yaml"),
            allow_real=env("FINETUNEPILOT_ALLOW_REAL", "") == "1",
        )


class Health(BaseModel):
    status: str
    version: str


class SessionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = Field(min_length=3, max_length=2000)
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_dir: str = "data/sample"
    gpu: str = "t4"
    max_runs: int = Field(default=3, ge=1, le=10)
    max_minutes: float = Field(default=90, gt=0, le=24 * 60)
    dry_run: bool = True


class SessionCreated(BaseModel):
    session_id: str
    status: str


class SessionBrief(BaseModel):
    session_id: str
    status: str
    objective: str
    dry_run: bool
    runs: int
    best_args_exact_match: float | None
    stop_reason: str | None = None
    error: str | None = None
    updated_at: str | None = None


class SessionDetail(BaseModel):
    session: dict[str, Any]
    report_markdown: str | None


def _brief(summary: dict[str, Any]) -> SessionBrief:
    runs = summary.get("runs") or []
    scores = [
        (r.get("metrics") or {}).get("args_exact_match")
        for r in runs
        if (r.get("metrics") or {}).get("args_exact_match") is not None
    ]
    settings = summary.get("settings") or {}
    return SessionBrief(
        session_id=summary["session_id"],
        status=summary.get("status", "unknown"),
        objective=settings.get("objective", ""),
        dry_run=bool(settings.get("dry_run")),
        runs=len(runs),
        best_args_exact_match=max(scores) if scores else None,
        stop_reason=summary.get("stop_reason"),
        error=summary.get("error"),
        updated_at=summary.get("updated_at"),
    )


DepsFactory = Callable[[AgentSettings], AgentDeps]


def _default_deps(settings: AgentSettings) -> AgentDeps:
    from finetunepilot.agent.factory import build_deps

    return build_deps(settings, approver=lambda text: True, say=lambda message: None)


def create_app(
    settings: ApiSettings | None = None, deps_factory: DepsFactory | None = None
) -> FastAPI:
    cfg = settings or ApiSettings.from_env()
    make_deps = deps_factory or _default_deps
    store = SessionStore(cfg.sessions_dir)
    app = FastAPI(title="FineTunePilot", version=__version__)

    def run_in_background(agent_settings: AgentSettings, session_id: str) -> None:
        try:
            run_session(agent_settings, make_deps(agent_settings), session_id=session_id)
        except Exception as e:  # keep the error visible in the session list
            store.write_summary(
                {
                    "session_id": session_id,
                    "settings": agent_settings.model_dump(),
                    "status": "failed",
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": "".join(traceback.format_exception(e))[-4000:],
                }
            )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return (files("finetunepilot.api") / "static" / "index.html").read_text(encoding="utf-8")

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health(status="ok", version=__version__)

    @app.post("/sessions", response_model=SessionCreated, status_code=202)
    def create_session(body: SessionCreate, background: BackgroundTasks) -> SessionCreated:
        if not body.dry_run and not cfg.allow_real:
            raise HTTPException(
                403,
                "Real sessions call the billed Anthropic API and need a GPU. Start the server "
                "with FINETUNEPILOT_ALLOW_REAL=1 to allow them, or send dry_run=true.",
            )
        for split in ("train.jsonl", "test.jsonl"):
            if not (Path(body.dataset_dir) / split).is_file():
                raise HTTPException(422, f"dataset_dir has no {split}")
        try:
            agent_settings = AgentSettings(
                **body.model_dump(),
                auto_approve=True,
                llm=load_llm_settings(cfg.agent_config),
                runs_dir=cfg.runs_dir,
                reports_dir=cfg.reports_dir,
                sessions_dir=cfg.sessions_dir,
                output_root=cfg.output_root,
            )
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        session_id = new_session_id()
        store.write_summary(
            {
                "session_id": session_id,
                "settings": agent_settings.model_dump(),
                "status": "queued",
                "runs": [],
            }
        )
        background.add_task(run_in_background, agent_settings, session_id)
        return SessionCreated(session_id=session_id, status="queued")

    @app.get("/sessions", response_model=list[SessionBrief])
    def list_sessions() -> list[SessionBrief]:
        return [_brief(s) for s in store.list_summaries()]

    @app.get("/sessions/{session_id}", response_model=SessionDetail)
    def get_session(session_id: str) -> SessionDetail:
        if not SAFE_ID.match(session_id) or not store.summary_path(session_id).is_file():
            raise HTTPException(404, "session not found")
        summary = store.read_summary(session_id)
        report = summary.get("report_path")
        text = (
            Path(report).read_text(encoding="utf-8") if report and Path(report).is_file() else None
        )
        return SessionDetail(session=summary, report_markdown=text)

    @app.get("/runs/{run_id}", response_model=RunLog)
    def get_run(run_id: str) -> RunLog:
        path = Path(cfg.runs_dir) / f"{run_id}.json"
        if not SAFE_ID.match(run_id) or not path.is_file():
            raise HTTPException(404, "run not found")
        return read_run(path)

    return app


app = create_app()
