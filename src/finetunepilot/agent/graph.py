"""LangGraph StateGraph for the agent, with a checkpoint after every node.

    prepare -> baseline -> plan -> estimate -> approve -> train -> evaluate -> diagnose -> decide
                             ^        |                    |         |                     |
                             +--------+ (doesn't fit)      +---------+--> decide           |
                                      estimate <------------------------------ (next run) -+
    any node error, rejection, or stop  ->  report -> END

State is checkpointed to <sessions_dir>/checkpoints.sqlite (thread id = session id), so a
crashed session continues from the last finished node with `--resume <session_id>`.
A small JSON summary per session (<sessions_dir>/<session_id>.json) is rewritten after every
node for the CLI and the API; LLM answers are appended to <session_id>.llm.jsonl.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from finetunepilot.agent import nodes
from finetunepilot.agent.nodes import AgentDeps
from finetunepilot.agent.state import AgentSettings, AgentState

NODES = {
    "prepare": nodes.prepare,
    "baseline": nodes.baseline,
    "plan": nodes.plan,
    "estimate": nodes.estimate,
    "approve": nodes.approve,
    "train": nodes.train,
    "evaluate": nodes.evaluate,
    "diagnose": nodes.diagnose,
    "decide": nodes.decide,
    "report": nodes.report,
}
RECURSION_LIMIT = 200


def new_session_id() -> str:
    return f"s-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


class SessionStore:
    """Files for one sessions directory: checkpoints, summaries, LLM logs."""

    def __init__(self, sessions_dir: str | Path):
        self.dir = Path(sessions_dir)  # created on first write, not on import

    def _ensure_dir(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def summary_path(self, session_id: str) -> Path:
        return self.dir / f"{session_id}.json"

    def write_summary(self, state: dict[str, Any]) -> None:
        summary = nodes.json_safe(state)
        for run in summary.get("runs", []):
            run.pop("failures", None)  # keep the summary small
        summary["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        self._ensure_dir()
        path = self.summary_path(state["session_id"])
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tmp.replace(path)

    def read_summary(self, session_id: str) -> dict[str, Any]:
        return json.loads(self.summary_path(session_id).read_text(encoding="utf-8"))

    def list_summaries(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted(self.dir.glob("s-*.json"), reverse=True):
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return out

    def llm_logger(self, session_id: str):
        path = self.dir / f"{session_id}.llm.jsonl"

        def log(entry: dict[str, Any]) -> None:
            self._ensure_dir()
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")

        return log

    @contextmanager
    def checkpointer(self):
        self._ensure_dir()
        conn = sqlite3.connect(self.dir / "checkpoints.sqlite", check_same_thread=False)
        try:
            yield SqliteSaver(conn)
        finally:
            conn.close()


def build_graph(deps: AgentDeps, checkpointer=None, store: SessionStore | None = None):
    graph = StateGraph(AgentState)

    def wrap(name: str, fn):
        def run_node(state: AgentState) -> dict:
            if state.get("error") and name != "report":
                return {}
            try:
                update = fn(state, deps)
            except Exception as e:  # a node failure ends the session with a report
                update = {"error": f"{name}: {type(e).__name__}: {e}"}
                deps.say(f"[{name}] failed: {update['error']}")
            update["last_node"] = name
            if store is not None:
                store.write_summary({**state, **update})
            return update

        return run_node

    for name, fn in NODES.items():
        graph.add_node(name, wrap(name, fn))

    def ok_then(target: str):
        return lambda state: "report" if state.get("error") else target

    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", ok_then("baseline"), ["baseline", "report"])
    graph.add_conditional_edges("baseline", ok_then("plan"), ["plan", "report"])
    graph.add_conditional_edges("plan", ok_then("estimate"), ["estimate", "report"])
    graph.add_conditional_edges(
        "estimate", nodes.route_after_estimate, ["approve", "plan", "report"]
    )
    graph.add_conditional_edges("approve", nodes.route_after_approve, ["train", "report"])
    graph.add_conditional_edges("train", nodes.route_after_train, ["evaluate", "decide", "report"])
    graph.add_conditional_edges(
        "evaluate", nodes.route_after_evaluate, ["diagnose", "decide", "report"]
    )
    graph.add_conditional_edges("diagnose", ok_then("decide"), ["decide", "report"])
    graph.add_conditional_edges("decide", nodes.route_after_decide, ["estimate", "report"])
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)


def run_session(settings: AgentSettings, deps: AgentDeps, session_id: str | None = None) -> dict:
    """Start a new session and run it to the end. Returns the final state."""
    sid = session_id or new_session_id()
    store = SessionStore(settings.sessions_dir)
    deps.llm_log = store.llm_logger(sid)
    initial: AgentState = {
        "session_id": sid,
        "settings": settings.model_dump(),
        "status": "running",
        "started_at": deps.clock(),
        "runs": [],
    }
    store.write_summary({**initial, "last_node": "start"})
    with store.checkpointer() as saver:
        graph = build_graph(deps, saver, store)
        return graph.invoke(
            initial, {"configurable": {"thread_id": sid}, "recursion_limit": RECURSION_LIMIT}
        )


def resume_session(session_id: str, deps: AgentDeps, sessions_dir: str | Path) -> dict:
    """Continue a session from its last checkpoint. A finished session is returned as is."""
    store = SessionStore(sessions_dir)
    deps.llm_log = store.llm_logger(session_id)
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": RECURSION_LIMIT}
    with store.checkpointer() as saver:
        graph = build_graph(deps, saver, store)
        snapshot = graph.get_state(config)
        if not snapshot.values:
            raise KeyError(f"no checkpoint for session {session_id!r} in {sessions_dir}")
        if not snapshot.next:
            return snapshot.values
        return graph.invoke(None, config)


def load_session_settings(session_id: str, sessions_dir: str | Path) -> AgentSettings:
    return AgentSettings.model_validate(
        SessionStore(sessions_dir).read_summary(session_id)["settings"]
    )
