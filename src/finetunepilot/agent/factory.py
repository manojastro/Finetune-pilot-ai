"""Build the agent's dependencies (LLM + toolbox) from its settings. Used by the CLI and API."""

from __future__ import annotations

from collections.abc import Callable

from finetunepilot.agent.llm import AnthropicClient
from finetunepilot.agent.nodes import AgentDeps
from finetunepilot.agent.state import AgentSettings
from finetunepilot.agent.toolbox import DryRunLLM, RealToolbox, SimulatedToolbox


def build_deps(
    settings: AgentSettings,
    approver: Callable[[str], bool],
    say: Callable[[str], None] = print,
) -> AgentDeps:
    """Dry run: rule-based LLM + simulated GPU work. Otherwise: Claude + real jobs."""
    if settings.dry_run:
        return AgentDeps(
            llm=DryRunLLM(),
            toolbox=SimulatedToolbox(settings.runs_dir, say=say),
            approver=approver,
            say=say,
        )
    return AgentDeps(
        llm=AnthropicClient(settings.llm),
        toolbox=RealToolbox(settings.runs_dir, say=say),
        approver=approver,
        say=say,
    )
