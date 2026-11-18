# FineTunePilot status

Labels: **built** = code written · **tested** = tests pass on laptop (CPU) · **measured** = real numbers from a GPU run · **not yet run**.

| stage | built | tested | measured | notes |
|---|---|---|---|---|
| 0 — setup | yes | yes (1 test) | n/a | uv project, folder layout, ruff/pytest config, `finetunepilot version` CLI |
| 1 — baseline QLoRA training | no | no | not yet run | |
| 2 — evaluation | no | no | not yet run | |
| 3 — VRAM estimate + OOM recovery | no | no | not yet run | |
| 4 — agent (LangGraph) | no | no | not yet run | |
| 5 — API, UI, Docker | no | no | not yet run | |
| 6 — portfolio polish | no | no | n/a | |

## Decisions
- Tool-call output format: Qwen native `<tool_call>{json}</tool_call>`.
- torch is unpinned; on the laptop `uv sync` installs the CPU-only wheel, Kaggle keeps its own CUDA build.
- Tests block all network access (see `tests/conftest.py`).
