# FineTunePilot

An AI agent that takes a base model, a dataset, an objective and a compute budget, then plans,
runs, diagnoses and compares QLoRA fine-tuning experiments and writes a report.

First target task: fine-tune `Qwen/Qwen2.5-0.5B-Instruct` for tool calling.

> Work in progress. See [STATUS.md](STATUS.md) for what is built, tested and measured.

## Setup (stage 0)

```bash
uv sync                      # installs deps (CPU-only torch on the laptop)
uv run finetunepilot version # check the CLI works
uv run pytest -q             # run tests
uv run ruff check . && uv run ruff format --check .
```
