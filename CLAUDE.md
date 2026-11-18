# FineTunePilot — project rules for Claude Code

## What this project is
FineTunePilot is an AI agent that takes a base model, a dataset, an objective and a compute budget.
It then plans, runs, diagnoses and compares QLoRA fine-tuning experiments, and writes a report.

First target task: fine-tune Qwen2.5-0.5B-Instruct for tool calling (pick the right tool, produce valid JSON arguments).

## Environment
- Development: my laptop, **no GPU**. Claude Code writes and tests code here.
- Training: free Kaggle or Colab notebook with a T4 GPU (16 GB). I run the notebook and paste results back.
- Python 3.11, managed with `uv`.
- Libraries: torch (unpinned), transformers, trl, peft, datasets, pydantic, pyyaml, typer, rich,
  langgraph, langgraph-checkpoint-sqlite, anthropic, fastapi, uvicorn.
  GPU-only extra `gpu`: bitsandbytes, accelerate. Dev: pytest, ruff, httpx.

## Hard rules
1. Never launch paid compute, push to Hugging Face, deploy anything, or call a paid API without my explicit approval in chat.
2. Tests must never need a GPU, a network connection or an API key. Use mocks and fakes.
3. Every GPU code path must also have a CPU or dry-run path so it can be tested here.
4. Every training or eval run writes `runs/<run_id>.json` with: config, hardware, each attempt, metrics, status, errors, duration,
   plus git commit, library versions and seed. Eval-only runs have an empty attempts list.
5. The LLM only decides configs and explains results. Deterministic Python tools do all the training, evaluation and file work.
6. Any LLM output is validated with pydantic before use. Invalid output means retry once, then fail clearly.
7. Every function in `tools/` has a unit test.
8. Report only measured numbers. When reporting status, label each item: built / tested / measured / not yet run.
9. Work one stage at a time. At the end of a stage: run tests and ruff, update `STATUS.md`, summarize, and stop for my review.
10. Explain changes in simple English. Keep code readable over clever.

## ML rules
- Train with loss on the assistant tokens only (completion-only loss), never on the prompt or tool schemas.
- Target output format: Qwen native `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.
  The parser must also accept bare JSON and code fences.
- Dataset JSONL stores raw `messages` + `tools`; apply the chat template at load time.
- Train and test rows must not overlap (no shared request text); a test checks this.
- On Kaggle use one GPU only (`CUDA_VISIBLE_DEVICES=0`), even on "T4 x2".
- Each training attempt runs in its own subprocess, so an OOM can't leave the next attempt with less memory.
- The VRAM estimate must include the logits (vocab size x seq length x batch size, fp32).
- Dry-run mode forces fp32 and turns off 4-bit loading. The dry-run model and tokenizer are built locally, never downloaded.

## Secrets
- API keys and HF tokens are read from env vars (Kaggle Secrets on Kaggle). Never commit or log them.
- The Anthropic model name comes from config; it is not hardcoded.

## Commands
- Install: `uv sync`
- Tests: `uv run pytest -q` (fast only: `uv run pytest -q -m "not slow"`)
- Lint: `uv run ruff check . && uv run ruff format --check .`
- CLI: `uv run finetunepilot --help`

## Repo layout
```
Finetune-pilot-ai/
  CLAUDE.md
  STATUS.md
  README.md
  pyproject.toml
  configs/            # YAML training configs
  data/sample/        # tiny tool-calling dataset committed to repo
  src/finetunepilot/
    config.py         # pydantic config models
    data.py           # load/validate dataset rows, apply chat template
    dryrun.py         # tiny local model + tokenizer for CPU dry runs
    templates/        # Qwen2.5 chat template copy (for offline dry runs)
    train.py          # QLoRA training via TRL + PEFT
    eval.py           # tool-call accuracy evaluation
    runner.py         # runs training with OOM retry + budget cap
    worker.py         # one training attempt in its own subprocess
    runlog.py         # writes/reads runs/*.json
    cli.py            # typer CLI
    tools/            # VRAM estimator, log parser, scoring, comparison
    agent/            # LangGraph graph, nodes, LLM client
    api/              # FastAPI app
  notebooks/          # Kaggle/Colab notebooks
  scripts/            # one-off scripts (e.g. sample data generator)
  runs/               # run logs (gitignored except runs/examples/)
  reports/            # agent reports (gitignored except reports/examples/)
  docs/               # design notes (stage 6)
  tests/
```

## Definition of done for each stage
- `uv run pytest -q` passes
- ruff clean
- `STATUS.md` updated with what was built, tested and measured
- README section for the stage written
