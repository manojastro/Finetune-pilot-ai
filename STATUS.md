# FineTunePilot status

Labels: **built** = code written · **tested** = tests pass on laptop (CPU) · **measured** = real numbers from a GPU run · **not yet run**.

| stage | built | tested | measured | notes |
|---|---|---|---|---|
| 0 — setup | yes | yes | n/a | uv project, folder layout, ruff/pytest config, `finetunepilot version` CLI |
| 1 — baseline QLoRA training | yes | yes (47 tests, 3 slow) | not yet run | dry run passes on CPU; Kaggle T4 run pending |
| 2 — evaluation | no | no | not yet run | |
| 3 — VRAM estimate + OOM recovery | no | no | not yet run | |
| 4 — agent (LangGraph) | no | no | not yet run | |
| 5 — API, UI, Docker | no | no | not yet run | |
| 6 — portfolio polish | no | no | n/a | |

## Stage 1 detail

| item | status |
|---|---|
| `TrainConfig` / `RunConfig` + YAML loading + range checks | built, tested |
| `configs/qwen05b_toolcall.yaml` (T4: 4-bit NF4, fp16, r=16, eff. batch 8, 5 epochs) | built, tested (loads); not yet run on GPU |
| `data/sample/` 60 train / 20 test rows, 6 tools, no overlap | built, tested |
| chat-template formatting, completion-only loss mask | built, tested (with real Qwen2.5 template, offline) |
| `train()` real GPU path | built; not yet run |
| `train()` dry run (tiny model, CPU, 2 steps) | built, tested |
| `runs/<run_id>.json` write/read | built, tested |
| `finetunepilot train [--dry-run]` | built, tested |
| `notebooks/01_train_kaggle.ipynb` | built; not yet run |

Checked against the real Qwen2.5 tokenizer during development (not a training measurement):
longest row = 762 tokens, mean ≈ 721, so `max_seq_length: 1024` truncates nothing.

## Decisions
- Tool-call output format: Qwen native `<tool_call>{json}</tool_call>`.
- torch is unpinned; on the laptop `uv sync` installs the CPU-only wheel, Kaggle keeps its own CUDA build.
- transformers >=5.20,<6 and trl >=1.18,<2: the code uses their current argument names
  (`warmup_steps` as a ratio, `activation_checkpointing`, `max_length`).
- Rows are rendered to strings before entering `datasets`, so Arrow can't add `null` argument keys.
- Rows longer than `max_seq_length` fail the run instead of being silently truncated
  (truncation would cut off the tool call). Stage 3's "reduce max_seq_length" fix must respect this.
- Stage 1 trains in-process; the one-subprocess-per-attempt rule arrives with the stage 3 runner.
- Tests block all network access (see `tests/conftest.py`).
