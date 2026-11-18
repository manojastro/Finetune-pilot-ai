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

## Stage 1 — baseline QLoRA training

```bash
# On the laptop: tiny random model on CPU, 2 steps, no GPU or download (~5 s)
uv run finetunepilot train --config configs/qwen05b_toolcall.yaml --dry-run

# On a GPU (Kaggle T4): real QLoRA run
finetunepilot train --config configs/qwen05b_toolcall.yaml
```

What happens:
1. `configs/qwen05b_toolcall.yaml` is validated (`finetunepilot.config`).
2. `data/sample/train.jsonl` rows (raw `messages` + `tools`) are rendered with the model's chat
   template into a prompt and a completion. The completion is the tool call in Qwen's format:
   `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.
3. The base model loads in 4-bit NF4, LoRA adapters (r=16) go on every linear layer, and TRL's
   `SFTTrainer` trains with the loss on the completion only.
4. The adapter and tokenizer are saved to `outputs/qwen05b_toolcall/<run_id>/`, and
   `runs/<run_id>.json` records the config, hardware, library versions, git commit, loss
   history, steps, duration and peak GPU memory.

On Kaggle, run `notebooks/01_train_kaggle.ipynb` (GPU T4, Internet on).

The sample data comes from `scripts/make_sample_data.py`. Train and test use different sentence
templates and argument values, and a test checks that no request appears in both.

Results: not yet measured.
