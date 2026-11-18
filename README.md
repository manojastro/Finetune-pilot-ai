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

## Stage 2 — evaluation

```bash
# Base model, then base + adapter, then compare (GPU)
finetunepilot eval --base-model Qwen/Qwen2.5-0.5B-Instruct --test data/sample/test.jsonl
finetunepilot eval --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/qwen05b_toolcall/<run_id>
finetunepilot compare runs/eval-base-<...>.json runs/eval-tuned-<...>.json

# Laptop: same commands with --dry-run (tiny random model, scores are meaningless)
uv run finetunepilot eval --base-model tiny --dry-run
```

Each test prompt is built exactly as in training. The model generates greedily (up to 128 new
tokens) and the output is parsed by `tools/scoring.py`:

| metric | meaning |
|---|---|
| `json_valid` | a JSON object was found in the output |
| `tool_name_accuracy` | the right tool was chosen |
| `args_key_match` | right tool, and the same set of argument keys |
| `args_exact_match` | right tool, and exactly the right arguments (the full call is correct) |
| `format_ok` | the output is exactly `<tool_call>{json}</tool_call>`, the trained format |

The eval run JSON stores every prediction, so wrong rows can be inspected later.
On Kaggle, run `notebooks/02_eval_kaggle.ipynb`.

Results: not yet measured.
