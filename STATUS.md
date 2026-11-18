# FineTunePilot status

Labels: **built** = code written · **tested** = tests pass on laptop (CPU) · **measured** = real numbers from a GPU run · **not yet run**.

| stage | built | tested | measured | notes |
|---|---|---|---|---|
| 0 — setup | yes | yes | n/a | uv project, folder layout, ruff/pytest config, `finetunepilot version` CLI |
| 1 — baseline QLoRA training | yes | yes | not yet run | dry run passes on CPU; Kaggle T4 run pending |
| 2 — evaluation | yes | yes | not yet run | 102 tests total (5 slow); base vs tuned on Kaggle pending |
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

## Stage 2 detail

| item | status |
|---|---|
| `tools/scoring.py`: parse `<tool_call>` / fenced / bare JSON, 5 metrics, per-tool breakdown | built, tested (36 output cases) |
| `tools/compare.py`: metric table with delta (last - first) | built, tested |
| `eval.py`: greedy generation, fixed `max_new_tokens=128`, per-row predictions in run JSON | built, tested (oracle model scores 1.0; dry run) |
| `eval.py` real GPU path (4-bit base, optional adapter) | built; not yet run |
| `finetunepilot eval` / `finetunepilot compare` | built, tested |
| `notebooks/02_eval_kaggle.ipynb` (trains first if no adapter) | built; not yet run |

Measured accuracy: **none yet**. Base vs tuned numbers come from the Kaggle run.

Metric definitions:
- `json_valid`: a JSON object was found (inside `<tool_call>`, a code fence, or free text).
- `tool_name_accuracy`: its `name` equals the target tool.
- `args_key_match`: right tool and the same set of argument keys.
- `args_exact_match`: right tool and equal arguments (key order ignored, `5 == 5.0`, `"5" != 5`).
- `format_ok`: output is exactly `<tool_call>\n{json}\n</tool_call>`.

## Decisions
- Tool-call output format: Qwen native `<tool_call>{json}</tool_call>`.
- torch is unpinned; on the laptop `uv sync` installs the CPU-only wheel, Kaggle keeps its own CUDA build.
- transformers >=5.20,<6 and trl >=1.18,<2: the code uses their current argument names
  (`warmup_steps` as a ratio, `activation_checkpointing`, `max_length`).
- Rows are rendered to strings before entering `datasets`, so Arrow can't add `null` argument keys.
- Rows longer than `max_seq_length` fail the run instead of being silently truncated
  (truncation would cut off the tool call). Stage 3's "reduce max_seq_length" fix must respect this.
- Stage 1 trains in-process; the one-subprocess-per-attempt rule arrives with the stage 3 runner.
- Eval loads the base model in 4-bit for both base and tuned runs, so the comparison only
  changes one thing: the adapter. `--no-load-in-4bit` exists for fp16 eval.
- `max_new_tokens=128` is fixed: the longest target tool call is 58 Qwen tokens.
- Inside `<tool_call>` or a code fence the payload must itself be a JSON object; only free text
  gets the "first JSON object anywhere" search.
- Tests block all network access (see `tests/conftest.py`).
