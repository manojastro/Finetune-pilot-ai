# FineTunePilot status

Labels: **built** = code written · **tested** = tests pass on laptop (CPU) · **measured** = real numbers from a GPU run · **not yet run**.

| stage | built | tested | measured | notes |
|---|---|---|---|---|
| 0 — setup | yes | yes | n/a | uv project, folder layout, ruff/pytest config, `finetunepilot version` CLI |
| 1 — baseline QLoRA training | yes | yes | not yet run | dry run passes on CPU; Kaggle T4 run pending |
| 2 — evaluation | yes | yes | not yet run | base vs tuned on Kaggle pending |
| 3 — VRAM estimate + OOM recovery | yes | yes | not yet run | `--force-oom` on Kaggle pending |
| 4 — agent (LangGraph) | yes | yes | not yet run | dry run completes on CPU; real session pending |
| 5 — API, UI, Docker | yes | yes | n/a | 198 tests total (7 slow); image built and a dry-run session served from `docker run`; not deployed |
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

## Stage 3 detail

| item | status |
|---|---|
| `tools/vram.py` estimate with breakdown + `fits` (incl. logits term) | built, tested |
| activation coefficients (16.5·h + 5·I per token per layer; h with checkpointing) | measured on CPU (autograd saved bytes, Qwen2.5-0.5B shapes + LoRA r=16); GPU accuracy not yet measured |
| `tools/log_parser.py` → CUDA_OOM / NAN_LOSS / DATASET_ERROR / TOKENIZER_ERROR / UNKNOWN + evidence | built, tested |
| NaN-loss callback in `train.py` | built, tested |
| `worker.py`: one attempt per subprocess, result file, crash + timeout handling | built, tested |
| `runner.py`: pre-check, OOM ladder, NaN fix, budget, allowed-changes guard, per-attempt log | built, tested (fake attempts) |
| `finetunepilot run [--force-oom] [--dry-run]` | built, tested (dry run: 16 OOM → 8 OOM → 8+GC succeeds, OOM simulated from the estimate) |
| `notebooks/03_recovery_kaggle.ipynb` | built; not yet run |

Estimates for the default T4 config (these are estimates, not measurements):

| config | estimate | limit (14.7 GiB − 15%) | fits |
|---|---|---|---|
| batch 4, longest row 762 tokens | 11.0 GiB | 12.5 GiB | yes |
| batch 4, padded to 1024 | 14.4 GiB | 12.5 GiB | no |
| batch 16 (`--force-oom`) | 40.0 GiB | 12.5 GiB | no |
| batch 8 | 20.7 GiB | 12.5 GiB | no |
| batch 8 + gradient checkpointing | 9.7 GiB | 12.5 GiB | yes |

## Stage 4 detail

| item | status |
|---|---|
| `agent/llm.py`: `LLMClient`, `AnthropicClient` (model from `configs/agent.yaml`), `FakeLLM`, `ask_json` (validate, 1 retry, fail) | built, tested (fake SDK client; no network) |
| `agent/state.py`: settings, `PlanProposal` / `ConfigChange` schemas, only 10 tunable fields | built, tested |
| `agent/nodes.py` + `graph.py`: prepare → baseline → plan → estimate → approve → train → evaluate → diagnose → decide → report | built, tested |
| checkpoint after every node (SQLite) + `--resume` | built, tested (simulated crash, then resume) |
| `agent/toolbox.py`: real jobs, or simulated GPU work for `--dry-run` | built, tested |
| `agent/report.py`: `reports/<session_id>.md` | built, tested |
| `finetunepilot agent ...` | built, tested (dry run) |
| `notebooks/04_agent_kaggle.ipynb` (real LLM + GPU) | built; not yet run |
| Anthropic API calls | **never made** (tests use fakes; a real session needs your key and approval) |

Dry-run numbers in agent reports are simulated by a toy quality function and are labelled so;
they are not measurements.

## Stage 5 detail

| item | status |
|---|---|
| `api/app.py`: `/health`, `POST/GET /sessions`, `GET /sessions/{id}`, `GET /runs/{id}` | built, tested (TestClient, dry-run sessions) |
| real sessions over the API | refused (403) unless `FINETUNEPILOT_ALLOW_REAL=1`; tested with a fake deps factory |
| `api/static/index.html` run-history UI (no build step) | built; checked in headless Chromium (no JS errors, sessions/runs/report render) |
| `Dockerfile` (CPU torch, non-root, healthcheck) | built locally (4.06 GB as reported by Docker); `docker run` served the UI and completed a dry-run session |
| `.github/workflows/ci.yml` (ruff + `pytest -m "not slow"`) | written; not yet run on GitHub |
| deployment | not done (left for later, as asked) |

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
- The pre-check estimates at the longest row in the data (measured with the tokenizer before
  launching), not `max_seq_length`, because batches are only padded to their longest row.
- OOM fixes are applied one per failure in the given order, then the batch keeps halving.
  If the pre-check can't make the config fit with allowed fixes, the run stops before launching.
- `--force-oom` sets batch 16 and skips the pre-check (otherwise the pre-check would prevent the OOM).
- Dry runs simulate OOM when the real config's estimate exceeds the GPU size, so the recovery
  loop can be shown on a laptop.
- The error parser reads only message lines, never indented traceback code lines (which caused
  false TOKENIZER_ERROR matches in testing).
- `finetunepilot train` stays single-attempt and in-process; `finetunepilot run` is the
  subprocess-per-attempt path.
- Agent graph change vs. the original prompt: `decide` loops back to `estimate` (not `approve`),
  so every diagnosed change is memory-checked before it runs. A `prepare` + `baseline` step was
  added so "improved" has a base-model score to compare with.
- The LLM can only set 10 hyperparameters; model, data paths, 4-bit, precision and seed are code-owned.
- `diagnose` must change exactly one field to a new value; `decide` continues only if the last
  run beat the best score so far (args_exact_match, then tool_name_accuracy, then json_valid).
- The Anthropic client uses `claude-opus-5-5` from `configs/agent.yaml`, effort `medium`, and
  server-side refusal fallback (`fallbacks: default`); set `fallbacks: null` to turn it off.
- Run ids got 32 random bits (was 16): a flaky test showed same-second collisions.
- The API treats posting a session as approval of all its runs (no terminal to ask), and
  refuses real sessions unless the server is started with `FINETUNEPILOT_ALLOW_REAL=1`.
- The UI escapes all text before rendering markdown (reports contain LLM-written reasons).
- The Docker base image is a build argument (`BASE_IMAGE`) so a Docker Hub mirror can be used
  when Docker Hub rate-limits anonymous pulls; uv is installed from PyPI.
- Tests block all network access (see `tests/conftest.py`).
