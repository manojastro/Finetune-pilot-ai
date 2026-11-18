# 2-minute demo script

Everything below runs on a laptop without a GPU (dry runs). Say clearly that dry-run numbers are
simulated; real GPU numbers are not yet measured.

**Before recording**

```bash
uv sync
rm -rf runs/sessions reports/s-*.md            # start with an empty session list
uv run uvicorn finetunepilot.api.app:app --port 8000 &
```

Have three terminal tabs ready, plus a browser tab on `http://localhost:8000/`.
Use a large terminal font and `COLUMNS=120`.

---

### 0:00–0:15 The problem (talk over the README architecture diagram)

> "Fine-tuning a small model is mostly babysitting: pick settings, run out of GPU memory, retry,
> evaluate, guess what to change. FineTunePilot automates that loop. The LLM only proposes
> configs; deterministic code trains, evaluates and records everything."

**Show:** the Mermaid architecture diagram in the README on GitHub.

### 0:15–0:45 OOM recovery

```bash
uv run finetunepilot run -c configs/qwen05b_toolcall.yaml --dry-run --force-oom
```

> "I force batch size 16 and skip the memory check. Each attempt is a fresh subprocess, so an
> out-of-memory crash can't poison the next one. The runner classifies the error and applies the
> next fix: halve the batch while keeping the effective batch, then gradient checkpointing.
> Third attempt succeeds."

**Show:** the attempts table (failed, failed, success, with the fix for each row).

### 0:45–1:25 The agent loop

```bash
uv run finetunepilot agent --objective "Pick the right tool with exact JSON arguments" --dry-run
```

Answer `y` to the first two runs.

> "It scores the base model first, then plans a config. Before every run it shows the config, the
> VRAM estimate and the time, and asks me. After each run it evaluates on held-out data, asks the
> LLM for exactly one change, and continues only if the score improved. In a real session the
> LLM is Claude, and every answer is validated with pydantic, retried once, or the session fails
> with a report."

**Show:** one approval prompt (config, VRAM estimate vs. limit), then the per-run
`args_exact_match` lines.

Answer `n` on the third prompt:

> "I can stop it at any point. It still writes a report."

### 1:25–1:50 The report and the UI

**Show:** the browser tab and refresh. Click the newest session.

> "Every session and run is listed with metrics, links to the raw run logs, and the report: what
> changed in each run and why, which attempt needed recovery, and the best run. Dry-run numbers
> are labelled as simulated. In a real report, every number is copied from a run log."

**Show:** the results table and the "What changed and why" section; click one run-log link to
show the JSON (config, hardware, attempts, git commit).

### 1:50–2:00 Close

```bash
uv run pytest -q -m "not slow"
```

> "Tested without a GPU, network or API key: tiny local models for every GPU path, CI on every
> push. Next step is the real T4 runs on Kaggle; the notebooks are ready."

---

**If something goes wrong live:** `reports/examples/dry-run-session.md` is a pre-generated dry-run
report to show instead.
