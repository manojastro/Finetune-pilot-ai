# Interview notes: 10 likely questions

Short answers, each pointing to the code. GPU results are **not yet measured**; say so if asked.

---

### 1. What does FineTunePilot do, in one minute?

It automates the fine-tuning loop for a small model:

1. Score the base model.
2. Ask an LLM for a QLoRA config.
3. Check the config fits the GPU (`tools/vram.py`).
4. Train with automatic OOM recovery (`runner.py`, one subprocess per attempt).
5. Score the adapter on a held-out set (`eval.py`, `tools/scoring.py`).
6. Ask the LLM for one change, and repeat while the score improves.
7. Write a report.

The first task teaches Qwen2.5-0.5B to call tools. The LLM only proposes configs; code does all
the work, and every number comes from a run log.

### 2. Why not let the LLM run the whole thing as a tool-using agent?

Containment and testability. A config proposal is the riskiest output I let through, and it gets
checked four times:

- pydantic ranges, and only 10 tunable fields (`agent/state.py`);
- the VRAM estimate (`nodes.estimate`);
- human approval (`nodes.approve`);
- the runner's allowed-changes guard.

Model, data paths, 4-bit and seed are code-owned. The flow is a fixed LangGraph graph
(`agent/graph.py`), so I can test every branch with a scripted `FakeLLM`: bad JSON, a plan that
doesn't fit, budget stop, crash and resume.

### 3. How do you handle invalid LLM output?

`agent/llm.ask_json` pulls out the JSON object (from a code fence or anywhere in the text),
validates it with the pydantic schema, then runs an extra check (for example "the new value must
differ" or "max_seq_length ≥ longest row"). On failure it sends the error back once; a second
failure raises `InvalidLLMOutput`. The node wrapper catches that and the session ends with a
report naming the error. Every LLM answer, valid or not, is logged to `<session>.llm.jsonl`.

### 4. Walk me through OOM recovery.

`runner.run_with_recovery`:

1. **Pre-check:** estimate at the longest real row. If it won't fit, apply fixes on paper; if
   nothing fits, refuse to start.
2. **Run** the attempt in a fresh subprocess (`worker.py`), so a crash releases all GPU memory.
3. **Classify** the failure with `tools/log_parser.classify_error`, which reads message lines
   only, not traceback code.
4. **Fix and retry.** For OOM, the fixes go in order: halve batch and double grad accumulation
   (same effective batch), then gradient checkpointing, then cut sequence length 25% (never below
   the longest row), then keep halving. For NaN: halve the learning rate. Anything else stops.
5. **Stop** at max attempts or minutes; the subprocess is killed at the deadline.

Every attempt is logged with its config diff, error class, evidence line and duration.
`--force-oom` demos it by forcing batch 16 *and* skipping the pre-check (otherwise the pre-check
would prevent the OOM).

### 5. Why a subprocess per attempt instead of `torch.cuda.empty_cache()`?

After an OOM, the exception's traceback, the trainer and the optimizer can still hold references,
and the allocator is fragmented. `empty_cache` doesn't reliably give that memory back, so the
retry can OOM at a size that would have fit in a fresh process. A subprocess makes recovery
deterministic, for a few seconds of start-up. The worker reports through a JSON file; if it dies
without one, the runner classifies the error from the captured output (`make_subprocess_attempt`).

### 6. How accurate is your VRAM estimate?

On a GPU: not yet measured, and I'd say that directly. The components:

- weights: 4-bit linear layers, plus fp32 embeddings because `prepare_model_for_kbit_training`
  upcasts them;
- LoRA parameters, gradients, paged 8-bit AdamW state;
- activations;
- logits at `b·s·vocab·4·2` bytes, often the biggest term with a 152k vocabulary;
- CUDA context, plus 10% fragmentation.

The activation coefficients aren't guessed: I measured autograd's saved tensors on CPU for the
real layer shapes with LoRA, giving 16.5·hidden + 5·MLP values per token per layer, or exactly
`hidden` with checkpointing. One finding mattered: estimating at `max_seq_length` (1024) says the
default config doesn't fit a T4, but batches are padded only to the longest row (762), where it
does. Every run log stores `peak_gpu_memory_reserved_mb`, so calibrating is a lookup once real
runs exist.

### 7. How do you make sure the evaluation is fair?

- **No leakage:** train and test use different templates and argument values, and a test asserts
  no request appears in both (`tests/test_data.py`).
- **Same conditions:** base and tuned are evaluated the same way (4-bit, greedy, same prompt
  builder as training, fixed `max_new_tokens=128`; the longest target is 58 tokens).
- **Strict scoring:** `args_exact_match` requires the right tool *and* equal arguments, with
  sensible number handling. A separate `format_ok` metric shows whether the exact trained format
  was used, so a lenient parser can't hide format drift.
- **Pipeline check:** an oracle fake model must score 1.0 through the real `evaluate()`, which
  checks batching, left padding and slicing.

The honest caveat: `diagnose` sees test failures, so across runs the agent's choices are tuned on
the test set. A validation split is the fix.

### 8. What was the trickiest bug?

Two to mention:

- **Arrow inventing null arguments.** Loading nested JSON straight into `datasets.Dataset` makes
  Arrow merge every tool's argument dict into one schema, so `"unit": null` would appear in
  training targets. Rows are rendered to strings first, and a test checks no `null` leaks.
- **A flaky test that exposed a real collision.** Run IDs had 16 random bits within one second;
  50 IDs in a burst collided about 2% of the time, which would overwrite run logs. Now 32 bits.

Both were caught by tests, not in production.

### 9. How is it tested without a GPU?

Every GPU path has a dry run:

- **Training and eval:** a 2-layer random Qwen2 model and a tokenizer trained locally in under a
  second, carrying the *real* Qwen2.5 chat template, so formatting and loss masking are tested
  exactly.
- **Recovery:** the worker can simulate OOM from the estimate, so the loop runs on a laptop.
- **The agent:** a rule-based LLM plus simulated GPU work, labelled "simulated" everywhere.

`conftest.py` blocks network sockets in all tests. There are 202 tests (7 slow); CI runs ruff and
the fast ones on every push.

### 10. What would you do next / what are the limits?

**Limits:**

- No measured GPU results yet; the Kaggle notebooks are ready.
- The data is tiny and synthetic (60/20 rows, 6 tools), so scores would show learning *these*
  tools, not general tool calling.
- The estimator isn't calibrated on real hardware.
- The API runs sessions in-process, with no queue or auth.

**Next:**

1. Run on Kaggle and calibrate the estimator.
2. A validation split for diagnosis.
3. A bigger, more varied dataset with multi-call and "no tool" cases.
4. A job queue behind the API.
