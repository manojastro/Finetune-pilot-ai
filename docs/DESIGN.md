# FineTunePilot: design decisions and trade-offs

This explains *why* the code is shaped the way it is. Numbers marked "estimate" come from
`tools/vram.py`, not from a GPU. Real GPU numbers have not been measured yet (see `STATUS.md`).

## 1. Why QLoRA

**Decision:** 4-bit NF4 base model (double quantization) + LoRA adapters (r=16, α=32) on all seven
linear layers, fp16 compute on the T4.

- **It fits free hardware.** The target is a free Kaggle/Colab T4 (≈14.7 GiB usable, no bf16).
  The 0.5B model's linear layers take about 0.17 GiB in 4-bit. The fp32 embedding table
  (`prepare_model_for_kbit_training` upcasts non-quantized parameters) takes another 0.5 GiB. So
  most of the memory goes to activations and logits, which batch size and sequence length control.
- **It's cheap to iterate.** LoRA trains 8,798,208 parameters (checked by a unit test against the
  architecture), about 1.8% of the model. Adapters are a few tens of MB, so each agent run produces
  a small artifact.
- **The trade-offs:**
  - 4-bit weights cost some quality versus a bf16 LoRA, and every forward pass dequantizes.
  - The T4 forces fp16, which overflows more easily than bf16. That's one reason there's a NaN
    detector and an automatic learning-rate halving.
- **Evaluation stays fair.** Base and tuned models are both evaluated in 4-bit, so the adapter is
  the only difference in a comparison.

**Loss on the tool call only.** The prompt holds the system text and six JSON tool schemas, about
700 tokens; the answer is at most 58 tokens. Training on the whole sequence would mostly teach the
model to reproduce schemas. `train.py` uses TRL's prompt/completion format with
`completion_only_loss=True`, and a test decodes the labelled tokens to check that they are exactly
the `<tool_call>` block plus one `<|im_end|>`.

**The model's native format.** The target is Qwen2.5's own `<tool_call>{json}</tool_call>`, which
its chat template already produces. Fine-tuning reinforces a format the base model knows, and the
base-vs-tuned comparison isn't skewed by a format only the tuned model saw. The parser still
accepts bare JSON and code fences, and a separate `format_ok` metric records whether the exact
format was used.

## 2. Why the LLM only picks configs

**Decision:** the LLM is called in exactly two places (`plan`, `diagnose`). It returns a small JSON
object, and deterministic code does everything else.

- **Failures stay contained.** A wrong LLM answer can only produce a bad config, and a bad config
  is caught by:
  - pydantic range checks;
  - the "not below the longest row" check;
  - the VRAM estimate;
  - your approval step;
  - the runner's recovery loop.

  It can't delete files, run code, or touch the model, data paths, 4-bit or seed. Those are
  code-owned in `agent/state.build_train_config`, and the `PlanProposal` and `ConfigChange`
  schemas only accept 10 tunable fields.
- **Validate, retry once, fail clearly.** `agent/llm.ask_json` parses, validates and runs
  extra checks. On failure it sends the error back once; a second failure raises
  `InvalidLLMOutput` and the session ends with a report. That makes LLM behaviour testable with
  scripted `FakeLLM` answers.
- **One change per diagnosis.** If two things change at once, you can't tell which one helped.
  `diagnose` must name a single field with a new value, and `decide` keeps going only while the
  score improves.
- **Everything is reproducible.** Every LLM answer is appended to `<session>.llm.jsonl`, and the
  numbers the LLM sees all come from run logs.
- **The trade-off:** the agent can't invent strategies outside the tunable fields (new data, a
  different model). That's on purpose for a first version.

The LLM runs on Claude through the Messages API: `claude-opus-5-5` in `configs/agent.yaml` (not
in code), effort `medium`, and server-side refusal fallback on. This is a code-controlled
workflow, not a free-roaming tool-using agent, so plain single calls fit better than an agent
framework.

## 3. Why VRAM is an estimate (and how it was built)

Exact peak memory depends on allocator fragmentation, kernel choices, bitsandbytes internals and
padding, which you only learn by running. An estimate is still worth having: it rejects obviously
impossible configs before you spend GPU minutes, and it tells the LLM *why* a plan was rejected.

- **Terms:** 4-bit weights, fp32 embeddings and norms, LoRA parameters, gradients, optimizer
  state, activations, logits, a fixed CUDA context and +10% fragmentation.
- **Activations were measured, not guessed.** Autograd's saved-tensor bytes were measured on CPU
  for Qwen2.5-0.5B shapes with LoRA: 16.5·hidden + 5·MLP values per token per layer, or exactly
  `hidden` with gradient checkpointing.
- **Logits are explicit.** With a 152k vocabulary, `batch × seq × vocab × 4 bytes × 2` is often
  the biggest term, about 3.4 GiB at batch 4 × 762 tokens.
- **Sequence length is the real longest row**, not `max_seq_length`. Batches are padded only to
  their longest row. At 1024 the default config "doesn't fit" a T4 (14.4 GiB estimate); at the
  real 762 tokens it does (11.0 GiB against a 12.5 GiB limit).
- **There's a safety margin:** 15% below the GPU's size.
- **The trade-off:** coefficients measured for one architecture may be off for others, and the
  estimate has not been checked against a real T4 yet. Every run log records
  `peak_gpu_memory_reserved_mb`, so that check is a lookup once the Kaggle runs exist.

## 4. Budget guards

| guard | where | why |
|---|---|---|
| `max_attempts` per training run | `runner.RecoveryPolicy` | recovery can't loop forever |
| `max_minutes` per run; the attempt subprocess is killed at the deadline | `runner.run_with_recovery`, `make_subprocess_attempt` | a hung attempt can't eat the budget |
| `max_runs`, `max_minutes` per session | `agent/nodes.decide` | caps GPU time and LLM calls |
| stop when a run doesn't improve | `decide` | no aimless spending |
| `allowed_changes` list + a final check that the config diff stays inside it | `runner` | recovery can't quietly change, say, the learning rate during an OOM fix |
| pre-check refuses a config that can't fit | `runner.precheck` | no GPU minutes on a guaranteed OOM |
| human approval per run (or explicit `--yes`) | `agent/nodes.approve` | you stay in control of spend |
| real API sessions off by default | `api/app.py` (`FINETUNEPILOT_ALLOW_REAL`) | a web request can't start billed work by accident |

## 5. One subprocess per training attempt

After a CUDA OOM, memory held by the dead trainer, the optimizer and the exception's traceback
isn't reliably freed, and the next in-process attempt often OOMs at a size that would have fit.
`worker.py` runs each attempt in a fresh Python process and reports through a JSON result file,
so a crash frees everything.

- **No result file?** If the process dies without writing one (killed, segfault), the runner
  classifies the error from the captured output.
- **The cost:** a few seconds of process start-up and model loading per attempt.

## 6. Error classification

`tools/log_parser.py` maps failures to `CUDA_OOM`, `NAN_LOSS`, `DATASET_ERROR`,
`TOKENIZER_ERROR` or `UNKNOWN`, together with the line that showed it. Only OOM and NaN have
automatic fixes; everything else stops the run.

The parser reads only *message* lines, never the indented code lines of a traceback. An early
version matched `tokenizer` in a traceback code line and misclassified a network error, and a
test now covers that case.

## 7. Data and evaluation

- **The data is synthetic.** 60/20 rows, 6 tools, made by a seeded script. It's small enough to
  train in minutes on a T4 and fully controlled.
- **Train and test don't share templates.** They use different sentence templates *and* different
  argument values, and a test asserts no request appears in both. Without that, template overlap
  would inflate test scores.
- **Raw data on disk.** Rows store raw `messages` + `tools`, and the chat template is applied at
  load time, so the files are model-agnostic. Rows become strings *before* entering
  `datasets.Dataset`. Otherwise Arrow merges the different tools' argument dicts into one schema
  and leaks `"unit": null` into the training text.
- **Strict scoring.** `args_exact_match` needs the right tool and equal arguments: key order is
  ignored, `5 == 5.0`, `"5" != 5`, `True != 1`. An "oracle" fake model must score 1.0 through the
  real eval pipeline, which checks batching, left padding and prompt slicing.
- **Known weakness:** `diagnose` sees test-set failures, so over several runs the agent's choices
  are tuned on the test set. A separate validation split is listed as a next step.

## 8. Run logs and checkpoints

- **Run logs are the source of truth.** Every number in a report, model card or the UI is
  copied from `runs/*.json`, written by the shared `jobs.py` for the CLI and the agent alike.
  Logs are written atomically and also *before* training starts, so a crash leaves a "running"
  record instead of nothing.
- **Checkpoints.** The agent's graph state is checkpointed to SQLite after every node
  (LangGraph `SqliteSaver`), so `--resume` continues from the last finished node without
  repeating finished runs, which a test checks.

## 9. Dry runs

Rule: every GPU path has a CPU path, and tests never need a GPU, network or API key (a
`conftest.py` fixture blocks sockets).

- **Training and eval dry runs** use a 2-layer random Qwen2 model and a byte-level BPE trained in
  under a second, carrying a copy of the *real* Qwen2.5 chat template. Data formatting, LoRA, the
  trainer and saving all run for real.
- **The agent dry run** swaps GPU work for a toy "config quality" function. That tests the control
  flow, not the agent's judgement, and every report and log says so.

## 10. What I would change next

- Measure the estimator against real T4 peaks and fit its constants per architecture.
- Add a validation split for `diagnose`, and keep the test set for the final report only.
- A job queue (separate worker process) behind the API, and authentication before sharing it.
- Let `plan` see measured throughput from earlier sessions to give better time estimates.
