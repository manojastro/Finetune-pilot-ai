"""Tool-call accuracy evaluation of a base model or a base model + LoRA adapter.

For each test row: build the same prompt used in training (system + tools + request), generate
greedily with a fixed max_new_tokens, cut the text at the first end-of-turn token, and score it
with the pure functions in tools/scoring.py.

Real run (GPU): base model in 4-bit NF4 by default, so base and tuned are measured the same way
                (the adapter was trained on the 4-bit base).
Dry run (CPU):  tiny random model + local tokenizer. Scores are meaningless (near 0); it only
                proves the code path. With --adapter pointing at a dry-run training output, the
                tiny base model is rebuilt with the same seed so the adapter fits.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from finetunepilot.data import ToolCallRow, build_prompt, load_rows
from finetunepilot.tools.scoring import score_prediction, summarize

DRY_RUN_MAX_NEW_TOKENS = 24  # keeps the CPU dry run fast; the random model emits junk anyway


class EvalSettings(BaseModel):
    base_model: str
    adapter: str | None = None
    test_path: str
    max_new_tokens: int = 128
    batch_size: int = 8
    load_in_4bit: bool = True
    seed: int = 42
    dry_run: bool = False


class EvalResult(BaseModel):
    summary: dict[str, Any]
    predictions: list[dict[str, Any]]
    duration_s: float
    peak_gpu_memory_mb: float | None


def clean_generation(text: str, stop_strings: list[str]) -> str:
    """Cut generated text at the first stop string (end of turn / padding) and strip spaces."""
    cut = len(text)
    for stop in stop_strings:
        if stop:
            idx = text.find(stop)
            if idx != -1:
                cut = min(cut, idx)
    return text[:cut].strip()


def _load_real(s: EvalSettings) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. Real evaluation needs a GPU; use --dry-run on CPU.")
    # The adapter dir holds the exact tokenizer used in training.
    tokenizer = AutoTokenizer.from_pretrained(s.adapter or s.base_model)
    quant = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        if s.load_in_4bit
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        s.base_model, quantization_config=quant, dtype=torch.float16, device_map={"": 0}
    )
    if s.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, s.adapter)
    return model, tokenizer


def _load_dry_run(s: EvalSettings) -> tuple[Any, Any]:
    from finetunepilot.dryrun import build_tiny_model, build_tiny_tokenizer

    if s.adapter:
        from peft import PeftModel
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(s.adapter)
        model = PeftModel.from_pretrained(build_tiny_model(tokenizer, seed=s.seed), s.adapter)
    else:
        tokenizer = build_tiny_tokenizer([s.test_path])
        model = build_tiny_model(tokenizer, seed=s.seed)
    return model, tokenizer


def generate(model: Any, tokenizer: Any, rows: list[ToolCallRow], s: EvalSettings) -> list[str]:
    """Greedy generation, one cleaned response per row, in row order."""
    import torch
    from transformers import GenerationConfig

    tokenizer.padding_side = "left"  # decoder-only models need left padding for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_ids = sorted(
        {i for i in (tokenizer.eos_token_id, tokenizer.pad_token_id) if i is not None}
    )
    gen_config = GenerationConfig(
        max_new_tokens=s.max_new_tokens,
        do_sample=False,  # greedy: deterministic
        eos_token_id=stop_ids,
        pad_token_id=tokenizer.pad_token_id,
    )
    stop_strings = [tokenizer.eos_token, tokenizer.pad_token]
    device = next(model.parameters()).device
    model.eval()

    outputs: list[str] = []
    for start in range(0, len(rows), s.batch_size):
        batch = rows[start : start + s.batch_size]
        prompts = [build_prompt(r, tokenizer) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(**enc, generation_config=gen_config)
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        for seq in new_tokens:
            text = tokenizer.decode(seq, skip_special_tokens=False)
            outputs.append(clean_generation(text, stop_strings))
    return outputs


def evaluate(s: EvalSettings) -> EvalResult:
    """Generate a response for every test row and score it."""
    import torch
    from transformers import set_seed

    start = time.perf_counter()
    if s.dry_run:
        s = s.model_copy(update={"max_new_tokens": min(s.max_new_tokens, DRY_RUN_MAX_NEW_TOKENS)})
    set_seed(s.seed)
    use_cuda = torch.cuda.is_available() and not s.dry_run
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    rows = load_rows(s.test_path)
    model, tokenizer = _load_dry_run(s) if s.dry_run else _load_real(s)
    outputs = generate(model, tokenizer, rows, s)

    scores = [
        score_prediction(text, r.target.name, r.target.arguments)
        for text, r in zip(outputs, rows, strict=True)
    ]
    predictions = [
        {
            "id": r.id,
            "request": r.request,
            "target": r.target.model_dump(),
            "output": text,
            **score.model_dump(),
        }
        for r, text, score in zip(rows, outputs, scores, strict=True)
    ]
    return EvalResult(
        summary=summarize(scores, [r.target.name for r in rows]),
        predictions=predictions,
        duration_s=round(time.perf_counter() - start, 2),
        peak_gpu_memory_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if use_cuda
        else None,
    )


def default_eval_name(adapter: str | Path | None) -> str:
    return "eval-tuned" if adapter else "eval-base"
