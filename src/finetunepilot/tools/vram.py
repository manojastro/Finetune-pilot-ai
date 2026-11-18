"""Estimate GPU memory for a QLoRA training config. THIS IS AN ESTIMATE, not a measurement.

All sizes are in GiB (2**30 bytes), the unit torch reports (a "16 GB" T4 shows ~14.7 GiB).

    total = (weights + lora + gradients + optimizer + activations + logits + cuda_context)
            * (1 + fragmentation)
    fits  = total <= gpu_memory_gb * (1 - safety_margin)

Terms (b = batch size, s = sequence length, h = hidden, I = MLP size, L = layers, V = vocab):

weights      4-bit: linear layers at 0.5 byte/param (+3% for NF4 double-quant constants).
             Embeddings, norms (and an untied lm_head) are fp32: prepare_model_for_kbit_training
             upcasts every non-quantized parameter. Not 4-bit: everything at the compute dtype.
lora         r * (in + out) params per target module, fp32 (PEFT keeps adapters in fp32).
gradients    lora params * 4 bytes (only adapters are trained).
optimizer    AdamW: 8 bytes/param (two fp32 states); paged 8-bit AdamW: 2 bytes/param.
activations  without checkpointing: L * b * s * (16.5h + 5I) * act_bytes.
             with checkpointing:    L * b * s * h * act_bytes  +  one layer recomputed in full.
             The coefficients were MEASURED on CPU (autograd saved-tensor bytes) for
             Qwen2.5-0.5B shapes with LoRA r=16 on all linear layers: 39,040 values per token
             per layer without checkpointing, exactly h with it.
logits       b * s * V * 4 bytes * 2: the fp32 logits kept for the loss plus their gradient.
             With a 152k vocab this is often the largest term.
cuda_context ~0.5 GiB fixed (CUDA/cuBLAS context, kernels).
fragmentation +10% on top of everything (allocator caching / fragmentation).

The sequence length should be the longest row actually in the data (batches are padded to the
longest row), not max_seq_length, or the estimate is far too pessimistic. Compare it with the
measured `peak_gpu_memory_reserved_mb` from real runs to see how far off it is.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from finetunepilot.config import TrainConfig

GIB = 2**30
ACT_PER_HIDDEN = 16.5  # activation values saved per token per layer, per unit of hidden size
ACT_PER_INTERMEDIATE = 5.0  # ... per unit of MLP intermediate size
QUANT_OVERHEAD = 1.03  # NF4 + double quantization constants
CUDA_CONTEXT_GIB = 0.5
FRAGMENTATION = 0.10

# Usable memory torch reports for common GPUs, in GiB.
GPU_MEMORY_GIB = {"t4": 14.7, "p100": 15.9, "l4": 22.0, "a10g": 22.0, "a100-40gb": 39.4}


class ModelSpec(BaseModel):
    """The architecture numbers the estimate needs."""

    hidden_size: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_kv_heads: int = Field(gt=0)
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def module_shapes(self) -> dict[str, tuple[int, int]]:
        """(in_features, out_features) of each linear layer in one decoder layer."""
        h, i, kv = self.hidden_size, self.intermediate_size, self.num_kv_heads * self.head_dim
        return {
            "q_proj": (h, h),
            "k_proj": (h, kv),
            "v_proj": (h, kv),
            "o_proj": (h, h),
            "gate_proj": (h, i),
            "up_proj": (h, i),
            "down_proj": (i, h),
        }


_QWEN25_05B = ModelSpec(
    hidden_size=896,
    num_layers=24,
    intermediate_size=4864,
    vocab_size=151936,
    num_attention_heads=14,
    num_kv_heads=2,
)
_QWEN25_15B = ModelSpec(
    hidden_size=1536,
    num_layers=28,
    intermediate_size=8960,
    vocab_size=151936,
    num_attention_heads=12,
    num_kv_heads=2,
)
KNOWN_MODEL_SPECS: dict[str, ModelSpec] = {
    "Qwen/Qwen2.5-0.5B-Instruct": _QWEN25_05B,
    "Qwen/Qwen2.5-0.5B": _QWEN25_05B,
    "Qwen/Qwen2.5-1.5B-Instruct": _QWEN25_15B,
    "Qwen/Qwen2.5-1.5B": _QWEN25_15B,
}


class VramEstimate(BaseModel):
    total_gb: float
    limit_gb: float  # gpu_memory_gb * (1 - safety_margin)
    gpu_memory_gb: float
    fits: bool
    headroom_gb: float  # limit - total (negative = does not fit)
    seq_len: int
    breakdown_gb: dict[str, float]
    lora_params: int


def spec_from_hf_config(config: dict[str, Any]) -> ModelSpec:
    """Build a ModelSpec from a Hugging Face config.json dict (Llama/Qwen/Mistral style)."""
    return ModelSpec(
        hidden_size=config["hidden_size"],
        num_layers=config["num_hidden_layers"],
        intermediate_size=config["intermediate_size"],
        vocab_size=config["vocab_size"],
        num_attention_heads=config["num_attention_heads"],
        num_kv_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
        tie_word_embeddings=config.get("tie_word_embeddings", False),
    )


def get_model_spec(base_model: str) -> ModelSpec:
    """Look up a known model. Raises KeyError naming the fix for unknown models."""
    try:
        return KNOWN_MODEL_SPECS[base_model]
    except KeyError:
        raise KeyError(
            f"No built-in architecture numbers for {base_model!r}. Pass a ModelSpec, e.g. "
            "spec_from_hf_config(AutoConfig.from_pretrained(name).to_dict())."
        ) from None


def lora_param_count(spec: ModelSpec, target_modules: list[str], r: int) -> int:
    shapes = spec.module_shapes()
    unknown = sorted(set(target_modules) - set(shapes))
    if unknown:
        raise ValueError(f"Unknown LoRA target modules for this architecture: {unknown}")
    per_layer = sum(r * (shapes[m][0] + shapes[m][1]) for m in set(target_modules))
    return per_layer * spec.num_layers


def _weight_bytes(spec: ModelSpec, load_in_4bit: bool, compute_bytes: int) -> float:
    linear = sum(i * o for i, o in spec.module_shapes().values()) * spec.num_layers
    embed = spec.vocab_size * spec.hidden_size * (1 if spec.tie_word_embeddings else 2)
    norms = spec.hidden_size * (2 * spec.num_layers + 1)
    if load_in_4bit:
        return linear * 0.5 * QUANT_OVERHEAD + (embed + norms) * 4
    return (linear + embed + norms) * compute_bytes


def estimate_vram(
    cfg: TrainConfig,
    gpu_memory_gb: float,
    spec: ModelSpec | None = None,
    seq_len: int | None = None,
    safety_margin: float = 0.15,
) -> VramEstimate:
    """Estimate peak training memory. `seq_len` defaults to cfg.max_seq_length (worst case);
    pass the longest row in the dataset for a realistic number."""
    if not 0 <= safety_margin < 1:
        raise ValueError("safety_margin must be in [0, 1)")
    spec = spec or get_model_spec(cfg.base_model)
    s = min(seq_len or cfg.max_seq_length, cfg.max_seq_length)
    b = cfg.per_device_batch_size
    half = cfg.fp16 or cfg.bf16
    act_bytes = 2 if half else 4
    tokens = b * s

    lora = lora_param_count(spec, cfg.target_modules, cfg.lora_r)
    per_token_layer = (
        ACT_PER_HIDDEN * spec.hidden_size + ACT_PER_INTERMEDIATE * spec.intermediate_size
    )
    if cfg.gradient_checkpointing:
        activations = (spec.num_layers * spec.hidden_size + per_token_layer) * tokens * act_bytes
    else:
        activations = spec.num_layers * per_token_layer * tokens * act_bytes

    breakdown_bytes = {
        "weights": _weight_bytes(spec, cfg.load_in_4bit, act_bytes),
        "lora_params": lora * 4,
        "gradients": lora * 4,
        "optimizer": lora * (2 if cfg.load_in_4bit else 8),  # paged 8-bit AdamW vs AdamW
        "activations": activations,
        "logits": tokens * spec.vocab_size * 4 * 2,
        "cuda_context": CUDA_CONTEXT_GIB * GIB,
    }
    subtotal = sum(breakdown_bytes.values())
    breakdown_bytes["fragmentation"] = subtotal * FRAGMENTATION
    total = subtotal * (1 + FRAGMENTATION)

    limit = gpu_memory_gb * (1 - safety_margin)
    total_gb = total / GIB
    return VramEstimate(
        total_gb=round(total_gb, 3),
        limit_gb=round(limit, 3),
        gpu_memory_gb=gpu_memory_gb,
        fits=total_gb <= limit,
        headroom_gb=round(limit - total_gb, 3),
        seq_len=s,
        breakdown_gb={k: round(v / GIB, 3) for k, v in breakdown_bytes.items()},
        lora_params=lora,
    )
