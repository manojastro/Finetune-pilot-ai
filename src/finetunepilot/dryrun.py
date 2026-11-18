"""A tiny random model and tokenizer, built locally, for dry runs and tests (no downloads).

- Tokenizer: a byte-level BPE trained in under a second on the dataset text, with Qwen's
  special tokens and the real Qwen2.5 chat template (copied into templates/). So dry-run
  data formatting matches what the real model sees.
- Model: a 2-layer Qwen2 architecture (~0.1M params) with random weights. It uses the same
  module names (q_proj, k_proj, ...) as the real model, so the LoRA config is exercised too.

The numbers a dry run produces (loss etc.) are meaningless; it only proves the code path works.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"]


def qwen_chat_template() -> str:
    return (files("finetunepilot") / "templates" / "qwen2_5_chat_template.jinja").read_text(
        encoding="utf-8"
    )


def build_tiny_tokenizer(corpus_files: list[str | Path], vocab_size: int = 1000):
    """Train a small byte-level BPE on the given text files and wrap it as an HF tokenizer."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    texts = [qwen_chat_template()]
    for path in corpus_files:
        texts.extend(Path(path).read_text(encoding="utf-8").splitlines())

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<tool_call>", "</tool_call>"],
    )
    hf_tok.chat_template = qwen_chat_template()
    return hf_tok


def build_tiny_model(tokenizer, seed: int = 0):
    """A 2-layer Qwen2 causal LM with random weights, sized for the tiny tokenizer."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=8192,
        tie_word_embeddings=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=None,
    )
    return Qwen2ForCausalLM(config)
