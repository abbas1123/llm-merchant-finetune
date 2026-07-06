"""Model loading and batched greedy generation.

This is the only module (besides train.py) that imports torch/transformers,
so everything CI tests stays importable without them.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from merchant_llm.prompts import build_messages

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model(
    model_name: str = DEFAULT_MODEL,
    adapter_path: str | None = None,
    device: str = "auto",
):
    """Load tokenizer + model, optionally with a LoRA adapter merged in.

    Returns (model, tokenizer, device_str).
    """
    device = resolve_device(device)
    dtype = resolve_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.to(device)
    model.eval()
    return model, tokenizer, device


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    raws: list[str],
    device: str,
    max_new_tokens: int = 64,
) -> list[str]:
    """Greedy-decode completions for a batch of raw descriptors."""
    prompts = [
        tokenizer.apply_chat_template(
            build_messages(raw), tokenize=False, add_generation_prompt=True
        )
        for raw in raws
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    completions = outputs[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(completions, skip_special_tokens=True)


def generate_all(
    model,
    tokenizer,
    raws: list[str],
    device: str,
    batch_size: int = 32,
    max_new_tokens: int = 64,
    progress: bool = True,
) -> list[str]:
    """Run generation over a full split in batches."""
    outputs: list[str] = []
    for start in range(0, len(raws), batch_size):
        batch = raws[start : start + batch_size]
        outputs.extend(generate_batch(model, tokenizer, batch, device, max_new_tokens))
        if progress:
            print(f"  generated {min(start + batch_size, len(raws))}/{len(raws)}", flush=True)
    return outputs
