"""LoRA fine-tuning on the merchant normalization dataset.

Uses a plain transformers Trainer with chat-template-formatted examples and
prompt tokens masked out of the loss (labels = -100), which keeps the script
independent of TRL API churn. Saves the adapter, the training config and a
loss curve CSV to the output directory.

Usage:
    python -m merchant_llm.train --train-data data/train.jsonl --val-data data/val.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from merchant_llm.config import DEFAULT_MODEL
from merchant_llm.data import read_jsonl
from merchant_llm.prompts import build_training_messages

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class SupervisedDataset(torch.utils.data.Dataset):
    """Chat-formatted examples with the prompt part masked out of the loss."""

    def __init__(self, examples: list[dict], tokenizer, max_len: int):
        self.rows = [self._encode(ex, tokenizer, max_len) for ex in examples]

    @staticmethod
    def _token_ids(encoded) -> list[int]:
        # transformers v5 returns a BatchEncoding here; v4 returned a bare list
        return encoded if isinstance(encoded, list) else encoded["input_ids"]

    @classmethod
    def _encode(cls, example: dict, tokenizer, max_len: int) -> dict:
        messages = build_training_messages(example)
        full_ids = cls._token_ids(tokenizer.apply_chat_template(messages, tokenize=True))
        prompt_ids = cls._token_ids(
            tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                "chat template does not render the prompt as a prefix of the full "
                "conversation; label masking would be wrong for this tokenizer"
            )
        full_ids = full_ids[:max_len]
        labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids) :]
        return {"input_ids": full_ids, "labels": labels}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


@dataclass
class PadCollator:
    """Right-pads input_ids with pad_token_id and labels with -100."""

    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention_mask = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attention_mask.append([1] * (max_len - pad) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def write_loss_curve(log_history: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "train_loss", "eval_loss"])
        for entry in log_history:
            if "loss" in entry:
                writer.writerow([entry["step"], entry["loss"], ""])
            elif "eval_loss" in entry:
                writer.writerow([entry["step"], "", entry["eval_loss"]])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--val-data", type=Path, default=Path("data/val.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("models/adapter"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--scheduler", default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=150)
    parser.add_argument(
        "--subsample", type=int, default=None, help="train on only the first N rows (CPU runs)"
    )
    args = parser.parse_args(argv)

    set_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    fp16 = use_cuda and not bf16

    train_rows = read_jsonl(args.train_data)
    val_rows = read_jsonl(args.val_data)
    if args.subsample:
        train_rows = train_rows[: args.subsample]

    print(f"loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if bf16 else torch.float32
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = SupervisedDataset(train_rows, tokenizer, args.max_len)
    val_dataset = SupervisedDataset(val_rows, tokenizer, args.max_len)
    print(f"train={len(train_dataset)} val={len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir="models/checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        # only track eval loss; otherwise the Trainer gathers full-vocab logits
        # for the whole val set, which is brutally slow and memory-hungry
        prediction_loss_only=True,
        save_strategy="no",
        bf16=bf16,
        fp16=fp16,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    start = time.perf_counter()
    trainer.train()
    wall_seconds = time.perf_counter() - start

    final_eval = trainer.evaluate()
    print(f"final eval loss: {final_eval['eval_loss']:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    write_loss_curve(trainer.state.log_history, args.out / "loss_curve.csv")

    import peft
    import transformers

    config_dump = {
        "base_model": args.model,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": LORA_TARGET_MODULES,
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "scheduler": args.scheduler,
        "warmup_steps": args.warmup_steps,
        "max_len": args.max_len,
        "seed": args.seed,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "precision": "bf16" if bf16 else ("fp16" if fp16 else "fp32"),
        "device": torch.cuda.get_device_name(0) if use_cuda else f"cpu ({platform.processor()})",
        "wall_seconds": round(wall_seconds, 1),
        "final_eval_loss": round(final_eval["eval_loss"], 4),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }
    (args.out / "training_config.json").write_text(
        json.dumps(config_dump, indent=2), encoding="utf-8"
    )
    print(f"saved adapter + config -> {args.out} (wall time {wall_seconds / 60:.1f} min)")


if __name__ == "__main__":
    main()
