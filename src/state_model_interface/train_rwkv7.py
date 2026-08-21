"""Full supervised fine-tuning entry point for the public RWKV7 SMI profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .compiler import compile_smi, install_smi_tokens
from .template import load_chat_template

DEFAULT_MODEL = "aabbdev/RWKV7-1.5B-20260805"
DEFAULT_REVISION = "5904f9d1cdb05a565e5da9304db0447c8a8eb938"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-fine-tune RWKV7 with SMI")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset ID")
    parser.add_argument("--dataset-config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--messages-column", default="messages")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--full-loss", action="store_true")
    parser.add_argument("--no-packing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedConfig
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        config=PreTrainedConfig(),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        dtype=dtype,
    )
    model.config.use_cache = False
    token_ids = install_smi_tokens(tokenizer, model)
    tokenizer.chat_template = load_chat_template(args.chat_template)

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
    )
    if args.messages_column not in dataset.column_names:
        raise KeyError(f"dataset has no {args.messages_column!r} column")

    def compile_row(example: dict[str, Any]) -> dict[str, list[int]]:
        template_kwargs = example.get("chat_template_kwargs") or {}
        compiled = compile_smi(
            tokenizer,
            example[args.messages_column],
            tools=example.get("tools"),
            smi_ctrl=example.get("smi_ctrl"),
            smi_caps=example.get("smi_caps"),
            token_ids=token_ids,
            assistant_only_loss=not args.full_loss,
            preserve_thinking=bool(template_kwargs.get("preserve_thinking", True)),
        )
        if not args.full_loss and all(label == -100 for label in compiled.labels):
            raise ValueError("training example contains no model-side target tokens")
        return {"input_ids": compiled.input_ids, "labels": compiled.labels}

    dataset = dataset.map(
        compile_row,
        remove_columns=dataset.column_names,
        desc="Compiling SMI token streams",
    )

    training_args = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        packing=not args.no_packing,
        packing_strategy="bfd",
        assistant_only_loss=False,
        completion_only_loss=False,
        use_cache=False,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        bf16=args.dtype == "bfloat16",
        fp16=False,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    args.output.joinpath("chat_template.jinja").write_text(
        tokenizer.chat_template,
        encoding="utf-8",
    )
    args.output.joinpath("smi_token_ids.json").write_text(
        json.dumps(token_ids, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_MODEL", "DEFAULT_REVISION", "main", "parse_args"]
