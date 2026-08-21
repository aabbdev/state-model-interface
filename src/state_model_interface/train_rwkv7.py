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


def _json_field(value: Any, name: str) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} contains invalid JSON") from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-fine-tune RWKV7 with SMI")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset ID")
    parser.add_argument("--dataset-config")
    parser.add_argument("--data-files", nargs="+")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-num-proc", type=int, default=8)
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
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--logging-dir", type=Path)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--report-to", choices=("tensorboard", "none"), default="tensorboard"
    )
    parser.add_argument("--full-loss", action="store_true")
    parser.add_argument("--no-packing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--wkv-implementation", choices=("chunked", "eager"), default="chunked"
    )
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    import torch
    from datasets import load_dataset
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedConfig
    from transformers.integrations.integration_utils import TensorBoardCallback
    from transformers.trainer_callback import TrainerCallback
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
    model.config.wkv_implementation = args.wkv_implementation
    token_ids = install_smi_tokens(tokenizer, model)
    tokenizer.chat_template = load_chat_template(args.chat_template)

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        data_files=args.data_files,
        split=args.split,
    )
    if args.messages_column not in dataset.column_names:
        raise KeyError(f"dataset has no {args.messages_column!r} column")

    def compile_row(example: dict[str, Any]) -> dict[str, list[int]]:
        template_kwargs = example.get("chat_template_kwargs") or {}
        messages = _json_field(example[args.messages_column], args.messages_column)
        tools = _json_field(example.get("tools", example.get("tools_json")), "tools")
        smi_ctrl = _json_field(example.get("smi_ctrl"), "smi_ctrl")
        smi_caps = _json_field(example.get("smi_caps"), "smi_caps")
        compiled = compile_smi(
            tokenizer,
            messages,
            tools=tools,
            smi_ctrl=smi_ctrl,
            smi_caps=smi_caps,
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
        num_proc=args.dataset_num_proc,
        desc="Compiling SMI token streams",
    )

    logging_dir = args.logging_dir or args.output / "tensorboard"

    class GpuMetricsCallback(TrainerCallback):
        def __init__(self, writer: SummaryWriter) -> None:
            self.writer = writer

        def on_train_begin(self, args, state, control, **kwargs):
            del args, state, control, kwargs
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def on_log(self, args, state, control, logs=None, **kwargs):
            del args, control, kwargs
            if not torch.cuda.is_available():
                return
            gib = 1024**3
            self.writer.add_scalar(
                "gpu/memory_allocated_gib",
                torch.cuda.memory_allocated() / gib,
                state.global_step,
            )
            self.writer.add_scalar(
                "gpu/memory_reserved_gib",
                torch.cuda.memory_reserved() / gib,
                state.global_step,
            )
            self.writer.add_scalar(
                "gpu/max_memory_allocated_gib",
                torch.cuda.max_memory_allocated() / gib,
                state.global_step,
            )
            self.writer.flush()

    callbacks = None
    if args.report_to == "tensorboard":
        writer = SummaryWriter(log_dir=str(logging_dir))
        callbacks = [TensorBoardCallback(tb_writer=writer), GpuMetricsCallback(writer)]

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
        shuffle_dataset=True,
        assistant_only_loss=False,
        completion_only_loss=False,
        use_cache=False,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        bf16=args.dtype == "bfloat16",
        fp16=False,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        include_num_input_tokens_seen=True,
        report_to="none",
        run_name=args.run_name,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
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
    args.output.joinpath("smi_training_config.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "base_revision": args.revision,
                "wkv_implementation": args.wkv_implementation,
                "max_length": args.max_length,
                "packing": not args.no_packing,
                "assistant_only_loss": not args.full_loss,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_MODEL", "DEFAULT_REVISION", "main", "parse_args"]
