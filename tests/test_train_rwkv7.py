from pathlib import Path

from state_model_interface.train_rwkv7 import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    parse_args,
)


def test_training_cli_defaults_to_pinned_model_and_tensorboard() -> None:
    args = parse_args(["--dataset", "owner/data", "--output", "run"])

    assert args.model == DEFAULT_MODEL
    assert args.revision == DEFAULT_REVISION
    assert args.output == Path("run")
    assert args.report_to == "tensorboard"
    assert args.logging_dir is None
    assert args.logging_steps == 10
    assert args.no_packing is False
    assert args.no_gradient_checkpointing is False
