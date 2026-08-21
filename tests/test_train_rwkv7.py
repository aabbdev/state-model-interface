from pathlib import Path

from state_model_interface.train_rwkv7 import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    _json_field,
    parse_args,
)


def test_training_cli_defaults_to_pinned_model_and_tensorboard() -> None:
    args = parse_args(["--dataset", "owner/data", "--output", "run"])

    assert args.model == DEFAULT_MODEL
    assert args.revision == DEFAULT_REVISION
    assert args.output == Path("run")
    assert args.data_files is None
    assert args.dataset_num_proc == 8
    assert args.report_to == "tensorboard"
    assert args.logging_dir is None
    assert args.logging_steps == 10
    assert args.save_steps == 500
    assert args.save_total_limit == 1
    assert args.no_packing is False
    assert args.gradient_checkpointing is False
    assert args.wkv_implementation == "auto"

    tilelang_args = parse_args(
        [
            "--dataset",
            "owner/data",
            "--output",
            "run",
            "--wkv-implementation",
            "chunked",
        ]
    )
    assert tilelang_args.wkv_implementation == "chunked"


def test_json_fields_accept_structures_and_reject_invalid_json() -> None:
    value = [{"role": "user", "content": "hello"}]
    assert _json_field(value, "messages") is value
    assert _json_field('[{"role":"user","content":"hello"}]', "messages") == value

    try:
        _json_field("{broken", "messages")
    except ValueError as error:
        assert "messages contains invalid JSON" in str(error)
    else:
        raise AssertionError("invalid JSON was accepted")
