import hashlib
import json
from pathlib import Path

from state_model_interface.train_rwkv7 import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    _json_field,
    _validate_precompiled_manifest,
    _validate_precompiled_row,
    parse_args,
)


def test_training_cli_defaults_to_pinned_model_and_tensorboard() -> None:
    args = parse_args(["--dataset", "owner/data", "--output", "run"])

    assert args.model == DEFAULT_MODEL
    assert args.revision == DEFAULT_REVISION
    assert args.output == Path("run")
    assert args.data_files is None
    assert args.precompiled_manifest is None
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


def test_precompiled_rows_are_fail_closed_and_support_full_loss() -> None:
    row = {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}
    assert (
        _validate_precompiled_row(row, vocab_size=4, max_length=3, full_loss=False)
        == row
    )
    assert _validate_precompiled_row(
        row, vocab_size=4, max_length=3, full_loss=True
    ) == {"input_ids": [1, 2, 3], "labels": [1, 2, 3]}

    for invalid in (
        {"input_ids": [], "labels": []},
        {"input_ids": [1], "labels": [2]},
        {"input_ids": [4], "labels": [4]},
        {"input_ids": [1], "labels": [-100]},
        {"input_ids": [1.5], "labels": [-100]},
        {"input_ids": [True], "labels": [True]},
    ):
        try:
            _validate_precompiled_row(
                invalid, vocab_size=4, max_length=3, full_loss=False
            )
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"invalid precompiled row was accepted: {invalid}")


def test_precompiled_manifest_is_pinned_and_complete(tmp_path: Path) -> None:
    data = tmp_path / "pilot.parquet"
    data.write_bytes(b"")
    manifest = {
        "complete": True,
        "pretokenized": True,
        "format_version": 2,
        "assistant_only_loss": True,
        "preserve_thinking": True,
        "columns": ["input_ids", "labels"],
        "tokenizer": DEFAULT_MODEL,
        "tokenizer_revision": DEFAULT_REVISION,
        "output_sha256": hashlib.sha256(b"").hexdigest(),
    }
    data.with_suffix(".parquet.manifest.json").write_text(json.dumps(manifest))

    assert (
        _validate_precompiled_manifest(
            [str(data)],
            model=DEFAULT_MODEL,
            revision=DEFAULT_REVISION,
            manifest_path=data.with_suffix(".parquet.manifest.json"),
        )
        == manifest
    )
