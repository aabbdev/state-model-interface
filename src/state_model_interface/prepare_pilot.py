"""Build the pinned, commercially usable 10M-token SMI pilot mixture."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .compiler import compile_smi, install_smi_tokens

TOKENIZER = "aabbdev/RWKV7-1.5B-20260805"
TOKENIZER_REVISION = "5904f9d1cdb05a565e5da9304db0447c8a8eb938"
NEMOTRON_REVISION = "7c804833427f633ccd53b582dbf02525fd680f78"


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Pinned source and token budget."""

    name: str
    dataset: str
    config: str | None
    split: str
    revision: str
    license: str
    quota: int
    adapter: str
    data_url: str | None = None


DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "ultrachat",
        "HuggingFaceH4/ultrachat_200k",
        None,
        "train_sft",
        "8049631c405ae6576f93f445c6b8166f76f5505a",
        "MIT",
        3_500_000,
        "ultrachat",
    ),
    SourceSpec(
        "aya",
        "CohereLabs/aya_dataset",
        "default",
        "train",
        "f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
        "Apache-2.0",
        1_000_000,
        "aya",
    ),
    SourceSpec(
        "nemotron_agentic",
        "nvidia/Nemotron-SFT-Agentic-v2",
        None,
        "tool_calling",
        NEMOTRON_REVISION,
        "CC-BY-4.0 / Apache-2.0 / MIT",
        2_810_000,
        "nemotron",
        (
            "https://huggingface.co/datasets/nvidia/"
            f"Nemotron-SFT-Agentic-v2/resolve/{NEMOTRON_REVISION}/"
            "data/tool_calling.jsonl"
        ),
    ),
    SourceSpec(
        "hermes_func_calling",
        "NousResearch/hermes-function-calling-v1",
        "func_calling",
        "train",
        "dae3e1d28cfbcf4b915c04ea1e072030529b4bda",
        "Apache-2.0",
        190_000,
        "hermes",
    ),
    SourceSpec(
        "openr1_math",
        "open-r1/OpenR1-Math-220k",
        "default",
        "train",
        "e4e141ec9dea9f8326f4d347be56105859b2bd68",
        "Apache-2.0",
        1_000_000,
        "openr1",
    ),
    SourceSpec(
        "opencodeinstruct",
        "nvidia/OpenCodeInstruct",
        "train",
        "train",
        "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d",
        "CC-BY-4.0",
        1_500_000,
        "opencode",
    ),
)

TOTAL_TARGET_TOKENS = sum(source.quota for source in DEFAULT_SOURCES)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
_THINK_RE = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*(.*)$", re.DOTALL)


def _json_value(value: Any, name: str) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} contains invalid JSON") from error


def _json_or_literal(value: Any, name: str) -> Any:
    """Read JSON, with a bounded literal-only fallback for legacy Hermes rows."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if len(value) > 1_000_000 or not value.lstrip().startswith(("{", "[")):
            raise ValueError(f"{name} contains invalid JSON") from None
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"{name} contains invalid JSON/literal") from error
        if not isinstance(parsed, (dict, list)):
            raise TypeError(f"{name} must be an object or array")
        return parsed


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("value is not finite canonical JSON") from error


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _normalize_arguments(value: Any) -> Mapping[str, Any]:
    parsed = _json_value(value, "tool arguments")
    if parsed is None:
        return {}
    if not isinstance(parsed, Mapping):
        raise TypeError("tool arguments must be an object")
    _canonical_json(parsed)
    return parsed


def _normalize_tools(value: Any) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    parsed = _json_or_literal(value, "tools")
    if isinstance(parsed, Mapping) and "tools" in parsed:
        parsed = parsed["tools"]
    if isinstance(parsed, Mapping):
        parsed = [parsed]
    if isinstance(parsed, (str, bytes)) or not isinstance(parsed, Sequence):
        raise TypeError("tools must be an array")
    result: list[dict[str, Any]] = []
    for raw_tool in parsed:
        if not isinstance(raw_tool, Mapping):
            raise TypeError("each tool must be an object")
        function = raw_tool.get("function", raw_tool)
        if not isinstance(function, Mapping):
            raise TypeError("tool function must be an object")
        name = _require_text(function.get("name"), "tool name")
        normalized_function = dict(function)
        normalized_function["name"] = name
        normalized_function.setdefault(
            "parameters", {"type": "object", "properties": {}}
        )
        result.append({"type": "function", "function": normalized_function})
    _canonical_json(result)
    return result


def _normalize_tool_call(value: Any, fallback_id: str) -> dict[str, Any]:
    parsed = _json_or_literal(value, "tool call")
    if not isinstance(parsed, Mapping):
        raise TypeError("tool call must be an object")
    function = parsed.get("function", parsed)
    if not isinstance(function, Mapping):
        raise TypeError("tool call function must be an object")
    name = _require_text(function.get("name"), "tool call name")
    arguments = _normalize_arguments(
        function.get("arguments", function.get("parameters", {}))
    )
    call_id = parsed.get("id") or fallback_id
    return {
        "id": _require_text(call_id, "tool call id"),
        "type": "function",
        "function": {"name": name, "arguments": dict(arguments)},
    }


def _normalize_messages(value: Any) -> list[dict[str, Any]]:
    parsed = _json_value(value, "messages")
    if isinstance(parsed, (str, bytes, Mapping)) or not isinstance(parsed, Sequence):
        raise TypeError("messages must be an array")
    aliases = {
        "human": "user",
        "prompter": "user",
        "gpt": "assistant",
        "bot": "assistant",
        "function": "tool",
    }
    result: list[dict[str, Any]] = []
    for index, raw_message in enumerate(parsed):
        if not isinstance(raw_message, Mapping):
            raise TypeError("each message must be an object")
        role_value = raw_message.get("role", raw_message.get("from"))
        role = aliases.get(str(role_value).lower(), str(role_value).lower())
        message: dict[str, Any] = {
            "role": role,
            "content": raw_message.get("content", raw_message.get("value", "")),
        }
        if role == "assistant" and raw_message.get("tool_calls") is not None:
            calls = _json_value(raw_message["tool_calls"], "tool_calls")
            if isinstance(calls, Mapping):
                calls = [calls]
            if isinstance(calls, (str, bytes)) or not isinstance(calls, Sequence):
                raise ValueError("tool_calls must be an array")
            message["tool_calls"] = [
                _normalize_tool_call(call, f"call_{index}_{call_index}")
                for call_index, call in enumerate(calls)
            ]
        if role in {"tool", "observation"}:
            call_id = raw_message.get("tool_call_id", raw_message.get("call_id"))
            if call_id is not None:
                message["tool_call_id"] = _require_text(call_id, "tool_call_id")
        for key in ("reasoning_content", "thinking"):
            if raw_message.get(key) is not None:
                message[key] = raw_message[key]
        result.append(message)
    return result


def adapt_ultrachat(row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list]:
    return _normalize_messages(row.get("messages")), []


def adapt_aya(row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list]:
    prompt = _require_text(row.get("inputs"), "Aya inputs")
    target = _require_text(row.get("targets"), "Aya targets")
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ], []


def adapt_nemotron(
    row: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messages = _normalize_messages(row.get("messages"))
    tools = _normalize_tools(row.get("tools"))
    return messages, tools


def _extract_tool_block(value: str, pattern: re.Pattern[str]) -> list[str]:
    matches = [match.strip() for match in pattern.findall(value)]
    if ("<tool_call>" in value and not matches and pattern is _TOOL_CALL_RE) or (
        "<tool_response>" in value and not matches and pattern is _TOOL_RESPONSE_RE
    ):
        raise ValueError("unclosed Hermes tool XML block")
    return matches


def adapt_hermes(
    row: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_messages = row.get("conversations", row.get("messages"))
    parsed = _json_or_literal(raw_messages, "Hermes conversations")
    if isinstance(parsed, (str, bytes, Mapping)) or not isinstance(parsed, Sequence):
        raise TypeError("Hermes conversations must be an array")
    tools = _normalize_tools(row.get("tools"))
    messages: list[dict[str, Any]] = []
    pending_ids: list[str] = []
    call_number = 0
    aliases = {"human": "user", "gpt": "assistant", "function": "tool"}
    for raw in parsed:
        if not isinstance(raw, Mapping):
            raise TypeError("Hermes conversation item must be an object")
        raw_role = str(raw.get("from", raw.get("role", ""))).lower()
        role = aliases.get(raw_role, raw_role)
        value = raw.get("value", raw.get("content", ""))
        if not isinstance(value, str):
            raise TypeError("Hermes message value must be text")
        call_blocks = (
            _extract_tool_block(value, _TOOL_CALL_RE) if role == "assistant" else []
        )
        response_blocks = (
            _extract_tool_block(value, _TOOL_RESPONSE_RE) if role == "tool" else []
        )
        if call_blocks:
            calls = []
            for block in call_blocks:
                call_id = f"hermes_call_{call_number}"
                call_number += 1
                call = _normalize_tool_call(block, call_id)
                calls.append(call)
                pending_ids.append(call["id"])
            content = _TOOL_CALL_RE.sub("", value).strip()
            messages.append(
                {"role": "assistant", "content": content, "tool_calls": calls}
            )
            continue
        if response_blocks or role == "tool":
            blocks = response_blocks or [value]
            for block in blocks:
                if not pending_ids:
                    raise ValueError("Hermes tool response has no preceding call")
                try:
                    content = _json_or_literal(block, "Hermes tool response")
                except ValueError:
                    # Tool stdout is frequently plain text rather than JSON.
                    content = block
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": pending_ids.pop(0),
                        "content": content,
                    }
                )
            continue
        messages.append({"role": role, "content": value})
    return messages, tools


def _flag_at(value: Any, index: int) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and (index < len(value) and bool(value[index]))
    )


def adapt_openr1(row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list]:
    generations = row.get("generations")
    if isinstance(generations, (str, bytes)) or not isinstance(generations, Sequence):
        raise TypeError("OpenR1 generations must be an array")
    selected: str | None = None
    for index, generation in enumerate(generations):
        complete = _flag_at(row.get("is_reasoning_complete"), index)
        correct = _flag_at(row.get("correctness_math_verify"), index) or _flag_at(
            row.get("correctness_llama"), index
        )
        if complete and correct and isinstance(generation, str) and generation.strip():
            selected = generation.strip()
            break
    if selected is None:
        raise ValueError("OpenR1 row has no complete, correct generation")
    match = _THINK_RE.match(selected)
    if match:
        reasoning, answer = match.groups()
        answer = answer or row.get("answer", "")
    else:
        reasoning = selected
        answer = row.get("answer", "")
    answer = _require_text(answer, "OpenR1 answer")
    return [
        {"role": "user", "content": _require_text(row.get("problem"), "problem")},
        {
            "role": "assistant",
            "reasoning_content": _require_text(reasoning, "reasoning"),
            "content": answer,
        },
    ], []


def adapt_opencode(
    row: Mapping[str, Any], *, minimum_score: float = 0.8
) -> tuple[list[dict[str, Any]], list]:
    raw_score = row.get("average_test_score")
    if not isinstance(raw_score, (str, int, float)) or isinstance(raw_score, bool):
        raise TypeError("OpenCodeInstruct score is not numeric")
    try:
        score = float(raw_score)
    except (TypeError, ValueError) as error:
        raise ValueError("OpenCodeInstruct score is not numeric") from error
    if not math.isfinite(score) or score < minimum_score:
        raise ValueError("OpenCodeInstruct score is below threshold")
    return [
        {
            "role": "user",
            "content": _require_text(row.get("input"), "OpenCodeInstruct input"),
        },
        {
            "role": "assistant",
            "content": _require_text(row.get("output"), "OpenCodeInstruct output"),
        },
    ], []


Adapter = Callable[
    [Mapping[str, Any]], tuple[list[dict[str, Any]], list[dict[str, Any]]]
]


def _adapter(spec: SourceSpec, minimum_code_score: float) -> Adapter:
    adapters: dict[str, Adapter] = {
        "ultrachat": adapt_ultrachat,
        "aya": adapt_aya,
        "nemotron": adapt_nemotron,
        "hermes": adapt_hermes,
        "openr1": adapt_openr1,
        "opencode": lambda row: adapt_opencode(row, minimum_score=minimum_code_score),
    }
    return adapters[spec.adapter]


def _source_rows(spec: SourceSpec, *, seed: int, buffer_size: int) -> Iterable:
    from datasets import load_dataset

    if spec.data_url:
        dataset = load_dataset(
            "json",
            data_files={spec.split: spec.data_url},
            split=spec.split,
            streaming=True,
        )
    else:
        dataset = load_dataset(
            spec.dataset,
            spec.config,
            split=spec.split,
            revision=spec.revision,
            streaming=True,
        )
    return dataset.shuffle(seed=seed, buffer_size=buffer_size)


def _quota_overrides(values: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    known = {source.name for source in DEFAULT_SOURCES}
    for value in values:
        name, separator, raw_quota = value.partition("=")
        if not separator or name not in known:
            raise ValueError(f"invalid quota override: {value}")
        quota = int(raw_quota)
        if quota < 0:
            raise ValueError("quotas must be non-negative")
        result[name] = quota
    return result


def _with_quotas(overrides: Mapping[str, int]) -> tuple[SourceSpec, ...]:
    return tuple(
        SourceSpec(
            **{**asdict(source), "quota": overrides.get(source.name, source.quota)}
        )
        for source in DEFAULT_SOURCES
    )


def _digest(messages: Any, tools: Any) -> bytes:
    value = _canonical_json({"messages": messages, "tools": tools})
    return hashlib.sha256(value.encode("utf-8")).digest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _maximum_decoded_token_chars(tokenizer: Any) -> int:
    maximum = max(
        len(
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        for token_id in range(len(tokenizer))
    )
    if maximum <= 0:
        raise ValueError("tokenizer has no non-empty decoded token")
    return maximum


def prepare(
    *,
    output: Path,
    manifest_path: Path,
    sources: Sequence[SourceSpec],
    tokenizer: Any,
    max_length: int,
    seed: int,
    shuffle_buffer: int,
    row_group_size: int,
    minimum_code_score: float,
    row_provider: Callable[..., Iterable] = _source_rows,
) -> dict[str, Any]:
    """Stream, validate, deduplicate, and write the pilot Parquet and manifest."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if max_length <= 0 or row_group_size <= 0 or shuffle_buffer <= 0:
        raise ValueError("length and buffer sizes must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing mixture: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.incomplete")
    if temporary_output.exists():
        raise FileExistsError(f"incomplete mixture already exists: {temporary_output}")
    token_ids = install_smi_tokens(tokenizer)
    maximum_token_chars = _maximum_decoded_token_chars(tokenizer)
    schema = pa.schema(
        [
            ("messages_json", pa.string()),
            ("tools_json", pa.string()),
            ("source", pa.string()),
            ("target_tokens", pa.int64()),
        ]
    )
    stats = {
        source.name: {
            "quota": source.quota,
            "target_tokens": 0,
            "rows": 0,
            "rejected": {},
        }
        for source in sources
    }
    seen: set[bytes] = set()
    buffer: list[dict[str, Any]] = []
    total_rows = 0
    writer = pq.ParquetWriter(temporary_output, schema, compression="zstd")

    def reject(source_name: str, reason: str) -> None:
        rejected = stats[source_name]["rejected"]
        rejected[reason] = rejected.get(reason, 0) + 1

    def flush() -> None:
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
            buffer.clear()

    try:
        source_order = list(sources)
        random.Random(seed).shuffle(source_order)
        for source_index, source in enumerate(source_order):
            if source.quota == 0:
                continue
            print(f"source {source.name}: target {source.quota} tokens", flush=True)
            adapter = _adapter(source, minimum_code_score)
            rows = row_provider(
                source, seed=seed + source_index, buffer_size=shuffle_buffer
            )
            source_seen = 0
            for row in rows:
                if stats[source.name]["target_tokens"] >= source.quota:
                    break
                source_seen += 1
                if source_seen % 1_000 == 0:
                    print(
                        f"source {source.name}: seen {source_seen}, accepted "
                        f"{stats[source.name]['rows']} rows / "
                        f"{stats[source.name]['target_tokens']} tokens",
                        flush=True,
                    )
                try:
                    if not isinstance(row, Mapping):
                        raise TypeError("row is not an object")
                    messages, tools = adapter(row)
                    canonical_messages = _canonical_json(messages)
                    canonical_tools = _canonical_json(tools)
                    if (
                        len(canonical_messages) + len(canonical_tools)
                        > max_length * maximum_token_chars
                    ):
                        reject(source.name, "oversized_text")
                        continue
                    fingerprint = _digest(messages, tools)
                    if fingerprint in seen:
                        reject(source.name, "duplicate")
                        continue
                    compiled = compile_smi(
                        tokenizer,
                        messages,
                        tools=tools or None,
                        token_ids=token_ids,
                    )
                    if len(compiled.input_ids) > max_length:
                        reject(source.name, "too_long")
                        continue
                    target_tokens = sum(label != -100 for label in compiled.labels)
                    if target_tokens == 0:
                        reject(source.name, "no_target")
                        continue
                except (KeyError, TypeError, ValueError, OverflowError):
                    reject(source.name, "invalid")
                    continue
                seen.add(fingerprint)
                buffer.append(
                    {
                        "messages_json": canonical_messages,
                        "tools_json": canonical_tools,
                        "source": source.name,
                        "target_tokens": target_tokens,
                    }
                )
                stats[source.name]["target_tokens"] += target_tokens
                stats[source.name]["rows"] += 1
                total_rows += 1
                if len(buffer) >= row_group_size:
                    flush()
            print(
                f"source {source.name}: complete after {source_seen} rows; accepted "
                f"{stats[source.name]['rows']} rows / "
                f"{stats[source.name]['target_tokens']} tokens",
                flush=True,
            )
        flush()
    finally:
        writer.close()

    temporary_output.rename(output)
    total_tokens = sum(value["target_tokens"] for value in stats.values())
    manifest = {
        "format_version": 1,
        "output": str(output),
        "output_sha256": _file_sha256(output),
        "columns": ["messages_json", "tools_json", "source", "target_tokens"],
        "tokenizer": TOKENIZER,
        "tokenizer_revision": TOKENIZER_REVISION,
        "max_length": max_length,
        "seed": seed,
        "shuffle_buffer": shuffle_buffer,
        "total_rows": total_rows,
        "target_tokens": total_tokens,
        "requested_target_tokens": sum(source.quota for source in sources),
        "complete": all(
            stats[source.name]["target_tokens"] >= source.quota for source in sources
        ),
        "sources": [{**asdict(source), **stats[source.name]} for source in sources],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the pinned commercial-safe 10M-token SMI pilot"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--row-group-size", type=int, default=1024)
    parser.add_argument("--minimum-code-score", type=float, default=0.8)
    parser.add_argument(
        "--quota",
        action="append",
        default=[],
        metavar="SOURCE=TOKENS",
        help="override a source target-token budget (repeatable)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from transformers import AutoTokenizer, PreTrainedConfig

    if not math.isfinite(args.minimum_code_score):
        raise ValueError("minimum code score must be finite")
    sources = _with_quotas(_quota_overrides(args.quota))
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER,
        revision=TOKENIZER_REVISION,
        config=PreTrainedConfig(),
    )
    manifest_path = args.manifest or args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    )
    manifest = prepare(
        output=args.output,
        manifest_path=manifest_path,
        sources=sources,
        tokenizer=tokenizer,
        max_length=args.max_length,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        row_group_size=args.row_group_size,
        minimum_code_score=args.minimum_code_score,
    )
    print(
        f"wrote {manifest['total_rows']} rows / {manifest['target_tokens']} "
        f"assistant target tokens to {args.output}"
    )
    if not manifest["complete"]:
        print("warning: one or more exact token quotas were not filled")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_SOURCES",
    "NEMOTRON_REVISION",
    "TOKENIZER",
    "TOKENIZER_REVISION",
    "TOTAL_TARGET_TOKENS",
    "SourceSpec",
    "adapt_aya",
    "adapt_hermes",
    "adapt_nemotron",
    "adapt_opencode",
    "adapt_openr1",
    "adapt_ultrachat",
    "main",
    "parse_args",
    "prepare",
]
