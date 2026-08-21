from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from state_model_interface.compiler import install_smi_tokens
from state_model_interface.prepare_pilot import (
    DEFAULT_SOURCES,
    NEMOTRON_REVISION,
    TOKENIZER_REVISION,
    TOTAL_TARGET_TOKENS,
    SourceSpec,
    _quota_overrides,
    adapt_aya,
    adapt_hermes,
    adapt_nemotron,
    adapt_opencode,
    adapt_openr1,
    adapt_ultrachat,
    parse_args,
    prepare,
)


class ByteTokenizer:
    def __init__(self) -> None:
        self.vocab = {"<eos>": 0}
        self.special = {"<eos>"}
        self.eos_token_id = 0

    def add_special_tokens(self, spec: dict[str, list[str]]) -> int:
        added = 0
        for token in spec["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
            self.special.add(token)
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, -1)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        split_special_tokens: bool,
    ) -> list[int]:
        assert add_special_tokens is False
        if not split_special_tokens and text in self.special:
            return [self.vocab[text]]
        result = []
        for byte in text.encode():
            key = f"b{byte}"
            if key not in self.vocab:
                self.vocab[key] = len(self.vocab)
            result.append(self.vocab[key])
        return result


def test_default_sources_are_pinned_and_quotas_sum_to_ten_million() -> None:
    assert TOTAL_TARGET_TOKENS == 10_000_000
    assert sum(source.quota for source in DEFAULT_SOURCES) == 10_000_000
    assert all(len(source.revision) == 40 for source in DEFAULT_SOURCES)
    assert DEFAULT_SOURCES[0].split == "train_sft"
    assert DEFAULT_SOURCES[1].split == "train"
    nemotron = next(
        source for source in DEFAULT_SOURCES if source.adapter == "nemotron"
    )
    assert nemotron.data_url is not None
    assert NEMOTRON_REVISION in nemotron.data_url
    assert TOKENIZER_REVISION == "5904f9d1cdb05a565e5da9304db0447c8a8eb938"
    assert all(
        "test" not in source.split and "eval" not in source.split
        for source in DEFAULT_SOURCES
    )


def test_ultrachat_and_aya_adapters() -> None:
    messages, tools = adapt_ultrachat(
        {
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ]
        }
    )
    assert messages[-1] == {"role": "assistant", "content": "A"}
    assert tools == []

    messages, tools = adapt_aya({"inputs": "question", "targets": "answer"})
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[-1]["content"] == "answer"
    assert tools == []


def test_nemotron_normalizes_tools_calls_and_string_arguments() -> None:
    messages, tools = adapt_nemotron(
        {
            "tools": json.dumps(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "parameters": {"type": "object"},
                        },
                    }
                ]
            ),
            "messages": [
                {"role": "user", "content": "find it"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "search", "arguments": '{"q":"x"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": {"result": "x"}},
            ],
        }
    )
    assert tools[0]["function"]["name"] == "search"
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"q": "x"}
    assert messages[2]["tool_call_id"] == "c1"


def test_hermes_sharegpt_xml_calls_and_responses_are_linked() -> None:
    messages, tools = adapt_hermes(
        {
            "tools": "[{'name': 'weather', 'parameters': {'type': 'object'}}]",
            "conversations": [
                {
                    "from": "system",
                    "value": (
                        'Return calls like <tool_call>{"name": <function-name>, '
                        '"arguments": <args-dict>}</tool_call>'
                    ),
                },
                {"from": "human", "value": "weather?"},
                {
                    "from": "gpt",
                    "value": 'checking\n<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>',
                },
                {"from": "tool", "value": '<tool_response>{"temp":20}</tool_response>'},
                {"from": "gpt", "value": "20 C"},
            ],
        }
    )
    call = messages[2]["tool_calls"][0]
    assert messages[0]["role"] == "system"
    assert "<function-name>" in messages[0]["content"]
    assert messages[2]["content"] == "checking"
    assert messages[3]["tool_call_id"] == call["id"]
    assert messages[3]["content"] == {"temp": 20}
    assert tools[0]["function"]["name"] == "weather"


def test_hermes_rejects_orphan_response_and_never_evaluates_code() -> None:
    with pytest.raises(ValueError, match="no preceding call"):
        adapt_hermes(
            {"tools": [], "conversations": [{"from": "tool", "value": "orphan"}]}
        )
    with pytest.raises(ValueError, match="invalid"):
        adapt_hermes(
            {
                "tools": "__import__('os').system('false')",
                "conversations": [],
            }
        )


def test_openr1_selects_first_complete_correct_trace_and_splits_think() -> None:
    messages, _ = adapt_openr1(
        {
            "problem": "2+2?",
            "answer": "4",
            "generations": [
                "<think>wrong</think> 5",
                "<think>full reasoning</think> 4",
                "<think>later</think> 4",
            ],
            "is_reasoning_complete": [True, True, True],
            "correctness_math_verify": [False, True, True],
            "correctness_llama": [False, False, False],
        }
    )
    assert messages[-1]["reasoning_content"] == "full reasoning"
    assert messages[-1]["content"] == "4"


def test_opencode_requires_high_finite_score() -> None:
    messages, _ = adapt_opencode(
        {"input": "write code", "output": "print(1)", "average_test_score": "0.95"}
    )
    assert messages[-1]["content"] == "print(1)"
    for score in ("0.5", "nan", "not-a-number"):
        with pytest.raises(ValueError):
            adapt_opencode({"input": "x", "output": "y", "average_test_score": score})


def test_cli_defaults_and_quota_validation() -> None:
    args = parse_args(["--output", "pilot.parquet"])
    assert args.max_length == 2048
    assert args.shuffle_buffer == 10_000
    assert args.minimum_code_score == 0.8
    assert _quota_overrides(["aya=12"]) == {"aya": 12}
    with pytest.raises(ValueError):
        _quota_overrides(["unknown=1"])


def test_prepare_writes_simple_parquet_and_manifest_offline(tmp_path: Path) -> None:
    tokenizer = ByteTokenizer()
    token_ids = install_smi_tokens(tokenizer)
    sample = {"inputs": "Q", "targets": "A"}
    messages, _ = adapt_aya(sample)
    from state_model_interface.compiler import compile_smi

    target_tokens = sum(
        label != -100
        for label in compile_smi(tokenizer, messages, token_ids=token_ids).labels
    )
    source = SourceSpec(
        "aya",
        "offline/aya",
        "default",
        "train",
        "a" * 40,
        "Apache-2.0",
        target_tokens,
        "aya",
    )

    def rows(spec: SourceSpec, *, seed: int, buffer_size: int) -> list[dict[str, Any]]:
        assert spec == source
        assert isinstance(seed, int) and buffer_size == 8
        return [sample, sample, {"inputs": "bad", "targets": ""}]

    output = tmp_path / "pilot.parquet"
    manifest_path = tmp_path / "manifest.json"
    manifest = prepare(
        output=output,
        manifest_path=manifest_path,
        sources=[source],
        tokenizer=tokenizer,
        max_length=1000,
        seed=7,
        shuffle_buffer=8,
        row_group_size=1,
        minimum_code_score=0.8,
        row_provider=rows,
    )

    table = pq.read_table(output)
    assert table.column_names == [
        "messages_json",
        "tools_json",
        "source",
        "target_tokens",
    ]
    assert table.num_rows == 1
    assert table["source"].to_pylist() == ["aya"]
    assert manifest["complete"] is True
    assert manifest["target_tokens"] == target_tokens
    assert len(manifest["output_sha256"]) == 64
    assert json.loads(manifest_path.read_text()) == manifest
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare(
            output=output,
            manifest_path=manifest_path,
            sources=[source],
            tokenizer=tokenizer,
            max_length=1000,
            seed=7,
            shuffle_buffer=8,
            row_group_size=1,
            minimum_code_score=0.8,
            row_provider=rows,
        )
