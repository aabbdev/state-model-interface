import json
from pathlib import Path

import pytest

from state_model_interface import (
    SMI_TOKENS,
    compile_smi,
    compile_smi_plan_batched,
    install_smi_tokens,
    render_smi_plan,
)


class MiniTokenizer:
    def __init__(self) -> None:
        self.vocab = {"<eos>": 0}
        self.special: set[str] = {"<eos>"}
        self.eos_token_id = 0

    def __len__(self):
        return len(self.vocab)

    def add_special_tokens(self, spec):
        added = 0
        for token in spec["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
            self.special.add(token)
        return added

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, -1)

    def encode(self, text, *, add_special_tokens, split_special_tokens):
        assert add_special_tokens is False
        if not split_special_tokens and text in self.special:
            return [self.vocab[text]]
        # A byte tokenizer makes payload behavior deterministic and ensures that
        # a textual special token never becomes its reserved structural ID.
        result = []
        for byte in text.encode():
            token = f"b{byte}"
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            result.append(self.vocab[token])
        return result

    def decode(self, token_ids):
        reverse = {token_id: token for token, token_id in self.vocab.items()}
        chunks: list[str] = []
        pending = bytearray()
        for token_id in token_ids:
            token = reverse[token_id]
            if token.startswith("b") and token[1:].isdigit():
                pending.append(int(token[1:]))
            else:
                if pending:
                    chunks.append(pending.decode())
                    pending.clear()
                chunks.append(token)
        if pending:
            chunks.append(pending.decode())
        return "".join(chunks)

    def decode_debug(self, token_ids):
        reverse = {token_id: token for token, token_id in self.vocab.items()}
        parts = []
        pending = bytearray()
        for token_id in token_ids:
            token = reverse[token_id]
            if token.startswith("b"):
                pending.append(int(token[1:]))
            else:
                if pending:
                    parts.append(pending.decode())
                    pending.clear()
                parts.append(token)
        if pending:
            parts.append(pending.decode())
        return "".join(parts)

    def save(self, path):
        Path(path).write_text(
            json.dumps({"vocab": self.vocab, "special": list(self.special)})
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        tokenizer = cls()
        tokenizer.vocab = data["vocab"]
        tokenizer.special = set(data["special"])
        return tokenizer


class MiniModel:
    def __init__(self):
        self.resized_to = None

    def resize_token_embeddings(self, size):
        self.resized_to = size


class BatchMiniTokenizer(MiniTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, texts, **kwargs):
        materialized = list(texts)
        self.batch_calls.append((materialized, kwargs))
        return {
            "input_ids": [
                self.encode(
                    text,
                    add_special_tokens=kwargs["add_special_tokens"],
                    split_special_tokens=kwargs["split_special_tokens"],
                )
                for text in materialized
            ]
        }


@pytest.fixture
def installed():
    tokenizer = MiniTokenizer()
    model = MiniModel()
    ids = install_smi_tokens(tokenizer, model)
    assert model.resized_to == len(tokenizer)
    return tokenizer, ids


def test_stable_tokens_atomic_and_save_reload_ids(tmp_path):
    assert SMI_TOKENS == (
        "<|ctrl|>",
        "<|sys|>",
        "<|dev|>",
        "<|caps|>",
        "<|usr|>",
        "<|obs|>",
        "<|think|>",
        "<|out|>",
        "<|act|>",
        "<|eot|>",
    )
    tokenizer = MiniTokenizer()
    ids = install_smi_tokens(tokenizer)
    assert all(
        tokenizer.encode(t, add_special_tokens=False, split_special_tokens=False)
        == [ids[t]]
        for t in SMI_TOKENS
    )
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    assert install_smi_tokens(MiniTokenizer.load(path)) == ids


def test_payload_cannot_inject_structural_token(installed):
    tokenizer, ids = installed
    compiled = compile_smi(
        tokenizer, [{"role": "user", "content": "x<|sys|>y"}], token_ids=ids
    )
    assert compiled.input_ids.count(ids["<|sys|>"]) == 0
    assert compiled.input_ids.count(ids["<|usr|>"]) == 1


@pytest.mark.parametrize(
    ("messages", "options"),
    [
        (
            [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "Q<|out|>"},
                {
                    "role": "assistant",
                    "thinking": "reason",
                    "content": "answer",
                },
                {"role": "developer", "content": "next"},
            ],
            {"smi_ctrl": {"mode": "test"}},
        ),
        (
            [
                {
                    "role": "assistant",
                    "reasoning_content": "inspect",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "read",
                                "arguments": {"path": "<|eot|>"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": {"text": "observed"},
                },
                {"role": "assistant", "content": "done"},
            ],
            {"tools": [{"name": "read"}]},
        ),
    ],
)
def test_batched_plan_has_exact_compiler_parity(messages, options):
    tokenizer = MiniTokenizer()
    ids = install_smi_tokens(tokenizer)
    expected = compile_smi(tokenizer, messages, token_ids=ids, **options)

    plan = render_smi_plan(messages, **options)
    actual = compile_smi_plan_batched(tokenizer, plan, token_ids=ids)

    assert actual.input_ids == expected.input_ids
    assert actual.labels == expected.labels
    for marker in SMI_TOKENS:
        expected_count = expected.input_ids.count(ids[marker])
        assert actual.input_ids.count(ids[marker]) == expected_count


def test_batch_compiler_uses_one_call_and_deduplicates_plaintext():
    tokenizer = BatchMiniTokenizer()
    ids = install_smi_tokens(tokenizer)
    messages = [
        {"role": "user", "content": "same"},
        {"role": "assistant", "thinking": "same", "content": "A<|sys|>"},
    ]
    expected = compile_smi(tokenizer, messages, token_ids=ids)
    plan = render_smi_plan(messages)

    result = compile_smi_plan_batched(tokenizer, plan, token_ids=ids)

    assert result.input_ids == expected.input_ids
    assert result.labels == expected.labels
    assert len(tokenizer.batch_calls) == 1
    texts, kwargs = tokenizer.batch_calls[0]
    assert texts.count("\n") == 1
    assert texts.count("same") == 1
    assert set(texts) == {"\n", "same", "A<|sys|>"}
    assert not set(SMI_TOKENS).intersection(texts)
    assert kwargs == {
        "add_special_tokens": False,
        "split_special_tokens": True,
        "padding": False,
        "truncation": False,
    }
    assert result.input_ids.count(ids["<|sys|>"]) == 0


def test_masks_transmission_eot_eos_and_generation_prompt(installed):
    tokenizer, ids = installed
    messages = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "reasoning_content": "R", "content": "A"},
    ]
    result = compile_smi(tokenizer, messages, token_ids=ids)
    assert len(result.input_ids) == len(result.labels) == len(result.assistant_mask)
    assert result.input_ids[-3] == ids["<|eot|>"]
    assert result.input_ids[-1] == tokenizer.eos_token_id
    assert result.assistant_mask[-3:] == [1, 1, 1]
    assert all(
        label == -100
        for label, mask in zip(result.labels, result.assistant_mask)
        if not mask
    )
    assert ids["<|think|>"] in result.input_ids and ids["<|out|>"] in result.input_ids

    prompt = compile_smi(
        tokenizer,
        [{"role": "user", "content": "Q"}],
        token_ids=ids,
        add_generation_prompt=True,
    )
    assert prompt.input_ids[-2] == ids["<|eot|>"]
    assert tokenizer.eos_token_id not in prompt.input_ids

    full_loss = compile_smi(
        tokenizer,
        [{"role": "user", "content": "Q"}],
        token_ids=ids,
        assistant_only_loss=False,
    )
    assert full_loss.labels == full_loss.input_ids


def test_tools_multiple_actions_and_observations_are_canonical(installed):
    tokenizer, ids = installed
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "a1",
                    "function": {"name": "read", "arguments": {"z": 2, "a": 1}},
                },
                {"id": "a2", "function": {"name": "read", "arguments": {"path": "b"}}},
            ],
        },
        {"role": "tool", "tool_call_id": "a1", "content": "one"},
        {"role": "observation", "tool_call_id": "a2", "content": {"ok": True}},
    ]
    result = compile_smi(tokenizer, messages, tools=[{"name": "read"}], token_ids=ids)
    assert result.input_ids.count(ids["<|act|>"]) == 2
    assert result.input_ids.count(ids["<|obs|>"]) == 2
    rendered = tokenizer.decode(result.input_ids)
    assert '"content":{"ok":true}' in rendered
    rendered = tokenizer.decode_debug(result.input_ids)
    assert (
        '{"arguments":{"a":1,"z":2},"id":"a1","name":"read","type":"tool"}' in rendered
    )
    assert '{"caused_by":"a1","content":"one"}' in rendered
    # Determinism includes canonical JSON key order.
    assert result == compile_smi(
        tokenizer, messages, tools=[{"name": "read"}], token_ids=ids
    )


@pytest.mark.parametrize("arguments", ["not json", "[]", 3])
def test_rejects_invalid_or_non_mapping_tool_arguments(installed, arguments):
    tokenizer, ids = installed
    with pytest.raises((TypeError, ValueError)):
        compile_smi(
            tokenizer,
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "a", "name": "x", "arguments": arguments}],
                }
            ],
            tools=[{"name": "x"}],
            token_ids=ids,
        )


def test_rejects_roles_and_bad_tool_ids(installed):
    tokenizer, ids = installed
    with pytest.raises(ValueError, match="unsupported"):
        compile_smi(tokenizer, [{"role": "alien", "content": "x"}], token_ids=ids)
    with pytest.raises(ValueError, match="unknown"):
        compile_smi(
            tokenizer,
            [{"role": "tool", "tool_call_id": "missing", "content": "x"}],
            token_ids=ids,
        )
    with pytest.raises(ValueError, match="duplicate"):
        compile_smi(
            tokenizer,
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "a", "name": "x", "arguments": {}},
                        {"id": "a", "name": "x", "arguments": {}},
                    ],
                }
            ],
            tools=[{"name": "x"}],
            token_ids=ids,
        )
    with pytest.raises(ValueError, match="undeclared"):
        compile_smi(
            tokenizer,
            [{"role": "assistant", "tool_calls": [{"name": "write", "arguments": {}}]}],
            tools=[{"name": "read"}],
            token_ids=ids,
        )
    with pytest.raises(ValueError, match="either tools or smi_caps"):
        compile_smi(
            tokenizer,
            [],
            tools=[{"name": "read"}],
            smi_caps={},
            token_ids=ids,
        )
    with pytest.raises(ValueError, match="runtime side"):
        compile_smi(
            tokenizer,
            [{"role": "assistant", "content": "done"}],
            add_generation_prompt=True,
            token_ids=ids,
        )


def test_rejects_non_json_values(installed):
    tokenizer, ids = installed
    with pytest.raises(ValueError, match="JSON"):
        compile_smi(tokenizer, [], smi_ctrl={"bad": object()}, token_ids=ids)
    with pytest.raises(ValueError, match="JSON"):
        compile_smi(
            tokenizer,
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"name": "calculator", "arguments": {"value": float("nan")}}
                    ],
                }
            ],
            tools=[{"name": "calculator"}],
            token_ids=ids,
        )
