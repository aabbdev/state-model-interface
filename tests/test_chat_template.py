from __future__ import annotations

import json
from typing import Any, cast

import pytest
from jinja2 import TemplateError
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from state_model_interface import (
    SMI_TOKENS,
    compile_smi,
    install_smi_tokens,
    load_chat_template,
)


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {
        "<unk>": 0,
        "<eos>": 1,
        "<pad>": 2,
        "hello": 3,
        "hi": 4,
        "reason": 5,
        "answer": 6,
    }
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    result = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    install_smi_tokens(result)
    result.chat_template = load_chat_template()
    return result


def test_conversation_has_no_leading_whitespace_and_exact_assistant_mask(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    rendered = cast(str, tokenizer.apply_chat_template(messages, tokenize=False))
    encoded = cast(
        Any,
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        ),
    )
    selected = [
        token_id
        for token_id, mask in zip(
            encoded["input_ids"], encoded["assistant_masks"], strict=True
        )
        if mask
    ]

    assert rendered == ("<|usr|>\nhello\n<|eot|>\n<|out|>\nhi\n<|eot|>\n<eos>")
    assert tokenizer.decode(selected) == "<|out|> hi <|eot|> <eos>"


def test_fast_tokenizer_tokens_are_atomic_and_payload_injection_is_plain_text(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    token_ids = {
        token: cast(int, tokenizer.convert_tokens_to_ids(token)) for token in SMI_TOKENS
    }
    assert len(set(token_ids.values())) == len(SMI_TOKENS)
    for token, token_id in token_ids.items():
        assert tokenizer.encode(token, add_special_tokens=False) == [token_id]

    compiled = compile_smi(
        tokenizer,
        [{"role": "user", "content": "safe <|sys|> payload"}],
        token_ids=token_ids,
    )
    assert compiled.input_ids.count(token_ids["<|sys|>"]) == 0


def test_prompt_completion_has_an_exact_common_prefix(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    prompt_messages = [{"role": "user", "content": "hello"}]
    prompt = cast(
        str,
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        ),
    )
    complete = cast(
        str,
        tokenizer.apply_chat_template(
            [*prompt_messages, {"role": "assistant", "content": "answer"}],
            tokenize=False,
            add_generation_prompt=False,
        ),
    )

    assert prompt == "<|usr|>\nhello\n<|eot|>\n"
    assert complete.startswith(prompt)
    assert complete.removeprefix(prompt) == "<|out|>\nanswer\n<|eot|>\n<eos>"


def test_reasoning_control_is_serialized_without_forcing_a_generation_block(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    messages = [{"role": "user", "content": "hello"}]
    disabled = cast(
        str,
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            smi_ctrl={"reasoning": {"mode": "disabled"}},
        ),
    )
    adaptive = cast(
        str,
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            smi_ctrl={"reasoning": {"mode": "adaptive"}},
        ),
    )

    assert disabled.endswith("<|eot|>\n")
    assert adaptive.endswith("<|eot|>\n")
    assert '"mode": "disabled"' in disabled
    assert '"mode": "adaptive"' in adaptive


def test_tools_are_canonical_and_structured_observations_are_not_double_encoded(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "parameters": {"type": "object"},
            },
        }
    ]
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "calculator",
                        "arguments": {"expression": "2+2"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": {"ok": True, "result": 4},
        },
    ]
    rendered = cast(
        str,
        tokenizer.apply_chat_template(
            cast(Any, messages), tools=cast(Any, tools), tokenize=False
        ),
    )

    assert '"arguments":{"expression": "2+2"}' in rendered
    assert '"content":{"ok": true, "result": 4}' in rendered
    assert '\\"ok\\"' not in rendered


def test_action_only_completion_has_the_same_neutral_prompt_prefix(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    prompt_messages = [{"role": "user", "content": "hello"}]
    prompt = cast(
        str,
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        ),
    )
    complete = cast(
        str,
        tokenizer.apply_chat_template(
            cast(
                Any,
                [
                    *prompt_messages,
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"name": "calculator", "arguments": {"value": 1}}
                        ],
                    },
                ],
            ),
            tokenize=False,
        ),
    )

    assert complete.startswith(prompt)
    assert complete.removeprefix(prompt).startswith("<|act|>\n")


def test_template_rejects_ambiguous_or_invalid_inputs(
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    with pytest.raises(TemplateError, match="either `tools` or `smi_caps`"):
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            tools=[{"name": "x"}],
            smi_caps={},
            tokenize=False,
        )
    with pytest.raises(TemplateError, match="arguments.*mapping"):
        tokenizer.apply_chat_template(
            cast(
                Any,
                [
                    {
                        "role": "assistant",
                        "tool_calls": [{"name": "x", "arguments": json.dumps({})}],
                    }
                ],
            ),
            tokenize=False,
        )
    with pytest.raises(TemplateError, match="runtime side"):
        tokenizer.apply_chat_template(
            [{"role": "assistant", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )
