from __future__ import annotations

import importlib.util
import json

import pytest

from state_model_interface.compiler import install_smi_tokens
from state_model_interface.linear_tokenizer import LinearRWKVTokenizer


class IdentityPreTokenizer:
    def pre_tokenize_str(self, text: str):
        return [(text, (0, len(text)))]


class FakeBackend:
    def __init__(self, max_input_chars: int) -> None:
        self.pre_tokenizer = IdentityPreTokenizer()
        self._config = {
            "normalizer": None,
            "pre_tokenizer": {"type": "ByteLevel", "use_regex": False},
            "model": {
                "type": "WordPiece",
                "unk_token": "<unk>",
                "continuing_subword_prefix": "",
                "max_input_chars_per_word": max_input_chars,
                "vocab": {"<unk>": 0, "a": 1},
            },
        }

    def to_str(self) -> str:
        return json.dumps(self._config)


class FakeTokenizer:
    def __init__(self, max_input_chars: int) -> None:
        self.backend_tokenizer = FakeBackend(max_input_chars)
        self.vocab_size = 2
        self.eos_token_id = 0


def test_linear_wordpiece_uses_longest_match_and_unknown_fallback() -> None:
    tokenizer = LinearRWKVTokenizer(
        vocab={"<unk>": 0, "a": 1, "ab": 2, "b": 3},
        unknown_token="<unk>",
        eos_token_id=0,
        pre_tokenizer=IdentityPreTokenizer(),
    )

    assert tokenizer.encode("abab") == [2, 2]
    assert tokenizer.encode("ac") == [0]
    assert tokenizer(
        ["ab", "b"],
        add_special_tokens=False,
        split_special_tokens=True,
        padding=False,
        truncation=False,
    ) == {"input_ids": [[2], [3]]}


def test_linear_tokenizer_rejects_different_wordpiece_limit() -> None:
    with pytest.raises(ValueError, match="supported RWKV"):
        LinearRWKVTokenizer.from_tokenizer(FakeTokenizer(2))


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="transformers is required",
)
def test_pinned_rwkv_tokenizer_has_exact_multilingual_parity() -> None:
    from transformers import AutoTokenizer, PreTrainedConfig

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "aabbdev/RWKV7-1.5B-20260805",
            revision="5904f9d1cdb05a565e5da9304db0447c8a8eb938",
            config=PreTrainedConfig(),
            local_files_only=True,
        )
    except OSError:
        pytest.skip("pinned tokenizer is not cached")
    install_smi_tokens(tokenizer)
    linear = LinearRWKVTokenizer.from_tokenizer(tokenizer)
    texts = [
        "hello world",
        "a\nb",
        "你好，世界",
        "தமிழ் மொழி",
        "اللغة العربية",
        "emoji 👩🏽‍💻",
        "e\u0301 café",
        "literal <|act|> marker",
    ]

    expected = [
        tokenizer.encode(
            text,
            add_special_tokens=False,
            split_special_tokens=True,
        )
        for text in texts
    ]
    assert linear.encode_batch(texts) == expected
