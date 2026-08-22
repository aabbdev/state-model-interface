"""Linear-time encoder for the pinned RWKV ByteLevel/WordPiece tokenizer."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import numpy as np


@dataclass(slots=True)
class _TrieNode:
    children: dict[str, _TrieNode] = field(default_factory=dict)
    token_id: int | None = None


class LinearRWKVTokenizer:
    """Bit-exact longest-match WordPiece encoding with bounded trie traversal."""

    def __init__(
        self,
        *,
        vocab: dict[str, int],
        unknown_token: str,
        eos_token_id: int,
        pre_tokenizer: Any,
    ) -> None:
        if unknown_token not in vocab:
            raise ValueError("WordPiece unknown token is absent from the model vocab")
        self.unk_token_id = int(vocab[unknown_token])
        self.eos_token_id = int(eos_token_id)
        self._pre_tokenizer = pre_tokenizer
        self._root = _TrieNode()
        for token, token_id in vocab.items():
            if not token:
                raise ValueError("WordPiece vocab contains an empty token")
            node = self._root
            for character in token:
                node = node.children.setdefault(character, _TrieNode())
            node.token_id = int(token_id)
        self._offsets, self._edge_chars, self._edge_targets, self._terminals = (
            self._flatten_trie()
        )
        self._root = _TrieNode()
        self._compiled_matcher: Any | None = None
        try:
            numba = import_module("numba")
            self._compiled_matcher = numba.njit(cache=True, nogil=True)(
                _match_wordpiece
            )
        except (ImportError, TypeError, ValueError):
            pass

    @property
    def uses_numba(self) -> bool:
        return self._compiled_matcher is not None

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> LinearRWKVTokenizer:
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None or backend.pre_tokenizer is None:
            raise TypeError(
                "RWKV tokenizer requires a tokenizers backend and pre-tokenizer"
            )
        config = json.loads(backend.to_str())
        model = config.get("model") or {}
        pre_tokenizer = config.get("pre_tokenizer") or {}
        if (
            model.get("type") != "WordPiece"
            or model.get("continuing_subword_prefix", "") != ""
            or model.get("max_input_chars_per_word") != 2_147_483_647
            or pre_tokenizer.get("type") != "ByteLevel"
            or pre_tokenizer.get("use_regex") is not False
            or config.get("normalizer") is not None
        ):
            raise ValueError(
                "tokenizer is not the supported RWKV ByteLevel/WordPiece form"
            )
        vocab = model.get("vocab")
        if not isinstance(vocab, dict) or len(vocab) != int(tokenizer.vocab_size):
            raise ValueError("tokenizer model vocab is missing or inconsistent")
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("tokenizer has no EOS token")
        return cls(
            vocab={str(token): int(token_id) for token, token_id in vocab.items()},
            unknown_token=str(model.get("unk_token")),
            eos_token_id=int(eos_token_id),
            pre_tokenizer=backend.pre_tokenizer,
        )

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        result: list[int] = []
        for piece, _ in self._pre_tokenizer.pre_tokenize_str(text):
            encoded_piece = self._encode_piece(piece)
            if encoded_piece is None:
                result.append(self.unk_token_id)
            else:
                result.extend(encoded_piece)
        return result

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def __call__(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool,
        split_special_tokens: bool,
        padding: bool,
        truncation: bool,
    ) -> dict[str, list[list[int]]]:
        if add_special_tokens or not split_special_tokens or padding or truncation:
            raise ValueError("unsupported options for linear RWKV tokenizer")
        return {"input_ids": self.encode_batch(texts)}

    def _encode_piece(self, piece: str) -> list[int] | None:
        codepoints = np.fromiter((ord(character) for character in piece), np.int32)
        matcher = (
            self._compiled_matcher
            if self._compiled_matcher is not None
            else _match_wordpiece
        )
        try:
            encoded = matcher(
                codepoints,
                self._offsets,
                self._edge_chars,
                self._edge_targets,
                self._terminals,
            )
        except Exception:  # noqa: BLE001 - optional JIT must fail back exactly
            self._compiled_matcher = None
            encoded = _match_wordpiece(
                codepoints,
                self._offsets,
                self._edge_chars,
                self._edge_targets,
                self._terminals,
            )
        return None if encoded.size == 1 and encoded[0] == -1 else encoded.tolist()

    def _flatten_trie(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nodes = [self._root]
        node_indices = {id(self._root): 0}
        edge_chars: list[int] = []
        edge_targets: list[int] = []
        offsets = [0]
        terminals: list[int] = []
        index = 0
        while index < len(nodes):
            node = nodes[index]
            terminals.append(-1 if node.token_id is None else node.token_id)
            for character, child in sorted(node.children.items()):
                child_index = node_indices.get(id(child))
                if child_index is None:
                    child_index = len(nodes)
                    node_indices[id(child)] = child_index
                    nodes.append(child)
                edge_chars.append(ord(character))
                edge_targets.append(child_index)
            offsets.append(len(edge_chars))
            index += 1
        return (
            np.asarray(offsets, dtype=np.int32),
            np.asarray(edge_chars, dtype=np.int32),
            np.asarray(edge_targets, dtype=np.int32),
            np.asarray(terminals, dtype=np.int32),
        )


def _match_wordpiece(
    codepoints: np.ndarray,
    offsets: np.ndarray,
    edge_chars: np.ndarray,
    edge_targets: np.ndarray,
    terminals: np.ndarray,
) -> np.ndarray:
    output = np.empty(codepoints.size, dtype=np.int32)
    output_size = 0
    start = 0
    while start < codepoints.size:
        node = 0
        cursor = start
        best_id = -1
        best_end = start
        while cursor < codepoints.size:
            low = int(offsets[node])
            high = int(offsets[node + 1]) - 1
            target = -1
            character = int(codepoints[cursor])
            while low <= high:
                middle = (low + high) // 2
                candidate = int(edge_chars[middle])
                if candidate < character:
                    low = middle + 1
                elif candidate > character:
                    high = middle - 1
                else:
                    target = int(edge_targets[middle])
                    break
            if target < 0:
                break
            node = target
            cursor += 1
            token_id = int(terminals[node])
            if token_id >= 0:
                best_id = token_id
                best_end = cursor
        if best_id < 0:
            return np.asarray([-1], dtype=np.int32)
        output[output_size] = best_id
        output_size += 1
        start = best_end
    return output[:output_size]


__all__ = ["LinearRWKVTokenizer"]
