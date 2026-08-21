"""Secure token-ID serialization for the State Model Interface (SMI)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SMI_TOKENS: tuple[str, ...] = (
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

_ROLE_TOKEN = {
    "system": "<|sys|>",
    "developer": "<|dev|>",
    "user": "<|usr|>",
    "tool": "<|obs|>",
    "observation": "<|obs|>",
}
_RUNTIME_ROLES = frozenset(_ROLE_TOKEN)
_VALID_ROLES = _RUNTIME_ROLES | {"assistant"}


@dataclass(frozen=True, slots=True)
class CompiledSMI:
    """Model-ready SMI sequence and aligned training metadata."""

    input_ids: list[int]
    labels: list[int]
    assistant_mask: list[int]


def install_smi_tokens(tokenizer: Any, model: Any | None = None) -> dict[str, int]:
    """Install SMI tokens, optionally resize a model, and return stable token IDs."""
    tokenizer.add_special_tokens({"additional_special_tokens": list(SMI_TOKENS)})
    token_ids = {
        token: int(tokenizer.convert_tokens_to_ids(token)) for token in SMI_TOKENS
    }
    if len(set(token_ids.values())) != len(SMI_TOKENS) or any(
        token_id < 0 for token_id in token_ids.values()
    ):
        raise ValueError("tokenizer did not assign ten distinct SMI token IDs")
    for token, token_id in token_ids.items():
        encoded = _encode(tokenizer, token, split_special_tokens=False)
        if encoded != [token_id]:
            raise ValueError(f"SMI token is not atomic after installation: {token}")
    if model is not None:
        model.resize_token_embeddings(len(tokenizer))
    return token_ids


def compile_smi(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    smi_ctrl: Any | None = None,
    smi_caps: Any | None = None,
    token_ids: Mapping[str, int] | None = None,
    assistant_only_loss: bool = True,
    add_generation_prompt: bool = False,
    preserve_thinking: bool = True,
) -> CompiledSMI:
    """Compile trusted structure plus injection-safe payloads directly to token IDs."""
    ids = dict(token_ids or _installed_ids(tokenizer))
    if set(ids) != set(SMI_TOKENS):
        raise ValueError("token_ids must map exactly the ten SMI tokens")
    if tools is not None and (isinstance(tools, (str, bytes, Mapping))):
        raise TypeError("tools must be a sequence of mappings")
    if tools and smi_caps is not None:
        raise ValueError("pass either tools or smi_caps, not both")
    allowed_tool_names = _declared_tool_names(tools, smi_caps)

    input_ids: list[int] = []
    assistant_mask: list[int] = []

    def append_structural(token: str, assistant: bool) -> None:
        input_ids.append(ids[token])
        assistant_mask.append(int(assistant))

    def append_plain(value: str, assistant: bool) -> None:
        encoded = _encode(tokenizer, value, split_special_tokens=True)
        input_ids.extend(encoded)
        assistant_mask.extend([int(assistant)] * len(encoded))

    def block(token: str, payload: str, assistant: bool) -> None:
        append_structural(token, assistant)
        append_plain("\n", assistant)
        append_plain(payload, assistant)
        append_plain("\n", assistant)

    side: str | None = None
    if smi_ctrl is not None:
        block("<|ctrl|>", _payload_or_json(smi_ctrl), False)
        side = "runtime"
    if smi_caps is not None:
        block("<|caps|>", _payload_or_json(smi_caps), False)
        side = "runtime"
    elif tools:
        normalized_tools = _mapping_sequence(tools, "tools")
        block("<|caps|>", _canonical_json({"tools": normalized_tools}), False)
        side = "runtime"

    known_tool_ids: set[str] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"message {index} must be a mapping")
        role = message.get("role")
        if role not in _VALID_ROLES:
            raise ValueError(f"unsupported SMI role: {role!r}")
        is_assistant = role == "assistant"
        new_side = "model" if is_assistant else "runtime"
        if side is not None and side != new_side:
            append_structural("<|eot|>", side == "model")
            append_plain("\n", side == "model")
        side = new_side

        if is_assistant:
            _append_assistant(
                message,
                block,
                known_tool_ids,
                allowed_tool_names,
                preserve_thinking=preserve_thinking,
            )
        else:
            payload = _runtime_payload(message, known_tool_ids)
            block(_ROLE_TOKEN[role], payload, False)

    if add_generation_prompt:
        if side == "model":
            raise ValueError(
                "add_generation_prompt requires history ending on the runtime side"
            )
        if side is not None:
            append_structural("<|eot|>", side == "model")
            append_plain("\n", side == "model")
    else:
        if side is not None:
            append_structural("<|eot|>", side == "model")
            append_plain("\n", side == "model")
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("tokenizer must define a native eos_token_id")
        input_ids.append(int(eos_token_id))
        assistant_mask.append(int(side == "model"))

    labels = [
        token_id if (not assistant_only_loss or mask) else -100
        for token_id, mask in zip(input_ids, assistant_mask, strict=True)
    ]
    return CompiledSMI(input_ids, labels, assistant_mask)


def _append_assistant(
    message: Mapping[str, Any],
    block: Any,
    known_tool_ids: set[str],
    allowed_tool_names: set[str],
    *,
    preserve_thinking: bool,
) -> None:
    reasoning = message.get("reasoning_content") or message.get("thinking")
    if preserve_thinking and reasoning:
        block("<|think|>", _render_content(reasoning), True)
    if message.get("content") is not None:
        rendered = _render_content(message["content"])
        if rendered.strip():
            block("<|out|>", rendered, True)
    tool_calls = message.get("tool_calls")
    if tool_calls is None:
        return
    if isinstance(tool_calls, (str, bytes, Mapping)) or not isinstance(
        tool_calls, Sequence
    ):
        raise TypeError("assistant tool_calls must be a sequence")
    for tool_call in tool_calls:
        if not isinstance(tool_call, Mapping):
            raise TypeError("each tool call must be a mapping")
        call_id = tool_call.get("id")
        if call_id is not None:
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("tool call id must be a non-empty string")
            if call_id in known_tool_ids:
                raise ValueError(f"duplicate tool call id: {call_id}")
            known_tool_ids.add(call_id)
        function = tool_call.get("function", tool_call)
        if not isinstance(function, Mapping):
            raise TypeError("tool call function must be a mapping")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool call name must be a non-empty string")
        if name not in allowed_tool_names:
            raise ValueError(f"tool call names undeclared capability: {name}")
        arguments = function.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise TypeError("tool arguments must be a mapping")
        action: dict[str, Any] = {"type": "tool", "name": name, "arguments": arguments}
        if call_id is not None:
            action = {"id": call_id, **action}
        block("<|act|>", _canonical_json(action), True)


def _runtime_payload(message: Mapping[str, Any], known_tool_ids: set[str]) -> str:
    role = message["role"]
    raw_content = message.get("content")
    content = _render_content(raw_content)
    if role not in {"tool", "observation"}:
        return content
    call_id = message.get("tool_call_id")
    if call_id is None:
        return content
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("tool_call_id must be a non-empty string")
    if call_id not in known_tool_ids:
        raise ValueError(f"unknown tool_call_id: {call_id}")
    return _canonical_json({"caused_by": call_id, "content": raw_content})


def _render_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        if "smi_payload" in value:
            if not isinstance(value["smi_payload"], str):
                raise TypeError("smi_payload must be a string")
            return value["smi_payload"]
        if "text" in value:
            if not isinstance(value["text"], str):
                raise TypeError("text content must be a string")
            return value["text"]
        return _canonical_json(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rendered: list[str] = []
        for item in value:
            if isinstance(item, str):
                rendered.append(item)
            elif isinstance(item, Mapping) and "smi_payload" in item:
                if not isinstance(item["smi_payload"], str):
                    raise TypeError("smi_payload must be a string")
                rendered.append(item["smi_payload"])
            elif isinstance(item, Mapping) and "text" in item:
                if not isinstance(item["text"], str):
                    raise TypeError("text content must be a string")
                rendered.append(item["text"])
            else:
                raise TypeError("unsupported multimodal content item")
        return "".join(rendered)
    if isinstance(value, (bool, int, float)):
        return _canonical_json(value)
    raise TypeError(f"unsupported SMI payload type: {type(value).__name__}")


def _payload_or_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (Mapping, list, tuple)):
        return _canonical_json(value)
    if isinstance(value, (bool, int, float)):
        return _canonical_json(value)
    raise TypeError(f"unsupported SMI control payload: {type(value).__name__}")


def _mapping_sequence(value: Sequence[Any], name: str) -> list[Mapping[str, Any]]:
    result = list(value)
    if not all(isinstance(item, Mapping) for item in result):
        raise TypeError(f"every item in {name} must be a mapping")
    return result


def _tool_names(tools: Sequence[Mapping[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in _mapping_sequence(tools, "tools"):
        function = tool.get("function", tool)
        if not isinstance(function, Mapping):
            raise TypeError("tool function must be a mapping")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("every declared tool requires a non-empty name")
        if name in names:
            raise ValueError(f"duplicate declared tool name: {name}")
        names.add(name)
    return names


def _declared_tool_names(
    tools: Sequence[Mapping[str, Any]] | None, smi_caps: Any | None
) -> set[str]:
    if tools:
        return _tool_names(tools)
    if isinstance(smi_caps, Mapping) and smi_caps.get("tools"):
        native_tools = smi_caps["tools"]
        if isinstance(native_tools, (str, bytes, Mapping)) or not isinstance(
            native_tools, Sequence
        ):
            raise TypeError("smi_caps.tools must be a sequence of mappings")
        return _tool_names(native_tools)
    return set()


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
        raise ValueError("value is not JSON serializable") from error


def _installed_ids(tokenizer: Any) -> dict[str, int]:
    result = {
        token: int(tokenizer.convert_tokens_to_ids(token)) for token in SMI_TOKENS
    }
    if len(set(result.values())) != len(SMI_TOKENS):
        raise ValueError("install SMI tokens before compiling")
    return result


def _encode(tokenizer: Any, text: str, *, split_special_tokens: bool) -> list[int]:
    try:
        encoded = tokenizer.encode(
            text,
            add_special_tokens=False,
            split_special_tokens=split_special_tokens,
        )
    except TypeError as error:
        raise TypeError(
            "tokenizer.encode must support split_special_tokens for secure compilation"
        ) from error
    return [int(token_id) for token_id in encoded]
