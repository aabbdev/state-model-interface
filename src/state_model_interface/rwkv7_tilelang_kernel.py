"""Training-only adapter for FLA's full DPLR TileLang RWKV7 kernel."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from typing import Any

import torch

FLA_REVISION = "27967b970eaaf982a6960abf6cba8add9c34c7cc"
DEFAULT_CHUNK_SIZE = 32


def rwkv7_smi_tilelang(
    r: torch.Tensor,
    w_log: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    normalized_key: torch.Tensor,
    gate_a: torch.Tensor,
    state: torch.Tensor,
    cu_seq_lens: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FLA's complete TileLang DPLR forward/backward for RWKV7 training.

    The public RWKV7 recurrence uses ``a=-normalized_key`` and
    ``b=normalized_key*gate_a``. Cache output is intentionally disabled: this
    backend is for label-bearing SFT where the model sets ``use_cache=False``.
    """
    del kwargs
    if not r.is_cuda:
        raise RuntimeError("SMI TileLang RWKV7 training requires CUDA tensors")
    if r.dtype not in {torch.bfloat16, torch.float16}:
        raise TypeError("SMI TileLang RWKV7 training requires BF16 or FP16 activations")
    if state.requires_grad:
        raise ValueError("SMI TileLang training does not accept a differentiable cache")
    try:
        from fla.ops.generalized_delta_rule.dplr.backends.tilelang.chunk import (  # type: ignore[import-not-found]
            chunk_dplr_delta_rule_tilelang,
        )
    except ImportError as error:
        raise RuntimeError(
            "install the pinned flash-linear-attention[tilelang] training extra"
        ) from error

    activation_dtype = r.dtype
    output, _ = chunk_dplr_delta_rule_tilelang(
        q=r,
        k=key.to(activation_dtype),
        v=value.to(activation_dtype),
        a=(-normalized_key).to(activation_dtype),
        b=(normalized_key * gate_a).to(activation_dtype),
        gk=w_log,
        scale=1.0,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seq_lens,
        chunk_size=DEFAULT_CHUNK_SIZE,
        disable_recompute=False,
    )
    return output, state


def register_rwkv7_tilelang_kernel(model: Any) -> Callable[[], None]:
    """Temporarily replace the chunked entry and return a restoration callback."""
    if importlib.util.find_spec("fla") is None:
        raise RuntimeError(
            "flash-linear-attention is unavailable; install the training extra"
        )
    if importlib.util.find_spec("tilelang") is None:
        raise RuntimeError("TileLang is unavailable; install the training extra")
    module = importlib.import_module(model.__class__.__module__)
    if not hasattr(module, "RWKV7_WKV_FUNCTIONS"):
        raise TypeError(
            "model is not the bundled RWKV7 remote implementation; load it before FLA"
        )
    registry = module.RWKV7_WKV_FUNCTIONS
    original = registry["chunked"]
    registry["chunked"] = rwkv7_smi_tilelang
    model.config.wkv_implementation = "chunked"

    def restore() -> None:
        if registry.get("chunked") is rwkv7_smi_tilelang:
            registry["chunked"] = original

    return restore


def tilelang_training_available() -> bool:
    return (
        torch.cuda.is_available()
        and importlib.util.find_spec("fla") is not None
        and importlib.util.find_spec("tilelang") is not None
    )


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "FLA_REVISION",
    "register_rwkv7_tilelang_kernel",
    "rwkv7_smi_tilelang",
    "tilelang_training_available",
]
