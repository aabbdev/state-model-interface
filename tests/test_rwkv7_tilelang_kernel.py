from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from state_model_interface import rwkv7_tilelang_kernel as kernel


def test_tilelang_adapter_is_importable_without_gpu_extra() -> None:
    assert len(kernel.FLA_REVISION) == 40
    assert kernel.DEFAULT_CHUNK_SIZE == 32
    assert callable(kernel.register_rwkv7_tilelang_kernel)
    assert isinstance(kernel.tilelang_training_available(), bool)


def test_registration_is_restorable(monkeypatch: pytest.MonkeyPatch) -> None:
    def original(*args, **kwargs):
        del args, kwargs

    registry = {"chunked": original}
    module = SimpleNamespace(RWKV7_WKV_FUNCTIONS=registry)

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(wkv_implementation="eager")

    FakeModel.__module__ = "dynamic_rwkv7"
    model = FakeModel()
    monkeypatch.setattr(kernel.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(kernel.importlib, "import_module", lambda name: module)

    restore = kernel.register_rwkv7_tilelang_kernel(model)
    assert registry["chunked"] is kernel.rwkv7_smi_tilelang
    assert model.config.wkv_implementation == "chunked"
    restore()
    assert registry["chunked"] is original


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("tilelang") is None
    or importlib.util.find_spec("fla") is None,
    reason="CUDA FLA TileLang extra is required",
)
def test_tilelang_adapter_has_finite_forward_and_backward() -> None:
    torch.manual_seed(0)
    shape = (1, 64, 1, 64)
    r = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w_source = torch.randn(
        shape, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    w_log = -0.6065306597126334 * w_source.sigmoid()
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    normalized_source = torch.randn(
        shape, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    normalized_key = torch.nn.functional.normalize(
        normalized_source.float(), dim=-1
    ).to(torch.bfloat16)
    gate_source = torch.randn(
        shape, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    gate_a = gate_source.sigmoid()
    inputs = (r, w_log, key, value, normalized_key, gate_a)
    state = torch.zeros(1, 1, 64, 64, device="cuda", dtype=torch.float32)
    output, returned_state = kernel.rwkv7_smi_tilelang(*inputs, state)
    output.float().square().mean().backward()

    assert output.shape == inputs[0].shape
    assert returned_state is state
    assert torch.isfinite(output).all()
    leaves = (r, w_source, key, value, normalized_source, gate_source)
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all() for value in leaves
    )
