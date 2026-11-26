"""CPU tests for ``patch_rocm_router_bf16x3``. vLLM is replaced by a stand-in module, so none is needed.

ROCm is simulated by setting ``torch.version.hip``, the build attribute ``is_rocm`` reads.

uv run --isolated --extra skyrl-train --extra dev pytest tests/backends/skyrl_train/patches/vllm/test_rocm_router_bf16x3.py
"""

import sys
import types

import pytest
import torch

from skyrl.backends.skyrl_train.patches.vllm import patch_rocm_router_bf16x3

_KERNEL = "vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_rocm"


@pytest.fixture
def kernel(monkeypatch):
    """Stands in for vLLM's kernel module, which reports gfx950 as supported."""
    monkeypatch.setattr(patch_rocm_router_bf16x3, "_PATCHED", False)
    parent = None
    for depth in range(1, _KERNEL.count(".") + 2):
        name = ".".join(_KERNEL.split(".")[:depth])
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        if parent is not None:
            setattr(parent, name.rsplit(".", 1)[1], module)
        parent = module
    parent.platform_supported = lambda: True
    return parent


def test_on_rocm_the_kernel_reports_its_platform_unsupported(monkeypatch, kernel):
    monkeypatch.setattr(torch.version, "hip", "7.15.26333", raising=False)

    assert patch_rocm_router_bf16x3.apply_rocm_router_bf16x3_patch() is True
    # GateLinear reads this through the module at model init, so no split copy is built.
    assert kernel.platform_supported() is False
    assert patch_rocm_router_bf16x3.apply_rocm_router_bf16x3_patch() is False  # once per process


def test_off_rocm_vllm_is_left_alone(monkeypatch, kernel):
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    assert patch_rocm_router_bf16x3.apply_rocm_router_bf16x3_patch() is False
    assert kernel.platform_supported() is True


def test_a_vllm_without_the_kernel_is_left_alone(monkeypatch, kernel):
    monkeypatch.setattr(torch.version, "hip", "7.15.26333", raising=False)
    monkeypatch.setitem(sys.modules, _KERNEL, None)  # importing it raises ImportError
    monkeypatch.delattr(sys.modules[_KERNEL.rsplit(".", 1)[0]], "bf16x3_router_gemm_rocm")

    assert patch_rocm_router_bf16x3.apply_rocm_router_bf16x3_patch() is False
