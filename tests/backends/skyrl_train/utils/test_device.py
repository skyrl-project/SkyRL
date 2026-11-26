"""
Unit tests for training device selection.
"""

import importlib.util
from types import SimpleNamespace

import pytest
import torch

import skyrl.backends.skyrl_train.utils.device as device_utils
from skyrl.backends.skyrl_train.utils.device import (
    _ENABLED_TRAINING_ACCELERATORS,
    device_module,
    device_type,
)


def _mock_accelerator(monkeypatch, accelerator_type):
    """Fake the accelerator torch was built for, with no device visible (to simulate a GPU-less Ray driver)."""
    accelerator = None if accelerator_type is None else SimpleNamespace(type=accelerator_type)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda check_available=False: accelerator)


@pytest.mark.parametrize(
    "accelerator_type, expected",
    [(a, a) for a in sorted(_ENABLED_TRAINING_ACCELERATORS)]
    + [
        # Accelerators that are not enabled for training fall back to cuda.
        ("mps", "cuda"),
        # CPU-only torch builds fall back to cuda.
        (None, "cuda"),
    ],
)
def test_device_type(monkeypatch, accelerator_type, expected):
    _mock_accelerator(monkeypatch, accelerator_type)
    assert device_type() == expected


def test_device_type_without_torch_accelerator(monkeypatch):
    monkeypatch.delattr(torch, "accelerator")
    assert device_type() == "cuda"


@pytest.mark.parametrize(
    "accelerator_type, expected",
    [(a, getattr(torch, a)) for a in sorted(_ENABLED_TRAINING_ACCELERATORS)] + [(None, torch.cuda)],
)
def test_device_module(monkeypatch, accelerator_type, expected):
    _mock_accelerator(monkeypatch, accelerator_type)
    assert device_module() is expected


@pytest.mark.parametrize("dev_type", sorted(_ENABLED_TRAINING_ACCELERATORS))
def test_vllm_tables_cover_enabled_accelerators(monkeypatch, dev_type):
    # Enabling an accelerator without adding it to both vLLM tables would raise KeyError only once vLLM starts.
    monkeypatch.setattr(device_utils, "device_type", lambda: dev_type)
    assert device_utils.vllm_device_control_env_var()
    assert device_utils.vllm_worker_cls()


@pytest.mark.vllm
@pytest.mark.parametrize("dev_type", sorted(_ENABLED_TRAINING_ACCELERATORS))
def test_vllm_worker_cls_module_exists(monkeypatch, dev_type):
    # Worker classes are hardcoded strings; catch a vLLM bump that moves a module (checks the module, not the class).
    # find_spec doesn't import it, so the XPU row also runs on hosts that can't import the XPU worker.
    monkeypatch.setattr(device_utils, "device_type", lambda: dev_type)
    worker_module = device_utils.vllm_worker_cls().rsplit(".", 1)[0]
    assert importlib.util.find_spec(worker_module) is not None
