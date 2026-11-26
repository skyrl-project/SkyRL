# An accelerator is the non-CPU device torch computes on: "cuda" for NVIDIA,
# "xpu" for Intel, "mps" for Apple, and so on. This helper only adds "xpu" for
# now; other accelerators such as MPS, and CPU-only torch builds, fall back to
# "cuda" so they keep their existing behaviour.

import torch

_ENABLED_TRAINING_ACCELERATORS = {"cuda", "xpu"}

# Env var that pins the mp backend's workers to their devices.
_VLLM_DEVICE_CONTROL_ENV_VAR = {
    "cuda": "CUDA_VISIBLE_DEVICES",
    "xpu": "ZE_AFFINITY_MASK",
}

# Worker class vLLM's platform picks for worker_cls="auto".
_VLLM_WORKER_CLS = {
    "cuda": "vllm.v1.worker.gpu_worker.Worker",
    "xpu": "vllm.v1.worker.xpu_worker.XPUWorker",
}

def device_type() -> str:
    """Return the device type training runs on. Eg: ``"cuda"`` or ``"xpu"``.
    Uses the accelerator torch was built for, so it works in a process that sees no device.
    """

    if hasattr(torch, "accelerator"):
        accelerator = torch.accelerator.current_accelerator()
        if accelerator is not None and accelerator.type in _ENABLED_TRAINING_ACCELERATORS:
            return accelerator.type
    return "cuda"


def device_module():
    """Return the torch device module for ``device_type()``. Eg: ``torch.cuda`` or ``torch.xpu``."""

    return torch.get_device_module(device_type())

def vllm_device_control_env_var() -> str:
    """Return the env var vLLM uses to pick devices for ``device_type()``. Eg: ``"CUDA_VISIBLE_DEVICES"``."""

    return _VLLM_DEVICE_CONTROL_ENV_VAR[device_type()]

def vllm_worker_cls() -> str:
    """Return vLLM's default worker class for ``device_type()``. Eg: ``"vllm.v1.worker.gpu_worker.Worker"``."""

    return _VLLM_WORKER_CLS[device_type()]
