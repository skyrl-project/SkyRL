"""Runtime patch: on AMD ROCm, keep vLLM's MoE router gate off its bf16x3 GEMM so weight sync can reload it.

On MI350X/MI355X (gfx950), vLLM gives a ``GateLinear`` whose weight is fp32 a second copy of that weight,
``_bf16x3_weight``, split into three bf16 terms for its ROCm bf16x3 router GEMM
(``model_executor/layers/fused_moe/router/gate_linear.py``). The gate reads that copy, not ``weight``, for
batches of 2048 tokens and more. NemotronH's router gate is such a layer; a gate with a bf16 weight
(Qwen3.5-35B-A3B) is not.

vLLM's layerwise reload, which every full-weight sync goes through, cannot put that copy back. It is absent
from the layer's restore metadata, so after ``restore_layer_on_meta`` the rebuilt split is an ordinary
attribute and no longer a buffer; ``_copy_and_restore_kernel_tensors`` then skips it and
``_place_kernel_tensors`` raises ``KeyError: "attribute '_bf16x3_weight' already exists"``. The first weight
sync of a run fails in every engine worker (vLLM 0.33.0, Nemotron-3.5-Lightning on MI355X).

This makes the kernel report its platform as unsupported, so the gate builds no split copy and computes its
logits from the fp32 weight itself (vLLM's low-token ROCm kernel, ``F.linear`` above its token limit). The
weight a sync replaces is then the only one the router reads. Making the reload tolerate the attribute
would not be enough: the copy would keep the old weights.

It must be installed before the model is built, and an engine that compiled its model with the kernel on
left a graph in vLLM's compile cache (``~/.cache/vllm/torch_compile_cache``) that expects the copy: empty
that cache when moving a machine from an unpatched vLLM to this.

TODO: remove this patch and its call in ``new_inference_worker_wrap.py`` once the pinned vLLM rebuilds
``_bf16x3_weight`` on a layerwise reload.
"""

from loguru import logger

from skyrl.backends.skyrl_train.gpu_platform import is_rocm

_PATCHED = False


def apply_rocm_router_bf16x3_patch() -> bool:
    """Install the patch once per process. Returns True if installed; does nothing off ROCm."""
    global _PATCHED
    if _PATCHED or not is_rocm():
        return False
    try:
        from vllm.model_executor.layers.fused_moe.router import bf16x3_router_gemm_rocm
    except ImportError:  # no vLLM, or a vLLM without this kernel
        return False
    bf16x3_router_gemm_rocm.platform_supported = lambda: False
    _PATCHED = True
    logger.info("Patched vLLM's ROCm bf16x3 router GEMM off: MoE router gates use their fp32 weight")
    return True
