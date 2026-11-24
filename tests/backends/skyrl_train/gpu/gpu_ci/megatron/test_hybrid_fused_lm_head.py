"""The fused LM head on a HybridModel through patch_hybrid_output_processor.py's ``output_processor`` hook.

The tiny Beam checkpoint in bf16 on one GPU, through ``MegatronModelWrapper``: the policy's log-probs, then a training
step's NLL and MTP draft loss. 1. With one MTP layer, the fused head matches the materialized logits. 2. Without MTP,
the hook is bit-identical to the post_process-off fallback it replaces.
"""

import functools
import inspect
import os
from contextlib import contextmanager

import pytest
import torch
from megatron.bridge import AutoBridge
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.utils import unwrap_model

from skyrl.backends.skyrl_train.mtp.native_loss_patch import disable_native_mtp_loss
from skyrl.backends.skyrl_train.patches.megatron import (
    patch_hybrid_output_processor as patch,
)
from skyrl.backends.skyrl_train.utils.ppo_utils import PolicyLossRegistry
from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import (
    MegatronModelWrapper,
)
from tests.backends.skyrl_train.gpu.gpu_ci.megatron.megatron_tiny_models.create_beam_tiny import (
    create_beam_tiny,
)
from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_mtp_draft_loss_tp_pp_cp import (
    _micro_batches,
    _trainer_cfg,
)
from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_mtp_hard_label_loss_vs_native import (
    SEQ,
    _make_rl_batch,
    init_megatron,
    run_on_one_node,
    teardown_megatron,
)

pytestmark = pytest.mark.megatron


def _beam(mtp: bool) -> list:
    # The checkpoint of test_megatron_models.py's beam-tiny rows, built on first use.
    path = create_beam_tiny(os.environ.get("SKYRL_BEAM_TINY_PATH", "/tmp/skyrl-beam-tiny-random"))
    provider = AutoBridge.from_hf_pretrained(path).to_megatron_provider(load_weights=not mtp)
    provider.gradient_accumulation_fusion, provider.attention_backend = False, "flash"  # as the policy worker sets them
    if mtp:
        # BeamBridge has no MTP layer: add one attention + dense MLP depth on megatron-core's hybrid MTP block, without
        # Beam's depth-scaled residuals (Bridge refuses them with MTP; they only rescale the residual adds).
        provider.mtp_num_layers, provider.mtp_hybrid_override_pattern = 1, "*-"
        provider.depth_scaled_residual = False
        spec = provider._resolve_hybrid_stack_spec()
        spec.submodules.mtp_block_spec = hybrid_stack_spec.submodules.mtp_block_spec
        provider._resolve_hybrid_stack_spec = lambda: spec
    provider.finalize()
    model = provider.provide_distributed_model(ddp_config=DistributedDataParallelConfig(), bf16=True)
    if mtp:  # BeamBridge maps no MTP weights: copy the checkpoint's from a model without MTP.
        trunk = unwrap_model(_beam(mtp=False)[0]).state_dict()
        missing, unexpected = unwrap_model(model[0]).load_state_dict(trunk, strict=False)
        assert missing and all(k.startswith("mtp.") for k in missing) and not unexpected, (missing, unexpected)
    return model


def _run(model, fused: bool) -> dict:
    cfg = _trainer_cfg("fused_triton" if fused else "logits", chunk=1024)
    cfg.fused_lm_head_logprob_backend = "torch"  # what Triton falls back to: it has no final logit softcap
    wrapper = MegatronModelWrapper(cfg, model, policy_loss_fn=PolicyLossRegistry.get("regular"))
    micros = _micro_batches(_make_rl_batch(torch.device("cuda")), "one_microbatch")
    with torch.no_grad():  # as the policy worker runs it
        log_probs = wrapper.forward([dict(m) for m in micros], seq_len=SEQ, micro_batch_size=4)
    (metrics,) = wrapper.forward_backward_mini_batch(micros, seq_len=SEQ, micro_batch_size=4, loss_fn="cross_entropy")
    return dict(log_probs=log_probs.cpu(), loss=metrics["loss"], mtp_loss=metrics.get("mtp_loss"))


@contextmanager
def _without_hook():
    """megatron-core's own HybridModel.forward, so ``call_model_with_fused_lm_head`` takes its old fallback."""
    forward = inspect.unwrap(HybridModel.forward)
    hooked = forward.__code__, forward.__defaults__, forward.__kwdefaults__
    forward.__code__, forward.__defaults__, forward.__kwdefaults__ = patch._ORIGINALS[forward][:3]
    try:
        yield
    finally:
        forward.__code__, forward.__defaults__, forward.__kwdefaults__ = hooked


def _on_one_gpu(fn, world_size, rank, init_method):
    init_megatron(world_size, rank, init_method)
    try:
        assert patch.apply_hybrid_output_processor_patch()
        return fn()
    finally:
        teardown_megatron()


def _logits_vs_fused_with_mtp():
    disable_native_mtp_loss()  # as the policy worker does for a model with MTP heads
    model = _beam(mtp=True)
    runs = _run(model, fused=False), _run(model, fused=True)
    with _without_hook(), pytest.raises(NotImplementedError, match="HybridModel with MTP"):
        _run(model, fused=True)  # what the hook fixes
    return runs


def _hook_vs_fallback():
    model = _beam(mtp=False)
    hook = _run(model, fused=True)
    with _without_hook():
        return hook, _run(model, fused=True)


def test_fused_lm_head_with_mtp_matches_logits(module_scoped_ray_init_fixture):
    ((logits, fused),) = run_on_one_node(functools.partial(_on_one_gpu, _logits_vs_fused_with_mtp), world_size=1)
    print(f"\nlogits: {logits}\nfused: {fused}")
    # Both paths round the logits to bf16 (Beam's softcap keeps the fused head on its torch matmul). Another GEMM tiling
    # can round a logit to the next bf16 value, 2^-5 apart below the top logits' ~7, so a log-prob (target logit minus
    # logsumexp) moves by up to 2^-4: per token, so for the draft loss (a token mean) too. The NLL is a token sum.
    tol, num_tokens = 2**-4, int(_make_rl_batch("cpu")["loss_mask"].sum())
    assert (fused["log_probs"] - logits["log_probs"]).abs().max() <= tol
    assert logits["mtp_loss"] is not None and abs(fused["mtp_loss"] - logits["mtp_loss"]) <= tol
    assert abs(fused["loss"] - logits["loss"]) <= tol * num_tokens


def test_hook_is_bit_identical_to_the_post_process_off_fallback(module_scoped_ray_init_fixture):
    ((hook, fallback),) = run_on_one_node(functools.partial(_on_one_gpu, _hook_vs_fallback), world_size=1)
    assert torch.equal(hook["log_probs"], fallback["log_probs"])
    assert hook["loss"] == fallback["loss"]
