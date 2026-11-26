"""Megatron's per-step MoE expert-bias update under ``freeze_moe_router`` and LoRA.

DeepSeek-style routers carry an expert bias (HF ``e_score_correction_bias``) that Megatron's
``finalize_model_grads`` moves after every optimizer step to balance expert load, without a
gradient. ``freeze_moe_router`` must stop it, and so must LoRA: the adapter-only sync
(``merge_lora=false``) never sends the bias to vLLM and LoRA checkpoints never save it. Each case
runs one real training step on the policy worker and compares every router's bias before and
after; the full fine-tune case is the control that shows the update runs at all.

Run with:
uv run --isolated --extra dev --extra megatron pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_moe_expert_bias_freeze.py
"""

from typing import Optional

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.train.config import SkyRLLoraConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import ray_init
from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_megatron_models import (
    get_test_actor_config,
)
from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
    make_dummy_training_batch,
)

pytestmark = pytest.mark.megatron

MODEL_NAME = "yujiepan/deepseek-v3-tiny-random"
BIAS_UPDATE_RATE = 1e-3  # Megatron's default moe_router_bias_update_rate, which the DeepSeek-V3 bridge keeps
BATCH_SIZE = 4


class _ExpertBiasPolicyWorkerBase(MegatronPolicyWorkerBase):
    def _routers(self):
        return [
            (f"chunk{index}.{name}", module)
            for index, chunk in enumerate(self.actor_module)
            for name, module in chunk.named_modules()
            if getattr(module, "expert_bias", None) is not None
        ]

    def expert_biases(self) -> dict:
        """Every router's expert bias on this rank, keyed by module name."""
        return {name: module.expert_bias.detach().float().cpu().clone() for name, module in self._routers()}

    def zero_expert_biases(self) -> None:
        """The random checkpoint stores its ``e_score_correction_bias`` uninitialized (all NaN), so
        start from zeros, Megatron's own initial value, to give the update something finite to move."""
        for _, module in self._routers():
            module.expert_bias.zero_()


_ExpertBiasPolicyWorker = ray.remote(num_gpus=1)(_ExpertBiasPolicyWorkerBase)


def _config(freeze_moe_router: bool, merge_lora: Optional[bool]):
    cfg = get_test_actor_config(MODEL_NAME)
    cfg.trainer.strategy = "megatron"
    # MLA has no packed (thd) Transformer Engine attention backend before Hopper; the same
    # fallback get_test_actor_config applies to its other MLA models.
    cfg.trainer.flash_attn = False
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:
        cfg.trainer.remove_microbatch_padding = False
    cfg.trainer.policy.megatron_config.freeze_moe_router = freeze_moe_router
    if merge_lora is not None:
        cfg.trainer.policy.model.lora = SkyRLLoraConfig(rank=8, alpha=8)
        cfg.trainer.policy.megatron_config.lora_config.merge_lora = merge_lora
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.train_batch_size = BATCH_SIZE
    cfg.trainer.policy_mini_batch_size = 1
    cfg.generator.n_samples_per_prompt = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    validate_cfg(cfg)
    return cfg


@pytest.mark.parametrize(
    ("freeze_moe_router", "merge_lora", "frozen"),
    [
        pytest.param(False, None, False, id="full_finetune"),
        pytest.param(True, None, True, id="freeze_moe_router"),
        pytest.param(False, False, True, id="lora_adapter_sync"),
        pytest.param(False, True, True, id="lora_merged_sync"),
    ],
)
def test_expert_bias_after_one_training_step(freeze_moe_router, merge_lora, frozen):
    cfg = _config(freeze_moe_router, merge_lora)
    batch = make_dummy_training_batch(batch_size=BATCH_SIZE, seq_len=32, num_actions=8)
    batch.metadata["global_step"] = 0

    # MLA needs cuDNN fused attention, which the GPU-test conftest turns off for every worker
    # (NVTE_FUSED_ATTN=0); re-enable it, as test_megatron_models does for its MLA models.
    with ray_init(extra_env_vars={"NVTE_FUSED_ATTN": "1"}):
        policy = init_worker_with_type("policy", cfg=cfg, num_gpus_per_node=1, worker_cls=_ExpertBiasPolicyWorker)
        ray.get(policy.async_run_ray_method("pass_through", "zero_expert_biases"))
        before = ray.get(policy.async_run_ray_method("pass_through", "expert_biases"))[0]
        assert before, f"{MODEL_NAME} built no router with an expert bias"

        ray.get(policy.async_run_ray_method("mesh", "forward_backward", batch))
        ray.get(policy.async_run_ray_method("pass_through", "optim_step"))
        after = ray.get(policy.async_run_ray_method("pass_through", "expert_biases"))[0]

    assert all(bias.isfinite().all() for bias in after.values()), f"non-finite expert bias after one step: {after}"
    moved = max((after[name] - before[name]).abs().max().item() for name in before)
    print(f"{len(before)} routers with an expert bias; largest move after one step: {moved:.2e}")
    if frozen:
        assert moved == 0.0, f"the expert bias moved by {moved:.2e} in one step although it should be frozen"
    else:
        assert moved == pytest.approx(BIAS_UPDATE_RATE, rel=1e-3)
