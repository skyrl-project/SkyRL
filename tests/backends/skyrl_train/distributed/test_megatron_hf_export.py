"""CPU unit tests for Megatron HF LoRA export (`merged`, `adapter`, `both`).

Exercises `MegatronStrategy.save_hf_model` and `MegatronWorker.save_hf_model`
without GPUs or distributed init, and validates `save_hf_adapter` call kwargs
against the pinned `AutoBridge.save_hf_adapter` signature.
"""

import os
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest

from skyrl.train.config import MegatronHFExportConfig

pytestmark = pytest.mark.megatron

try:
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.conversion.mapping_registry import (
        MegatronMappingRegistry,
    )
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        FusedExpertMapping,
        FusedGatedExpertMapping,
        GatedMLPMapping,
    )

    from skyrl.backends.skyrl_train.distributed.megatron import (
        megatron_strategy as strategy_mod,
    )
    from skyrl.backends.skyrl_train.distributed.megatron.lora_import import SkyRLLoRA
    from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
        has_packed_hf_expert_mappings,
    )
    from skyrl.backends.skyrl_train.workers.megatron import (
        megatron_worker as worker_mod,
    )
except ModuleNotFoundError as e:
    if not e.name or (e.name != "megatron" and not e.name.startswith("megatron.")):
        raise
    pytest.skip(f"megatron unavailable: {e}", allow_module_level=True)


def _make_strategy(*, is_lora: bool = True, is_rank_0: bool = True):
    owner = strategy_mod.MegatronStrategy.__new__(strategy_mod.MegatronStrategy)
    owner.is_lora = is_lora
    owner.hf_config = Mock()
    owner.is_rank_0 = Mock(return_value=is_rank_0)
    owner.save_hf_configs = Mock()
    owner.print = Mock()
    return owner


def _make_bridge():
    # Autospec against the pinned AutoBridge.save_hf_adapter / save_hf_weights
    # so signature drift fails on CPU.
    save_hf_adapter = create_autospec(AutoBridge.save_hf_adapter, return_value=None)
    save_hf_weights = create_autospec(AutoBridge.save_hf_weights, return_value=None)

    bridge = SimpleNamespace(
        hf_pretrained=SimpleNamespace(save_artifacts=Mock()),
    )
    bridge.save_hf_adapter = lambda *args, **kwargs: save_hf_adapter(bridge, *args, **kwargs)
    bridge.save_hf_weights = lambda *args, **kwargs: save_hf_weights(bridge, *args, **kwargs)
    return bridge, save_hf_adapter, save_hf_weights


def test_megatron_strategy_save_hf_model_merged_only(tmp_path, monkeypatch):
    monkeypatch.setattr(strategy_mod.dist, "barrier", Mock())
    owner = _make_strategy(is_lora=True, is_rank_0=True)
    bridge, save_adapter, save_weights = _make_bridge()
    model = SimpleNamespace(actor_module=["chunk0"])
    tokenizer = Mock()
    out_dir = str(tmp_path / "policy")

    owner.save_hf_model(
        bridge,
        model,
        out_dir,
        tokenizer=tokenizer,
        distributed_save=True,
        save_every_n_ranks=2,
        lora_export_format="merged",
        peft_config=Mock(),
    )

    save_weights.assert_called_once_with(
        bridge,
        ["chunk0"],
        out_dir,
        strict=False,
        distributed_save=True,
        save_every_n_ranks=2,
    )
    bridge.hf_pretrained.save_artifacts.assert_called_once_with(out_dir)
    owner.save_hf_configs.assert_called_once_with(owner.hf_config, out_dir, tokenizer)
    save_adapter.assert_not_called()
    tokenizer.save_pretrained.assert_not_called()


def test_megatron_strategy_save_hf_model_adapter_only(tmp_path, monkeypatch):
    monkeypatch.setattr(strategy_mod.dist, "barrier", Mock())
    owner = _make_strategy(is_lora=True, is_rank_0=True)
    bridge, save_adapter, save_weights = _make_bridge()
    model = SimpleNamespace(actor_module=["chunk0"])
    peft_config = Mock()
    tokenizer = Mock()
    out_dir = str(tmp_path / "policy")

    owner.save_hf_model(
        bridge,
        model,
        out_dir,
        tokenizer=tokenizer,
        lora_export_format="adapter",
        peft_config=peft_config,
        base_model_name_or_path="Qwen/Qwen3-8B",
        expand_shared_outer=True,
        allow_serving_layout=False,
    )

    save_weights.assert_not_called()
    bridge.hf_pretrained.save_artifacts.assert_not_called()
    owner.save_hf_configs.assert_not_called()
    save_adapter.assert_called_once_with(
        bridge,
        ["chunk0"],
        out_dir,
        peft_config=peft_config,
        base_model_name_or_path="Qwen/Qwen3-8B",
        expand_shared_outer=True,
        allow_serving_layout=False,
    )
    tokenizer.save_pretrained.assert_called_once_with(out_dir)


def test_megatron_strategy_save_hf_model_both(tmp_path, monkeypatch):
    monkeypatch.setattr(strategy_mod.dist, "barrier", Mock())
    owner = _make_strategy(is_lora=True, is_rank_0=True)
    bridge, save_adapter, save_weights = _make_bridge()
    model = SimpleNamespace(actor_module=["chunk0"])
    peft_config = Mock()
    tokenizer = Mock()
    out_dir = str(tmp_path / "policy")
    expected_adapter_dir = os.path.join(out_dir, "lora_adapter")

    owner.save_hf_model(
        bridge,
        model,
        out_dir,
        tokenizer=tokenizer,
        lora_export_format="both",
        peft_config=peft_config,
        base_model_name_or_path="Qwen/Qwen3-8B",
        allow_serving_layout=True,
    )

    save_weights.assert_called_once_with(
        bridge,
        ["chunk0"],
        out_dir,
        strict=False,
        distributed_save=False,
        save_every_n_ranks=1,
    )
    bridge.hf_pretrained.save_artifacts.assert_called_once_with(out_dir)
    owner.save_hf_configs.assert_called_once_with(owner.hf_config, out_dir, tokenizer)
    save_adapter.assert_called_once_with(
        bridge,
        ["chunk0"],
        expected_adapter_dir,
        peft_config=peft_config,
        base_model_name_or_path="Qwen/Qwen3-8B",
        expand_shared_outer=False,
        allow_serving_layout=True,
    )
    tokenizer.save_pretrained.assert_called_once_with(expected_adapter_dir)


def test_megatron_strategy_save_hf_model_non_rank0_skips_tokenizer_write(tmp_path, monkeypatch):
    monkeypatch.setattr(strategy_mod.dist, "barrier", Mock())
    monkeypatch.setattr(strategy_mod.dist, "is_initialized", Mock(return_value=True))
    monkeypatch.setattr(strategy_mod.dist, "broadcast_object_list", Mock())
    owner = _make_strategy(is_lora=True, is_rank_0=False)
    bridge, save_adapter, _ = _make_bridge()
    model = SimpleNamespace(actor_module=["chunk0"])
    peft_config = Mock()
    tokenizer = Mock()
    out_dir = str(tmp_path / "policy")

    owner.save_hf_model(
        bridge,
        model,
        out_dir,
        tokenizer=tokenizer,
        lora_export_format="adapter",
        peft_config=peft_config,
    )

    # All ranks must still participate in the collective bridge call.
    save_adapter.assert_called_once()
    tokenizer.save_pretrained.assert_not_called()


def test_megatron_strategy_rank0_tokenizer_failure_propagates_to_all_ranks(tmp_path, monkeypatch):
    barrier = Mock()
    monkeypatch.setattr(strategy_mod.dist, "barrier", barrier)
    monkeypatch.setattr(strategy_mod.dist, "is_initialized", Mock(return_value=True))

    broadcast_payload = []

    def _capture_broadcast(obj_list, src=0):
        broadcast_payload.append(list(obj_list))

    monkeypatch.setattr(strategy_mod.dist, "broadcast_object_list", _capture_broadcast)

    owner_r0 = _make_strategy(is_lora=True, is_rank_0=True)
    bridge, _, _ = _make_bridge()
    model = SimpleNamespace(actor_module=["chunk0"])
    tokenizer = Mock()
    tokenizer.save_pretrained.side_effect = OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        owner_r0.save_hf_model(
            bridge,
            model,
            str(tmp_path / "policy"),
            tokenizer=tokenizer,
            lora_export_format="adapter",
            peft_config=Mock(),
        )
    assert broadcast_payload == [["OSError: disk full"]]
    # Must raise before the trailing barrier so ranks do not hang.
    assert barrier.call_count == 1

    # Non-rank-0 receives the broadcast error status even if tokenizer=None locally.
    def _deliver_error(obj_list, src=0):
        obj_list[0] = "OSError: disk full"

    monkeypatch.setattr(strategy_mod.dist, "broadcast_object_list", _deliver_error)
    owner_r1 = _make_strategy(is_lora=True, is_rank_0=False)
    with pytest.raises(RuntimeError, match="Rank 0 failed to save tokenizer.*OSError: disk full"):
        owner_r1.save_hf_model(
            bridge,
            model,
            str(tmp_path / "policy"),
            tokenizer=None,
            lora_export_format="adapter",
            peft_config=Mock(),
        )


def test_megatron_strategy_both_mode_rank0_metadata_failure_aborts_before_adapter_export(tmp_path, monkeypatch):
    monkeypatch.setattr(strategy_mod.dist, "barrier", Mock())
    monkeypatch.setattr(strategy_mod.dist, "is_initialized", Mock(return_value=True))

    def _deliver_error(obj_list, src=0):
        obj_list[0] = "OSError: metadata write failed"

    monkeypatch.setattr(strategy_mod.dist, "broadcast_object_list", _deliver_error)
    owner_r1 = _make_strategy(is_lora=True, is_rank_0=False)
    bridge, save_adapter, save_weights = _make_bridge()
    model = SimpleNamespace(actor_module=["chunk0"])

    with pytest.raises(RuntimeError, match="Rank 0 failed to save HF config and tokenizer.*metadata write failed"):
        owner_r1.save_hf_model(
            bridge,
            model,
            str(tmp_path / "policy"),
            tokenizer=None,
            lora_export_format="both",
            peft_config=Mock(),
        )

    save_weights.assert_called_once()
    save_adapter.assert_not_called()


def test_megatron_strategy_adapter_only_allows_cloud_path_with_distributed_save(tmp_path, monkeypatch):
    """Adapter export only writes from rank 0, so distributed_save=True must not block cloud paths."""
    monkeypatch.setattr(strategy_mod.dist, "barrier", Mock())
    monkeypatch.setattr(strategy_mod.io, "is_cloud_path", lambda p: p.startswith("s3://"))
    monkeypatch.setattr(strategy_mod.io, "makedirs", Mock())

    staged_dir = str(tmp_path / "staged")
    os.makedirs(staged_dir)

    from contextlib import contextmanager

    @contextmanager
    def _fake_local_work_dir(_path):
        yield staged_dir

    monkeypatch.setattr(strategy_mod.io, "local_work_dir", _fake_local_work_dir)

    owner = _make_strategy(is_lora=True, is_rank_0=True)
    bridge, save_adapter, save_weights = _make_bridge()
    model = SimpleNamespace(actor_module=["chunk0"])
    peft_config = Mock()

    owner.save_hf_model(
        bridge,
        model,
        "s3://bucket/exports/step_1/policy",
        tokenizer=None,
        distributed_save=True,
        lora_export_format="adapter",
        peft_config=peft_config,
    )

    save_weights.assert_not_called()
    save_adapter.assert_called_once_with(
        bridge,
        ["chunk0"],
        staged_dir,
        peft_config=peft_config,
        base_model_name_or_path=None,
        expand_shared_outer=False,
        allow_serving_layout=False,
    )


@pytest.mark.parametrize(
    "is_lora, kwargs, match",
    [
        (True, {"lora_export_format": "unknown"}, "lora_export_format must be one of"),
        (False, {"lora_export_format": "adapter", "peft_config": Mock()}, "requires a LoRA policy"),
        (True, {"lora_export_format": "adapter", "peft_config": None}, "requires a LoRA policy"),
        (
            True,
            {
                "lora_export_format": "adapter",
                "peft_config": Mock(),
                "expand_shared_outer": True,
                "allow_serving_layout": True,
            },
            "cannot both be True",
        ),
        (
            True,
            {"lora_export_format": "merged", "expand_shared_outer": True},
            "require lora_export_format to be 'adapter' or 'both'",
        ),
    ],
)
def test_megatron_strategy_save_hf_model_rejects_invalid_args(tmp_path, is_lora, kwargs, match):
    owner = _make_strategy(is_lora=is_lora, is_rank_0=True)
    bridge, _, _ = _make_bridge()
    model = SimpleNamespace(actor_module=["chunk0"])

    with pytest.raises(ValueError, match=match):
        owner.save_hf_model(bridge, model, str(tmp_path / "policy"), **kwargs)


def test_megatron_worker_save_hf_model_forwards_adapter_export_config(monkeypatch):
    monkeypatch.setattr(worker_mod, "sync_params_for_export", Mock())
    strategy_save = Mock()
    peft_config = Mock()
    tokenizer = Mock()
    bridge = Mock()
    model = Mock()

    worker = SimpleNamespace(
        actor_module=["chunk0"],
        optimizer=None,
        bridge=bridge,
        model=model,
        lora_cls=peft_config,
        _logical_model_path="Qwen/Qwen3-8B",
        megatron_config=SimpleNamespace(
            hf_export_config=MegatronHFExportConfig(
                distributed_save=False,
                save_every_n_ranks=1,
                lora_export_format="both",
                expand_shared_outer=True,
                allow_serving_layout=False,
            )
        ),
        strategy=SimpleNamespace(save_hf_model=strategy_save),
    )

    worker_mod.MegatronWorker.save_hf_model(worker, "/tmp/export/policy", tokenizer=tokenizer)

    strategy_save.assert_called_once_with(
        bridge,
        model,
        "/tmp/export/policy",
        tokenizer=tokenizer,
        distributed_save=False,
        save_every_n_ranks=1,
        lora_export_format="both",
        peft_config=peft_config,
        base_model_name_or_path="Qwen/Qwen3-8B",
        expand_shared_outer=True,
        allow_serving_layout=False,
        moe_grouped_gemm=True,
    )


def _make_bridge_with_registry(registry: MegatronMappingRegistry):
    bridge, save_adapter, save_weights = _make_bridge()
    bridge._model_bridge = SimpleNamespace(mapping_registry=lambda: registry)
    return bridge, save_adapter, save_weights


def _packed_expert_registry(*, megatron_prefix: str = "", sequential: bool = False) -> MegatronMappingRegistry:
    if sequential:
        fc1 = f"{megatron_prefix}decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight"
        fc2 = f"{megatron_prefix}decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight"
    else:
        fc1 = f"{megatron_prefix}decoder.layers.*.mlp.experts.linear_fc1.weight*"
        fc2 = f"{megatron_prefix}decoder.layers.*.mlp.experts.linear_fc2.weight*"
    return MegatronMappingRegistry(
        FusedGatedExpertMapping(
            megatron_param=fc1,
            hf_param="model.language_model.layers.*.mlp.experts.gate_up_proj",
        ),
        FusedExpertMapping(
            megatron_param=fc2,
            hf_param="model.language_model.layers.*.mlp.experts.down_proj",
        ),
    )


def _per_expert_registry() -> MegatronMappingRegistry:
    return MegatronMappingRegistry(
        GatedMLPMapping(
            megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
            gate="model.layers.*.mlp.experts.*.gate_proj.weight",
            up="model.layers.*.mlp.experts.*.up_proj.weight",
        ),
        AutoMapping(
            megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
            hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
        ),
    )


@pytest.mark.parametrize(
    "registry, expected",
    [
        pytest.param(_packed_expert_registry(), True, id="packed_grouped"),
        pytest.param(_packed_expert_registry(megatron_prefix="language_model."), True, id="packed_vl_prefixed"),
        pytest.param(_packed_expert_registry(sequential=True), True, id="packed_sequential"),
        pytest.param(_per_expert_registry(), False, id="per_expert"),
        pytest.param(
            MegatronMappingRegistry(
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc2.weight",
                    hf_param="model.layers.*.mlp.down_proj.weight",
                )
            ),
            False,
            id="dense",
        ),
    ],
)
def test_has_packed_hf_expert_mappings(registry, expected):
    bridge, _, _ = _make_bridge_with_registry(registry)
    assert has_packed_hf_expert_mappings(bridge) is expected


@pytest.mark.parametrize("fmt", ["adapter", "both"])
@pytest.mark.parametrize(
    "expand_shared_outer, allow_serving_layout",
    [(True, False), (False, True)],
)
def test_megatron_strategy_rejects_shared_outer_adapter_export_on_packed_experts(
    tmp_path, monkeypatch, fmt, expand_shared_outer, allow_serving_layout
):
    """Packed-HF expert checkpoints must fail instead of silently dropping earlier experts."""
    import torch

    barrier = Mock()
    monkeypatch.setattr(strategy_mod.dist, "barrier", barrier)
    owner = _make_strategy(is_lora=True, is_rank_0=True)
    bridge, save_adapter, save_weights = _make_bridge_with_registry(_packed_expert_registry())
    # Distinct per-expert weights across experts 0 and 1 that would collide under one HF key.
    distinct_experts = SimpleNamespace(
        expert_0=torch.ones(4, 2),
        expert_1=torch.full((4, 2), 2.0),
    )
    model = SimpleNamespace(actor_module=[distinct_experts])
    peft_config = SkyRLLoRA(dim=4, alpha=8, experts_shared_outer_loras=True)

    with pytest.raises(ValueError, match="packed HF expert weights"):
        owner.save_hf_model(
            bridge,
            model,
            str(tmp_path / "policy"),
            lora_export_format=fmt,
            peft_config=peft_config,
            expand_shared_outer=expand_shared_outer,
            allow_serving_layout=allow_serving_layout,
        )

    save_adapter.assert_not_called()
    save_weights.assert_not_called()
    barrier.assert_not_called()


def test_megatron_strategy_allows_packed_experts_without_shared_outer_and_per_expert_with_shared_outer(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(strategy_mod.dist, "barrier", Mock())
    owner = _make_strategy(is_lora=True, is_rank_0=True)
    model = SimpleNamespace(actor_module=["chunk0"])

    # Packed experts without shared-outer use 3D ParamWrapper export and stay allowed.
    packed_bridge, save_adapter_packed, _ = _make_bridge_with_registry(_packed_expert_registry())
    owner.save_hf_model(
        packed_bridge,
        model,
        str(tmp_path / "packed_non_shared"),
        lora_export_format="adapter",
        peft_config=SkyRLLoRA(dim=4, alpha=8, experts_shared_outer_loras=False),
    )
    save_adapter_packed.assert_called_once()

    # Per-expert layout with shared-outer stays allowed.
    per_expert_bridge, save_adapter_per_expert, _ = _make_bridge_with_registry(_per_expert_registry())
    owner.save_hf_model(
        per_expert_bridge,
        model,
        str(tmp_path / "per_expert_shared"),
        lora_export_format="adapter",
        peft_config=SkyRLLoRA(dim=4, alpha=8, experts_shared_outer_loras=True),
        expand_shared_outer=True,
    )
    save_adapter_per_expert.assert_called_once()


@pytest.mark.parametrize("fmt", ["adapter", "both"])
def test_megatron_worker_configure_lora_rejects_shared_outer_adapter_export_on_packed_experts(monkeypatch, fmt):
    from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig

    monkeypatch.setattr(worker_mod, "apply_packed_per_expert_sharded_state_dict_patch", Mock())
    cfg = SkyRLTrainConfig().trainer
    cfg.strategy = "megatron"
    cfg.policy.model.lora = SkyRLLoraConfig(rank=8, alpha=8)
    cfg.policy.megatron_config.lora_config.experts_shared_outer_loras = True
    cfg.policy.megatron_config.hf_export_config.lora_export_format = fmt
    cfg.policy.megatron_config.hf_export_config.expand_shared_outer = True

    bridge, _, _ = _make_bridge_with_registry(_packed_expert_registry())
    worker = worker_mod.MegatronPolicyWorkerBase.__new__(worker_mod.MegatronPolicyWorkerBase)
    worker.cfg = cfg
    worker.megatron_config = cfg.policy.megatron_config
    worker.bridge = bridge
    worker.provider = SimpleNamespace(num_moe_experts=4, vision_config=None, multi_latent_attention=False)
    worker.is_vlm = False
    worker._rank = 0

    with pytest.raises(ValueError, match="packed HF expert weights"):
        worker.configure_lora(cfg.policy.model.lora, lora_type="lora", experts_shared_outer_loras=True)


def test_megatron_worker_configure_lora_allows_merged_export_on_packed_experts_with_shared_outer(monkeypatch):
    from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig

    monkeypatch.setattr(worker_mod, "apply_packed_per_expert_sharded_state_dict_patch", Mock())
    cfg = SkyRLTrainConfig().trainer
    cfg.strategy = "megatron"
    cfg.policy.model.lora = SkyRLLoraConfig(rank=8, alpha=8)
    cfg.policy.megatron_config.lora_config.experts_shared_outer_loras = True
    cfg.policy.megatron_config.hf_export_config.lora_export_format = "merged"

    bridge, _, _ = _make_bridge_with_registry(_packed_expert_registry())
    worker = worker_mod.MegatronPolicyWorkerBase.__new__(worker_mod.MegatronPolicyWorkerBase)
    worker.cfg = cfg
    worker.megatron_config = cfg.policy.megatron_config
    worker.bridge = bridge
    worker.provider = SimpleNamespace(num_moe_experts=4, vision_config=None, multi_latent_attention=False)
    worker.is_vlm = False
    worker._rank = 0

    worker.configure_lora(cfg.policy.model.lora, lora_type="lora", experts_shared_outer_loras=True)
    assert worker.lora_cls.experts_shared_outer_loras is True


@pytest.mark.parametrize(
    "lora_kwargs",
    [
        pytest.param({"train_mlp": False}, id="attn_only_group"),
        pytest.param({"target_modules": ["linear_qkv", "linear_proj"]}, id="explicit_attn_targets"),
    ],
)
def test_packed_experts_allow_shared_outer_adapter_export_when_experts_not_targeted(tmp_path, monkeypatch, lora_kwargs):
    """Attention-only adapters do not wrap routed experts, so packed-expert checkpoints stay allowed."""
    from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig

    monkeypatch.setattr(worker_mod, "apply_packed_per_expert_sharded_state_dict_patch", Mock())
    monkeypatch.setattr(strategy_mod.dist, "barrier", Mock())
    cfg = SkyRLTrainConfig().trainer
    cfg.strategy = "megatron"
    cfg.policy.model.lora = SkyRLLoraConfig(rank=8, alpha=8, **lora_kwargs)
    cfg.policy.megatron_config.lora_config.experts_shared_outer_loras = True
    cfg.policy.megatron_config.hf_export_config.lora_export_format = "adapter"
    cfg.policy.megatron_config.hf_export_config.expand_shared_outer = True

    bridge, save_adapter, _ = _make_bridge_with_registry(_packed_expert_registry())
    worker = worker_mod.MegatronPolicyWorkerBase.__new__(worker_mod.MegatronPolicyWorkerBase)
    worker.cfg = cfg
    worker.megatron_config = cfg.policy.megatron_config
    worker.bridge = bridge
    worker.provider = SimpleNamespace(num_moe_experts=4, vision_config=None, multi_latent_attention=False)
    worker.is_vlm = False
    worker._rank = 0

    worker.configure_lora(cfg.policy.model.lora, lora_type="lora", experts_shared_outer_loras=True)

    owner = _make_strategy(is_lora=True, is_rank_0=True)
    owner.save_hf_model(
        bridge,
        SimpleNamespace(actor_module=["chunk0"]),
        str(tmp_path / "attn_only_adapter"),
        lora_export_format="adapter",
        peft_config=worker.lora_cls,
        expand_shared_outer=True,
    )
    save_adapter.assert_called_once()


@pytest.mark.parametrize(
    "target_modules",
    [
        pytest.param(["*.linear_fc*"], id="wildcard_linear_fc"),
        pytest.param(["*.layers.3.mlp.experts.linear_fc1"], id="layer_specific_expert"),
        pytest.param(["*.layers.[2-9][0-9].mlp.experts.linear_fc1"], id="regex_layer_range_expert"),
        pytest.param(["*.layers.1*0.mlp.experts.linear_fc1"], id="partial_wildcard_layer_expert"),
        pytest.param(["*.experts.*"], id="wildcard_experts"),
    ],
)
def test_packed_experts_reject_wildcard_targets_matching_routed_experts(tmp_path, monkeypatch, target_modules):
    """Wildcard patterns matching routed experts must be rejected at both configure_lora and save_hf_model."""
    from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig

    monkeypatch.setattr(worker_mod, "apply_packed_per_expert_sharded_state_dict_patch", Mock())
    barrier = Mock()
    monkeypatch.setattr(strategy_mod.dist, "barrier", barrier)
    cfg = SkyRLTrainConfig().trainer
    cfg.strategy = "megatron"
    cfg.policy.model.lora = SkyRLLoraConfig(rank=8, alpha=8, target_modules=target_modules)
    cfg.policy.megatron_config.lora_config.experts_shared_outer_loras = True
    cfg.policy.megatron_config.hf_export_config.lora_export_format = "adapter"
    cfg.policy.megatron_config.hf_export_config.expand_shared_outer = True

    bridge, save_adapter, save_weights = _make_bridge_with_registry(_packed_expert_registry())
    worker = worker_mod.MegatronPolicyWorkerBase.__new__(worker_mod.MegatronPolicyWorkerBase)
    worker.cfg = cfg
    worker.megatron_config = cfg.policy.megatron_config
    worker.bridge = bridge
    worker.provider = SimpleNamespace(num_moe_experts=4, vision_config=None, multi_latent_attention=False)
    worker.is_vlm = False
    worker._rank = 0

    with pytest.raises(ValueError, match="packed HF expert weights"):
        worker.configure_lora(cfg.policy.model.lora, lora_type="lora", experts_shared_outer_loras=True)

    owner = _make_strategy(is_lora=True, is_rank_0=True)
    peft_config = SkyRLLoRA(dim=8, alpha=8, target_modules=target_modules, experts_shared_outer_loras=True)
    with pytest.raises(ValueError, match="packed HF expert weights"):
        owner.save_hf_model(
            bridge,
            SimpleNamespace(actor_module=["chunk0"]),
            str(tmp_path / "wildcard_packed"),
            lora_export_format="adapter",
            peft_config=peft_config,
            expand_shared_outer=True,
        )

    save_adapter.assert_not_called()
    save_weights.assert_not_called()
    barrier.assert_not_called()


@pytest.mark.parametrize("fmt", ["adapter", "both"])
def test_packed_experts_reject_sequential_experts_adapter_export_without_shared_outer(tmp_path, monkeypatch, fmt):
    """Sequential MoE experts (`moe_grouped_gemm=False`) on packed HF checkpoints must fail even without shared-outer."""
    import torch

    from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig

    barrier = Mock()
    monkeypatch.setattr(strategy_mod.dist, "barrier", barrier)
    cfg = SkyRLTrainConfig().trainer
    cfg.strategy = "megatron"
    cfg.policy.model.lora = SkyRLLoraConfig(rank=8, alpha=8)
    cfg.policy.megatron_config.moe_grouped_gemm = False
    cfg.policy.megatron_config.lora_config.experts_shared_outer_loras = False
    cfg.policy.megatron_config.hf_export_config.lora_export_format = fmt

    bridge, save_adapter, save_weights = _make_bridge_with_registry(_packed_expert_registry(sequential=True))
    worker = worker_mod.MegatronPolicyWorkerBase.__new__(worker_mod.MegatronPolicyWorkerBase)
    worker.cfg = cfg
    worker.megatron_config = cfg.policy.megatron_config
    worker.bridge = bridge
    worker.provider = SimpleNamespace(
        num_moe_experts=2,
        moe_grouped_gemm=False,
        vision_config=None,
        multi_latent_attention=False,
    )
    worker.is_vlm = False
    worker._rank = 0

    with pytest.raises(ValueError, match="moe_grouped_gemm=False"):
        worker.configure_lora(cfg.policy.model.lora, lora_type="lora", experts_shared_outer_loras=False)

    # Distinct per-expert sequential weights (local_experts.0 != local_experts.1) that would collide on export.
    distinct_seq_experts = SimpleNamespace(
        local_experts=[
            SimpleNamespace(linear_fc1=torch.ones(4, 2), linear_fc2=torch.full((2, 4), 1.5)),
            SimpleNamespace(linear_fc1=torch.full((4, 2), 2.0), linear_fc2=torch.full((2, 4), 3.0)),
        ]
    )
    owner = _make_strategy(is_lora=True, is_rank_0=True)
    owner.megatron_config = cfg.policy.megatron_config
    peft_config = SkyRLLoRA(dim=8, alpha=8, experts_shared_outer_loras=False)
    with pytest.raises(ValueError, match="moe_grouped_gemm=False"):
        owner.save_hf_model(
            bridge,
            SimpleNamespace(actor_module=[distinct_seq_experts]),
            str(tmp_path / "seq_packed"),
            lora_export_format=fmt,
            peft_config=peft_config,
            moe_grouped_gemm=False,
        )

    save_adapter.assert_not_called()
    save_weights.assert_not_called()
    barrier.assert_not_called()


def test_packed_experts_sequential_allows_attention_only_adapter_export(tmp_path, monkeypatch):
    from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig

    monkeypatch.setattr(strategy_mod.dist, "barrier", Mock())
    cfg = SkyRLTrainConfig().trainer
    cfg.strategy = "megatron"
    cfg.policy.model.lora = SkyRLLoraConfig(rank=8, alpha=8, train_mlp=False)
    cfg.policy.megatron_config.moe_grouped_gemm = False
    cfg.policy.megatron_config.hf_export_config.lora_export_format = "adapter"

    bridge, save_adapter, _ = _make_bridge_with_registry(_packed_expert_registry(sequential=True))
    worker = worker_mod.MegatronPolicyWorkerBase.__new__(worker_mod.MegatronPolicyWorkerBase)
    worker.cfg = cfg
    worker.megatron_config = cfg.policy.megatron_config
    worker.bridge = bridge
    worker.provider = SimpleNamespace(
        num_moe_experts=2,
        moe_grouped_gemm=False,
        vision_config=None,
        multi_latent_attention=False,
    )
    worker.is_vlm = False
    worker._rank = 0

    worker.configure_lora(cfg.policy.model.lora, lora_type="lora", experts_shared_outer_loras=False)

    owner = _make_strategy(is_lora=True, is_rank_0=True)
    owner.save_hf_model(
        bridge,
        SimpleNamespace(actor_module=["chunk0"]),
        str(tmp_path / "seq_packed_attn_only"),
        lora_export_format="adapter",
        peft_config=worker.lora_cls,
        moe_grouped_gemm=False,
    )
    save_adapter.assert_called_once()
