"""CPU tests for the MTP heads' own learning rate (trainer.mtp.lr).

The heads get their own Megatron optimizer parameter group, matched by name, and Megatron's scheduler drives it
between ``mtp_lr`` and a proportionally scaled ``min_lr`` with the policy's warmup and decay. A ``skyrl_mtp_heads`` tag
on the heads' groups finds them for the ``mtp_lr`` metric.
"""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")

from megatron.core.optimizer import (  # noqa: E402
    OptimizerConfig,
    _get_param_groups,
    get_standard_config_overrides,
)
from megatron.core.optimizer_param_scheduler import (
    OptimizerParamScheduler,  # noqa: E402
)

from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (  # noqa: E402
    get_megatron_optimizer,
    mtp_heads_lr,
    mtp_lr_config_overrides,
)
from skyrl.train.config.config import (  # noqa: E402
    MegatronOptimizerOffloadToDiskConfig,
)

pytestmark = pytest.mark.megatron


@pytest.fixture(scope="module", autouse=True)
def _single_process_group():
    if not torch.distributed.is_initialized():
        store = tempfile.NamedTemporaryFile(delete=False)
        torch.distributed.init_process_group("gloo", init_method=f"file://{store.name}", rank=0, world_size=1)
        yield
        torch.distributed.destroy_process_group()
        if os.path.exists(store.name):  # gloo's FileStore may have removed it already
            os.unlink(store.name)
    else:
        yield


class _Toy(torch.nn.Module):
    """Megatron-style names: an embedding, decoder layers, the MTP block, and a 1-D norm in each."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Linear(4, 4, bias=False)
        self.embedding.weight.is_embedding_or_output_parameter = True  # Megatron's decoupled_lr target
        self.decoder = torch.nn.Module()
        self.decoder.proj = torch.nn.Linear(4, 4, bias=False)
        self.decoder.norm = torch.nn.LayerNorm(4, bias=False)
        self.mtp = torch.nn.Module()
        self.mtp.eh_proj = torch.nn.Linear(8, 4, bias=False)
        self.mtp.enorm = torch.nn.LayerNorm(4, bias=False)


def _groups(lr=1e-6, min_lr=5e-7, mtp_lr=3e-5, decoupled_lr=None):
    """Megatron's param groups for the toy model; ``mtp_lr=None`` builds the standard ones, without the heads' own."""
    config = OptimizerConfig(lr=lr, min_lr=min_lr, weight_decay=0.01, decoupled_lr=decoupled_lr)
    model = _Toy()
    overrides = get_standard_config_overrides(config) if mtp_lr is None else mtp_lr_config_overrides(config, mtp_lr)
    groups = _get_param_groups([model], config, overrides)
    by_param = {}
    for g in groups:
        for p in g["params"]:
            by_param[id(p)] = g
    return model, groups, by_param


def test_heads_get_their_own_lr_and_scaled_min_lr():
    model, _, by_param = _groups()
    for p in (model.mtp.eh_proj.weight, model.mtp.enorm.weight):
        assert by_param[id(p)]["max_lr"] == pytest.approx(3e-5)
        assert by_param[id(p)]["min_lr"] == pytest.approx(1.5e-5)  # the policy's min_lr / lr = 0.5
    for p in (model.decoder.proj.weight, model.decoder.norm.weight):
        assert by_param[id(p)]["max_lr"] == pytest.approx(1e-6)
        assert by_param[id(p)]["min_lr"] == pytest.approx(5e-7)
    # The standard overrides merge: 1-D params keep Megatron's zero weight decay, in the heads' group too.
    assert by_param[id(model.mtp.enorm.weight)]["wd_mult"] == 0.0
    assert by_param[id(model.mtp.eh_proj.weight)]["wd_mult"] == 1.0
    assert by_param[id(model.decoder.norm.weight)]["wd_mult"] == 0.0


def _tagged_param_ids(groups):
    return {id(p) for g in groups if g.get("skyrl_mtp_heads") for p in g["params"]}


def _mtp_param_ids(model):
    return {id(model.mtp.eh_proj.weight), id(model.mtp.enorm.weight)}


def test_megatron_keeps_the_heads_tag_on_their_groups_only():
    # _get_param_groups copies the override's keys into the param group, so the tag reaches the optimizer.
    model, groups, _ = _groups()
    assert _tagged_param_ids(groups) == _mtp_param_ids(model)


def _scheduler(optimizer, lr_decay_style="constant"):
    return OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=1e-6,
        min_lr=5e-7,
        lr_warmup_steps=2,
        lr_decay_steps=100,
        lr_decay_style=lr_decay_style,
        start_wd=0.01,
        end_wd=0.01,
        wd_incr_steps=100,
        wd_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=True,
    )


def test_scheduler_drives_the_heads_group_at_its_own_lr():
    _, groups, _ = _groups()
    sched = _scheduler(SimpleNamespace(param_groups=groups))
    sched.step(1)  # halfway through warmup: both groups at half their peak
    lrs = sorted({round(g["lr"], 12) for g in groups})
    assert lrs == [pytest.approx(5e-7), pytest.approx(1.5e-5)]
    sched.step(5)  # past warmup: each at its own peak
    lrs = sorted({round(g["lr"], 12) for g in groups})
    assert lrs == [pytest.approx(1e-6), pytest.approx(3e-5)]


def _scheduled_without_heads_group():
    """An optimizer with Megatron's standard param groups only, scheduled halfway through warmup."""
    _, groups, _ = _groups(mtp_lr=None)
    optimizer = SimpleNamespace(param_groups=groups)
    _scheduler(optimizer).step(1)
    return optimizer


def test_mtp_lr_metric_reads_the_heads_group():
    model, groups, by_param = _groups()
    optimizer = SimpleNamespace(param_groups=groups)
    _scheduler(optimizer).step(1)  # halfway through warmup
    assert mtp_heads_lr(optimizer, 3e-5) == by_param[id(model.mtp.eh_proj.weight)]["lr"] == pytest.approx(1.5e-5)
    # Without an own lr the heads sit in the policy's group.
    assert mtp_heads_lr(optimizer, None) == by_param[id(model.decoder.proj.weight)]["lr"] == pytest.approx(5e-7)
    # A missing heads' group fails rather than silently dropping the metric.
    with pytest.raises(AssertionError, match="no optimizer param group"):
        mtp_heads_lr(_scheduled_without_heads_group(), 3e-5)


def test_a_rank_without_heads_still_has_their_tagged_groups(monkeypatch):
    # A PP stage without the MTP layer: Megatron aligns the param groups across ranks by all-gathering their
    # override keys, so this rank gets the heads' groups too, empty but tagged and scheduled.
    config = OptimizerConfig(lr=1e-6, min_lr=5e-7, weight_decay=0.01)
    overrides = mtp_lr_config_overrides(config, 3e-5)
    heads_rank_keys = []

    def record_heads_rank(gathered, keys, group=None):
        heads_rank_keys.extend(keys)
        gathered[:] = [keys]

    def gather_with_heads_rank(gathered, keys, group=None):
        gathered[:] = [keys, heads_rank_keys]

    monkeypatch.setattr(torch.distributed, "all_gather_object", record_heads_rank)
    _get_param_groups([_Toy()], config, overrides)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_with_heads_rank)
    no_heads = _Toy()
    del no_heads.mtp
    groups = _get_param_groups([no_heads], config, overrides)
    tagged = [g for g in groups if g.get("skyrl_mtp_heads")]
    assert tagged and not any(g["params"] for g in tagged)
    optimizer = SimpleNamespace(param_groups=groups)
    _scheduler(optimizer).step(1)  # halfway through warmup
    assert mtp_heads_lr(optimizer, 3e-5) == pytest.approx(1.5e-5)


def test_mtp_lr_metric_needs_the_heads_group_at_the_policy_lr():
    # trainer.mtp.lr equal to the policy's lr: the policy's group, at the same max_lr, does not stand in for a
    # missing heads' group.
    with pytest.raises(AssertionError, match="no optimizer param group"):
        mtp_heads_lr(_scheduled_without_heads_group(), 1e-6)
    # The heads still get their own groups, at the policy's max_lr, and the metric reads them.
    model, groups, by_param = _groups(mtp_lr=1e-6)
    assert _tagged_param_ids(groups) == _mtp_param_ids(model)
    optimizer = SimpleNamespace(param_groups=groups)
    _scheduler(optimizer).step(1)
    assert mtp_heads_lr(optimizer, 1e-6) == by_param[id(model.mtp.eh_proj.weight)]["lr"]


def test_mtp_lr_metric_reads_the_heads_group_at_the_decoupled_lr():
    # trainer.mtp.lr equal to decoupled_lr: the embedding's group has the heads' max_lr but the policy's min_lr,
    # so the two groups decay apart.
    model, groups, by_param = _groups(mtp_lr=3e-5, decoupled_lr=3e-5)
    optimizer = SimpleNamespace(param_groups=groups)
    _scheduler(optimizer, lr_decay_style="linear").step(51)  # halfway through the decay
    heads, embedding = by_param[id(model.mtp.eh_proj.weight)], by_param[id(model.embedding.weight)]
    assert embedding["max_lr"] == heads["max_lr"] == pytest.approx(3e-5)
    assert heads["lr"] == pytest.approx(2.25e-5)  # halfway between 3e-5 and the scaled min_lr 1.5e-5
    assert embedding["lr"] == pytest.approx(1.525e-5)  # halfway between 3e-5 and the policy's min_lr 5e-7
    assert mtp_heads_lr(optimizer, 3e-5) == heads["lr"]


def test_disk_offloaded_optimizer_refuses_an_own_mtp_lr():
    with pytest.raises(ValueError, match="does not support trainer.mtp.lr"):
        get_megatron_optimizer(
            model=[],
            config=OptimizerConfig(lr=1e-6),
            offload_to_disk=MegatronOptimizerOffloadToDiskConfig(),
            mtp_lr=3e-5,
        )
