# SPDX-License-Identifier: Apache-2.0
"""Lanes: trainer copies as an axis of producer selection.

With slot sharing every deployment overlays one carve, so worker ``w`` of all
``R`` deployments pulls a name from the same producer. A name held once per
trainer copy (an EP rank's experts) is then served by one of its copies while
the others idle: on the 1024-GPU GLM-5.3 run (DP6 trainer, 80 DP8/EP8 engines)
only 64 of 384 producers served. Deployment ``d`` now reads from the producers
in lane ``d % num_lanes``.

CPU-only: ``RdtRouter`` and the planning helpers import nothing but torch.
"""

import math
from collections import Counter, defaultdict

import pytest

from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_common import (
    RdtRouter,
    TrainerCopy,
    assign_producer_indices,
    count_lanes,
    plan_producer_lanes,
    share_group_of,
)


class _Layout:
    """A trainer layout and the fleet's per-name needs, over few layers."""

    def __init__(self, num_producers, owner_sets, names, classes, group_lens, needs, copies):
        self.num_producers = num_producers
        self.owner_sets = owner_sets
        self.names = names
        self.classes = classes
        self.group_lens = group_lens
        self.needs = needs  # name -> worker indices that pull it
        self.copies = copies  # per producer, its GroupedWeightSource.trainer_copy()

    def group_owners(self):
        out, pos = [], 0
        for glen in self.group_lens:
            union = set()
            for c in self.classes[pos : pos + glen]:
                union.update(self.owner_sets[c])
            out.append(sorted(union))
            pos += glen
        return out

    def plan(self, wpr):
        return plan_producer_lanes(self.names, self.classes, self.owner_sets, self.group_owners(), self.copies, wpr)

    def router(self, num_consumers, wpr, producer_lane=None):
        return RdtRouter(
            self.num_producers,
            num_consumers,
            self.owner_sets,
            self.classes,
            self.names,
            self.group_lens,
            workers_per_replica=wpr,
            producer_lane=producer_lane,
        )


def _glm_layout(dp=6, pp=4, ep=16, layers_per_stage=2, wpr=8):
    """The GLM-5.3 1024-GPU trainer, TP1/CP16/PP4/EP16 at ``dp``, over few layers.

    Global rank ``stage * 16 * dp + 16 * dp_rank + cp_rank`` (Megatron order,
    PP outermost); with CP16 = EP16 the EP rank is the CP rank. Each layer group
    holds one replicated name and one expert block per EP rank. Worker ``w`` of
    an EP8 engine needs the dense name and the experts of EP ranks ``2w, 2w+1``.
    """
    per_stage = 16 * dp
    owner_sets, names, classes, group_lens, needs = [], [], [], [], {}
    layer = 0
    for stage in range(pp):
        base = stage * per_stage
        dense_cls = len(owner_sets)
        owner_sets.append(list(range(base, base + per_stage)))
        expert_cls = []
        for e in range(ep):
            expert_cls.append(len(owner_sets))
            owner_sets.append([base + e + 16 * d for d in range(dp)])
        for _ in range(layers_per_stage):
            n = f"model.layers.{layer}.attn.w"
            names.append(n)
            classes.append(dense_cls)
            needs[n] = set(range(wpr))
            for e in range(ep):
                n = f"model.layers.{layer}.mlp.experts.{e}.w"
                names.append(n)
                classes.append(expert_cls[e])
                needs[n] = {e * wpr // ep}
            group_lens.append(1 + ep)
            layer += 1
    copies = [TrainerCopy((r % per_stage) // 16, dp) for r in range(pp * per_stage)]
    return _Layout(pp * per_stage, owner_sets, names, classes, group_lens, needs, copies)


def _dense_layout(num_producers, dp, wpr=8, n_groups=4):
    """A dense trainer (every rank holds everything), ``num_producers / dp``
    TP x CP ranks per DP rank; every worker pulls every name."""
    names = [f"model.layers.{i}.w" for i in range(n_groups)]
    per_dp = num_producers // dp
    copies = [TrainerCopy(r // per_dp, dp) for r in range(num_producers)]
    return _Layout(
        num_producers,
        [list(range(num_producers))],
        names,
        [0] * n_groups,
        [1] * n_groups,
        {n: set(range(wpr)) for n in names},
        copies,
    )


def _serve_stats(router, layout, num_consumers, wpr):
    """Readers per (producer, name), and serve rings (share groups) per producer."""
    readers: Counter = Counter()
    rings: dict[int, set] = defaultdict(set)
    for c in range(num_consumers):
        for n in layout.names:
            if c % wpr in layout.needs[n]:
                p = router.producer_for(c, n)
                readers[(p, n)] += 1
                rings[p].add(router.share_group(c))
    return readers, rings


def _historical(router, consumer_id, name, wpr):
    """The routing rule before lanes."""
    own = router.owners(name)
    block = assign_producer_indices(len(own), wpr, consumer_id % wpr)
    return own[block[router._group_of[name] % len(block)]]


class TestHelpers:
    def test_one_lane_is_the_worker_index(self):
        assert [share_group_of(c, 8, 1) for c in range(32)] == [c % 8 for c in range(32)]

    def test_lanes_split_deployments_round_robin(self):
        assert [share_group_of(d * 8 + 3, 8, 3) for d in range(4)] == [3, 11, 19, 3]

    def test_sharing_off_is_identity(self):
        assert [share_group_of(c, 0, 6) for c in range(10)] == list(range(10))

    def test_lanes_cap_at_the_deployment_count(self):
        assert count_lanes([0, 1, 2, 3, 4, 5], 640, 8) == 6
        assert count_lanes([0, 1, 2, 3, 4, 5], 32, 8) == 4
        assert count_lanes([0, 1, 2, 3, 4, 5], 8, 8) == 1
        assert count_lanes([0, 1, 2, 3, 4, 5], 640, 0) == 1
        assert count_lanes([], 640, 8) == 1


class TestPlan:
    def test_glm_dp6_gets_six_lanes(self):
        layout = _glm_layout()
        plan = layout.plan(8)
        assert plan.producer_lane == [c.copy_rank for c in layout.copies]
        assert (plan.num_copies, plan.lane_cap) == (6, 12)

    def test_one_copy_is_one_lane(self):
        layout = _glm_layout(dp=1)
        assert layout.plan(8).producer_lane == [0] * layout.num_producers

    def test_sharing_off_is_one_lane(self):
        layout = _glm_layout()
        assert layout.plan(0) == ([0] * layout.num_producers, 6, 1)

    def test_the_cap_keeps_one_owner_per_worker(self):
        """Dense DP2 over 16 ranks: 8 workers fit two lanes of 8 owners; 16
        workers do not, so the plan keeps one lane rather than doubling rings."""
        layout = _dense_layout(16, dp=2)
        assert layout.plan(8).producer_lane == [r // 8 for r in range(16)]
        assert _dense_layout(16, dp=2, wpr=16).plan(16) == ([0] * 16, 2, 1)

    def test_the_cap_folds_copies_onto_fewer_lanes(self):
        # Dense DP4 over 16 ranks with 8 workers: 4 copies, but only 2 lanes
        # keep 8 owners each.
        plan = _dense_layout(16, dp=4).plan(8)
        assert plan.producer_lane == [(r // 4) % 2 for r in range(16)]
        assert (plan.num_copies, plan.lane_cap) == (4, 2)


class TestPlanRejectsBadCopies:
    """Trainer copies that are not disjoint and full raise at ownership
    resolution, with or without slot sharing."""

    @pytest.mark.parametrize("wpr", [8, 0])
    def test_a_copy_missing_a_name(self, wpr):
        layout = _glm_layout(dp=2, pp=1, layers_per_stage=1)
        # Expert 3 is held only by copy 0.
        layout.owner_sets[1 + 3] = [3]
        with pytest.raises(ValueError, match=r"trainer copies \[1\] of 2 do not hold 'model.layers.0.mlp.experts.3.w'"):
            layout.plan(wpr)

    @pytest.mark.parametrize("wpr", [8, 0])
    def test_disagreeing_num_copies(self, wpr):
        layout = _glm_layout(dp=2, pp=1, layers_per_stage=1)
        layout.copies[0] = TrainerCopy(0, 4)
        with pytest.raises(ValueError, match=r"disagree on num_copies: 2 on ranks \[1, 2,.*; 4 on ranks \[0\]"):
            layout.plan(wpr)

    def test_an_out_of_range_copy_rank(self):
        layout = _glm_layout(dp=2, pp=1, layers_per_stage=1)
        layout.copies[5] = TrainerCopy(2, 2)
        with pytest.raises(ValueError, match=r"copy_rank out of range \[0, 2\) \(rank, copy_rank\): \[\(5, 2\)\]"):
            layout.plan(8)


class TestParity:
    """With one lane no pull moves."""

    @pytest.mark.parametrize("dp", [1, 2])
    def test_no_table_and_one_lane_are_the_historical_routing(self, dp):
        layout = _glm_layout(dp=dp, layers_per_stage=1)
        wpr, n_cons = 8, 80
        none = layout.router(n_cons, wpr)
        zeros = layout.router(n_cons, wpr, [0] * layout.num_producers)
        assert none.num_lanes == zeros.num_lanes == 1
        for c in range(n_cons):
            for n in layout.names:
                want = _historical(none, c, n, wpr)
                assert none.producer_for(c, n) == zeros.producer_for(c, n) == want
                assert none.share_group(c) == c % wpr

    def test_one_deployment_ignores_the_table(self):
        layout = _glm_layout(dp=6, layers_per_stage=1)
        plan = layout.plan(8)
        one = layout.router(8, 8, plan.producer_lane)
        assert one.num_lanes == 1
        for c in range(8):
            for n in layout.names:
                assert one.producer_for(c, n) == _historical(one, c, n, 8)


class TestGlm1024:
    """The regression: 80 DP8/EP8 engines over the DP6 trainer."""

    def test_every_producer_serves_and_fan_in_drops(self):
        layout = _glm_layout(dp=6)
        wpr, n_cons = 8, 640
        old = layout.router(n_cons, wpr)
        old_readers, _ = _serve_stats(old, layout, n_cons, wpr)
        experts = [n for n in layout.names if ".experts." in n]
        assert max(old_readers[k] for k in old_readers if k[1] in experts) == 80
        assert len({p for p, _ in old_readers}) < layout.num_producers

        new = layout.router(n_cons, wpr, layout.plan(wpr).producer_lane)
        assert new.num_lanes == 6
        readers, rings = _serve_stats(new, layout, n_cons, wpr)
        assert len({p for p, _ in readers}) == layout.num_producers == 384
        assert max(readers.values()) <= math.ceil(80 / 6)
        assert max(len(r) for r in rings.values()) == 1
        for n in experts:
            assert {p for p, n2 in readers if n2 == n} == set(new.owners(n))


class TestDenseNoRingGrowth:
    @pytest.mark.parametrize("num_producers,dp,wpr", [(16, 1, 8), (8, 1, 8), (16, 2, 8), (16, 4, 8), (16, 2, 16)])
    def test_rings_never_grow(self, num_producers, dp, wpr):
        layout = _dense_layout(num_producers, dp, wpr=wpr)
        n_cons = wpr * 16
        _, old_rings = _serve_stats(layout.router(n_cons, wpr), layout, n_cons, wpr)
        new = layout.router(n_cons, wpr, layout.plan(wpr).producer_lane)
        _, rings = _serve_stats(new, layout, n_cons, wpr)
        assert max(len(r) for r in rings.values()) <= max(len(r) for r in old_rings.values())
        if dp == 1:
            assert new.num_lanes == 1


class TestUnevenDeployments:
    @pytest.mark.parametrize("deployments", [7, 4])
    def test_routes_stay_owned_and_consistent(self, deployments):
        layout = _glm_layout(dp=6, layers_per_stage=1)
        wpr = 8
        n_cons = wpr * deployments
        _, old_rings = _serve_stats(layout.router(n_cons, wpr), layout, n_cons, wpr)
        r = layout.router(n_cons, wpr, layout.plan(wpr).producer_lane)
        assert r.num_lanes == min(6, deployments)
        readers, rings = _serve_stats(r, layout, n_cons, wpr)
        assert all(p in r.owners(n) for p, n in readers)
        assert max(len(g) for g in rings.values()) <= max(len(g) for g in old_rings.values())
        assert max(readers.values()) <= math.ceil(deployments / r.num_lanes)


class TestSharersRouteIdentically:
    """The slot-sharing contract: every consumer of one share group pulls every
    name from the same producer, so their plans (and digests) agree."""

    @pytest.mark.parametrize(
        "layout,deployments",
        [
            (_glm_layout(dp=6, layers_per_stage=1), 80),
            (_glm_layout(dp=6, layers_per_stage=1), 7),
            (_glm_layout(dp=6, layers_per_stage=1), 4),
            (_glm_layout(dp=2, layers_per_stage=1), 8),
            (_dense_layout(16, dp=4), 10),
        ],
    )
    def test_same_group_same_producer(self, layout, deployments):
        wpr = 8
        n_cons = wpr * deployments
        r = layout.router(n_cons, wpr, layout.plan(wpr).producer_lane)
        by_group: dict[int, list[int]] = defaultdict(list)
        for c in range(n_cons):
            by_group[r.share_group(c)].append(c)
        assert len(by_group) == wpr * r.num_lanes
        for members in by_group.values():
            for n in layout.names:
                assert len({r.producer_for(c, n) for c in members}) == 1, (n, members)


class TestRouterTable:
    def test_wrong_length_table_is_rejected(self):
        with pytest.raises(ValueError, match="producer_lane has 3 entries for 4 producers"):
            RdtRouter(4, 16, None, None, ["a"], [1], workers_per_replica=8, producer_lane=[0, 1, 0]).validate()

    def test_a_lane_without_owners_is_rejected(self):
        # Name "b" is held only by producer 0, so lane 1 cannot serve it.
        r = RdtRouter(
            4, 16, [[0, 1, 2, 3], [0]], [0, 1], ["a", "b"], [2], workers_per_replica=8, producer_lane=[0, 0, 1, 1]
        )
        with pytest.raises(ValueError, match=r"lane\(s\) \[1\] of 2 hold no owner of owner set 1"):
            r.validate()
