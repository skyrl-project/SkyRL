# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared pieces of the sharded-RDT backend: the op-chain allowlist, buffer
sizing and the minimum Ray version, which both sides derive from here so the
two cannot drift, plus ``RdtRouter`` — consumer-only, since routing is a
consumer decision.

The gather-group partition itself is ``base.layerwise_groups``: it defines what a
group index means for any ``WeightSource``, not just this transport.
"""

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import torch

# The op-chain contract, with two enforcers. The consumer's ``FakeRDTTensor``
# intercepts exactly these methods during the bake and records the string name;
# anything else reaches ``__torch_dispatch__`` and raises, so a loader needing
# real data fails at init rather than transferring wrong bytes. The producer
# replays a chain as ``getattr(tensor, op)(...)`` and refuses anything outside
# ``ALLOWED_OPS``, so a spoofed consumer cannot invoke arbitrary methods.
#
# Every entry must be a pure view. Both sets derive from this one table: written
# out separately they drifted, leaving ``t`` consumer-emittable but
# producer-rejected, and ``to`` allowed but unreachable -- ``to`` being exactly
# the dtype/device escape the bake exists to reject.
SUPPORTED_OPS: dict[Callable, str] = {
    torch.Tensor.narrow: "narrow",
    torch.Tensor.view: "view",
    torch.Tensor.reshape: "reshape",
    torch.Tensor.__getitem__: "__getitem__",
    torch.Tensor.unsqueeze: "unsqueeze",
    torch.Tensor.squeeze: "squeeze",
    torch.Tensor.transpose: "transpose",
    torch.Tensor.t: "t",
    torch.Tensor.permute: "permute",
    torch.Tensor.flatten: "flatten",
    torch.Tensor.contiguous: "contiguous",
    torch.Tensor.chunk: "chunk",
    # Multi-return like chunk: one child per output, trailing __getitem__(i).
    torch.Tensor.unbind: "unbind",
}

ALLOWED_OPS = frozenset(SUPPORTED_OPS.values())

# Ray Direct Transport landed in halves: the actor opt-in
# ``enable_tensor_transport`` in 2.49, and ``ray.experimental``'s
# ``register_nixl_memory`` / ``set_target_for_ref`` — which this transport calls
# directly — in 2.55. The floor is the tested version rather than the earliest
# one that could import; 2.55 has never been run against.
RDT_MIN_RAY_VERSION = "2.56.0"


def check_ray_rdt_version() -> None:
    """Refuse an installed Ray older than the one this backend is tested on.

    vLLM does not depend on Ray, so there is no pin to carry this. Without the
    check the failure is an opaque option-validation error out of
    ``.options(enable_tensor_transport=True)`` (below 2.49) or an ImportError
    raised deep in the first pull, long after init reported success (below 2.55).

    Raises:
        ValueError: the installed Ray predates ``RDT_MIN_RAY_VERSION``.
    """
    import importlib.metadata

    from packaging import version

    required = version.parse(RDT_MIN_RAY_VERSION)
    current = version.parse(importlib.metadata.version("ray"))
    if current < required:
        raise ValueError(
            f"The 'sharded_rdt' weight transfer backend requires Ray >= {required}, the version it "
            f"is tested against; Ray Direct Transport (ray.experimental.register_nixl_memory / "
            f"set_target_for_ref) needs at least 2.55. Found {current}. "
            f"Run `pip install -U 'ray>={required}'`."
        )


def assign_producer_indices(num_producers: int, num_consumers: int, consumer_idx: int) -> list[int]:
    """Producers (global indices) that consumer ``consumer_idx`` binds."""
    p = max(1, num_producers)
    c = max(1, num_consumers)
    if p >= c:
        return list(range(consumer_idx * p // c, (consumer_idx + 1) * p // c))
    return [consumer_idx * p // c]


class TrainerCopy(NamedTuple):
    """Where a trainer rank sits among the trainer's disjoint full model copies.

    Lanes (``plan_producer_lanes``) are built from these: a producer's lane is
    ``copy_rank % num_lanes``.
    """

    copy_rank: int
    """Which copy this rank belongs to, in ``[0, num_copies)``. Shared by every
    rank of that copy; together those ranks hold every name."""
    num_copies: int
    """Disjoint full copies across the trainer; the same on every rank. 1 means
    unreplicated."""


class LanePlan(NamedTuple):
    """The trainer's lane assignment, shipped to the consumers and producers.

    A lane pairs a subset of inference deployments with a subset of trainer
    copies: deployment ``d`` pulls only from producers in lane
    ``d % num_lanes``. Lanes spread the deployments' pulls over every copy
    instead of all reading one.
    """

    producer_lane: list[int]
    """Per producer rank, its lane; all 0 with one lane."""
    num_copies: int
    """Disjoint full model copies the trainer ranks reported."""
    lane_cap: int
    """Most lanes that keep ``workers_per_replica`` owners of every gather
    group; 1 with slot sharing off."""


def count_lanes(producer_lane: Sequence[int], num_consumers: int, workers_per_replica: int) -> int:
    """Lanes in use: one per producer lane, at most one per deployment. The
    router and the producer both derive the lane count here, from the same wire
    data, so their share groups cannot disagree."""
    consumers = max(1, num_consumers)
    deployments = consumers // min(consumers, max(1, workers_per_replica or consumers))
    return max(1, min(max(producer_lane, default=0) + 1, deployments))


def share_group_of(consumer_id: int, workers_per_replica: int, num_lanes: int) -> int:
    """The slot-sharing group of ``consumer_id``: its worker index within its
    lane (deployment ``d`` is in lane ``d % num_lanes``).

    Every consumer of one group routes every name to the same producer
    (``RdtRouter.producer_for``), so their pulls are byte-identical. With
    ``workers_per_replica <= 0`` nothing is shared: each consumer is its own
    group.
    """
    if workers_per_replica <= 0:
        return consumer_id
    deployment, worker = divmod(consumer_id, workers_per_replica)
    return worker + workers_per_replica * (deployment % max(1, num_lanes))


def _check_copies(
    names: Sequence[str],
    name_owner_class: Sequence[int],
    owner_sets: Sequence[Sequence[int]],
    rank_copies: Sequence[TrainerCopy],
) -> int:
    """The ``num_copies`` every rank reports, once the copies are checked to be
    disjoint and full: one count, ranks in range, every name held by every copy.

    Raises:
        ValueError: naming the disagreeing counts, the out-of-range copy rank,
            or the first name some copy does not hold.
    """
    counts = sorted({c.num_copies for c in rank_copies})
    if len(counts) > 1:
        by_count = {n: [p for p, c in enumerate(rank_copies) if c.num_copies == n] for n in counts}
        raise ValueError(
            "trainer ranks disagree on num_copies: "
            + "; ".join(f"{n} on ranks {ranks[:8]}" for n, ranks in by_count.items())
        )
    num_copies = counts[0]
    bad = [(p, c.copy_rank) for p, c in enumerate(rank_copies) if not 0 <= c.copy_rank < num_copies]
    if bad:
        raise ValueError(f"copy_rank out of range [0, {num_copies}) (rank, copy_rank): {bad[:8]}")
    copy_rank = [c.copy_rank for c in rank_copies]
    first_name: dict[int, str] = {}
    for n, oc in zip(names, name_owner_class):
        first_name.setdefault(oc, n)
    every = set(range(num_copies))
    for oc, row in enumerate(owner_sets):
        missing = sorted(every - {copy_rank[p] for p in row})
        if missing:
            raise ValueError(
                f"trainer copies {missing} of {num_copies} do not hold {first_name[oc]!r} "
                f"(owners {list(row)[:8]}, their copy ranks {sorted({copy_rank[p] for p in row})}). "
                "GroupedWeightSource.trainer_copy() must partition the ranks into full model copies."
            )
    return num_copies


def plan_producer_lanes(
    names: Sequence[str],
    name_owner_class: Sequence[int],
    owner_sets: Sequence[Sequence[int]],
    group_owners: Sequence[Sequence[int]],
    rank_copies: Sequence[TrainerCopy],
    workers_per_replica: int,
) -> LanePlan:
    """Assign each producer the lane its trainer copy serves.

    Deployments in lane ``k`` read only from producers in lane ``k``, so a name
    held once per trainer copy is served by every copy instead of one. The lane
    count is capped so each lane still holds at least ``workers_per_replica``
    owners of every group: a name every worker reads then keeps one producer
    per worker, so no producer gains serve rings. Copies fold onto lanes by
    ``copy_rank % num_lanes``, so every lane is a union of whole copies.

    Args:
        names: Every parameter name, in metadata order.
        name_owner_class: Per name, its index into ``owner_sets``.
        owner_sets: The distinct producer sets, as shipped to the consumers.
        group_owners: Per gather group, every producer holding any of its names.
        rank_copies: Per producer rank, its ``GroupedWeightSource.trainer_copy()``.
        workers_per_replica: Consumers per inference deployment; 0 when slot
            sharing is off, which leaves one lane.

    Raises:
        ValueError: the copies are not disjoint and full (``_check_copies``).
    """
    num_copies = _check_copies(names, name_owner_class, owner_sets, rank_copies)
    if workers_per_replica <= 0:
        cap = 1
    else:
        cap = min((len(g) // workers_per_replica for g in group_owners if g), default=num_copies)
    num_lanes = max(1, min(num_copies, cap))
    return LanePlan([c.copy_rank % num_lanes for c in rank_copies], num_copies, cap)


class RdtRouter:
    """Decides which producer serves each weight name, and — once bound — sends.

    Ownership is per NAME. ``owner_sets`` holds the few distinct producer sets
    that occur, and ``name_owner_class[i]`` indexes it for ``names[i]``; a name
    determines both its owner set and its gather group, so neither appears in the
    routing API. Empty tables mean every producer holds everything.

    That subsumes the coordinate schemes it replaced: pipeline-stage ownership is
    "the names of these groups have this owner set", expert parallelism is "these
    expert names have this one-rank owner set", and layouts that fit neither
    (a group produced by two stages) are expressible for free.

    Both engines derive the same tables from the same wire data, so they agree on
    who serves what. Disagreement is not a wrong answer but a hang or a loud
    misroute: a pull sent to a producer that never gathered the name trips its
    served-names guard.

    Routing is consumer-only. The trainer publishes what it holds and answers
    whatever arrives; it never asks who serves what.
    """

    def __init__(
        self,
        num_producers: int,
        num_consumers: int,
        owner_sets: list[list[int]] | None = None,
        name_owner_class: list[int] | None = None,
        names: list[str] | None = None,
        group_lens: list[int] | None = None,
        workers_per_replica: int = 0,
        producer_lane: list[int] | None = None,
    ) -> None:
        """Build the routing tables from the wire data both engines receive.

        The table arguments travel together and default together: with all of
        them empty the router degrades to "every producer holds everything", in
        one lane.

        Args:
            num_producers: Trainer ranks. Owner indices are positions in this
                range, and ``validate`` rejects any that fall outside it.
            num_consumers: Inference workers across the whole fleet. Fixes the
                id space; the block carve uses ``workers_per_replica`` of it.
            owner_sets: The distinct producer sets that occur, one row per
                owner class; each row is deduplicated and sorted here. Empty
                means a single class owning every producer.
            name_owner_class: Parallel to ``names`` — ``name_owner_class[i]``
                indexes ``owner_sets`` for ``names[i]``. A name with no entry
                falls back to class 0.
            names: Every parameter name, in group-major order: concatenating
                the gather groups reproduces this list exactly. The other two
                tables are keyed by this order.
            group_lens: Length of each gather group, consecutive over
                ``names``, so they sum to ``len(names)``. Fixes name -> group
                and the per-group owner union the free barrier fans out to.
            workers_per_replica: Consumers per inference DEPLOYMENT. The block
                carve spreads this many consumers over an owner set, and every
                deployment reuses the same carve — see ``producer_for``. 0 (the
                default) means one deployment, i.e. carve over the whole fleet.
            producer_lane: Per producer, its lane (``plan_producer_lanes``).
                Deployment ``d`` reads only from the owners in lane
                ``d % num_lanes``. None puts every producer in lane 0.

        Raises:
            ValueError: ``workers_per_replica`` does not divide
                ``num_consumers``, so the deployments cannot be uniform and the
                overlay would map two workers of one deployment onto the same
                block index; or ``producer_lane`` is not one entry per
                producer.
        """
        self.num_producers = max(1, num_producers)
        self.num_consumers = max(1, num_consumers)
        # [RDT-SHARE-SLOTS] Width of the block carve: one deployment's worth of
        # consumers, so deployments OVERLAY rather than spread (producer_for).
        self._block_consumers = min(self.num_consumers, max(1, workers_per_replica or self.num_consumers))
        if self.num_consumers % self._block_consumers:
            raise ValueError(
                f"workers_per_replica={workers_per_replica} does not divide num_consumers="
                f"{self.num_consumers}; the inference deployments must be uniform for the "
                "per-deployment block carve to be well defined."
            )
        self._owner_sets = [sorted(set(row)) for row in owner_sets] if owner_sets else [list(range(self.num_producers))]
        self._producer_lane = list(producer_lane) if producer_lane is not None else [0] * self.num_producers
        if len(self._producer_lane) != self.num_producers:
            raise ValueError(f"producer_lane has {len(self._producer_lane)} entries for {self.num_producers} producers")
        self.num_lanes = count_lanes(self._producer_lane, self.num_consumers, self._block_consumers)
        # owner class -> lane -> its owners in that lane. Out-of-range owners
        # are left to validate().
        lane_of = [lane % self.num_lanes for lane in self._producer_lane]
        self._lane_owners = [
            [[p for p in row if 0 <= p < self.num_producers and lane_of[p] == k] for k in range(self.num_lanes)]
            for row in self._owner_sets
        ]
        names = list(names or [])
        self.num_groups = len(group_lens) if group_lens else 0
        classes = list(name_owner_class or [])
        self._class_of = {n: (classes[i] if i < len(classes) else 0) for i, n in enumerate(names)}
        # name -> gather group, and the per-group owner union the free barrier
        # fans out to. Both are pure functions of the tables, so precompute once.
        self._group_of: dict[str, int] = {}
        self._group_owners: list[list[int]] = []
        pos = 0
        for gi, glen in enumerate(group_lens or []):
            union: set[int] = set()
            for n in names[pos : pos + glen]:
                self._group_of[n] = gi
                c = self._class_of.get(n, 0)
                # Bounds-checked: a bad class must surface from validate() as a
                # named ValueError, not as an IndexError from this precompute.
                if 0 <= c < len(self._owner_sets):
                    union.update(self._owner_sets[c])
            pos += glen
            self._group_owners.append(sorted(union))
        # Bound at init on the consumer; the rule methods never touch these.
        self._actors: list[Any] = []
        self._produce_methods: list[Any] = []
        self.consumer_id = 0

    # ---------------- rules (pure; safe to use unbound) ----------------

    def class_of(self, name: str) -> int:
        """The name's owner class — the planner's bucketing key, since all names
        of a chunk must share one producer."""
        return self._class_of.get(name, 0)

    def owners(self, name: str) -> list[int]:
        """Every producer holding ``name``."""
        return list(self._owner_sets[self.class_of(name)])

    def group_owners(self, group_idx: int) -> list[int]:
        """Every producer holding any name of ``group_idx``.

        The one group-keyed rule, because the free barrier is per group: a
        consumer signals ``free_group(gi)`` at each of these and the producer
        counts signals against the consumer total. Names route; groups free.
        """
        if not self._group_owners:
            return list(range(self.num_producers))
        return list(self._group_owners[group_idx])

    def producer_for(self, consumer_id: int, name: str) -> int:
        """The single producer ``consumer_id`` pulls ``name`` from.

        Blocks consumers across the name's owner set with the same rule that
        binds producers globally, then rotates by the name's GROUP index so a
        consumer spreads its groups over its block instead of hammering one NIC.
        Rotating per group (not per name) keeps every name of a chunk on one
        producer, which is what lets a chunk be a single pull.

        [RDT-SHARE-SLOTS] The carve is over ONE DEPLOYMENT (the consumer's index
        within its own deployment, over ``workers_per_replica`` of them), so
        worker w of every deployment resolves the SAME producer for every name.
        Carving over the whole fleet instead would spread the multi-owner
        (replicated) names of different deployments onto different producers,
        while the single-owner names still route by ownership -- so a producer
        would serve several distinct worker indices, each needing its own serve
        ring, and no two of them could share a slot (their plans differ). With
        the deployments overlaid, the consumers that meet on a producer are
        exactly the copies of one worker, whose pulls are byte-identical, which
        is what the producer's rendezvous collapses onto one slot.

        With one deployment the width IS the fleet, so this is the historical
        rule unchanged.

        Lanes are the other axis: deployment ``d`` carves over only the owners
        in lane ``d % num_lanes``, so the deployments' pulls of a name spread
        over the trainer copies holding it. Sharers (same ``share_group``) still
        resolve the same producer. One lane is the rule above.
        """
        if not self.owners(name):
            raise ValueError(f"{name!r} has no owner")
        deployment, worker = divmod(consumer_id, self._block_consumers)
        own = self._lane_owners[self.class_of(name)][deployment % self.num_lanes]
        block = assign_producer_indices(len(own), self._block_consumers, worker)
        return own[block[self._group_of.get(name, 0) % len(block)]]

    def share_group(self, consumer_id: int) -> int:
        """The slot-sharing group ``consumer_id`` routes as (``share_group_of``)."""
        return share_group_of(consumer_id, self._block_consumers, self.num_lanes)

    def validate(self) -> None:
        """Check the ownership tables can be served.

        Raises:
            ValueError: an owner set is empty or out of range, an owner class
                does not resolve, or a lane holds no owner of an owner set.
        """
        for c, row in enumerate(self._owner_sets):
            if not row:
                raise ValueError(f"owner set {c} is empty: no producer holds it")
            bad = [p for p in row if not 0 <= p < self.num_producers]
            if bad:
                raise ValueError(f"owner set {c} out of range: {bad}")
            empty = [k for k, own in enumerate(self._lane_owners[c]) if not own]
            if empty:
                raise ValueError(
                    f"lane(s) {empty} of {self.num_lanes} hold no owner of owner set {c} "
                    f"({row[:8]}): producer_lane does not partition the producers into full copies"
                )
        for name, c in self._class_of.items():
            if not 0 <= c < len(self._owner_sets):
                raise ValueError(
                    f"{name!r} has owner class {c}, but only " f"{len(self._owner_sets)} owner set(s) were shipped"
                )

    # ---------------- send surface (needs the binding) ----------------

    def bind(self, actors: list, produce_methods: list, consumer_id: int) -> None:
        """Attach this consumer's producer handles. Rebuilt wholesale per init,
        never appended to: every owner index is a position in these lists, so a
        rejoining engine that re-inits must not shift them."""
        if len(actors) != len(produce_methods):
            raise ValueError("actors and produce_methods must be parallel")
        self._actors = list(actors)
        self._produce_methods = list(produce_methods)
        self.consumer_id = consumer_id

    def _bound(self) -> None:
        if not self._produce_methods:
            raise RuntimeError("RdtRouter has no producers bound; call bind() first.")

    def pull(self, owner: int, keys: list, seq: int) -> Any:
        """Issue one packed pull to ``owner``.

        The consumer id keys the producer's serve ring; without it every worker is
        served out of ring 0 and concurrent pulls overwrite each other's blob.
        ``seq`` is this call's index in the stream to ``owner`` (see ``_Chunk``):
        it selects the serve slot in ISSUE order, so a slot is never repacked
        while the read of its previous contents is still in flight."""
        self._bound()
        return self._produce_methods[owner].remote(keys, consumer_id=self.consumer_id, seq=seq)

    def free_group(self, group_idx: int) -> list:
        """Signal the group done at every producer holding any of its names."""
        self._bound()
        return [self._actors[p].free_group.remote(group_idx) for p in self.group_owners(group_idx)]

    def reserve_serve_buffers(self, bytes_by_producer: list[int], plan_digests: list[str] | None = None) -> list:
        """Ask each producer to pre-register a serve ring sized to the most this
        consumer will pull from it.

        ``plan_digests[p]`` describes the chunks this consumer pulls from producer
        ``p``, in pull order. Producers that share one serve ring between the
        consumers of several inference deployments compare it across those
        consumers, so a fleet whose deployments are not actually identical fails
        at init instead of stalling mid-sync. Optional: a producer that does not
        share rings ignores it.
        """
        self._bound()
        digests: list = list(plan_digests) if plan_digests else [None] * len(bytes_by_producer)
        return [
            self._actors[p].reserve_serve_buffer.remote(self.consumer_id, nb, digests[p])
            for p, nb in enumerate(bytes_by_producer)
            if nb > 0
        ]


def buffer_alloc_bytes(nbytes: int, presize: int = 0) -> int:
    """Size a NIXL buffer / ring slot for ``nbytes``: the max of the request, an
    optional ``presize`` floor, and a coarse 256MB round-up, so the buffer is
    allocated ONCE and never regrows. Regrowth is a correctness hazard, not just a
    perf one -- see ``buffer_presize_gb``. Shared by the consumer's receive buffers
    and the producer's serve rings."""
    return max(nbytes, presize, -(-nbytes // (256 << 20)) * (256 << 20))
