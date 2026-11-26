"""Tests for distributed barrier recovery during LoRA / weight sync failures.

Covers issue #1827: when rank 0 (or a writer rank) raises inside
``_save_lora_adapters_and_sync``, ``_publish_lora_adapter_in_memory``, or
prefix-cache reset before a ``torch.distributed.barrier()``, peer ranks must
not hang until Gloo times out and poisons the process group for subsequent
multi-tenant ``save_weights_for_sampler`` requests.
"""

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    InferenceServerTimeoutError,
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase
from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase
from skyrl.backends.skyrl_train.workers.worker_utils import (
    PeerRankSyncError,
    barrier_with_error_propagation,
    cleanup_unawaited_task,
)


@pytest.fixture(autouse=True)
def _isolate_control_plane_registry():
    """Keep process-local _ACTIVE_CLIENTS isolated between tests."""
    from skyrl.backends.skyrl_train.weight_sync import control_plane

    with control_plane._CONTROL_PLANE_LOCK:
        control_plane._ACTIVE_CLIENTS.clear()
    yield
    with control_plane._CONTROL_PLANE_LOCK:
        control_plane._ACTIVE_CLIENTS.clear()


class _FakeTaskType(Enum):
    CAUSAL_LM = "CAUSAL_LM"


class _FakePeftType(Enum):
    LORA = "LORA"


@dataclass
class _FakePeftConfig:
    r: int
    lora_alpha: int
    target_modules: tuple
    task_type: _FakeTaskType
    peft_type: _FakePeftType


def _init_gloo_group(rank: int, world_size: int, store_path: str, timeout_s: float = 5.0) -> None:
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_s),
    )


def _run_mp(fn, world_size: int, tmp_path: Path, *extra_args) -> None:
    store_path = str(tmp_path / "dist_store")
    mp.spawn(fn, args=(world_size, store_path, *extra_args), nprocs=world_size, join=True)


# ---------------------------------------------------------------------------
# Multi-rank Gloo tests: barrier_with_error_propagation
# ---------------------------------------------------------------------------


def _worker_barrier_rank0_failure_and_recovery(rank: int, world_size: int, store_path: str) -> None:
    _init_gloo_group(rank, world_size, store_path)
    try:
        # Round 1: rank 0 fails with a connection error before the barrier.
        local_error = RuntimeError("ClientConnectorError: Cannot connect to vLLM server") if rank == 0 else None
        if rank == 0:
            with pytest.raises(RuntimeError, match="Cannot connect to vLLM server"):
                barrier_with_error_propagation(local_error, stage="LoRA adapter load")
        else:
            with pytest.raises(PeerRankSyncError, match=r"Rank 0 failed during LoRA adapter load"):
                barrier_with_error_propagation(local_error, stage="LoRA adapter load")

        # Rounds 2 and 3: process group must still be healthy (no pair closure / timeout).
        barrier_with_error_propagation(None, stage="LoRA adapter load")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_barrier_with_error_propagation_rank0_failure_and_recovery(tmp_path):
    _run_mp(_worker_barrier_rank0_failure_and_recovery, 4, tmp_path)


def _worker_barrier_nonzero_rank_failure(rank: int, world_size: int, store_path: str) -> None:
    _init_gloo_group(rank, world_size, store_path)
    try:
        # Simulate a non-zero writer rank (e.g. second node leader) hitting disk full.
        local_error = OSError("No space left on device") if rank == 2 else None
        if rank == 2:
            with pytest.raises(OSError, match="No space left on device"):
                barrier_with_error_propagation(local_error, stage="LoRA adapter export")
        else:
            with pytest.raises(PeerRankSyncError, match=r"Rank 2 failed during LoRA adapter export"):
                barrier_with_error_propagation(local_error, stage="LoRA adapter export")

        # Subsequent sync succeeds across all 4 ranks.
        barrier_with_error_propagation(None, stage="LoRA adapter export")
    finally:
        dist.destroy_process_group()


def test_barrier_with_error_propagation_nonzero_rank_failure(tmp_path):
    _run_mp(_worker_barrier_nonzero_rank_failure, 4, tmp_path)


# ---------------------------------------------------------------------------
# Multi-rank Gloo tests: FSDP LoRA sync + prefix cache reset recovery
# ---------------------------------------------------------------------------


def _make_fsdp_lora_worker(base_sync_path: str) -> FSDPPolicyWorkerBase:
    peft_cfg = _FakePeftConfig(
        r=4,
        lora_alpha=8,
        target_modules=("q_proj",),
        task_type=_FakeTaskType.CAUSAL_LM,
        peft_type=_FakePeftType.LORA,
    )
    peft_model = SimpleNamespace(peft_config={"default": peft_cfg})
    worker = FSDPPolicyWorkerBase.__new__(FSDPPolicyWorkerBase)
    worker._is_lora = True
    worker._is_multimodal_lm_only = False
    worker._weight_sync_inference_client = None
    worker.model = SimpleNamespace(model=peft_model)
    worker.cfg = SimpleNamespace(
        fully_async=SimpleNamespace(enabled=False, clear_kv_cache_on_weight_sync=False),
        placement=SimpleNamespace(colocate_all=False),
        policy=SimpleNamespace(
            model=SimpleNamespace(lora=SimpleNamespace(lora_sync_path=base_sync_path)),
        ),
    )
    return worker


def _worker_fsdp_lora_load_failure_recovers(rank: int, world_size: int, store_path: str, sync_dir: str) -> None:
    from skyrl.backends.skyrl_train.distributed import fsdp_utils

    _init_gloo_group(rank, world_size, store_path)
    orig_collect = fsdp_utils.collect_lora_params
    orig_empty_cache = torch.cuda.empty_cache
    try:
        fsdp_utils.collect_lora_params = lambda module: {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 4),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(4, 2),
        }
        torch.cuda.empty_cache = lambda: None

        worker = _make_fsdp_lora_worker(sync_dir)
        ie_cfg = SimpleNamespace(enable_prefix_caching=True)

        loaded = []
        resets = []

        async def _load_lora(name: str, path: str):
            if name == "tenant-a":
                raise RuntimeError("Failed to load LoRA adapter 'tenant-a': ClientConnectorError")
            loaded.append((name, path))

        async def _reset_cache(reset_running_requests: bool = True):
            resets.append(reset_running_requests)

        client = Mock(spec=RemoteInferenceClient)
        client.load_lora_adapter = AsyncMock(side_effect=_load_lora)
        client.reset_prefix_cache = AsyncMock(side_effect=_reset_cache)

        async def _run():
            # Request 1 fails on rank 0 during load_lora_adapter; rank 1 must exit cleanly.
            if rank == 0:
                with pytest.raises(RuntimeError, match="ClientConnectorError"):
                    await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-a")
            else:
                await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-a")

            # Requests 2 & 3 for other tenants must succeed without barrier timeouts.
            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-b")
            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-c")

        asyncio.run(_run())

        if rank == 0:
            assert [name for name, _ in loaded] == ["tenant-b", "tenant-c"]
            assert len(resets) == 2
            assert (Path(sync_dir) / "tenant-b" / "adapter_model.safetensors").exists()
            assert (Path(sync_dir) / "tenant-c" / "adapter_model.safetensors").exists()
        else:
            assert loaded == []
            assert resets == []
    finally:
        fsdp_utils.collect_lora_params = orig_collect
        torch.cuda.empty_cache = orig_empty_cache
        dist.destroy_process_group()


def test_fsdp_lora_sync_load_failure_recovers_for_subsequent_tenants(tmp_path):
    _run_mp(_worker_fsdp_lora_load_failure_recovers, 2, tmp_path, str(tmp_path / "lora_sync"))


def _worker_fsdp_prefix_cache_reset_failure_recovers(
    rank: int, world_size: int, store_path: str, sync_dir: str
) -> None:
    from skyrl.backends.skyrl_train.distributed import fsdp_utils

    _init_gloo_group(rank, world_size, store_path)
    orig_collect = fsdp_utils.collect_lora_params
    orig_empty_cache = torch.cuda.empty_cache
    try:
        fsdp_utils.collect_lora_params = lambda module: {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 4),
        }
        torch.cuda.empty_cache = lambda: None

        worker = _make_fsdp_lora_worker(sync_dir)
        ie_cfg = SimpleNamespace(enable_prefix_caching=True)

        reset_attempts = 0

        async def _reset_cache(reset_running_requests: bool = True):
            nonlocal reset_attempts
            reset_attempts += 1
            if reset_attempts == 1:
                raise RuntimeError("Prefix cache reset failed on http://vllm:8001")

        client = Mock(spec=RemoteInferenceClient)
        client.load_lora_adapter = AsyncMock()
        client.reset_prefix_cache = AsyncMock(side_effect=_reset_cache)

        async def _run():
            if rank == 0:
                with pytest.raises(RuntimeError, match="Prefix cache reset failed"):
                    await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-a")
            else:
                await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-a")

            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-b")

        asyncio.run(_run())
        if rank == 0:
            assert reset_attempts == 2
    finally:
        fsdp_utils.collect_lora_params = orig_collect
        torch.cuda.empty_cache = orig_empty_cache
        dist.destroy_process_group()


def test_fsdp_lora_sync_prefix_cache_reset_failure_recovers(tmp_path):
    _run_mp(_worker_fsdp_prefix_cache_reset_failure_recovers, 2, tmp_path, str(tmp_path / "lora_sync"))


# ---------------------------------------------------------------------------
# Multi-rank Gloo tests: Worker._sync_weights_to_inference_engines
# ---------------------------------------------------------------------------


def _worker_full_weight_sync_prefix_cache_failure_recovers(rank: int, world_size: int, store_path: str) -> None:
    _init_gloo_group(rank, world_size, store_path)
    orig_empty_cache = torch.cuda.empty_cache
    try:
        torch.cuda.empty_cache = lambda: None

        class _Engine:
            skyrl_handles_prefix_cache_reset = False
            skyrl_force_disable_expandable_segments = False
            skyrl_empty_cache_after_send = False

            def __init__(self):
                self.sends = 0

            def send_weights(self):
                self.sends += 1

        worker = PolicyWorkerBase.__new__(PolicyWorkerBase)
        worker._weight_sync_engine = _Engine()
        worker.cfg = SimpleNamespace(
            fully_async=SimpleNamespace(enabled=False, clear_kv_cache_on_weight_sync=False),
            placement=SimpleNamespace(colocate_all=False),
        )

        @contextmanager
        def _noop_ctx(*args, **kwargs):
            yield

        worker._expandable_segments_disabled_for_sync = _noop_ctx

        reset_attempts = 0

        async def _reset_cache(reset_running_requests: bool = True):
            nonlocal reset_attempts
            reset_attempts += 1
            if reset_attempts == 1:
                raise RuntimeError("vLLM unreachable during reset_prefix_cache")

        client = Mock()
        client.reset_prefix_cache = AsyncMock(side_effect=_reset_cache)
        ie_cfg = SimpleNamespace(enable_prefix_caching=True)

        async def _run():
            if rank == 0:
                with pytest.raises(RuntimeError, match="vLLM unreachable"):
                    await worker._sync_weights_to_inference_engines(client, ie_cfg)
            else:
                await worker._sync_weights_to_inference_engines(client, ie_cfg)

            await worker._sync_weights_to_inference_engines(client, ie_cfg)

        asyncio.run(_run())
        assert worker._weight_sync_engine.sends == 2
    finally:
        torch.cuda.empty_cache = orig_empty_cache
        dist.destroy_process_group()


@pytest.mark.vllm
def test_sync_weights_prefix_cache_reset_failure_recovers(tmp_path):
    _run_mp(_worker_full_weight_sync_prefix_cache_failure_recovers, 2, tmp_path)


# ---------------------------------------------------------------------------
# Multi-rank Gloo tests: Megatron LoRA disk & in-memory sync recovery
# ---------------------------------------------------------------------------


def _make_megatron_lora_worker(base_sync_path: str, *, sync_mode: str = "disk"):
    from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
        MegatronPolicyWorkerBase,
    )

    worker = MegatronPolicyWorkerBase.__new__(MegatronPolicyWorkerBase)
    worker._is_lora = True
    worker._weight_sync_inference_client = None
    worker._weight_sync_engine = None
    worker._lora_weight_source = None
    worker._logical_model_path = "/models/base"
    worker.actor_module = []
    worker.provider = SimpleNamespace(num_moe_experts=None)
    worker.lora_cls = SimpleNamespace(dim=4, alpha=8, target_modules=["q_proj"])
    worker.bridge = SimpleNamespace(
        hf_pretrained=SimpleNamespace(model_name_or_path="/models/base"),
        export_adapter_weights=lambda actor_module, cpu=True, show_progress=False: [
            ("layers.0.self_attn.q_proj.lora_A.weight", torch.ones(4, 8, dtype=torch.bfloat16)),
            ("layers.0.self_attn.q_proj.lora_B.weight", torch.ones(8, 4, dtype=torch.bfloat16)),
        ],
        _model_bridge=SimpleNamespace(_adapter_target_modules=lambda names: ["q_proj"]),
    )
    worker.cfg = SimpleNamespace(
        fully_async=SimpleNamespace(enabled=False, clear_kv_cache_on_weight_sync=False),
        placement=SimpleNamespace(colocate_all=False),
        policy=SimpleNamespace(
            model=SimpleNamespace(lora=SimpleNamespace(lora_sync_path=base_sync_path, sync_mode=sync_mode)),
            megatron_config=SimpleNamespace(lora_config=SimpleNamespace(merge_lora=False)),
        ),
    )

    @contextmanager
    def _noop_ctx(*args, **kwargs):
        yield

    worker._expandable_segments_disabled_for_sync = _noop_ctx
    return worker


def _worker_megatron_disk_load_failure_recovers(rank: int, world_size: int, store_path: str, sync_dir: str) -> None:
    import megatron.bridge.models.conversion.peft_bridge as peft_bridge

    _init_gloo_group(rank, world_size, store_path)
    orig_empty_cache = torch.cuda.empty_cache
    orig_build_cfg = peft_bridge.build_adapter_config_dict
    try:
        torch.cuda.empty_cache = lambda: None
        peft_bridge.build_adapter_config_dict = lambda lora_cls, target_modules, base_model_name_or_path: {
            "r": lora_cls.dim,
            "lora_alpha": lora_cls.alpha,
            "target_modules": list(target_modules),
            "base_model_name_or_path": base_model_name_or_path,
        }

        worker = _make_megatron_lora_worker(sync_dir, sync_mode="disk")
        ie_cfg = SimpleNamespace(enable_prefix_caching=True)

        loaded = []
        resets = []

        async def _load_lora(name: str, path: str):
            if name == "model_1":
                raise RuntimeError("Failed to load LoRA adapter 'model_1' on 1/4 server(s): ClientConnectorError")
            loaded.append((name, path))

        async def _reset_cache(reset_running_requests: bool = True):
            resets.append(reset_running_requests)

        client = Mock(spec=RemoteInferenceClient)
        client.load_lora_adapter = AsyncMock(side_effect=_load_lora)
        client.reset_prefix_cache = AsyncMock(side_effect=_reset_cache)

        async def _run():
            # Request 1 fails during rank-0 load_lora_adapter.
            if rank == 0:
                with pytest.raises(RuntimeError, match="ClientConnectorError"):
                    await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_1")
            else:
                await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_1")

            # Subsequent multi-tenant requests must succeed without barrier timeouts.
            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_2")
            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_3")

        asyncio.run(_run())

        if rank == 0:
            assert [name for name, _ in loaded] == ["model_2", "model_3"]
            assert len(resets) == 2
        else:
            assert loaded == []
            assert resets == []
    finally:
        torch.cuda.empty_cache = orig_empty_cache
        peft_bridge.build_adapter_config_dict = orig_build_cfg
        dist.destroy_process_group()


@pytest.mark.megatron
def test_megatron_lora_disk_sync_load_failure_recovers_multi_tenant(tmp_path):
    _run_mp(_worker_megatron_disk_load_failure_recovers, 2, tmp_path, str(tmp_path / "lora_sync"))


def _worker_megatron_disk_writer_failure_skips_load(rank: int, world_size: int, store_path: str, sync_dir: str) -> None:
    import megatron.bridge.models.conversion.peft_bridge as peft_bridge

    _init_gloo_group(rank, world_size, store_path)
    orig_empty_cache = torch.cuda.empty_cache
    orig_build_cfg = peft_bridge.build_adapter_config_dict
    try:
        torch.cuda.empty_cache = lambda: None
        fail_first = {"flag": True}

        def _build_cfg(lora_cls, target_modules, base_model_name_or_path):
            if fail_first["flag"]:
                fail_first["flag"] = False
                raise OSError("disk full while exporting adapter")
            return {
                "r": lora_cls.dim,
                "lora_alpha": lora_cls.alpha,
                "target_modules": list(target_modules),
                "base_model_name_or_path": base_model_name_or_path,
            }

        peft_bridge.build_adapter_config_dict = _build_cfg

        worker = _make_megatron_lora_worker(sync_dir, sync_mode="disk")
        ie_cfg = SimpleNamespace(enable_prefix_caching=False)

        client = Mock(spec=RemoteInferenceClient)
        client.load_lora_adapter = AsyncMock()

        async def _run():
            if rank == 0:
                with pytest.raises(OSError, match="disk full"):
                    await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_1")
                client.load_lora_adapter.assert_not_awaited()
            else:
                await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_1")

            # Next request succeeds on both ranks.
            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_2")
            if rank == 0:
                client.load_lora_adapter.assert_awaited_once()

        asyncio.run(_run())
    finally:
        torch.cuda.empty_cache = orig_empty_cache
        peft_bridge.build_adapter_config_dict = orig_build_cfg
        dist.destroy_process_group()


@pytest.mark.megatron
def test_megatron_lora_disk_sync_writer_failure_skips_load_and_recovers(tmp_path):
    _run_mp(_worker_megatron_disk_writer_failure_skips_load, 2, tmp_path, str(tmp_path / "lora_sync"))


def _worker_megatron_memory_sync_failures_recover(rank: int, world_size: int, store_path: str, sync_dir: str) -> None:
    from skyrl.backends.skyrl_train.weight_sync.sources import LoraAdapterWeightSource

    _init_gloo_group(rank, world_size, store_path)
    orig_empty_cache = torch.cuda.empty_cache
    try:
        torch.cuda.empty_cache = lambda: None

        class _FakeLoraSource(LoraAdapterWeightSource):
            def __init__(self):
                super().__init__(dtype=torch.bfloat16, experts_per_shared_adapter=1)
                self.prepare_calls = []

            def export_adapter_stream(self):
                self.prepare_calls.append(self._lora_name)
                if self._lora_name == "tenant-prep-fail" and rank == 1:
                    raise RuntimeError("rank-1 prepare failure")
                return [("base_model.model.q_proj.lora_A.weight", torch.ones(2, 4, dtype=torch.bfloat16))]

        class _FakeEngine:
            skyrl_force_disable_expandable_segments = False

            def __init__(self, source):
                self.source = source
                self.sends = []

            def send_weights(self):
                self.sends.append(self.source._lora_name)
                list(self.source.iter_full())

        worker = _make_megatron_lora_worker(sync_dir, sync_mode="memory")
        source = _FakeLoraSource()
        engine = _FakeEngine(source)
        worker._lora_weight_source = source
        worker._weight_sync_engine = engine
        ie_cfg = SimpleNamespace(enable_prefix_caching=False)

        async def _set_target(target):
            if target["lora_name"] == "tenant-a":
                raise RuntimeError("Cannot connect to server during set_lora_receive_target")

        async def _load_lora(lora_name: str, in_memory: bool = False):
            assert in_memory is True
            if lora_name == "tenant-b":
                raise RuntimeError("Cannot connect to server during in-memory load_lora_adapter")

        client = Mock(spec=RemoteInferenceClient)
        client.set_lora_receive_target = AsyncMock(side_effect=_set_target)
        client.load_lora_adapter = AsyncMock(side_effect=_load_lora)

        async def _run():
            # Request 0: rank-1 fails during source.prepare(); rank 0 must NOT arm receivers.
            if rank == 1:
                with pytest.raises(RuntimeError, match="rank-1 prepare failure"):
                    await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-prep-fail")
            else:
                await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-prep-fail")
                client.set_lora_receive_target.assert_not_awaited()
            assert engine.sends == []
            assert source._prepared is None

            # Request 1: set_lora_receive_target fails on rank 0.
            # Neither rank may enter send_weights(), and prepared state must be discarded.
            if rank == 0:
                with pytest.raises(RuntimeError, match="set_lora_receive_target"):
                    await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-a")
            else:
                await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-a")
            assert engine.sends == []
            assert source._prepared is None

            # Request 2: send_weights succeeds, then load_lora_adapter fails on rank 0.
            if rank == 0:
                with pytest.raises(RuntimeError, match="in-memory load_lora_adapter"):
                    await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-b")
            else:
                await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-b")
            assert engine.sends == ["tenant-b"]

            # Request 3: full in-memory sync succeeds with fresh export for tenant-c.
            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-c")
            assert engine.sends == ["tenant-b", "tenant-c"]
            assert source.prepare_calls == ["tenant-prep-fail", "tenant-a", "tenant-b", "tenant-c"]

        asyncio.run(_run())
    finally:
        torch.cuda.empty_cache = orig_empty_cache
        dist.destroy_process_group()


@pytest.mark.megatron
def test_megatron_lora_memory_sync_arm_and_load_failures_recover(tmp_path):
    _run_mp(_worker_megatron_memory_sync_failures_recover, 2, tmp_path, str(tmp_path / "lora_sync"))


def _worker_megatron_probe_failure_does_not_strand_peers(
    rank: int, world_size: int, store_path: str, sync_dir: str
) -> None:
    _init_gloo_group(rank, world_size, store_path)
    try:
        # Use a regular file path as base_sync_path so rank 0's os.makedirs fails with FileExistsError/NotADirectoryError.
        bad_path = str(Path(sync_dir) / "not_a_dir")
        if rank == 0:
            Path(sync_dir).mkdir(parents=True, exist_ok=True)
            Path(bad_path).write_text("blocker", encoding="utf-8")
        dist.barrier()

        worker = _make_megatron_lora_worker(bad_path, sync_mode="disk")
        if rank == 0:
            with pytest.raises(OSError):
                worker._is_lora_sync_writer_rank()
        else:
            with pytest.raises(PeerRankSyncError, match="writer probe"):
                worker._is_lora_sync_writer_rank()

        # Once pointed at a valid directory, probe succeeds on both ranks.
        worker.cfg.policy.model.lora.lora_sync_path = str(Path(sync_dir) / "valid_dir")
        is_writer = worker._is_lora_sync_writer_rank()
        assert is_writer == (rank == 0)
    finally:
        dist.destroy_process_group()


@pytest.mark.megatron
def test_megatron_writer_rank_probe_failure_does_not_strand_peers(tmp_path):
    _run_mp(_worker_megatron_probe_failure_does_not_strand_peers, 2, tmp_path, str(tmp_path / "probe"))


# ---------------------------------------------------------------------------
# Fast in-process unit & concurrency tests
# ---------------------------------------------------------------------------


def test_barrier_with_error_propagation_uninitialized_falls_back_to_barrier(monkeypatch):
    barriers = []
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "barrier", lambda: barriers.append(1))

    barrier_with_error_propagation(None, stage="test")
    assert len(barriers) == 1

    with pytest.raises(ValueError, match="boom"):
        barrier_with_error_propagation(ValueError("boom"), stage="test")
    assert len(barriers) == 2


@pytest.mark.asyncio
async def test_cleanup_unawaited_task_handles_coro_and_task():
    async def _never_awaited():
        return 1

    coro = _never_awaited()
    await cleanup_unawaited_task(coro)
    assert coro.cr_frame is None

    cancelled_cleanly = False

    async def _background_task():
        nonlocal cancelled_cleanly
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled_cleanly = True
            raise

    task = asyncio.create_task(_background_task())
    await asyncio.sleep(0)
    await cleanup_unawaited_task(task)
    assert task.cancelled()
    assert cancelled_cleanly is True

    loop = asyncio.get_running_loop()
    failed_fut = loop.create_future()
    failed_fut.set_exception(RuntimeError("already failed"))
    await cleanup_unawaited_task(failed_fut)
    assert isinstance(failed_fut.exception(), RuntimeError)


@pytest.mark.vllm
def test_lora_source_discard_prepared_forces_reexport():
    from skyrl.backends.skyrl_train.weight_sync.sources import LoraAdapterWeightSource

    class _CountingSource(LoraAdapterWeightSource):
        def __init__(self):
            super().__init__(dtype=torch.float32, experts_per_shared_adapter=1)
            self.exports = 0

        def export_adapter_stream(self):
            self.exports += 1
            return [("base_model.model.q_proj.lora_A.weight", torch.full((2, 2), float(self.exports)))]

    source = _CountingSource()
    source.set_lora_name("tenant-a")
    source.prepare()
    assert source.exports == 1
    assert len(source.metadata) == 1

    # Discarding clears both _prepared and cached metadata so the next tenant re-exports.
    source.discard_prepared()
    source.set_lora_name("tenant-b")
    source.prepare()
    assert source.exports == 2
    assert source.receive_target["lora_name"] == "tenant-b"


@pytest.mark.asyncio
async def test_await_weight_sync_thread_drains_on_cancellation_before_discard():
    """Cancellation must wait for the worker thread before finally clears prepared state."""
    import threading
    import time

    worker = PolicyWorkerBase.__new__(PolicyWorkerBase)
    started = threading.Event()
    state = {"prepared": None, "thread_done": False}

    def _slow_prepare():
        started.set()
        time.sleep(0.15)
        state["prepared"] = "stale-tenant-a"
        state["thread_done"] = True

    async def _run_with_cleanup():
        try:
            await worker._await_weight_sync_thread(_slow_prepare)
        finally:
            # Must observe thread_done=True so _slow_prepare cannot repopulate after discard.
            assert state["thread_done"] is True
            state["prepared"] = None

    task = asyncio.create_task(_run_with_cleanup())
    while not started.is_set():
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state["thread_done"] is True
    assert state["prepared"] is None


def _worker_full_weight_sync_send_failure_on_rank0_recovers(rank: int, world_size: int, store_path: str) -> None:
    _init_gloo_group(rank, world_size, store_path)
    orig_empty_cache = torch.cuda.empty_cache
    try:
        torch.cuda.empty_cache = lambda: None

        class _Engine:
            skyrl_handles_prefix_cache_reset = False
            skyrl_force_disable_expandable_segments = False
            skyrl_empty_cache_after_send = False

            def __init__(self):
                self.sends = 0

            def send_weights(self):
                self.sends += 1
                if rank == 0 and self.sends == 1:
                    raise RuntimeError("rank-0 HTTP finish RPC failed after collectives")

        worker = PolicyWorkerBase.__new__(PolicyWorkerBase)
        worker._weight_sync_engine = _Engine()
        worker.cfg = SimpleNamespace(
            fully_async=SimpleNamespace(enabled=False, clear_kv_cache_on_weight_sync=False),
            placement=SimpleNamespace(colocate_all=False),
        )

        @contextmanager
        def _noop_ctx(*args, **kwargs):
            yield

        worker._expandable_segments_disabled_for_sync = _noop_ctx

        resets = []

        async def _reset_cache(reset_running_requests: bool = True):
            resets.append(reset_running_requests)

        client = Mock()
        client.reset_prefix_cache = AsyncMock(side_effect=_reset_cache)
        ie_cfg = SimpleNamespace(enable_prefix_caching=True)

        async def _run():
            if rank == 0:
                with pytest.raises(RuntimeError, match="rank-0 HTTP finish RPC failed"):
                    await worker._sync_weights_to_inference_engines(client, ie_cfg)
            else:
                await worker._sync_weights_to_inference_engines(client, ie_cfg)

            # Second sync must succeed cleanly on both ranks.
            await worker._sync_weights_to_inference_engines(client, ie_cfg)

        asyncio.run(_run())
        assert worker._weight_sync_engine.sends == 2
        if rank == 0:
            assert len(resets) == 1
    finally:
        torch.cuda.empty_cache = orig_empty_cache
        dist.destroy_process_group()


@pytest.mark.vllm
def test_sync_weights_send_failure_on_rank0_recovers(tmp_path):
    _run_mp(_worker_full_weight_sync_send_failure_on_rank0_recovers, 2, tmp_path)


@pytest.mark.asyncio
async def test_set_lora_receive_target_settles_and_disarms_on_partial_failure():
    """Partial arm failure (including error-after-arm) settles and disarms all servers."""
    client = RemoteInferenceClient.__new__(RemoteInferenceClient)
    client.server_urls = ["http://server-lost-resp:8000", "http://server-good:8001"]

    armed = {}
    calls = []

    async def _fake_call_server(url: str, endpoint: str, json=None, method="POST", params=None):
        method_name = json["method"]
        calls.append((url, method_name))
        if method_name == "skyrl_set_lora_receive_target":
            if "server-lost-resp" in url:
                # Server armed locally before losing the HTTP response.
                armed[url] = json["kwargs"]["receive_target"]["lora_name"]
                raise ConnectionResetError("connection lost after arm")
            await asyncio.sleep(0.08)
            if url in armed:
                raise RuntimeError(f"already armed for {armed[url]}")
            armed[url] = json["kwargs"]["receive_target"]["lora_name"]
            return url, {"status": 200, "body": {}}
        if method_name == "skyrl_clear_lora_receive_target":
            armed.pop(url, None)
            return url, {"status": 200, "body": {}}
        raise AssertionError(f"Unexpected method {method_name}")

    client._call_server = _fake_call_server

    with pytest.raises(ConnectionResetError, match="connection lost after arm"):
        await client.set_lora_receive_target({"kind": "lora", "lora_name": "tenant-a"})

    # Both servers (including the one whose HTTP response failed) must be disarmed.
    assert armed == {}
    assert ("http://server-lost-resp:8000", "skyrl_clear_lora_receive_target") in calls
    assert ("http://server-good:8001", "skyrl_clear_lora_receive_target") in calls

    # Next tenant can arm both servers cleanly.
    client.server_urls = ["http://server-good:8001"]
    await client.set_lora_receive_target({"kind": "lora", "lora_name": "tenant-b"})
    assert armed == {"http://server-good:8001": "tenant-b"}


@pytest.mark.asyncio
async def test_set_lora_receive_target_cancellation_drains_and_disarms():
    """Parent cancellation drains in-flight arm RPCs and disarms all servers."""
    client = RemoteInferenceClient.__new__(RemoteInferenceClient)
    client.server_urls = ["http://server-fast:8000", "http://server-slow:8001"]

    armed = {}
    fast_armed = asyncio.Event()

    async def _fake_call_server(url: str, endpoint: str, json=None, method="POST", params=None):
        method_name = json["method"]
        if method_name == "skyrl_set_lora_receive_target":
            if "server-fast" in url:
                armed[url] = json["kwargs"]["receive_target"]["lora_name"]
                fast_armed.set()
                return url, {"status": 200, "body": {}}
            await asyncio.sleep(0.1)
            armed[url] = json["kwargs"]["receive_target"]["lora_name"]
            return url, {"status": 200, "body": {}}
        if method_name == "skyrl_clear_lora_receive_target":
            armed.pop(url, None)
            return url, {"status": 200, "body": {}}
        raise AssertionError(f"Unexpected method {method_name}")

    client._call_server = _fake_call_server

    task = asyncio.create_task(client.set_lora_receive_target({"kind": "lora", "lora_name": "tenant-a"}))
    await fast_armed.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert armed == {}


@pytest.mark.vllm
def test_skyrl_clear_lora_receive_target_allows_rearm():
    from skyrl.backends.skyrl_train.weight_sync.lora_target import (
        build_lora_receive_target,
    )
    from skyrl.backends.skyrl_train.weight_sync.weight_receivers import (
        SkyrlLoraStagingMixin,
    )

    class _DummyReceiver(SkyrlLoraStagingMixin):
        vllm_config = SimpleNamespace(lora_config=object())

    receiver = _DummyReceiver()
    receiver.skyrl_set_lora_receive_target(build_lora_receive_target("tenant-a", {}, {}))
    assert receiver.skyrl_lora_armed() is True

    receiver.skyrl_clear_lora_receive_target()
    assert receiver.skyrl_lora_armed() is False

    # Re-arming for the next tenant succeeds cleanly.
    receiver.skyrl_set_lora_receive_target(build_lora_receive_target("tenant-b", {}, {}))
    assert receiver.skyrl_lora_armed() is True


@pytest.mark.asyncio
async def test_load_lora_adapter_settles_all_servers_before_raising(monkeypatch):
    """When one server fails fast, load_lora_adapter settles sibling loads before raising."""
    from skyrl.backends.skyrl_train.inference_servers import (
        remote_inference_client as ric_mod,
    )

    client = RemoteInferenceClient.__new__(RemoteInferenceClient)
    client.server_urls = ["http://server-bad:8000", "http://server-slow:8001"]
    client._get_session = AsyncMock(return_value=object())

    slow_settled = False

    class _FakeResp:
        status = 200

        async def text(self):
            return "Success"

    class _FakeRequestCtx:
        def __init__(self, url: str):
            self.url = url

        async def __aenter__(self):
            nonlocal slow_settled
            if "server-bad" in self.url:
                raise ConnectionRefusedError("Cannot connect to server-bad")
            await asyncio.sleep(0.1)
            slow_settled = True
            return _FakeResp()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(ric_mod, "_control_plane_request", lambda session, method, url, **kw: _FakeRequestCtx(url))

    with pytest.raises(ConnectionRefusedError, match="server-bad"):
        await client.load_lora_adapter("tenant-a", "/tmp/lora/tenant-a")

    assert slow_settled is True


# ---------------------------------------------------------------------------
# Regression tests for bounded phase timeout, send-failure disarm, and dirty-server recovery
# ---------------------------------------------------------------------------


def _worker_stalled_arm_and_cleanup_bounded_below_gloo_timeout(rank: int, world_size: int, store_path: str) -> None:
    import time

    from skyrl.backends.skyrl_train.inference_servers import (
        remote_inference_client as ric_mod,
    )

    pg_timeout_s = 2.5
    _init_gloo_group(rank, world_size, store_path, timeout_s=pg_timeout_s)
    orig_pg_timeout = ric_mod.SKYRL_WORKER_NCCL_TIMEOUT_IN_S
    try:
        ric_mod.SKYRL_WORKER_NCCL_TIMEOUT_IN_S = pg_timeout_s

        client = RemoteInferenceClient.__new__(RemoteInferenceClient)
        client.server_urls = ["http://server-stalled:8000", "http://server-ok:8001"]
        stalled = {"flag": True}
        armed = {}

        async def _fake_call_server(url: str, endpoint: str, json=None, method="POST", params=None):
            method_name = json["method"]
            if "server-stalled" in url and stalled["flag"]:
                # Both arm and rollback stall far beyond the 2.5s Gloo PG timeout.
                await asyncio.sleep(30.0)
            if method_name == "skyrl_set_lora_receive_target":
                if url in armed:
                    raise RuntimeError(f"already armed for {armed[url]}")
                armed[url] = json["kwargs"]["receive_target"]["lora_name"]
                return url, {"status": 200, "body": {}}
            if method_name == "skyrl_clear_lora_receive_target":
                armed.pop(url, None)
                return url, {"status": 200, "body": {}}
            raise AssertionError(f"Unexpected method {method_name}")

        client._call_server = _fake_call_server

        async def _run():
            started = time.monotonic()
            arm_error = None
            if rank == 0:
                try:
                    await client.set_lora_receive_target({"kind": "lora", "lora_name": "tenant-a"})
                except Exception as exc:
                    arm_error = exc
            if rank == 0:
                with pytest.raises(InferenceServerTimeoutError) as exc_info:
                    barrier_with_error_propagation(arm_error, stage="in-memory LoRA receive target setup")
                # Must finish strictly below the 2.5s Gloo timeout AND note the rollback timeout.
                elapsed = time.monotonic() - started
                assert elapsed < pg_timeout_s
                assert any("skyrl_clear_lora_receive_target" in n for n in getattr(exc_info.value, "__notes__", []))
                assert "http://server-stalled:8000" in client._dirty_lora_servers()
            else:
                with pytest.raises(PeerRankSyncError, match="Rank 0 failed"):
                    barrier_with_error_propagation(None, stage="in-memory LoRA receive target setup")

            # Server recovers; preflight disarm clears dirty state and round 2 succeeds on the SAME Gloo PG.
            stalled["flag"] = False
            arm_error = None
            if rank == 0:
                await client.set_lora_receive_target({"kind": "lora", "lora_name": "tenant-b"})
            barrier_with_error_propagation(arm_error, stage="in-memory LoRA receive target setup")
            if rank == 0:
                assert armed == {
                    "http://server-stalled:8000": "tenant-b",
                    "http://server-ok:8001": "tenant-b",
                }
                assert client._dirty_lora_servers() == set()

        asyncio.run(_run())
    finally:
        ric_mod.SKYRL_WORKER_NCCL_TIMEOUT_IN_S = orig_pg_timeout
        dist.destroy_process_group()


def test_stalled_arm_and_cleanup_bounded_below_gloo_timeout(tmp_path):
    _run_mp(_worker_stalled_arm_and_cleanup_bounded_below_gloo_timeout, 2, tmp_path)


def _worker_megatron_send_failure_disarms_real_receivers(
    rank: int, world_size: int, store_path: str, sync_dir: str
) -> None:
    from skyrl.backends.skyrl_train.weight_sync.sources import LoraAdapterWeightSource
    from skyrl.backends.skyrl_train.weight_sync.weight_receivers import (
        SkyrlLoraStagingMixin,
    )

    _init_gloo_group(rank, world_size, store_path)
    orig_empty_cache = torch.cuda.empty_cache
    try:
        torch.cuda.empty_cache = lambda: None

        class _RealStagingReceiver(SkyrlLoraStagingMixin):
            vllm_config = SimpleNamespace(lora_config=object())

        receivers = {
            "http://server-0:8000": _RealStagingReceiver(),
            "http://server-1:8001": _RealStagingReceiver(),
        }

        class _FakeLoraSource(LoraAdapterWeightSource):
            def __init__(self):
                super().__init__(dtype=torch.bfloat16, experts_per_shared_adapter=1)

            def export_adapter_stream(self):
                return [("base_model.model.q_proj.lora_A.weight", torch.ones(2, 4, dtype=torch.bfloat16))]

        class _FailFirstSendEngine:
            skyrl_force_disable_expandable_segments = False

            def __init__(self, source):
                self.source = source
                self.calls = 0

            def send_weights(self):
                self.calls += 1
                if self.calls == 1:
                    # Simulate rank-0 failing in start_weight_update before finish_weight_update disarms receivers.
                    if rank == 0:
                        assert all(r.skyrl_lora_armed() for r in receivers.values())
                        raise RuntimeError("start_weight_update failed before finish_weight_update")
                    return
                list(self.source.iter_full())
                if rank == 0:
                    # Normal finish_weight_update disarms every receiver.
                    for r in receivers.values():
                        r.skyrl_clear_lora_receive_target()

        worker = _make_megatron_lora_worker(sync_dir, sync_mode="memory")
        source = _FakeLoraSource()
        engine = _FailFirstSendEngine(source)
        worker._lora_weight_source = source
        worker._weight_sync_engine = engine
        ie_cfg = SimpleNamespace(enable_prefix_caching=False)

        client = RemoteInferenceClient.__new__(RemoteInferenceClient)
        client.server_urls = list(receivers.keys())
        loaded = []

        async def _fake_call_server(url: str, endpoint: str, json=None, method="POST", params=None):
            method_name = json["method"]
            recv = receivers[url]
            if method_name == "skyrl_set_lora_receive_target":
                recv.skyrl_set_lora_receive_target(json["kwargs"]["receive_target"])
                return url, {"status": 200, "body": {}}
            if method_name == "skyrl_clear_lora_receive_target":
                round_id = (json.get("kwargs") or {}).get("round_id")
                recv.skyrl_clear_lora_receive_target(round_id=round_id)
                return url, {"status": 200, "body": {}}
            raise AssertionError(f"Unexpected method {method_name}")

        async def _fake_load_lora(lora_name: str, lora_path=None, *, in_memory: bool = False):
            assert in_memory is True
            loaded.append(lora_name)
            return {}

        client._call_server = _fake_call_server
        client.load_lora_adapter = AsyncMock(side_effect=_fake_load_lora)

        async def _run():
            # Round 1: arm succeeds on real staging receivers, then rank-0 send_weights fails before finish.
            if rank == 0:
                with pytest.raises(RuntimeError, match="start_weight_update failed"):
                    await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-a")
                assert all(not r.skyrl_lora_armed() for r in receivers.values())
                assert loaded == []
            else:
                await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-a")

            # Round 2: next tenant arms the same real staging receivers without hitting "already armed".
            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-b")
            if rank == 0:
                assert loaded == ["tenant-b"]
                assert all(not r.skyrl_lora_armed() for r in receivers.values())

        asyncio.run(_run())
    finally:
        torch.cuda.empty_cache = orig_empty_cache
        dist.destroy_process_group()


@pytest.mark.megatron
def test_megatron_in_memory_send_failure_disarms_real_staging_receivers(tmp_path):
    _run_mp(_worker_megatron_send_failure_disarms_real_receivers, 2, tmp_path, str(tmp_path / "lora_sync"))


@pytest.mark.asyncio
async def test_failed_disarm_reported_blocks_reuse_until_recovery():
    """A failed disarm preserves the original error with a note and blocks reuse until cleanup succeeds."""
    client = RemoteInferenceClient.__new__(RemoteInferenceClient)
    client.server_urls = ["http://server-a:8000", "http://server-b:8001"]

    armed = {}
    state = {"fail_b_arm": True, "fail_a_clear": True}

    async def _fake_call_server(url: str, endpoint: str, json=None, method="POST", params=None):
        method_name = json["method"]
        if method_name == "skyrl_set_lora_receive_target":
            if "server-b" in url and state["fail_b_arm"]:
                raise ConnectionRefusedError("server-b unreachable during arm")
            if url in armed:
                raise RuntimeError(f"already armed for {armed[url]}")
            armed[url] = json["kwargs"]["receive_target"]["lora_name"]
            return url, {"status": 200, "body": {}}
        if method_name == "skyrl_clear_lora_receive_target":
            if "server-a" in url and state["fail_a_clear"]:
                raise ConnectionError("server-a dropped clear RPC")
            armed.pop(url, None)
            return url, {"status": 200, "body": {}}
        raise AssertionError(f"Unexpected method {method_name}")

    client._call_server = _fake_call_server

    # Round 1: server-a arms, server-b fails arm, and rollback clear fails on server-a.
    with pytest.raises(ConnectionRefusedError, match="server-b unreachable") as exc_info:
        await client.set_lora_receive_target({"kind": "lora", "lora_name": "tenant-a"})
    assert any("server-a:8000" in n for n in getattr(exc_info.value, "__notes__", []))
    assert "http://server-a:8000" in client._dirty_lora_servers()
    assert armed == {"http://server-a:8000": "tenant-a"}

    # Round 2: server-b is healthy now, but server-a's clear still fails -> must reject BEFORE arming server-b.
    state["fail_b_arm"] = False
    with pytest.raises(RuntimeError, match="prior disarm still failing"):
        await client.set_lora_receive_target({"kind": "lora", "lora_name": "tenant-b"})
    assert "http://server-b:8001" not in armed

    # Round 3: server-a recovers -> preflight disarm clears server-a, then both servers arm for tenant-b.
    state["fail_a_clear"] = False
    await client.set_lora_receive_target({"kind": "lora", "lora_name": "tenant-b"})
    assert client._dirty_lora_servers() == set()
    assert armed == {
        "http://server-a:8000": "tenant-b",
        "http://server-b:8001": "tenant-b",
    }


def _worker_blocking_client_stalled_finish_recovers_gloo(rank: int, world_size: int, store_path: str) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor

    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        rank0_phase_timeout_s,
    )
    from skyrl.backends.skyrl_train.weight_sync.control_plane import (
        START_UPDATE_ENDPOINT,
        SkyrlWeightSyncClient,
        weight_sync_client_timeout,
    )
    from skyrl.backends.skyrl_train.workers.worker_utils import exchange_rank_errors

    pg_timeout_s = 2.5
    _init_gloo_group(rank, world_size, store_path, timeout_s=pg_timeout_s)
    try:
        import threading

        release_stalled = threading.Event()
        stalled = {"flag": True}

        class _Resp:
            status_code = 200
            reason = "OK"
            text = "{}"

            def json(self):
                return {"status": "ok"}

        class _BlockingSession:
            def post(self, url, json=None, timeout=None):
                if url.endswith("/finish_weight_update") and stalled["flag"]:
                    # Simulate a server that stalls past the phase budget, then finishes and confirms settlement.
                    release_stalled.wait(timeout=30.0)
                return _Resp()

            def close(self):
                pass

        client = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
        client._urls = ["http://server-0:8000"]
        client._dp = 1
        client._init_payload_fn = None
        client._timeout_s = None
        client._start_endpoint = START_UPDATE_ENDPOINT
        client._session = _BlockingSession()
        client._pool = ThreadPoolExecutor(max_workers=1)

        try:
            # Round 1: rank 0 stalls in blocking /finish_weight_update while rank 1 returns early
            # into exchange_rank_errors. Timeout must fire strictly below the 2.5s Gloo PG timeout.
            started = time.monotonic()
            send_error = None
            if rank == 0:
                with weight_sync_client_timeout(rank0_phase_timeout_s(pg_timeout_s)):
                    try:
                        client.finish_weight_update()
                    except Exception as exc:
                        send_error = exc
            any_failed, peer_err = exchange_rank_errors(send_error)
            elapsed = time.monotonic() - started
            assert elapsed < pg_timeout_s
            assert any_failed is True
            if rank == 0:
                assert isinstance(send_error, TimeoutError)
            else:
                assert peer_err is not None and peer_err[0] == 0

            # Round 2: stalled server RPC settles with server confirmation; same Gloo PG and client succeed.
            release_stalled.set()
            stalled["flag"] = False
            send_error = None
            if rank == 0:
                client.drain_inflight(deadline_at=time.monotonic() + 2.0)
                with weight_sync_client_timeout(rank0_phase_timeout_s(pg_timeout_s)):
                    client.finish_weight_update()
            barrier_with_error_propagation(send_error, stage="blocking finish recovery")
        finally:
            release_stalled.set()
            client.drain_inflight(deadline_at=time.monotonic() + 2.0)
            client.close()
    finally:
        dist.destroy_process_group()


def test_blocking_weight_sync_client_stalled_finish_recovers_gloo_group(tmp_path):
    _run_mp(_worker_blocking_client_stalled_finish_recovers_gloo, 2, tmp_path)


@pytest.mark.vllm
@pytest.mark.asyncio
async def test_late_server_arm_and_clear_protected_by_round_id():
    """Delayed server-side clear/arm tasks that continue after client timeout cannot corrupt later rounds."""
    from skyrl.backends.skyrl_train.weight_sync.lora_target import (
        build_lora_receive_target,
    )
    from skyrl.backends.skyrl_train.weight_sync.weight_receivers import (
        SkyrlLoraStagingMixin,
    )

    class _RealReceiver(SkyrlLoraStagingMixin):
        vllm_config = SimpleNamespace(lora_config=object())

    receiver = _RealReceiver()
    client = RemoteInferenceClient.__new__(RemoteInferenceClient)
    client.server_urls = ["http://server-0:8000"]

    late_tasks: list[asyncio.Task] = []
    release_late_clear = asyncio.Event()
    release_late_arm = asyncio.Event()
    mode = {"scenario": "delay_clear"}
    late_arm_rejected = {"error": None}

    async def _fake_call_server(url: str, endpoint: str, json=None, method="POST", params=None):
        method_name = json["method"]
        kwargs = json.get("kwargs") or {}
        if method_name == "skyrl_set_lora_receive_target":
            target = kwargs["receive_target"]
            if mode["scenario"] == "delay_arm" and target["lora_name"] == "tenant-c":
                # Independent server task that continues even after client times out/cancels.
                async def _server_apply_late_arm():
                    await release_late_arm.wait()
                    try:
                        receiver.skyrl_set_lora_receive_target(target)
                    except Exception as exc:
                        late_arm_rejected["error"] = exc

                late_tasks.append(asyncio.create_task(_server_apply_late_arm()))
                await asyncio.sleep(30.0)
            receiver.skyrl_set_lora_receive_target(target)
            return url, {"status": 200, "body": {}}

        if method_name == "skyrl_clear_lora_receive_target":
            rid = kwargs.get("round_id")
            if mode["scenario"] == "delay_clear":
                mode["scenario"] = "normal"

                # Independent server task that applies the stale clear AFTER tenant-b arms.
                async def _server_apply_late_clear():
                    await release_late_clear.wait()
                    receiver.skyrl_clear_lora_receive_target(round_id=rid)

                late_tasks.append(asyncio.create_task(_server_apply_late_clear()))
                await asyncio.sleep(30.0)
            receiver.skyrl_clear_lora_receive_target(round_id=rid)
            return url, {"status": 200, "body": {}}

        raise AssertionError(f"Unexpected method {method_name}")

    client._call_server = _fake_call_server

    # Scenario A: tenant-a's clear times out on client while continuing independently on the server.
    # Preflight clear + tenant-b arm run first, then tenant-a's stale clear arrives on the server.
    await client.set_lora_receive_target(build_lora_receive_target("tenant-a", {}, {}), timeout_s=1.0)
    assert receiver.skyrl_lora_armed() is True
    failed = await client.clear_lora_receive_target(timeout_s=0.15)
    assert "http://server-0:8000" in failed

    # Preflight clears tenant-a cleanly, then arms tenant-b.
    await client.set_lora_receive_target(build_lora_receive_target("tenant-b", {}, {}), timeout_s=1.0)
    assert receiver._skyrl_lora_target["lora_name"] == "tenant-b"

    # Now release the stale clear from tenant-a's round: it must NOT disarm tenant-b.
    release_late_clear.set()
    await asyncio.gather(*late_tasks)
    late_tasks.clear()
    assert receiver.skyrl_lora_armed() is True
    assert receiver._skyrl_lora_target["lora_name"] == "tenant-b"

    # Disarm tenant-b cleanly before Scenario B.
    assert await client.clear_lora_receive_target(timeout_s=1.0) == {}
    assert receiver.skyrl_lora_armed() is False

    # Scenario B: tenant-c's arm times out on client while continuing independently on the server;
    # rollback clear lands on server first, then the delayed arm arrives and must be rejected.
    mode["scenario"] = "delay_arm"
    with pytest.raises(InferenceServerTimeoutError):
        await client.set_lora_receive_target(build_lora_receive_target("tenant-c", {}, {}), timeout_s=0.25)

    release_late_arm.set()
    await asyncio.gather(*late_tasks)
    late_tasks.clear()
    assert isinstance(late_arm_rejected["error"], RuntimeError)
    assert "already cancelled" in str(late_arm_rejected["error"])
    assert receiver.skyrl_lora_armed() is False

    # Tenant-d can now arm cleanly.
    mode["scenario"] = "normal"
    await client.set_lora_receive_target(build_lora_receive_target("tenant-d", {}, {}), timeout_s=1.0)
    assert receiver._skyrl_lora_target["lora_name"] == "tenant-d"


def _worker_megatron_export_copy_failure_drains_collectives(
    rank: int, world_size: int, store_path: str, sync_dir: str
) -> None:
    from megatron.bridge.models.conversion import peft_bridge

    _init_gloo_group(rank, world_size, store_path)
    orig_empty_cache = torch.cuda.empty_cache
    orig_build_cfg = peft_bridge.build_adapter_config_dict
    try:
        torch.cuda.empty_cache = lambda: None
        peft_bridge.build_adapter_config_dict = lambda lora_cls, target_modules, base_model_name_or_path: {
            "r": lora_cls.dim,
            "lora_alpha": lora_cls.alpha,
            "target_modules": list(target_modules),
            "base_model_name_or_path": base_model_name_or_path,
        }

        worker = _make_megatron_lora_worker(sync_dir, sync_mode="disk")
        ie_cfg = SimpleNamespace(enable_prefix_caching=False)
        fail_first = {"flag": True}
        seen_cpu_args = []
        second_collective_reached = {"count": 0}

        class _FailingCopyTensor:
            def detach(self):
                return self

            def to(self, *args, **kwargs):
                raise MemoryError("simulated host OOM during writer-only CPU copy")

        def _export_with_collective(actor_module, cpu=True, show_progress=False):
            seen_cpu_args.append(cpu)
            if fail_first["flag"] and rank == 0:
                yield ("layers.0.self_attn.q_proj.lora_A.weight", _FailingCopyTensor())
            else:
                yield ("layers.0.self_attn.q_proj.lora_A.weight", torch.ones(4, 8, dtype=torch.bfloat16))
            # A second collective remains in the stream after the first tensor fails on rank 0.
            dist.barrier()
            second_collective_reached["count"] += 1
            yield ("layers.0.self_attn.q_proj.lora_B.weight", torch.ones(8, 4, dtype=torch.bfloat16))

        worker.bridge.export_adapter_weights = _export_with_collective

        client = Mock(spec=RemoteInferenceClient)
        client.load_lora_adapter = AsyncMock()

        async def _run():
            # Round 1: writer rank hits MemoryError copying tensor 1 to CPU, must still drain
            # tensor 2's collective barrier without deadlocking peer rank 1.
            if rank == 0:
                with pytest.raises(MemoryError, match="host OOM"):
                    await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_1")
                client.load_lora_adapter.assert_not_awaited()
            else:
                await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_1")

            assert second_collective_reached["count"] == 1
            assert seen_cpu_args == [False]

            # Round 2: export succeeds on both ranks on the same Gloo PG.
            fail_first["flag"] = False
            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="model_2")
            assert second_collective_reached["count"] == 2
            if rank == 0:
                client.load_lora_adapter.assert_awaited_once()

        asyncio.run(_run())
    finally:
        torch.cuda.empty_cache = orig_empty_cache
        peft_bridge.build_adapter_config_dict = orig_build_cfg
        dist.destroy_process_group()


@pytest.mark.megatron
def test_megatron_disk_export_copy_failure_drains_remaining_collectives(tmp_path):
    _run_mp(_worker_megatron_export_copy_failure_drains_collectives, 2, tmp_path, str(tmp_path / "lora_sync"))


def test_blocking_client_deadline_propagates_to_nested_executor():
    """Nested ThreadPoolExecutor submissions (like NCCL sender update_weights) must inherit the deadline."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from skyrl.backends.skyrl_train.weight_sync.control_plane import (
        START_UPDATE_ENDPOINT,
        SkyrlWeightSyncClient,
        weight_sync_client_deadline,
    )

    release = threading.Event()
    seen_timeouts = []

    class _Resp:
        status_code = 200
        reason = "OK"
        text = "{}"

    class _HangingSession:
        def post(self, url, json=None, timeout=None):
            seen_timeouts.append(timeout)
            release.wait(timeout=5.0)
            return _Resp()

        def close(self):
            pass

    client = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
    client._urls = ["http://server-0:8000"]
    client._dp = 1
    client._init_payload_fn = None
    client._timeout_s = None
    client._start_endpoint = START_UPDATE_ENDPOINT
    client._session = _HangingSession()
    client._pool = ThreadPoolExecutor(max_workers=1)

    nested_pool = ThreadPoolExecutor(max_workers=1)
    try:
        started = time.monotonic()
        with weight_sync_client_deadline(started + 0.25):
            # Submit from an independent thread pool that does NOT copy ContextVars.
            fut = nested_pool.submit(client.update_weights, {"names": ["w"]})
            with pytest.raises(TimeoutError, match="exceeded weight-sync phase deadline"):
                fut.result(timeout=2.0)
        elapsed = time.monotonic() - started
        assert elapsed < 1.0
        assert seen_timeouts and isinstance(seen_timeouts[0], tuple)
        connect_t, read_t = seen_timeouts[0]
        assert connect_t is not None and connect_t <= 0.3
        assert read_t is None
    finally:
        release.set()
        client.drain_inflight(deadline_at=time.monotonic() + 2.0)
        nested_pool.shutdown(wait=True)
        client.close()


def _worker_shared_deadline_bounds_combined_sequential_rpcs(rank: int, world_size: int, store_path: str) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor

    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        rank0_phase_timeout_s,
    )
    from skyrl.backends.skyrl_train.weight_sync.control_plane import (
        START_UPDATE_ENDPOINT,
        SkyrlWeightSyncClient,
        weight_sync_client_deadline,
    )
    from skyrl.backends.skyrl_train.workers.worker_utils import exchange_rank_errors

    pg_timeout_s = 2.5
    phase_budget = rank0_phase_timeout_s(pg_timeout_s)  # 1.5s
    _init_gloo_group(rank, world_size, store_path, timeout_s=pg_timeout_s)
    try:
        slow = {"flag": True}

        class _Resp:
            status_code = 200
            reason = "OK"
            text = "{}"

        class _SlowSequentialSession:
            def post(self, url, json=None, timeout=None):
                if slow["flag"]:
                    # Each RPC takes 1.0s (< 1.5s budget alone), so start + finish = 2.0s > 1.5s.
                    # Lifecycle RPCs use (connect_timeout, None) so the socket stays open until
                    # the server finishes executing and confirms settlement.
                    time.sleep(1.0)
                return _Resp()

            def close(self):
                pass

        client = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
        client._urls = ["http://server-0:8000"]
        client._dp = 1
        client._init_payload_fn = None
        client._timeout_s = None
        client._start_endpoint = START_UPDATE_ENDPOINT
        client._session = _SlowSequentialSession()
        client._pool = ThreadPoolExecutor(max_workers=1)

        try:
            started = time.monotonic()
            send_error = None
            if rank == 0:
                with weight_sync_client_deadline(started + phase_budget):
                    try:
                        client.start_weight_update()
                        client.finish_weight_update()
                    except Exception as exc:
                        send_error = exc
            any_failed, peer_err = exchange_rank_errors(send_error)
            elapsed = time.monotonic() - started
            assert elapsed < pg_timeout_s
            assert any_failed is True
            if rank == 0:
                assert isinstance(send_error, TimeoutError)
            else:
                assert peer_err is not None and peer_err[0] == 0

            # Drain any finishing thread, recover, and verify the same Gloo PG succeeds on round 2.
            if rank == 0:
                client.drain_inflight(deadline_at=time.monotonic() + 2.0)
            slow["flag"] = False
            if rank == 0:
                with weight_sync_client_deadline(time.monotonic() + phase_budget):
                    client.start_weight_update()
                    client.finish_weight_update()
            barrier_with_error_propagation(None, stage="shared deadline recovery")
        finally:
            client.close()
            del client
    finally:
        dist.destroy_process_group()


def test_shared_deadline_bounds_combined_sequential_rpcs_on_gloo(tmp_path):
    _run_mp(_worker_shared_deadline_bounds_combined_sequential_rpcs, 2, tmp_path)


def test_timed_out_blocking_rpc_blocks_reuse_until_settled():
    """A blocking POST that ignores socket timeout stays tracked and blocks reuse until it exits."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from skyrl.backends.skyrl_train.weight_sync.control_plane import (
        START_UPDATE_ENDPOINT,
        SkyrlWeightSyncClient,
        weight_sync_client_deadline,
    )

    release_old_post = threading.Event()
    post_count = {"n": 0}

    class _Resp:
        status_code = 200
        reason = "OK"
        text = "{}"

    class _StickyBlockingSession:
        def post(self, url, json=None, timeout=None):
            post_count["n"] += 1
            if post_count["n"] == 1:
                # Ignores timeout (simulates a blocking request outliving client timeout).
                release_old_post.wait(timeout=5.0)
            return _Resp()

        def close(self):
            pass

    client = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
    client._urls = ["http://server-0:8000"]
    client._dp = 1
    client._init_payload_fn = None
    client._timeout_s = None
    client._start_endpoint = START_UPDATE_ENDPOINT
    client._session = _StickyBlockingSession()
    client._pool = ThreadPoolExecutor(max_workers=1)

    try:
        # Round 1 times out while the blocking POST thread is still running.
        with weight_sync_client_deadline(time.monotonic() + 0.15):
            with pytest.raises(TimeoutError, match="exceeded weight-sync phase deadline"):
                client.finish_weight_update()
        assert post_count["n"] == 1

        # Round 2 while Round 1's thread is still running MUST fail without issuing a second POST.
        with weight_sync_client_deadline(time.monotonic() + 0.15):
            with pytest.raises(TimeoutError, match="prior timed-out request\\(s\\) are still in flight"):
                client.start_weight_update()
        assert post_count["n"] == 1

        # Release the stuck POST thread -> Round 3 drains it cleanly and succeeds.
        release_old_post.set()
        with weight_sync_client_deadline(time.monotonic() + 2.0):
            client.start_weight_update()
        assert post_count["n"] == 2
    finally:
        release_old_post.set()
        client.drain_inflight(deadline_at=time.monotonic() + 2.0)
        client.close()


def test_read_timeout_on_lifecycle_rpc_blocks_subsequent_round_without_server_confirmation():
    """Ambiguous read-timeout or post-connect transport errors on lifecycle RPCs must block reuse."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    from skyrl.backends.skyrl_train.weight_sync.control_plane import (
        START_UPDATE_ENDPOINT,
        SkyrlWeightSyncClient,
        _is_unconfirmed_lifecycle_error,
        drain_inflight_weight_sync_requests,
        weight_sync_client_deadline,
    )

    calls = []
    mode = {"exc": TimeoutError("client read timed out while server may still execute finish")}

    class _Resp:
        status_code = 200
        reason = "OK"
        text = "{}"

    class _TransportFailureSession:
        def post(self, url, json=None, timeout=None):
            calls.append(url)
            if url.endswith("/finish_weight_update"):
                raise mode["exc"]
            return _Resp()

        def close(self):
            pass

    for exc_to_raise, match_str in [
        (TimeoutError("client read timed out while server may still execute finish"), "read timed out"),
        (ConnectionResetError("connection reset after server accepted finish"), "connection reset"),
    ]:
        mode["exc"] = exc_to_raise
        client = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
        client._urls = ["http://server-0:8000"]
        client._dp = 1
        client._init_payload_fn = None
        client._timeout_s = None
        client._start_endpoint = START_UPDATE_ENDPOINT
        client._session = _TransportFailureSession()
        client._pool = ThreadPoolExecutor(max_workers=1)

        try:
            with weight_sync_client_deadline(time.monotonic() + 1.0):
                with pytest.raises(type(exc_to_raise), match=match_str):
                    client.finish_weight_update()

            # Even though the HTTP future finished, server completion is unconfirmed.
            unconfirmed = drain_inflight_weight_sync_requests(deadline_at=time.monotonic() + 0.2)
            assert any("finish_weight_update" in entry and "unconfirmed" in entry for entry in unconfirmed)

            # A subsequent tenant's lifecycle RPC must be rejected without contacting the server.
            calls_before = len(calls)
            with weight_sync_client_deadline(time.monotonic() + 1.0):
                with pytest.raises(TimeoutError, match="unconfirmed"):
                    client.start_weight_update()
            assert len(calls) == calls_before
        finally:
            client.close()
            del client

    # Pre-connect failures and HTTP status RuntimeErrors are NOT unconfirmed.
    assert _is_unconfirmed_lifecycle_error(ConnectionRefusedError("refused")) is False
    assert _is_unconfirmed_lifecycle_error(RuntimeError("HTTP 500")) is False
    assert _is_unconfirmed_lifecycle_error(ConnectionResetError("reset")) is True
    try:
        import requests

        assert _is_unconfirmed_lifecycle_error(requests.exceptions.ConnectTimeout("connect")) is False
        assert (
            _is_unconfirmed_lifecycle_error(requests.exceptions.ConnectionError(ConnectionRefusedError("refused")))
            is False
        )
        assert _is_unconfirmed_lifecycle_error(requests.exceptions.ReadTimeout("read")) is True
        assert _is_unconfirmed_lifecycle_error(requests.exceptions.ConnectionError("RemoteDisconnected")) is True
    except ImportError:
        pass


def test_inflight_drain_cancellation_keeps_unfinished_request_visible():
    """Draining must keep unfinished lifecycle futures visible in the registry even while waiting or cancelled."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from skyrl.backends.skyrl_train.weight_sync.control_plane import (
        START_UPDATE_ENDPOINT,
        SkyrlWeightSyncClient,
        drain_inflight_weight_sync_requests,
        weight_sync_client_deadline,
    )

    release_rpc = threading.Event()
    drain_started = threading.Event()

    class _Resp:
        status_code = 200
        reason = "OK"
        text = "{}"

    class _StuckSession:
        def post(self, url, json=None, timeout=None):
            release_rpc.wait(timeout=5.0)
            return _Resp()

        def close(self):
            pass

    client = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
    client._urls = ["http://server-0:8000"]
    client._dp = 1
    client._init_payload_fn = None
    client._timeout_s = None
    client._start_endpoint = START_UPDATE_ENDPOINT
    client._session = _StuckSession()
    client._pool = ThreadPoolExecutor(max_workers=1)

    try:
        with weight_sync_client_deadline(time.monotonic() + 0.1):
            with pytest.raises(TimeoutError, match="exceeded weight-sync phase deadline"):
                client.finish_weight_update()

        cancel_event = threading.Event()
        drain_result: list[str] = []

        def _run_drain():
            drain_started.set()
            drain_result.extend(
                drain_inflight_weight_sync_requests(deadline_at=time.monotonic() + 5.0, cancel_event=cancel_event)
            )

        t = threading.Thread(target=_run_drain)
        t.start()
        assert drain_started.wait(timeout=1.0)
        time.sleep(0.08)

        # While _run_drain is actively waiting, a concurrent caller must STILL see the unfinished request.
        visible = drain_inflight_weight_sync_requests(deadline_at=time.monotonic())
        assert visible == ["http://server-0:8000/finish_weight_update"]

        # Cancelling the drain stops it quickly while keeping the unfinished request registered.
        cancel_t0 = time.monotonic()
        cancel_event.set()
        t.join(timeout=1.0)
        assert not t.is_alive()
        assert time.monotonic() - cancel_t0 < 0.5
        assert drain_result == ["http://server-0:8000/finish_weight_update"]
        assert drain_inflight_weight_sync_requests(deadline_at=time.monotonic()) == [
            "http://server-0:8000/finish_weight_update"
        ]

        # Once the stuck RPC settles, the registry clears cleanly.
        release_rpc.set()
        assert drain_inflight_weight_sync_requests(deadline_at=time.monotonic() + 2.0) == []
    finally:
        release_rpc.set()
        client.drain_inflight(deadline_at=time.monotonic() + 2.0)
        client.close()


def test_close_does_not_hang_with_stuck_inflight_rpc_and_preserves_tracking():
    """close() must not hang indefinitely on a stuck stateful RPC and must keep it visible in global draining until settled."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from skyrl.backends.skyrl_train.weight_sync.control_plane import (
        START_UPDATE_ENDPOINT,
        SkyrlWeightSyncClient,
        drain_inflight_weight_sync_requests,
        weight_sync_client_deadline,
    )

    release_rpc = threading.Event()
    session_closed = threading.Event()

    class _Resp:
        status_code = 200
        reason = "OK"
        text = "{}"

    class _StuckIgnoreCloseSession:
        def post(self, url, json=None, timeout=None):
            release_rpc.wait(timeout=10.0)
            return _Resp()

        def close(self):
            session_closed.set()

    client = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
    client._urls = ["http://server-0:8000"]
    client._dp = 1
    client._init_payload_fn = None
    client._timeout_s = None
    client._start_endpoint = START_UPDATE_ENDPOINT
    client._session = _StuckIgnoreCloseSession()
    client._pool = ThreadPoolExecutor(max_workers=1)

    try:
        with weight_sync_client_deadline(time.monotonic() + 0.1):
            with pytest.raises(TimeoutError, match="exceeded weight-sync phase deadline"):
                client.finish_weight_update()

        t0 = time.monotonic()
        client.close()
        elapsed = time.monotonic() - t0
        assert session_closed.is_set()
        # Must return after bounded drain instead of hanging 10s in _pool.shutdown(wait=True).
        assert elapsed < 2.0
        # Unresolved request must remain visible to global draining.
        assert "http://server-0:8000/finish_weight_update" in drain_inflight_weight_sync_requests(
            deadline_at=time.monotonic()
        )

        # Once the stuck thread finally settles, global draining clears it.
        release_rpc.set()
        assert drain_inflight_weight_sync_requests(deadline_at=time.monotonic() + 2.0) == []
    finally:
        release_rpc.set()
        client.drain_inflight(deadline_at=time.monotonic() + 2.0)
        client.close()


def test_pause_endpoint_tracks_unconfirmed_transport_errors_and_deadline_race():
    """Query-bearing /pause uses connect-only timeout and tracks ambiguous transport errors; successful deadline races stay clean."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from skyrl.backends.skyrl_train.weight_sync.control_plane import (
        START_UPDATE_ENDPOINT,
        SkyrlWeightSyncClient,
        drain_inflight_weight_sync_requests,
        weight_sync_client_deadline,
    )

    seen_timeouts = []
    mode = {"fail_pause": True}

    class _Resp:
        status_code = 200
        reason = "OK"
        text = "{}"

    class _PauseSession:
        def post(self, url, json=None, timeout=None):
            seen_timeouts.append(timeout)
            if mode["fail_pause"] and "/pause" in url:
                raise ConnectionResetError("connection reset after server accepted pause")
            return _Resp()

        def close(self):
            pass

    client = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
    client._urls = ["http://server-0:8000"]
    client._dp = 1
    client._init_payload_fn = None
    client._timeout_s = None
    client._start_endpoint = START_UPDATE_ENDPOINT
    client._session = _PauseSession()
    client._pool = ThreadPoolExecutor(max_workers=1)

    try:
        with weight_sync_client_deadline(time.monotonic() + 1.0):
            with pytest.raises(ConnectionResetError, match="accepted pause"):
                client.pause_generation()
        assert seen_timeouts and isinstance(seen_timeouts[0], tuple) and seen_timeouts[0][1] is None
        unconfirmed = drain_inflight_weight_sync_requests(deadline_at=time.monotonic() + 0.1)
        assert any("/pause" in entry and "unconfirmed" in entry for entry in unconfirmed)
    finally:
        client.close()
        del client
        from skyrl.backends.skyrl_train.weight_sync import control_plane

        with control_plane._CONTROL_PLANE_LOCK:
            control_plane._ACTIVE_CLIENTS.clear()

    # Successful RPC finishing right as the deadline expires must NOT be marked unconfirmed.
    mode["fail_pause"] = False
    ok_client = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
    ok_client._urls = ["http://server-0:8000"]
    ok_client._dp = 1
    ok_client._init_payload_fn = None
    ok_client._timeout_s = None
    ok_client._start_endpoint = START_UPDATE_ENDPOINT
    ok_client._session = _PauseSession()
    ok_client._pool = ThreadPoolExecutor(max_workers=1)
    try:
        # Tight deadline where the POST completes cleanly right at the boundary.
        with weight_sync_client_deadline(time.monotonic() + 0.05):
            ok_client.finish_weight_update()
        assert drain_inflight_weight_sync_requests(deadline_at=time.monotonic() + 0.1) == []
        ok_client.start_weight_update()
    finally:
        ok_client.close()

    # Deferred /resume transport failure must record as unconfirmed so subsequent rounds cannot overlap it.
    release_stuck = threading.Event()
    deferred_attempted = threading.Event()
    deferred_timeouts = []

    class _DeferredFailSession:
        def post(self, url, json=None, timeout=None):
            if "/pause" in url:
                release_stuck.wait(timeout=5.0)
                return _Resp()
            if url.endswith("/resume"):
                deferred_timeouts.append(timeout)
                deferred_attempted.set()
                raise ConnectionResetError("deferred resume lost connection")
            return _Resp()

        def close(self):
            pass

    def_client = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
    def_client._urls = ["http://server-0:8000"]
    def_client._dp = 1
    def_client._init_payload_fn = None
    def_client._timeout_s = None
    def_client._start_endpoint = START_UPDATE_ENDPOINT
    def_client._session = _DeferredFailSession()
    def_client._pool = ThreadPoolExecutor(max_workers=1)
    try:
        with weight_sync_client_deadline(time.monotonic() + 0.1):
            with pytest.raises(TimeoutError, match="exceeded weight-sync phase deadline"):
                def_client.pause_generation()
        with weight_sync_client_deadline(time.monotonic() + 0.1):
            with pytest.raises(TimeoutError, match="Deferred /resume"):
                def_client.resume_generation()
        release_stuck.set()
        assert deferred_attempted.wait(timeout=2.0)
        # Wait for deferred_fut to finish recording unconfirmed transport error.
        time.sleep(0.05)
        assert deferred_timeouts and isinstance(deferred_timeouts[0], tuple) and deferred_timeouts[0][1] is None
        unconfirmed_resume = drain_inflight_weight_sync_requests(deadline_at=time.monotonic() + 0.2)
        assert any("/resume" in entry and "unconfirmed" in entry for entry in unconfirmed_resume)
    finally:
        release_stuck.set()
        def_client.close()


@pytest.mark.vllm
def test_delta_reload_failure_skips_resume_and_pre_reload_failure_resumes():
    """Delta sync must NOT resume after reload starts (weights may be incomplete), but MUST resume on pre-reload failures."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    from skyrl.backends.skyrl_train.weight_sync.control_plane import (
        START_UPDATE_ENDPOINT,
        SkyrlWeightSyncClient,
        weight_sync_client_deadline,
    )
    from skyrl.backends.skyrl_train.weight_sync.delta.trainer import (
        DeltaTrainerWeightTransferEngine,
    )

    class _Resp:
        status_code = 200
        reason = "OK"
        text = "{}"

        def json(self):
            return {"status": "ok", "success": True}

    # Case 1: Failure during start_weight_update / update_weights must NOT call /resume.
    seen_reload_fail = []

    class _ReloadFailSession:
        def post(self, url, json=None, timeout=None):
            endpoint = "/" + url.split("://", 1)[1].split("/", 1)[1]
            seen_reload_fail.append(endpoint)
            if endpoint == "/update_weights":
                raise TimeoutError("update_weights timed out on server")
            return _Resp()

        def close(self):
            pass

    client1 = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
    client1._urls = ["http://server-0:8000"]
    client1._dp = 1
    client1._init_payload_fn = None
    client1._timeout_s = None
    client1._start_endpoint = START_UPDATE_ENDPOINT
    client1._session = _ReloadFailSession()
    client1._pool = ThreadPoolExecutor(max_workers=1)

    engine1 = DeltaTrainerWeightTransferEngine.__new__(DeltaTrainerWeightTransferEngine)
    engine1.client = client1
    engine1._init_info = SimpleNamespace(sync_dir="/tmp/delta")
    engine1._reset_prefix_cache = False
    engine1.skyrl_draft_source = None

    try:
        with weight_sync_client_deadline(time.monotonic() + 0.8):
            with pytest.raises(TimeoutError, match="update_weights") as exc_info:
                engine1._apply_receiver_update({"target_version": 1, "sync_dir": "/tmp/delta"})
        assert "/pause?mode=keep&clear_cache=false" in seen_reload_fail
        assert "/update_weights" in seen_reload_fail
        assert "/resume" not in seen_reload_fail
        assert any("Skipping delta sync resume_generation" in note for note in getattr(exc_info.value, "__notes__", []))
        with pytest.raises(TimeoutError, match="unconfirmed"):
            client1.start_weight_update()
    finally:
        client1.close()

    # Case 2: Pre-reload failure (/reset_prefix_cache timeout exhausting update budget) MUST still /resume.
    seen_pre_reload_fail = []

    class _PreReloadFailSession:
        def post(self, url, json=None, timeout=None):
            endpoint = "/" + url.split("://", 1)[1].split("/", 1)[1]
            seen_pre_reload_fail.append(endpoint)
            if endpoint.startswith("/reset_prefix_cache"):
                time.sleep(min(timeout, 0.5) if isinstance(timeout, (int, float)) else 0.5)
                raise TimeoutError("reset_prefix_cache timed out")
            return _Resp()

        def close(self):
            pass

    client2 = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
    client2._urls = ["http://server-0:8000"]
    client2._dp = 1
    client2._init_payload_fn = None
    client2._timeout_s = None
    client2._start_endpoint = START_UPDATE_ENDPOINT
    client2._session = _PreReloadFailSession()
    client2._pool = ThreadPoolExecutor(max_workers=1)

    engine2 = DeltaTrainerWeightTransferEngine.__new__(DeltaTrainerWeightTransferEngine)
    engine2.client = client2
    engine2._init_info = SimpleNamespace(sync_dir="/tmp/delta")
    engine2._reset_prefix_cache = True
    engine2.skyrl_draft_source = None

    try:
        with weight_sync_client_deadline(time.monotonic() + 0.8):
            with pytest.raises(TimeoutError, match="reset_prefix_cache"):
                engine2._apply_receiver_update({"target_version": 1, "sync_dir": "/tmp/delta"})
        assert "/pause?mode=keep&clear_cache=false" in seen_pre_reload_fail
        assert "/resume" in seen_pre_reload_fail
    finally:
        client2.close()

    # Case 3: Persistent _reload_incomplete survives across retries after an HTTP 500 reload error
    # until a complete reload succeeds (a subsequent pre-reload pause failure must NOT resume).
    seen_retry = []
    retry_mode = {"phase": "fail_update_500"}

    class _RetryResp:
        def __init__(self, status_code=200, reason="OK", body=None):
            self.status_code = status_code
            self.reason = reason
            self._body = body if body is not None else {"status": "ok"}
            self.text = str(self._body)

        def json(self):
            return self._body

    class _PersistentReloadSession:
        def post(self, url, json=None, timeout=None):
            endpoint = "/" + url.split("://", 1)[1].split("/", 1)[1]
            seen_retry.append(endpoint)
            if retry_mode["phase"] == "fail_update_500" and endpoint == "/update_weights":
                return _RetryResp(500, "Internal Server Error", {"detail": "reload boom"})
            if retry_mode["phase"] == "fail_pause_500" and endpoint.startswith("/pause"):
                return _RetryResp(500, "Internal Server Error", {"detail": "pause boom"})
            return _RetryResp()

        def close(self):
            pass

    client3 = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
    client3._urls = ["http://server-0:8000"]
    client3._dp = 1
    client3._init_payload_fn = None
    client3._timeout_s = None
    client3._start_endpoint = START_UPDATE_ENDPOINT
    client3._session = _PersistentReloadSession()
    client3._pool = ThreadPoolExecutor(max_workers=1)

    engine3 = DeltaTrainerWeightTransferEngine.__new__(DeltaTrainerWeightTransferEngine)
    engine3.client = client3
    engine3._init_info = SimpleNamespace(sync_dir="/tmp/delta")
    engine3._reset_prefix_cache = False
    engine3.skyrl_draft_source = None

    try:
        # Round 1: HTTP 500 during update_weights -> leaves _reload_incomplete=True, skips /resume.
        with pytest.raises(RuntimeError, match="reload boom"):
            engine3._apply_receiver_update({"target_version": 1, "sync_dir": "/tmp/delta"})
        assert "/resume" not in seen_retry
        assert engine3._reload_incomplete is True

        # Round 2: Pre-reload pause failure -> MUST STILL skip /resume because Round 1 never completed.
        retry_mode["phase"] = "fail_pause_500"
        with pytest.raises(RuntimeError, match="pause boom"):
            engine3._apply_receiver_update({"target_version": 2, "sync_dir": "/tmp/delta"})
        assert "/resume" not in seen_retry
        assert engine3._reload_incomplete is True

        # Round 3: Complete reload succeeds -> clears _reload_incomplete and resumes generation.
        retry_mode["phase"] = "ok"
        engine3._apply_receiver_update({"target_version": 3, "sync_dir": "/tmp/delta"})
        assert engine3._reload_incomplete is False
        assert "/resume" in seen_retry
    finally:
        client3.close()


@pytest.mark.vllm
def test_delta_partial_pause_failure_still_resumes_all_servers():
    """If pause_generation succeeds on one server and fails on another, finally must still resume all servers."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    from skyrl.backends.skyrl_train.weight_sync.control_plane import (
        START_UPDATE_ENDPOINT,
        SkyrlWeightSyncClient,
        weight_sync_client_deadline,
    )
    from skyrl.backends.skyrl_train.weight_sync.delta.trainer import (
        DeltaTrainerWeightTransferEngine,
    )

    seen_calls = []

    class _Resp:
        def __init__(self, status_code=200, reason="OK", body=None):
            self.status_code = status_code
            self.reason = reason
            self._body = body if body is not None else {"status": "ok"}
            self.text = str(self._body)

        def json(self):
            return self._body

    class _PartialPauseSession:
        def post(self, url, json=None, timeout=None):
            seen_calls.append(url)
            if url.startswith("http://server-1:8000/pause"):
                return _Resp(500, "Internal Server Error", {"detail": "pause failed"})
            return _Resp()

        def close(self):
            pass

    client = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
    client._urls = ["http://server-0:8000", "http://server-1:8000"]
    client._dp = 1
    client._init_payload_fn = None
    client._timeout_s = None
    client._start_endpoint = START_UPDATE_ENDPOINT
    client._session = _PartialPauseSession()
    client._pool = ThreadPoolExecutor(max_workers=2)

    engine = DeltaTrainerWeightTransferEngine.__new__(DeltaTrainerWeightTransferEngine)
    engine.client = client
    engine._init_info = SimpleNamespace(sync_dir="/tmp/delta")
    engine._reset_prefix_cache = False
    engine.skyrl_draft_source = None

    try:
        with weight_sync_client_deadline(time.monotonic() + 2.0):
            with pytest.raises(RuntimeError, match="pause failed"):
                engine._apply_receiver_update({"target_version": 1, "sync_dir": "/tmp/delta"})
        assert "http://server-0:8000/resume" in seen_calls
        assert "http://server-1:8000/resume" in seen_calls
    finally:
        client.close()


@pytest.mark.vllm
def test_delta_resume_unpauses_healthy_servers_and_tracks_deferred_resume():
    """If one server's /pause stays stuck past cleanup deadline, healthy servers resume immediately and deferred /resume blocks new rounds until finished."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from skyrl.backends.skyrl_train.weight_sync.control_plane import (
        START_UPDATE_ENDPOINT,
        SkyrlWeightSyncClient,
        drain_inflight_weight_sync_requests,
        weight_sync_client_deadline,
    )
    from skyrl.backends.skyrl_train.weight_sync.delta.trainer import (
        DeltaTrainerWeightTransferEngine,
    )

    release_server1_pause = threading.Event()
    server1_resume_entered = threading.Event()
    release_server1_resume = threading.Event()
    seen_calls = []

    class _Resp:
        status_code = 200
        reason = "OK"
        text = "{}"

        def json(self):
            return {"status": "ok"}

    class _StuckOneServerPauseSession:
        def post(self, url, json=None, timeout=None):
            seen_calls.append(url)
            if url.startswith("http://server-1:8000/pause"):
                release_server1_pause.wait(timeout=5.0)
            if url == "http://server-1:8000/resume":
                server1_resume_entered.set()
                release_server1_resume.wait(timeout=5.0)
            return _Resp()

        def close(self):
            pass

    client = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
    client._urls = ["http://server-0:8000", "http://server-1:8000"]
    client._dp = 1
    client._init_payload_fn = None
    client._timeout_s = None
    client._start_endpoint = START_UPDATE_ENDPOINT
    client._session = _StuckOneServerPauseSession()
    client._pool = ThreadPoolExecutor(max_workers=2)

    engine = DeltaTrainerWeightTransferEngine.__new__(DeltaTrainerWeightTransferEngine)
    engine.client = client
    engine._init_info = SimpleNamespace(sync_dir="/tmp/delta")
    engine._reset_prefix_cache = False
    engine.skyrl_draft_source = None

    try:
        with weight_sync_client_deadline(time.monotonic() + 0.5):
            with pytest.raises(TimeoutError, match="pause"):
                engine._apply_receiver_update({"target_version": 1, "sync_dir": "/tmp/delta"})

        # Healthy server-0 must have been resumed synchronously during cleanup.
        assert "http://server-0:8000/resume" in seen_calls
        assert not server1_resume_entered.is_set()

        # Unblock server-1's /pause; deferred /resume starts, and while it is in flight,
        # a new round must STILL be blocked from overlapping it.
        release_server1_pause.set()
        assert server1_resume_entered.wait(timeout=2.0)
        still = drain_inflight_weight_sync_requests(deadline_at=time.monotonic())
        assert any("/resume" in entry for entry in still)

        # Once deferred /resume finishes, the registry clears cleanly.
        release_server1_resume.set()
        assert drain_inflight_weight_sync_requests(deadline_at=time.monotonic() + 2.0) == []
        assert "http://server-1:8000/resume" in seen_calls
    finally:
        release_server1_pause.set()
        release_server1_resume.set()
        client.drain_inflight(deadline_at=time.monotonic() + 2.0)
        client.close()


def _worker_skyrl_ipc_trainer_coordinates_failures(rank: int, world_size: int, store_path: str) -> None:
    from skyrl.backends.skyrl_train.weight_sync.weight_senders import (
        get_skyrl_ipc_trainer,
    )

    _init_gloo_group(rank, world_size, store_path)
    try:
        _, engine_cls = get_skyrl_ipc_trainer()
        engine = engine_cls.__new__(engine_cls)
        engine.is_sender = rank == 0
        engine.packed = True
        engine.skyrl_draft_source = None
        engine.source = SimpleNamespace()
        engine._skyrl_send_error = None

        fail_stage = {"stage": "start"}
        calls = []

        class _FakeIPCClient:
            def start_weight_update(self):
                calls.append("start")
                if fail_stage["stage"] == "start":
                    raise RuntimeError("IPC start_weight_update failed")

            def update_weights(self, update_info):
                calls.append(("update", update_info["names"]))
                if fail_stage["stage"] == "update":
                    raise RuntimeError("IPC update_weights failed on chunk 1")

            def finish_weight_update(self):
                calls.append("finish")
                if fail_stage["stage"] == "finish":
                    raise RuntimeError("IPC finish_weight_update failed")

        engine.client = _FakeIPCClient()

        # Simulate packed _send with two chunks and a real all_gather_object handle gather per chunk.
        def _fake_packed_send(source):
            for idx in range(2):
                gathered = [None] * world_size
                dist.all_gather_object(gathered, {f"gpu-{rank}": idx})
                merged_handles = {k: v for d in gathered for k, v in d.items()}
                engine._do_send(
                    names=[f"w{idx}"],
                    dtype_names=["float32"],
                    shapes=[[2, 2]],
                    ipc_handles=merged_handles,
                    tensor_sizes=[4],
                )
                engine._post_send_sync()
            return None

        engine._send = _fake_packed_send

        # 1) start_weight_update fails on rank 0 -> rank 1 never enters handle all_gather_object.
        if rank == 0:
            with pytest.raises(RuntimeError, match="IPC start_weight_update failed"):
                engine.send_weights()
        else:
            with pytest.raises(PeerRankSyncError, match="IPC start_weight_update"):
                engine.send_weights()

        # 2) update_weights fails on chunk 1 of 2 -> rank 1 exits at chunk 1's _post_send_sync.
        fail_stage["stage"] = "update"
        if rank == 0:
            with pytest.raises(RuntimeError, match="IPC update_weights failed"):
                engine.send_weights()
        else:
            with pytest.raises(PeerRankSyncError, match="IPC update_weights"):
                engine.send_weights()

        # 3) finish_weight_update fails on rank 0 -> rank 1 exits cleanly without barrier hang.
        fail_stage["stage"] = "finish"
        if rank == 0:
            with pytest.raises(RuntimeError, match="IPC finish_weight_update failed"):
                engine.send_weights()
        else:
            with pytest.raises(PeerRankSyncError, match="IPC finish_weight_update"):
                engine.send_weights()

        # 4) Full round succeeds on the same Gloo PG.
        fail_stage["stage"] = None
        engine.send_weights()
        if rank == 0:
            assert calls[-4:] == ["start", ("update", ["w0"]), ("update", ["w1"]), "finish"]
    finally:
        dist.destroy_process_group()


@pytest.mark.vllm
def test_skyrl_ipc_trainer_coordinates_start_update_finish_failures_on_gloo(tmp_path):
    _run_mp(_worker_skyrl_ipc_trainer_coordinates_failures, 2, tmp_path)


@pytest.mark.vllm
def test_delayed_finish_weight_update_does_not_disarm_newly_armed_tenant():
    """A delayed finish_weight_update from an earlier timed-out round must not disarm a newly armed tenant."""
    from skyrl.backends.skyrl_train.patches.vllm import (
        patch_lora_in_memory as lora_patch,
    )
    from skyrl.backends.skyrl_train.weight_sync.lora_target import (
        build_lora_receive_target,
    )
    from skyrl.backends.skyrl_train.weight_sync.weight_receivers import (
        SkyrlReceiveLifecycleMixin,
    )

    class _BaseEngine:
        def __init__(self):
            self.device = torch.device("cpu")
            self.vllm_config = SimpleNamespace(lora_config=object())
            self.model = SimpleNamespace(load_weights=lambda weights: set())

        def start_weight_update(self):
            pass

        def receive_weights(self, update_info):
            self.model.load_weights([("base_model.model.q_proj.lora_A.weight", torch.ones(2, 2))])

        def finish_weight_update(self):
            pass

    class _Receiver(SkyrlReceiveLifecycleMixin, _BaseEngine):
        pass

    receiver = _Receiver()
    # Tenant A arms and starts, then times out before finish completes; rollback disarms round r1.
    target_a = {**build_lora_receive_target("tenant-a", {}, {}), "round_id": "r1"}
    receiver.skyrl_set_lora_receive_target(target_a)
    receiver.start_weight_update()
    receiver.receive_weights(None)
    receiver.skyrl_clear_lora_receive_target(round_id="r1")
    assert receiver.skyrl_lora_armed() is False

    # Tenant B arms with round r2 (start_weight_update has not run yet).
    target_b = {**build_lora_receive_target("tenant-b", {}, {}), "round_id": "r2"}
    receiver.skyrl_set_lora_receive_target(target_b)
    assert receiver.skyrl_lora_armed() is True

    # Tenant A's delayed finish_weight_update now arrives on the server.
    # It must fail without disarming Tenant B.
    with pytest.raises(RuntimeError, match="before start_weight_update"):
        receiver.finish_weight_update()
    assert receiver.skyrl_lora_armed() is True
    assert receiver._skyrl_lora_target["lora_name"] == "tenant-b"

    # Tenant B's own full round now completes cleanly.
    try:
        receiver.start_weight_update()
        receiver.receive_weights(None)
        receiver.finish_weight_update()
        assert receiver.skyrl_lora_armed() is False
        assert "tenant-b" in lora_patch.staged_adapter_names()
    finally:
        lora_patch.discard_in_memory_adapter("tenant-b")


def _worker_skyrl_nccl_trainer_coordinates_failures(rank: int, world_size: int, store_path: str) -> None:
    from contextlib import contextmanager

    from vllm.distributed.weight_transfer.base import ParamMeta

    from skyrl.backends.skyrl_train.weight_sync.weight_senders import (
        get_skyrl_nccl_trainer,
    )

    _init_gloo_group(rank, world_size, store_path)
    try:
        _, engine_cls = get_skyrl_nccl_trainer()
        engine = engine_cls.__new__(engine_cls)
        engine.is_sender = rank == 0
        engine._sent_meta = None

        fail_stage = {"stage": "start"}
        calls = []
        iter_counts = {"main": 0, "draft": 0}

        class _CollectiveSource:
            def __init__(self, tag: str):
                self._tag = tag
                self._meta = [ParamMeta(f"{tag}.w0", torch.float32, (2, 2))]

            @property
            def metadata(self):
                return self._meta

            def iter_full(self):
                iter_counts[self._tag] += 1
                # Real Gloo collective standing in for FSDP/Megatron gather collectives.
                dist.barrier()
                yield f"{self._tag}.w0", torch.ones(2, 2)

        class _FakeNCCLClient:
            def __init__(self):
                self._in_draft = False

            @contextmanager
            def draft_session(self):
                self._in_draft = True
                try:
                    yield
                finally:
                    self._in_draft = False

            def start_weight_update(self):
                calls.append("draft_start" if self._in_draft else "start")
                if not self._in_draft and fail_stage["stage"] == "start":
                    raise RuntimeError("NCCL start_weight_update failed")

            def update_weights(self, update_info):
                calls.append(("update", update_info["names"]))
                if not self._in_draft and fail_stage["stage"] == "update_reject":
                    raise RuntimeError("NCCL update_weights rejected before broadcast")

            def finish_weight_update(self):
                calls.append("draft_finish" if self._in_draft else "finish")
                if not self._in_draft and fail_stage["stage"] == "finish":
                    raise RuntimeError("NCCL finish_weight_update failed")

        engine.client = _FakeNCCLClient()
        engine.source = _CollectiveSource("main")
        engine.skyrl_draft_source = _CollectiveSource("draft")

        # Drain source.iter_full() on both ranks (like real _broadcast) without needing CUDA NCCL.
        def _fake_broadcast(source, meta):
            for _ in source.iter_full():
                pass

        engine._broadcast = _fake_broadcast
        engine._post_send_sync = lambda: None

        # 1) Rank 0 fails start_weight_update -> rank 1 must abort BEFORE entering main iter_full() barrier.
        if rank == 0:
            with pytest.raises(RuntimeError, match="NCCL start_weight_update failed"):
                engine.send_weights()
        else:
            with pytest.raises(PeerRankSyncError, match="NCCL start_weight_update"):
                engine.send_weights()
        assert iter_counts["main"] == 0
        assert iter_counts["draft"] == 0

        # 2) Rank 0 /update_weights rejects before broadcast starts -> rank 1 aborts BEFORE entering main iter_full().
        fail_stage["stage"] = "update_reject"
        if rank == 0:
            with pytest.raises(RuntimeError, match="NCCL update_weights rejected"):
                engine.send_weights()
        else:
            with pytest.raises(PeerRankSyncError, match="NCCL update_weights|NCCL start_weight_update"):
                engine.send_weights()
        assert iter_counts["main"] == 0
        assert iter_counts["draft"] == 0

        # 3) Rank 0 fails main finish_weight_update -> rank 1 must abort BEFORE entering draft session.
        fail_stage["stage"] = "finish"
        if rank == 0:
            with pytest.raises(RuntimeError, match="NCCL finish_weight_update failed"):
                engine.send_weights()
        else:
            with pytest.raises(PeerRankSyncError, match="NCCL finish_weight_update"):
                engine.send_weights()
        assert iter_counts["main"] == 1
        assert iter_counts["draft"] == 0

        # 3) Full round (main + draft) succeeds on the same Gloo PG.
        fail_stage["stage"] = None
        engine.send_weights()
        assert iter_counts["main"] == 2
        assert iter_counts["draft"] == 1
        if rank == 0:
            assert calls[-6:] == [
                "start",
                ("update", ["main.w0"]),
                "finish",
                "draft_start",
                ("update", ["draft.w0"]),
                "draft_finish",
            ]
    finally:
        dist.destroy_process_group()


@pytest.mark.vllm
def test_skyrl_nccl_trainer_coordinates_start_and_finish_failures_on_gloo(tmp_path):
    _run_mp(_worker_skyrl_nccl_trainer_coordinates_failures, 2, tmp_path)


@pytest.mark.megatron
@pytest.mark.asyncio
async def test_megatron_in_memory_send_failure_with_inflight_rpc_defers_clear(tmp_path, monkeypatch):
    """When send_weights fails with an in-flight /start_weight_update, cleanup must NOT disarm receivers early."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from skyrl.backends.skyrl_train.inference_servers import (
        remote_inference_client as remote_inference_client_mod,
    )
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.backends.skyrl_train.weight_sync.control_plane import (
        START_UPDATE_ENDPOINT,
        SkyrlWeightSyncClient,
    )
    from skyrl.backends.skyrl_train.weight_sync.sources import LoraAdapterWeightSource
    from skyrl.backends.skyrl_train.weight_sync.weight_receivers import (
        SkyrlLoraStagingMixin,
    )

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "barrier", lambda: None)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(remote_inference_client_mod, "rank0_phase_timeout_s", lambda *a, **kw: 0.3)

    class _RealStagingReceiver(SkyrlLoraStagingMixin):
        def __init__(self):
            self.vllm_config = SimpleNamespace(lora_config=object())

    receiver = _RealStagingReceiver()
    release_start = threading.Event()
    delayed_start_saw_armed = []

    class _Resp:
        status_code = 200
        reason = "OK"
        text = "{}"

    class _DelayedStartSession:
        def post(self, url, json=None, timeout=None):
            if url.endswith("/start_weight_update") and not release_start.is_set():
                release_start.wait(timeout=5.0)
                # When the delayed /start_weight_update finally executes, the receiver MUST
                # still be LoRA-armed so it cannot fall through to a base-model reload.
                delayed_start_saw_armed.append(receiver.skyrl_lora_armed())
            return _Resp()

        def close(self):
            pass

    blocking_client = SkyrlWeightSyncClient.__new__(SkyrlWeightSyncClient)
    blocking_client._urls = ["http://server-0:8000"]
    blocking_client._dp = 1
    blocking_client._init_payload_fn = None
    blocking_client._timeout_s = None
    blocking_client._start_endpoint = START_UPDATE_ENDPOINT
    blocking_client._session = _DelayedStartSession()
    blocking_client._pool = ThreadPoolExecutor(max_workers=1)

    class _FakeLoraSource(LoraAdapterWeightSource):
        def __init__(self):
            super().__init__(dtype=torch.bfloat16, experts_per_shared_adapter=1)

        def export_adapter_stream(self):
            return [("base_model.model.q_proj.lora_A.weight", torch.ones(2, 4, dtype=torch.bfloat16))]

    class _Engine:
        skyrl_force_disable_expandable_segments = False

        def __init__(self, source):
            self.source = source

        def send_weights(self):
            blocking_client.start_weight_update()
            list(self.source.iter_full())
            receiver.skyrl_clear_lora_receive_target()

    worker = _make_megatron_lora_worker(str(tmp_path / "lora_sync"), sync_mode="memory")
    source = _FakeLoraSource()
    worker._lora_weight_source = source
    worker._weight_sync_engine = _Engine(source)
    ie_cfg = SimpleNamespace(enable_prefix_caching=False)

    clear_calls = []
    client = RemoteInferenceClient.__new__(RemoteInferenceClient)
    client.server_urls = ["http://server-0:8000"]

    async def _fake_call_server(url: str, endpoint: str, json=None, method="POST", params=None):
        method_name = json["method"]
        if method_name == "skyrl_set_lora_receive_target":
            receiver.skyrl_set_lora_receive_target(json["kwargs"]["receive_target"])
            return url, {"status": 200, "body": {}}
        if method_name == "skyrl_clear_lora_receive_target":
            clear_calls.append(json.get("kwargs"))
            receiver.skyrl_clear_lora_receive_target(round_id=(json.get("kwargs") or {}).get("round_id"))
            return url, {"status": 200, "body": {}}
        raise AssertionError(f"Unexpected method {method_name}")

    client._call_server = _fake_call_server
    client.load_lora_adapter = AsyncMock(return_value={})

    try:
        # Round 1: /start_weight_update exceeds phase budget while still running on the server.
        with pytest.raises(TimeoutError, match="exceeded weight-sync phase deadline") as exc_info:
            await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-a")

        # Cleanup must NOT have disarmed the receiver while /start_weight_update is still in flight.
        assert clear_calls == []
        assert receiver.skyrl_lora_armed() is True
        assert any("deferring target clear" in note for note in getattr(exc_info.value, "__notes__", []))

        # Delayed /start_weight_update now executes on the server: it MUST still see LoRA armed.
        release_start.set()
        blocking_client.drain_inflight(deadline_at=time.monotonic() + 2.0)
        assert delayed_start_saw_armed == [True]

        # Round 2: preflight disarms the dirty server now that the RPC settled, and tenant-b succeeds.
        await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-b")
        assert len(clear_calls) == 1
        assert receiver.skyrl_lora_armed() is False
        client.load_lora_adapter.assert_awaited_once_with("tenant-b", in_memory=True)

        # Round 3: cancel rank 0 mid-drain during send-failure cleanup. The old round MUST already be
        # marked dirty so Round 4 preflight-disarms it after settlement instead of hitting "already armed".
        release_start.clear()
        task = asyncio.create_task(worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-c"))
        await asyncio.sleep(0.35)
        task.cancel()
        with pytest.raises((asyncio.CancelledError, TimeoutError)):
            await task
        assert receiver.skyrl_lora_armed() is True
        release_start.set()
        blocking_client.drain_inflight(deadline_at=time.monotonic() + 2.0)

        await worker.broadcast_to_inference_engines(client, ie_cfg, model_id="tenant-d")
        assert receiver.skyrl_lora_armed() is False
        assert client.load_lora_adapter.await_count == 2
    finally:
        release_start.set()
        blocking_client.drain_inflight(deadline_at=time.monotonic() + 2.0)
        blocking_client.close()


@pytest.mark.asyncio
async def test_timed_out_load_lora_adapter_stays_tracked_and_blocks_overlap(monkeypatch):
    """A timed-out load_lora_adapter keeps its task tracked until server confirmation, blocking reload, unload, and arm."""
    from skyrl.backends.skyrl_train.inference_servers import (
        remote_inference_client as ric_mod,
    )
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        control_plane_timeout_budget,
    )

    client = RemoteInferenceClient.__new__(RemoteInferenceClient)
    client.server_urls = ["http://server-0:8000"]
    client._get_session = AsyncMock(return_value=object())

    release_load = asyncio.Event()
    load_calls = {"n": 0}
    seen_connect_only = []

    class _FakeResp:
        status = 200

        async def text(self):
            return "Success"

    class _SlowLoadCtx:
        async def __aenter__(self):
            load_calls["n"] += 1
            if load_calls["n"] == 1:
                await release_load.wait()
            return _FakeResp()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def _fake_request(session, method, url, *, connect_only=False, **kw):
        seen_connect_only.append(connect_only)
        return _SlowLoadCtx()

    monkeypatch.setattr(ric_mod, "_control_plane_request", _fake_request)

    # Round 1: load_lora_adapter times out within budget while server request keeps running.
    with control_plane_timeout_budget(0.1):
        with pytest.raises(InferenceServerTimeoutError, match="load_lora_adapter"):
            await client.load_lora_adapter("tenant-a", "/tmp/lora/tenant-a")
    assert seen_connect_only == [True]
    assert load_calls["n"] == 1

    # While Round 1's load is still running on the server, unload / reload / arm MUST all be blocked.
    with control_plane_timeout_budget(0.1):
        with pytest.raises(InferenceServerTimeoutError, match="still in flight"):
            await client.unload_lora_adapter("tenant-a")
        with pytest.raises(InferenceServerTimeoutError, match="still in flight"):
            await client.load_lora_adapter("tenant-a", "/tmp/lora/tenant-a")
        with pytest.raises(InferenceServerTimeoutError, match="prior load_lora_adapter"):
            await client.set_lora_receive_target({"lora_name": "tenant-b"})
    assert load_calls["n"] == 1

    # Once Round 1's server load actually finishes, Round 2 succeeds cleanly.
    release_load.set()
    with control_plane_timeout_budget(2.0):
        resp = await client.load_lora_adapter("tenant-a", "/tmp/lora/tenant-a")
    assert "http://server-0:8000" in resp
    assert load_calls["n"] == 2


def test_overlapping_deadline_scopes_exit_out_of_order_without_corrupting_budgets():
    """Overlapping weight_sync_client_deadline scopes exiting out of order must not pop each other's budgets."""
    import threading
    import time

    from skyrl.backends.skyrl_train.weight_sync.control_plane import (
        current_weight_sync_deadline,
        weight_sync_client_deadline,
    )

    a_entered = threading.Event()
    b_entered = threading.Event()
    a_exited = threading.Event()
    observed = {}

    t_now = time.monotonic()
    deadline_a = t_now + 5.0
    deadline_b = t_now + 20.0

    def _thread_a():
        with weight_sync_client_deadline(deadline_a):
            a_entered.set()
            assert b_entered.wait(timeout=2.0)
            observed["a_during"] = current_weight_sync_deadline()
        a_exited.set()

    def _thread_b():
        assert a_entered.wait(timeout=2.0)
        with weight_sync_client_deadline(deadline_b):
            b_entered.set()
            observed["b_before_a_exit"] = current_weight_sync_deadline()
            assert a_exited.wait(timeout=2.0)
            observed["b_after_a_exit"] = current_weight_sync_deadline()
            # Unscoped worker thread should see B's remaining deadline, not None or A's.
            worker_seen = []
            wt = threading.Thread(target=lambda: worker_seen.append(current_weight_sync_deadline()))
            wt.start()
            wt.join(timeout=2.0)
            observed["worker_after_a_exit"] = worker_seen[0]

    ta = threading.Thread(target=_thread_a)
    tb = threading.Thread(target=_thread_b)
    ta.start()
    tb.start()
    ta.join(timeout=3.0)
    tb.join(timeout=3.0)

    assert observed["a_during"] == deadline_a
    assert observed["b_before_a_exit"] == deadline_b
    assert observed["b_after_a_exit"] == deadline_b
    assert observed["worker_after_a_exit"] == deadline_b
    assert current_weight_sync_deadline() is None


@pytest.mark.vllm
def test_monotonic_round_fence_never_evicts_old_cancelled_arm():
    """Per-client monotonic <client_id>:<seq> tokens reject stale arms even after >1024 later cancellations."""
    from skyrl.backends.skyrl_train.weight_sync.lora_target import (
        build_lora_receive_target,
    )
    from skyrl.backends.skyrl_train.weight_sync.weight_receivers import (
        SkyrlLoraStagingMixin,
    )

    class _Receiver(SkyrlLoraStagingMixin):
        def __init__(self):
            self.vllm_config = SimpleNamespace(lora_config=object())

    recv = _Receiver()
    client_id = "trainerclient1"
    stale_target = {**build_lora_receive_target("tenant-stale", {}, {}), "round_id": f"{client_id}:1"}

    # Cancel round 1, then cancel 1,200 newer rounds (exceeding the old 1,024 tombstone cap).
    recv.skyrl_clear_lora_receive_target(round_id=f"{client_id}:1")
    for seq in range(2, 1205):
        recv.skyrl_clear_lora_receive_target(round_id=f"{client_id}:{seq}")

    # Round 1 must STILL be rejected forever.
    with pytest.raises(RuntimeError, match="already cancelled or superseded"):
        recv.skyrl_set_lora_receive_target(stale_target)
    assert recv.skyrl_lora_armed() is False

    # Round 1205 succeeds.
    fresh_target = {**build_lora_receive_target("tenant-fresh", {}, {}), "round_id": f"{client_id}:1205"}
    recv.skyrl_set_lora_receive_target(fresh_target)
    assert recv.skyrl_lora_armed() is True
