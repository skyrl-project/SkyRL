"""Tests for receive-side packed-buffer memory lifecycle in ``weight_sync/weight_receivers.py``.

vLLM 0.33's ``packed_nccl_broadcast_consumer`` unpacks each chunk as views into a multi-GB
``packed`` buffer. When ``layerwise`` reload buffers a small tensor across chunks (padded
layers like Qwen4Exp's ``GatedResidual.input_mix_weight_down_block_inject``, deferred
attention layers, or layers split across chunk boundaries), that view would otherwise pin the
entire chunk buffer until ``finish_weight_update``.
"""

from __future__ import annotations

import gc
import inspect
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync import weight_receivers
from skyrl.backends.skyrl_train.weight_sync.checkpoint_shapes import max_param_numel
from skyrl.backends.skyrl_train.weight_sync.weight_receivers import (
    SkyrlReceiveLifecycleMixin,
    detach_buffered_layerwise_views,
)


def _bind_loader(param, loaded_weight, *extra, **kwargs):
    def _loader(param, loaded_weight, *extra, **kwargs):
        return (param, loaded_weight, extra, kwargs)

    sig = inspect.signature(_loader)
    bound = sig.bind(param, loaded_weight, *extra, **kwargs)
    bound.apply_defaults()
    return bound


def _fake_layerwise_info(entries):
    return SimpleNamespace(loaded_weights=list(entries))


def _tracked_uint8_buffer(nbytes: int, freed_log: list[int], tag: int, *, fill_seed: int = 0) -> torch.Tensor:
    """Create a uint8 tensor backed by a numpy array whose deallocation appends ``tag`` to ``freed_log``.

    Unlike ``weakref.ref(tensor)`` (which only observes the Python tensor wrapper and clears as
    soon as ``tensor`` is deleted even while a dtype-reinterpreted view still pins the backing
    storage), ``weakref.finalize`` on the numpy backing buffer fires only when the underlying
    ``UntypedStorage`` itself is freed.
    """
    rng = np.random.default_rng(fill_seed)
    raw = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
    weakref.finalize(raw, freed_log.append, tag)
    return torch.from_numpy(raw)


class _FakeNcclChunkEngine(SkyrlReceiveLifecycleMixin):
    """Drives ``skyrl_checkpoint_load`` across multiple packed chunks like vLLM 0.33 NCCL."""

    def __init__(self, model, chunk_producer):
        self.device = "cpu"
        self.model = model
        self._chunk_producer = chunk_producer

    def receive_weights(self, update_info):
        with torch.device(self.device), self.skyrl_checkpoint_load():
            it = iter(self._chunk_producer())
            while True:
                try:
                    unpacked = next(it)
                except StopIteration:
                    break
                self.model.load_weights(unpacked)
                # Drop our reference to the chunk's views before resuming the producer,
                # matching packed_nccl_broadcast_consumer.receive_and_load returning.
                del unpacked


def test_packed_chunks_released_when_layerwise_reload_buffers_padded_layers(monkeypatch):
    """Each packed chunk's backing storage must be freed immediately after ``load_weights`` even
    when a padded layer buffers a small view of that chunk until ``finish_weight_update``."""
    layerwise_info: dict[object, SimpleNamespace] = {}
    monkeypatch.setattr(weight_receivers, "_get_layerwise_info_map", lambda: layerwise_info)

    expected_hc_values: dict[int, torch.Tensor] = {}
    freed_chunks: list[int] = []

    class _PaddedLayerModel:
        def load_weights(self, weights=None, **kwargs):
            for name, tensor in weights:
                if "hyper_connection" in name:
                    layer_idx = int(name.split(".")[1])
                    layer_key = f"hc_layer_{layer_idx}"
                    meta_param = torch.empty((8, 4), dtype=torch.bfloat16, device="meta")
                    bound = _bind_loader(meta_param, tensor)
                    layerwise_info.setdefault(layer_key, _fake_layerwise_info([])).loaded_weights.append(
                        ("weight", bound)
                    )
                # Large completed layers (like FusedMoE) copy and discard their view immediately.
            return {name for name, _ in weights}

        def named_parameters(self):
            return iter(())

    def chunk_producer():
        for layer_idx in range(4):
            # Simulate a large packed chunk: 4096 bytes of "experts" + 32 bytes of "hyper_connection".
            packed = _tracked_uint8_buffer(4096 + 32, freed_chunks, layer_idx, fill_seed=layer_idx)
            expert_view = packed[:4096].view(torch.bfloat16).view(32, 64)
            hc_view = packed[4096:].view(torch.bfloat16).view(4, 4)
            expected_hc_values[layer_idx] = hc_view.clone()

            yield [
                (f"layers.{layer_idx}.mlp.experts.gate_up_proj", expert_view),
                (f"layers.{layer_idx}.attn_hyper_connection.input_mix_weight_down.weight", hc_view),
            ]
            # Drop producer refs and verify the backing storage itself was freed (not pinned).
            del expert_view, hc_view, packed
            gc.collect()
            assert layer_idx in freed_chunks, f"chunk {layer_idx} backing storage remained pinned"

    engine = _FakeNcclChunkEngine(_PaddedLayerModel(), chunk_producer)
    engine.receive_weights(None)

    assert freed_chunks == [0, 1, 2, 3]

    # Every buffered HC tensor kept its exact bytes in a tight standalone storage for finalize replay.
    for layer_idx in range(4):
        info = layerwise_info[f"hc_layer_{layer_idx}"]
        assert len(info.loaded_weights) == 1
        _, bound = info.loaded_weights[0]
        buffered = bound.arguments["loaded_weight"]
        assert buffered.untyped_storage().nbytes() == buffered.nbytes
        assert buffered.shape == expected_hc_values[layer_idx].shape
        assert buffered.dtype == expected_hc_values[layer_idx].dtype
        assert torch.equal(buffered.view(torch.uint8), expected_hc_values[layer_idx].view(torch.uint8))
        assert torch.equal(bound.args[1].view(torch.uint8), expected_hc_values[layer_idx].view(torch.uint8))


def test_negative_control_unpatched_view_pins_backing_storage():
    """Confirm our storage-lifetime harness detects a pinned backing buffer when a view is kept."""
    freed: list[int] = []
    packed = _tracked_uint8_buffer(4096 + 32, freed, 0)
    hc_view = packed[4096:].view(torch.bfloat16).view(4, 4)
    bound = _bind_loader(None, hc_view)
    del hc_view, packed
    gc.collect()
    assert freed == [], "backing storage must remain alive while bound_args still holds a view"

    detach_buffered_layerwise_views({"hc": _fake_layerwise_info([("weight", bound)])})
    gc.collect()
    assert freed == [0]


def test_large_buffered_tensor_is_not_redundantly_cloned_while_small_sibling_is():
    """A pending tensor occupying >50% of its chunk stays in place (avoiding a 2x memory spike),
    while a small sibling view in the same chunk is detached so the chunk frees once the large
    layer completes."""
    freed: list[int] = []
    packed = _tracked_uint8_buffer(1000, freed, 0)

    large_view = packed[:800].view(torch.bfloat16)  # 80% of chunk
    small_view = packed[800:832].view(torch.bfloat16)  # 3.2% of chunk
    large_ptr = large_view.data_ptr()
    expected_small = small_view.clone()

    layerwise_info = {
        "moe_layer": _fake_layerwise_info([("w13", _bind_loader(None, large_view))]),
        "hc_layer": _fake_layerwise_info([("weight", _bind_loader(None, small_view))]),
    }

    cloned = detach_buffered_layerwise_views(layerwise_info)

    assert len(cloned) == 1
    # Large tensor kept in place without doubling.
    assert layerwise_info["moe_layer"].loaded_weights[0][1].arguments["loaded_weight"].data_ptr() == large_ptr
    # Small tensor detached into tight storage with identical values.
    detached_small = layerwise_info["hc_layer"].loaded_weights[0][1].arguments["loaded_weight"]
    assert detached_small.untyped_storage().nbytes() == detached_small.nbytes
    assert torch.equal(detached_small, expected_small)

    # Storage stays alive while large_view is still pending, then frees as soon as it completes.
    del large_view, small_view, packed
    gc.collect()
    assert freed == []
    layerwise_info["moe_layer"].loaded_weights.clear()
    gc.collect()
    assert freed == [0]


def test_empty_view_into_large_chunk_is_detached():
    """Even a 0-element slice of a packed chunk holds a reference to its storage and must be detached."""
    freed: list[int] = []
    packed = _tracked_uint8_buffer(2048, freed, 0)
    empty_view = packed[:0].view(torch.bfloat16)

    layerwise_info = {
        "empty_layer": _fake_layerwise_info([("w", _bind_loader(None, empty_view))]),
    }

    del empty_view, packed
    gc.collect()
    assert freed == [], "empty view must pin storage before detach"

    cloned = detach_buffered_layerwise_views(layerwise_info)
    assert len(cloned) == 1
    gc.collect()
    assert freed == [0]
    # Second pass is a no-op because the detached 0-byte tensor has 0-byte storage.
    assert detach_buffered_layerwise_views(layerwise_info) == []


def test_nested_bound_arguments_meta_params_and_eager_none_entries():
    """Handles ``*args`` tuples, ``**kwargs`` dicts, list values, meta ``param``, and ``None`` entries."""
    freed: list[int] = []
    packed = _tracked_uint8_buffer(1024, freed, 0)

    meta_param = torch.empty((4,), dtype=torch.bfloat16, device="meta")
    v1 = packed[:16].view(torch.bfloat16)
    v2 = packed[16:32].view(torch.bfloat16)
    v3 = packed[32:48].view(torch.bfloat16)
    expected = (v1.clone(), v2.clone(), v3.clone())

    bound = _bind_loader(meta_param, [v1], v2, extra_tensor=v3)
    layerwise_info = {
        "nested": _fake_layerwise_info([("weight", bound), ("eager_marker", None)]),
    }

    cloned = detach_buffered_layerwise_views(layerwise_info)
    assert len(cloned) == 3

    del v1, v2, v3, packed
    gc.collect()
    assert freed == [0]
    assert bound.arguments["param"] is meta_param
    assert torch.equal(bound.arguments["loaded_weight"][0], expected[0])
    assert torch.equal(bound.args[2], expected[1])
    assert torch.equal(bound.kwargs["extra_tensor"], expected[2])


def test_max_param_numel_skips_qwen4_ple_embedding_tables(tmp_path):
    """Qwen4Exp's frozen PLE n-gram table is never synced and must not inflate the packed buffer."""
    from safetensors.torch import save_file

    save_file(
        {
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": torch.zeros(128, 16),
            "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets": torch.zeros(16, dtype=torch.int64),
            "model.language_model.layers.1.ple.key_proj.weight": torch.zeros(12, 8),
            "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.zeros(4, 8, 8),
        },
        str(tmp_path / "model.safetensors"),
    )
    assert max_param_numel(str(tmp_path)) == 4 * 8 * 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_alternating_streams_multi_receive_and_deferred_finalization(monkeypatch):
    """Exercise CUDA event ordering, cross-stream ``record_stream``, cross-chunk consumption on
    alternating streams, multiple ``receive_weights`` calls within one update round, and
    post-finalization allocator churn without accumulating packed chunk memory."""
    layerwise_info: dict[object, SimpleNamespace] = {}
    monkeypatch.setattr(weight_receivers, "_get_layerwise_info_map", lambda: layerwise_info)
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.patches.vllm.patch_moe_kernel_constants.restore_moe_kernel_constants",
        lambda _model: None,
    )
    monkeypatch.setattr(weight_receivers, "skyrl_after_weight_update", lambda: None)
    monkeypatch.setattr(weight_receivers, "empty_cuda_cache_rocm", lambda: None)

    device = torch.device("cuda:0")
    chunk_expert_bytes = 64 * 1024 * 1024  # 64 MiB per chunk
    chunk_hc_bytes = 4 * 1024  # 4 KiB padded-layer slice per chunk
    chunk_split_bytes = 4 * 1024  # 4 KiB split-layer slice consumed on the next chunk's stream
    num_chunks_per_call = 6
    expected_hc: dict[int, torch.Tensor] = {}
    expected_split: dict[int, torch.Tensor] = {}
    replayed_hc: dict[int, torch.Tensor] = {}
    consumed_split: dict[int, torch.Tensor] = {}

    class _CudaFakeBaseEngine:
        def __init__(self, model):
            self.device = device
            self.model = model
            self._streams = (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device))

        def start_weight_update(self):
            layerwise_info.clear()

        def receive_weights(self, update_info):
            start_idx = update_info["start_idx"]
            caller_stream = torch.cuda.current_stream(device)
            prev_received = None
            for offset in range(num_chunks_per_call):
                idx = start_idx + offset
                stream = self._streams[offset % 2]
                stream.synchronize()
                with torch.cuda.stream(stream):
                    packed = torch.empty(
                        chunk_expert_bytes + chunk_hc_bytes + chunk_split_bytes, dtype=torch.uint8, device=device
                    )
                    hc_val = float((idx % 17) + 1)
                    split_val = float((idx % 13) + 21)
                    hc_end = chunk_expert_bytes + chunk_hc_bytes
                    packed[chunk_expert_bytes:hc_end].view(torch.bfloat16).fill_(hc_val)
                    packed[hc_end:].view(torch.bfloat16).fill_(split_val)
                    # Build expected tensors on CPU directly so no blocking D2H sync runs in the loop.
                    expected_hc[idx] = torch.full((chunk_hc_bytes // 2,), hc_val, dtype=torch.bfloat16)
                    expected_split[idx] = torch.full((chunk_split_bytes // 2,), split_val, dtype=torch.bfloat16)
                    if prev_received is not None:
                        stream.wait_event(prev_received)
                    prev_received = torch.cuda.Event()
                    prev_received.record(stream)
                    for s in (*self._streams, caller_stream):
                        packed.record_stream(s)
                    # Delay after prev_received so this stream's detached clone is still pending
                    # when the alternate stream enters load_weights on the next iteration.
                    torch.cuda._sleep(2_000_000)
                    expert_view = packed[:chunk_expert_bytes].view(torch.bfloat16)
                    hc_view = packed[chunk_expert_bytes:hc_end].view(torch.bfloat16)
                    split_view = packed[hc_end:].view(torch.bfloat16)
                    self.model.load_weights(
                        [(f"l{idx}.experts", expert_view), (f"l{idx}.hc", hc_view), (f"l{idx}.split", split_view)]
                    )
                    del expert_view, hc_view, split_view, packed
            if update_info.get("sync_streams", True):
                for stream in self._streams:
                    stream.synchronize()

        def finish_weight_update(self):
            # Delay copies on the caller stream, drop clone refs immediately, then churn
            # same-sized allocations on the producer streams before syncing to exercise
            # record_stream(caller_stream) protection against premature block reuse.
            torch.cuda._sleep(2_000_000)
            for key, info in list(layerwise_info.items()):
                if not info.loaded_weights:
                    continue
                _, bound = info.loaded_weights[0]
                src = bound.arguments["loaded_weight"]
                dst = torch.empty_like(src)
                dst.copy_(src)
                if key[0] == "hc":
                    replayed_hc[key[1]] = dst
                else:
                    consumed_split[key[1]] = dst
                info.loaded_weights.clear()
            layerwise_info.clear()
            for stream in self._streams:
                with torch.cuda.stream(stream):
                    for _ in range(16):
                        churn = torch.full((chunk_hc_bytes // 2,), -99.0, dtype=torch.bfloat16, device=device)
                        del churn

    class _CudaModel:
        def load_weights(self, weights=None, **kwargs):
            for name, tensor in weights:
                idx = int(name[1:].split(".")[0])
                if name.endswith(".hc"):
                    layerwise_info[("hc", idx)] = _fake_layerwise_info([("weight", _bind_loader(None, tensor))])
                elif name.endswith(".split"):
                    # Consume the previous chunk's detached clone on this alternate stream, then
                    # buffer this chunk's split tensor for the next chunk's stream.
                    prev_key = ("split", idx - 1)
                    if prev_key in layerwise_info and layerwise_info[prev_key].loaded_weights:
                        _, prev_bound = layerwise_info[prev_key].loaded_weights[0]
                        prev_src = prev_bound.arguments["loaded_weight"]
                        dst = torch.empty_like(prev_src)
                        dst.copy_(prev_src)
                        consumed_split[idx - 1] = dst
                        layerwise_info[prev_key].loaded_weights.clear()
                    layerwise_info[("split", idx)] = _fake_layerwise_info([("weight", _bind_loader(None, tensor))])
            return {name for name, _ in weights}

        def named_parameters(self):
            return iter(())

    class _Engine(SkyrlReceiveLifecycleMixin, _CudaFakeBaseEngine):
        pass

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)

    engine = _Engine(_CudaModel())
    engine.start_weight_update()
    # Two separate receive_weights calls in one round (12 x 64 MiB = 768 MiB streamed total).
    # Leave streams unsynchronized across the boundary so cross-call and finalization waits are exercised.
    engine.receive_weights({"start_idx": 0, "sync_streams": False})
    engine.receive_weights({"start_idx": num_chunks_per_call, "sync_streams": False})
    peak_delta = torch.cuda.max_memory_allocated(device) - baseline
    engine.finish_weight_update()
    torch.cuda.synchronize(device)

    # Without detaching views, 12 chunks x 64 MiB = 768 MiB would stay pinned. With detaching,
    # peak delta stays bounded to the 2-buffer receive ring (~128-200 MiB), well under half of 768 MiB.
    assert peak_delta < 320 * 1024 * 1024, f"expected bounded peak delta, got {peak_delta / 1024**2:.1f} MiB"
    assert len(replayed_hc) == 2 * num_chunks_per_call
    assert len(consumed_split) == 2 * num_chunks_per_call
    for idx, exp in expected_hc.items():
        assert torch.equal(replayed_hc[idx].cpu(), exp)
    for idx, exp in expected_split.items():
        assert torch.equal(consumed_split[idx].cpu(), exp)


@pytest.mark.vllm
def test_nccl_finish_weight_update_empties_cuda_cache(monkeypatch):
    """``SkyrlNCCLWeightTransferEngine.finish_weight_update`` releases cached receive-stream buffers."""
    pytest.importorskip("vllm", reason="SkyrlNCCLWeightTransferEngine subclasses vLLM's NCCL engine")
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

    from skyrl.backends.skyrl_train.weight_sync.weight_receivers import (
        get_skyrl_nccl_engine,
    )

    calls: list[str] = []
    monkeypatch.setattr(NCCLWeightTransferEngine, "finish_weight_update", lambda self: calls.append("super_finish"))
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.patches.vllm.patch_moe_kernel_constants.restore_moe_kernel_constants",
        lambda _model: None,
    )
    monkeypatch.setattr(weight_receivers, "skyrl_after_weight_update", lambda: None)
    monkeypatch.setattr(weight_receivers, "empty_cuda_cache_rocm", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))

    vllm_config = SimpleNamespace(parallel_config=None, model_config=None, lora_config=None)
    engine = get_skyrl_nccl_engine()(None, vllm_config, torch.device("cpu"), SimpleNamespace())
    engine.finish_weight_update()

    assert calls == ["super_finish", "empty_cache"]
