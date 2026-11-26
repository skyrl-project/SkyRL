"""SkyRL's receive-side weight-transfer engines (the inference-worker half).

vLLM's NCCL and IPC engines, subclassed to add two things that hang off the
same seam: the engines call ``self.model.load_weights(...)`` directly with no
callback, so the only injection point is the ``self.model`` handle they read.
That handle is whichever model the session targets -- the main model, or the
spec-decode drafter under ``/start_draft_weight_update``.

* **Checkpoint loading.** Split compact batched-MoE FP8 wire tensors and stacked
  NVFP4 expert layers from ordinary checkpoint weights before calling the target
  model's loader.
* **LoRA staging.** When the round has been armed with a LoRA receive target
  (``weight_sync/lora_target.py``), the stream is a PEFT adapter rather than the
  model: the tensors are staged for vLLM's LoRA manager instead of loaded, and
  the base model is not touched at all.

Registered (in ``weight_sync/register.py``) under ``skyrl_nccl`` / ``skyrl_ipc``
rather than shadowing vLLM's
``nccl`` / ``ipc``: ``register_engine`` raises on a duplicate name, and
``WeightTransferConfig.backend`` is typed ``Literal[...] | str`` and validated
against the registry, so a new name is all that is needed.
"""

from __future__ import annotations

import logging
import weakref
from contextlib import contextmanager
from typing import Any, Iterator

import torch

logger = logging.getLogger(__name__)

SKYRL_NCCL_BACKEND = "skyrl_nccl"
SKYRL_IPC_BACKEND = "skyrl_ipc"


def empty_cuda_cache_rocm() -> None:
    """Release unused ROCm cached blocks after a full-weight sync.

    ROCm's allocator does not return the reload's transient blocks on its own.
    """
    if torch.version.hip is None or not torch.cuda.is_available():
        return
    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


class _LoadWeightsProxy:
    """Wraps a model, overriding only ``load_weights``.

    Every other attribute access falls through to the real model.
    """

    def __init__(self, model: Any, load_weights: Any) -> None:
        self._model = model
        self.load_weights = load_weights

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not set on the proxy itself.
        return getattr(self._model, name)


def _get_layerwise_info_map() -> Any:
    """Return vLLM's ``LAYERWISE_INFO`` mapping, or ``None`` if unavailable."""
    try:
        from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO
    except ImportError:
        return None
    return LAYERWISE_INFO


def _detach_oversized_storage_views(obj: Any, cloned: list[torch.Tensor]) -> Any:
    """Clone tensors that pin a much larger underlying storage (e.g. a packed NCCL chunk).

    vLLM 0.33's ``packed_nccl_broadcast_consumer`` unpacks each chunk as views into one
    multi-GB ``packed`` buffer (unlike ``packed_ipc_consumer``, which clones per tensor).
    When ``layerwise`` reload retains any tensor across chunks -- e.g. padded layers
    like Qwen4Exp's ``GatedResidual.input_mix_weight_down_block_inject`` (``pad_size=12``),
    deferred attention layers, or layers split across chunk boundaries -- that small view
    pins the entire multi-GB chunk buffer until ``finalize_layerwise_processing`` runs in
    ``/finish_weight_update``.

    Detaching whenever ``storage_bytes >= 2 * tensor.nbytes`` frees the oversized chunk
    buffer immediately while avoiding a redundant copy of a large tensor that already
    occupies more than half of its chunk.
    """
    if isinstance(obj, torch.Tensor):
        if obj.is_meta:
            return obj
        try:
            storage_bytes = obj.untyped_storage().nbytes()
        except Exception:  # noqa: BLE001 - tensor subclasses without untyped storage
            return obj
        if storage_bytes > 0 and storage_bytes >= 2 * obj.nbytes:
            copy = obj.detach().clone()
            cloned.append(copy)
            return copy
        return obj
    if isinstance(obj, tuple):
        changed = False
        items = []
        for item in obj:
            new_item = _detach_oversized_storage_views(item, cloned)
            if new_item is not item:
                changed = True
            items.append(new_item)
        return tuple(items) if changed else obj
    if isinstance(obj, list):
        changed = False
        items = []
        for item in obj:
            new_item = _detach_oversized_storage_views(item, cloned)
            if new_item is not item:
                changed = True
            items.append(new_item)
        return items if changed else obj
    if isinstance(obj, dict):
        changed = False
        out = {}
        for k, v in obj.items():
            new_v = _detach_oversized_storage_views(v, cloned)
            if new_v is not v:
                changed = True
            out[k] = new_v
        return out if changed else obj
    return obj


def detach_buffered_layerwise_views(layerwise_info: Any = None) -> list[torch.Tensor]:
    """Detach any views buffered in vLLM's ``LAYERWISE_INFO`` that pin a larger chunk storage.

    Returns the newly cloned tensors so the caller can synchronize CUDA streams when
    packed receive alternates across multiple streams.
    """
    if layerwise_info is None:
        layerwise_info = _get_layerwise_info_map()
    if not layerwise_info:
        return []
    cloned: list[torch.Tensor] = []
    for info in list(layerwise_info.values()):
        loaded_weights = getattr(info, "loaded_weights", None)
        if not loaded_weights:
            continue
        for entry in loaded_weights:
            if not isinstance(entry, tuple) or len(entry) < 2:
                continue
            bound_args = entry[1]
            arguments = getattr(bound_args, "arguments", None)
            if not isinstance(arguments, dict):
                continue
            for key, val in list(arguments.items()):
                if key == "param":
                    continue
                new_val = _detach_oversized_storage_views(val, cloned)
                if new_val is not val:
                    arguments[key] = new_val
    return cloned


class _DetachedCloneStreamTracker:
    """Round-scoped CUDA stream ordering and allocator-lifetime tracking for detached views.

    vLLM 0.33's packed NCCL consumer alternates across multiple receive streams and records
    its inter-stream event *before* calling ``load_weights``; later, ``finish_weight_update``
    replays any remaining buffered tensors on the caller stream. This tracker lives for the
    whole update round (across one or more ``receive_weights`` calls through
    ``finish_weight_update``) so every consumer stream waits for async clones from other
    streams and is registered via ``record_stream`` before any clone can be freed.
    """

    def __init__(self) -> None:
        self._last_clone_events: dict[int, Any] = {}
        self._waited_events: dict[tuple[int, int], Any] = {}
        self._seen_streams: dict[int, Any] = {}
        self._detached_refs: list[weakref.ref[torch.Tensor]] = []

    def prepare_stream(self, device: Any) -> None:
        """Order ``current_stream(device)`` after other streams' clones and register it."""
        if not torch.cuda.is_available():
            return
        dev = torch.device(device)
        if dev.type != "cuda":
            return
        cur_stream = torch.cuda.current_stream(dev)
        cur_id = cur_stream.cuda_stream
        for producer_id, ev in self._last_clone_events.items():
            if producer_id != cur_id and self._waited_events.get((cur_id, producer_id)) is not ev:
                cur_stream.wait_event(ev)
                self._waited_events[(cur_id, producer_id)] = ev
        if cur_id not in self._seen_streams:
            self._seen_streams[cur_id] = cur_stream
            alive: list[weakref.ref[torch.Tensor]] = []
            for ref in self._detached_refs:
                tensor = ref()
                if tensor is not None:
                    tensor.record_stream(cur_stream)
                    alive.append(ref)
            self._detached_refs[:] = alive

    def record_clones(self, tensors: list[torch.Tensor]) -> None:
        """Record a completion event on the producing stream and register all known streams."""
        cuda_tensors = [t for t in tensors if t.is_cuda]
        if not cuda_tensors:
            return
        cur_stream = torch.cuda.current_stream(cuda_tensors[0].device)
        cur_id = cur_stream.cuda_stream
        self._seen_streams.setdefault(cur_id, cur_stream)
        ev = torch.cuda.Event()
        ev.record(cur_stream)
        self._last_clone_events[cur_id] = ev
        for tensor in cuda_tensors:
            self._detached_refs.append(weakref.ref(tensor))
            for sid, stream in self._seen_streams.items():
                if sid != cur_id:
                    tensor.record_stream(stream)


class SkyrlCheckpointLoadMixin:
    """Split the compact batched-MoE FP8 wire tensors from ordinary checkpoint weights."""

    @contextmanager
    def skyrl_checkpoint_load(self) -> Iterator[None]:
        """Install the FP8-aware loader proxy over ``self.model``."""
        from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import (
            _load_checkpoint_weights,
        )
        from skyrl.backends.skyrl_train.weight_sync.nvfp4_stacked_receive import (
            Nvfp4StackedReceiver,
        )

        model = self.model
        # Stacked NVFP4 expert layers (``expert_nvfp4_stacked_sync``) may span receive batches and, under
        # vLLM 0.33's packed IPC, separate ``receive_weights`` calls (one per packed chunk). So the receiver
        # lives for the whole update round: it holds a partial layer until it is complete, and
        # ``finish_weight_update`` checks that nothing is left over. Its fallback is the per-expert path.
        stacked = getattr(self, "_skyrl_nvfp4_stacked", None)
        if stacked is None or stacked.model is not model:
            if stacked is not None:
                stacked.finish()
            stacked = Nvfp4StackedReceiver(model, fallback_load=lambda w: model.load_weights(weights=w))
            self._skyrl_nvfp4_stacked = stacked

        tracker = getattr(self, "_skyrl_detached_clone_tracker", None)
        if tracker is None:
            tracker = _DetachedCloneStreamTracker()
            self._skyrl_detached_clone_tracker = tracker
        # Register the caller stream up front (before packed NCCL switches to its receive
        # streams) so every detached clone records the caller stream for finalization.
        tracker.prepare_stream(self.device)

        def load_weights(weights: Any, **kwargs: Any) -> Any:
            tracker.prepare_stream(self.device)
            result = _load_checkpoint_weights(model, weights, nvfp4_stacked_receiver=stacked, **kwargs)
            cloned = detach_buffered_layerwise_views()
            if cloned:
                tracker.record_clones(cloned)
            return result

        # The proxy scopes the override to the `WeightTransferEngine` context
        # instead of mutating `load_weights` on the model object itself.
        self.model = _LoadWeightsProxy(model, load_weights)
        try:
            yield
        finally:
            # Restore the exact object we found, so this composes with the
            # worker's set_weight_update_target / reset_weight_update_target.
            self.model = model


def skyrl_before_weight_update() -> None:
    """Serialized-FP8 hooks that must be in place before the reload runs.

    MXFP8 TRT-LLM MoE prepare re-derives a fixed per-layer weight/scale
    relocation on every sync. Replace it with a learned, bitwise-validated
    permutation cache; it falls back to the original on any mismatch, and is a
    no-op for non-MXFP8 wires. Installed here rather than at worker-extension
    import so the wrap lands in the process that owns the engine, and only once
    a sync actually happens.
    """
    from skyrl.backends.skyrl_train.inference_servers.trtllm_moe_prepare_cache import (
        install as install_trtllm_moe_prepare_cache,
    )

    install_trtllm_moe_prepare_cache()


def skyrl_after_weight_update() -> None:
    """Re-assert the serialized-FP8 wire's KV/attention scale contract.

    The wire ships no KV/attention scale calibration, so those scales are 1.0 by
    contract. vLLM corrupts them at boot (compressed-tensors copies the
    dummy-load placeholders verbatim) and after a level-2 wake
    (``init_fp8_kv_scales`` resets only the k/v tensors — q wakes as 0.0 and the
    float mirrors keep garbage), which serves NaN on the quantized-Q path or
    silently wrong logprobs on the bf16-Q path under ``kv_cache_dtype=fp8_*``.
    Both of those are patched in ``vllm_compat``; a reload re-runs
    ``process_weights_after_loading``, so re-assert it here too.

    Gated on a dummy-weight boot: these engines also serve models started from a
    real FP8 checkpoint, whose calibrated k/v scales must survive a sync that
    carries no replacement for them.
    """
    from skyrl.backends.skyrl_train.inference_servers.vllm_compat import (
        booted_without_checkpoint_weights,
        normalize_serialized_fp8_kv_scales,
    )

    if not booted_without_checkpoint_weights():
        return
    from skyrl.backends.skyrl_train.patches.vllm.patch_model_runner_registry import (
        current_model_runner,
    )

    model_runner = current_model_runner()
    if model_runner is None:
        return
    count = normalize_serialized_fp8_kv_scales(model_runner)
    if count:
        logger.info("Normalized FP8 KV/attention scales to 1.0 on %d layers after weight sync", count)


class SkyrlLoraStagingMixin:
    """Apply one update round to a LoRA adapter instead of the base model.

    Armed per round by the trainer, before ``send_weights()``, through
    ``NewInferenceWorkerWrap.skyrl_set_lora_receive_target`` -- the round trip
    itself is vLLM's and carries only names, dtypes and shapes (see
    ``weight_sync/lora_target.py`` on why the target rides ``/collective_rpc``).

    Arming lasts exactly one round: ``finish`` disarms, so a failed or missing
    arm can never silently apply an adapter stream to the base model, or the
    reverse.
    """

    # Class-level defaults rather than an __init__: vLLM's factory constructs
    # these engines through the base __init__, which SkyRL does not override.
    _skyrl_lora_target: dict[str, Any] | None = None
    _skyrl_lora_staged: dict[str, torch.Tensor] | None = None

    def skyrl_set_lora_receive_target(self, receive_target: dict[str, Any]) -> None:
        """Arm the next update round to build the named adapter."""
        from skyrl.backends.skyrl_train.weight_sync.lora_target import (
            is_lora_receive_target,
        )

        if not is_lora_receive_target(receive_target):
            raise ValueError(f"Not a LoRA receive target: {receive_target!r}")
        if self._skyrl_lora_target is not None:
            raise RuntimeError(
                f"A LoRA receive target is already armed for "
                f"{self._skyrl_lora_target['lora_name']!r}; finish that update first."
            )
        if not self._skyrl_lora_capable():
            raise RuntimeError(
                "Received a LoRA weight update but this engine was started without --enable-lora "
                "(trainer.policy.model.lora.rank > 0 with merge_lora=false sets it)."
            )
        self._skyrl_lora_target = dict(receive_target)

    def _skyrl_lora_capable(self) -> bool:
        """Whether this vLLM worker can register an adapter at all."""
        return getattr(self.vllm_config, "lora_config", None) is not None

    def skyrl_lora_armed(self) -> bool:
        return self._skyrl_lora_target is not None

    def skyrl_begin_lora_update(self) -> None:
        self._skyrl_lora_staged = {}

    @contextmanager
    def skyrl_lora_staging(self) -> Iterator[None]:
        """Install a proxy whose ``load_weights`` stages instead of loading."""
        staged = self._skyrl_lora_staged
        if staged is None:
            raise RuntimeError("skyrl_begin_lora_update must run before receiving a LoRA stream.")
        model = self.model

        def load_weights(weights: Any, **kwargs: Any) -> set:
            loaded = set()
            for name, tensor in weights:
                if name in staged:
                    raise ValueError(f"LoRA tensor {name!r} received twice in one weight update")
                # The transport buffers (the IPC-mapped view, the packed NCCL
                # buffer) are reused or freed right after the chunk, so the
                # private copy is required, not an optimization.
                staged[name] = tensor.detach().clone()
                loaded.add(name)
            return loaded

        # Restore the exact object found, so this composes with
        # set_weight_update_target.
        self.model = _LoadWeightsProxy(model, load_weights)
        try:
            yield
        finally:
            self.model = model

    def skyrl_finish_lora_update(self) -> None:
        """Hand the round's tensors to vLLM's LoRA manager, and disarm."""
        from skyrl.backends.skyrl_train.patches.vllm.patch_lora_in_memory import (
            stage_in_memory_adapter,
        )
        from skyrl.backends.skyrl_train.weight_sync.lora_target import (
            expand_lora_aliases,
        )

        target = self._skyrl_lora_target
        staged = self._skyrl_lora_staged
        # Disarm before the work: one round per arm, even if this raises.
        self._skyrl_lora_target = None
        self._skyrl_lora_staged = None
        if target is None:
            raise RuntimeError("skyrl_finish_lora_update called without an armed LoRA receive target.")
        if not staged:
            raise RuntimeError(f"LoRA weight update for {target['lora_name']!r} finished without receiving any tensors")
        tensors = expand_lora_aliases(staged, target.get("aliases") or {})
        stage_in_memory_adapter(target["lora_name"], tensors, target["adapter_config"])
        # Deliberately not logged here. Only loggers under vLLM's own namespace
        # are configured in a worker process, so an INFO line from this module
        # would be dropped without a trace; the trainer logs the same counts
        # (it has them before the send) where the output is captured.


class SkyrlReceiveLifecycleMixin(SkyrlLoraStagingMixin, SkyrlCheckpointLoadMixin):
    """The update lifecycle both push engines share.

    Each engine brackets its lifecycle in ``torch.device(self.device)``. vLLM's
    own path passes ``device=`` where it matters instead; SkyRL's loaders rely on
    it being the default device that weight loading sees.

    Written once rather than per engine: NCCL and IPC differ only in the vLLM
    base class they mix into, and every step here is the same for both.
    """

    def start_weight_update(self) -> None:
        if self.skyrl_lora_armed():
            # No layerwise reload: an adapter update never writes to the base
            # model's parameters, so there is nothing to reload.
            self.skyrl_begin_lora_update()
            return
        skyrl_before_weight_update()
        # A round that failed part-way must not leak its partial stacked NVFP4 layers or
        # detached-clone stream tracking into this one.
        self._skyrl_nvfp4_stacked = None
        self._skyrl_detached_clone_tracker = _DetachedCloneStreamTracker()
        with torch.device(self.device):
            super().start_weight_update()

    def receive_weights(self, update_info: Any) -> None:
        try:
            if self.skyrl_lora_armed():
                with torch.device(self.device), self.skyrl_lora_staging():
                    super().receive_weights(update_info)
                return
            with torch.device(self.device), self.skyrl_checkpoint_load():
                super().receive_weights(update_info)
        finally:
            # vLLM's NCCL receiver caches each round's update info keyed by `self.model`, which here
            # is a per-round proxy whose `load_weights` closure holds the round's staged tensors (a
            # LoRA round's adapter copy on the GPU). SkyRL's trainer always ships the names, so the
            # cache is never read: drop it so no proxy, or what it holds, outlives its round.
            last = getattr(self, "_last_update_info", None)
            if last is not None:
                last.clear()

    def skyrl_release_receive_buffers(self) -> None:
        """Release receive buffers the vLLM engine holds until the end of a round.

        vLLM's ``finish_weight_update`` releases them after its layerwise
        finalize. For an adapter round, ``finish_weight_update`` calls this hook
        in place of vLLM's. No-op for transports whose receive buffers do not
        outlive ``receive_weights``.
        """

    def finish_weight_update(self) -> None:
        if self.skyrl_lora_armed():
            # No layerwise finalize: an adapter update never writes to the base
            # model's parameters.
            try:
                self.skyrl_finish_lora_update()
            finally:
                self.skyrl_release_receive_buffers()
            return
        from skyrl.backends.skyrl_train.patches.vllm.patch_moe_kernel_constants import (
            restore_moe_kernel_constants,
        )

        stacked = getattr(self, "_skyrl_nvfp4_stacked", None)
        self._skyrl_nvfp4_stacked = None
        tracker = getattr(self, "_skyrl_detached_clone_tracker", None)
        self._skyrl_detached_clone_tracker = None
        if stacked is not None:
            stacked.finish()  # raises on a stacked layer that never completed
        with torch.device(self.device):
            if tracker is not None:
                # Ensure the finalization stream waits on any remaining clone events and is
                # recorded on every surviving detached clone before finalize frees them.
                tracker.prepare_stream(self.device)
            super().finish_weight_update()
            # A level-2 sleep zeroes MoE kernel constants that no reload rebuilds.
            restore_moe_kernel_constants(self.model)
        skyrl_after_weight_update()
        empty_cuda_cache_rocm()


def _build_skyrl_nccl_engine() -> type:
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

    class SkyrlNCCLWeightTransferEngine(SkyrlReceiveLifecycleMixin, NCCLWeightTransferEngine):
        """vLLM's dense NCCL receive engine plus SkyRL receive targets."""

        def finish_weight_update(self) -> None:
            super().finish_weight_update()
            # vLLM 0.33's NCCL receiver allocates packed buffers on dedicated receive streams
            # without releasing cached blocks on finish (fixed upstream in vllm#64691).
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return SkyrlNCCLWeightTransferEngine


def _build_skyrl_ipc_engine() -> type:
    from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine

    class SkyrlIPCWeightTransferEngine(SkyrlReceiveLifecycleMixin, IPCWeightTransferEngine):
        """vLLM's CUDA IPC receive engine plus SkyRL receive targets."""

        def skyrl_release_receive_buffers(self) -> None:
            # The importer holds this worker's mapping of the trainer's packed
            # IPC buffer. Closing it once per export releases the producer-side
            # refcount, so the trainer's ``ipc_collect`` can free the buffer.
            self._packed_importer.close()

    return SkyrlIPCWeightTransferEngine


# Built lazily and cached: the classes subclass vLLM's engines, and this module
# must stay importable without the wheel.
_ENGINE_CACHE: dict[str, type] = {}


def get_skyrl_nccl_engine() -> type:
    if SKYRL_NCCL_BACKEND not in _ENGINE_CACHE:
        _ENGINE_CACHE[SKYRL_NCCL_BACKEND] = _build_skyrl_nccl_engine()
    return _ENGINE_CACHE[SKYRL_NCCL_BACKEND]


def get_skyrl_ipc_engine() -> type:
    if SKYRL_IPC_BACKEND not in _ENGINE_CACHE:
        _ENGINE_CACHE[SKYRL_IPC_BACKEND] = _build_skyrl_ipc_engine()
    return _ENGINE_CACHE[SKYRL_IPC_BACKEND]
