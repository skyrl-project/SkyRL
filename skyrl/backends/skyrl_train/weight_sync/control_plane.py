"""Blocking control-plane client for the trainer-side weight transfer engines.

vLLM's ``VLLMWeightSyncClient`` is a synchronous four-method protocol
(``init_weight_transfer_engine`` / ``start_weight_update`` / ``update_weights`` /
``finish_weight_update``), and the engine that drives it runs off the worker's
event loop. So this client uses blocking HTTP against the same routes rather
than SkyRL's async ``RemoteInferenceClient``, keeping the whole engine
sync-to-sync with no event-loop involvement.

Four properties are load-bearing:

* **Per-call fresh connections** (``Connection: close``). A full training step
  elapses between syncs, so a pooled keep-alive connection is stale by the next
  call and races the server's ``timeout_keep_alive`` (uvicorn 5s) into
  ECONNRESET.
* **Concurrent fan-out** across servers. Required for correctness on the RDT
  path, not just speed: consumers pull in lockstep and a producer frees a served
  group only once every bound consumer has pulled, so a serial
  ``update_weights`` stalls the producer's gather loop and deadlocks.
* **Body-aware error messages** (:func:`_error_message`) — surface the response
  body's error detail, not the bare HTTP reason phrase.
* **No timeout** (an RDT bake + NIXL pull are long) and **no retry** (retrying a
  half-done stateful call would be wrong).

``requests`` rather than a new dependency: Ray provides it, and Ray is a hard
requirement of every path that reaches here. The import is local so nothing else
pays for it.
"""

import logging
import threading
import time
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

if TYPE_CHECKING:
    from requests import Response

logger = logging.getLogger(__name__)

# Per-context deadline (so concurrent/overlapping scopes in different threads
# never corrupt each other's budget) plus a lock-protected token-keyed fallback
# table so nested ThreadPoolExecutors that do not copy ContextVars (such as
# vLLM's NCCLTrainerWeightTransferEngine) still inherit the tightest active
# finite deadline.
_UNSET_DEADLINE = object()
_THREAD_DEADLINE_VAR: ContextVar[Any] = ContextVar("_THREAD_DEADLINE_VAR", default=_UNSET_DEADLINE)
_CONTROL_PLANE_LOCK = threading.Lock()
_WEIGHT_SYNC_DEADLINE_SCOPES: List[Tuple[object, Optional[float]]] = []
_ACTIVE_CLIENTS: "weakref.WeakSet[SkyrlWeightSyncClient]" = weakref.WeakSet()


@contextmanager
def weight_sync_client_deadline(deadline_at: Optional[float]) -> Iterator[None]:
    """Bind blocking ``SkyrlWeightSyncClient`` calls in this scope to one absolute deadline.

    ``deadline_at`` is a ``time.monotonic()`` timestamp (or ``None`` for no
    deadline). Uses a ContextVar for same-thread / nested scopes and a
    token-keyed active-scope table for nested thread pools, so overlapping
    scopes exiting out of order cannot pop each other's deadlines.
    """
    scope_token = object()
    cv_token = _THREAD_DEADLINE_VAR.set(deadline_at)
    with _CONTROL_PLANE_LOCK:
        _WEIGHT_SYNC_DEADLINE_SCOPES.append((scope_token, deadline_at))
    try:
        yield
    finally:
        _THREAD_DEADLINE_VAR.reset(cv_token)
        with _CONTROL_PLANE_LOCK:
            _WEIGHT_SYNC_DEADLINE_SCOPES[:] = [
                entry for entry in _WEIGHT_SYNC_DEADLINE_SCOPES if entry[0] is not scope_token
            ]


@contextmanager
def weight_sync_client_timeout(timeout_s: Optional[float]) -> Iterator[None]:
    """Convenience wrapper converting a duration into an absolute ``weight_sync_client_deadline``."""
    deadline_at = None if (timeout_s is None or timeout_s <= 0) else (time.monotonic() + timeout_s)
    with weight_sync_client_deadline(deadline_at):
        yield


def current_weight_sync_deadline() -> Optional[float]:
    """Return the active weight-sync monotonic deadline for this context (or tightest active scope)."""
    local_val = _THREAD_DEADLINE_VAR.get()
    if local_val is not _UNSET_DEADLINE:
        return local_val
    with _CONTROL_PLANE_LOCK:
        finite = [d for _, d in _WEIGHT_SYNC_DEADLINE_SCOPES if d is not None]
        return min(finite) if finite else None


def _current_process_deadline() -> Optional[float]:
    return current_weight_sync_deadline()


def drain_inflight_weight_sync_requests(
    deadline_at: Optional[float] = None,
    cancel_event: Optional[threading.Event] = None,
) -> List[str]:
    """Wait for any timed-out blocking weight-sync HTTP requests to settle up to ``deadline_at``.

    Returns a list of ``"<url><endpoint>"`` strings for requests still running
    when ``deadline_at`` expires (or ``cancel_event`` is set) plus any prior
    lifecycle RPCs whose completion on the server is unconfirmed.
    """
    with _CONTROL_PLANE_LOCK:
        clients = list(_ACTIVE_CLIENTS)
    still_running: List[str] = []
    for client in clients:
        still_running.extend(client.drain_inflight(deadline_at=deadline_at, cancel_event=cancel_event))
    return still_running


# vLLM dev-mode routes, plus /fetch_weights, which SkyRL adds in vllm_server_actor.
INIT_ENGINE_ENDPOINT = "/init_weight_transfer_engine"
START_UPDATE_ENDPOINT = "/start_weight_update"
START_DRAFT_UPDATE_ENDPOINT = "/start_draft_weight_update"
UPDATE_WEIGHTS_ENDPOINT = "/update_weights"
FINISH_UPDATE_ENDPOINT = "/finish_weight_update"
FETCH_WEIGHTS_ENDPOINT = "/fetch_weights"
RESET_PREFIX_CACHE_ENDPOINT = "/reset_prefix_cache"
PAUSE_ENDPOINT = "/pause"
RESUME_ENDPOINT = "/resume"

_STATEFUL_ENDPOINTS = frozenset(
    {
        START_UPDATE_ENDPOINT,
        START_DRAFT_UPDATE_ENDPOINT,
        UPDATE_WEIGHTS_ENDPOINT,
        FINISH_UPDATE_ENDPOINT,
        PAUSE_ENDPOINT,
        RESUME_ENDPOINT,
    }
)


def _is_stateful_endpoint(endpoint: str) -> bool:
    """Return True for endpoints whose server-side state transition must be confirmed before reuse."""
    return endpoint.split("?", 1)[0] in _STATEFUL_ENDPOINTS


def _has_exception_in_chain(exc: BaseException, types: Tuple[type, ...]) -> bool:
    seen: set[int] = set()
    stack: List[BaseException] = [exc]
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if isinstance(cur, types):
            return True
        for arg in getattr(cur, "args", ()):
            if isinstance(arg, BaseException):
                stack.append(arg)
        if cur.__cause__ is not None:
            stack.append(cur.__cause__)
        if cur.__context__ is not None:
            stack.append(cur.__context__)
    return False


def _is_unconfirmed_lifecycle_error(exc: BaseException) -> bool:
    """Return True when ``exc`` is an ambiguous transport/timeout error that may have left a lifecycle RPC running."""
    import socket

    # Pre-connect failures never established a TCP connection, so the server never received the request.
    if _has_exception_in_chain(exc, (ConnectionRefusedError, socket.gaierror)):
        return False
    try:
        import requests

        if isinstance(exc, requests.exceptions.ConnectTimeout):
            return False
        if isinstance(
            exc,
            (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
            ),
        ):
            return True
    except ImportError:
        pass
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


class SkyrlWeightSyncClient:
    """``VLLMWeightSyncClient`` over the inference servers' native HTTP routes.

    Args:
        server_urls: every inference server, ordered
            ``[engine0_dp0, engine0_dp1, ..., engine1_dp0, ...]`` — the order the
            per-server init rewrite (``init_payload_fn``) depends on.
        data_parallel_size: DP replicas per deployment. Servers are grouped into
            deployments of this many.
        init_payload_fn: rewrites the engine's single worker-side init dict into
            one payload per server, for the backends that need per-server fields:
            NCCL a cumulative ``rank_offset``, sharded RDT a deployment ordinal.
            See :func:`nccl_init_payloads` / :func:`rdt_init_payloads`. ``None``
            sends the same dict to every server.
    """

    def __init__(
        self,
        server_urls: Sequence[str],
        data_parallel_size: int = 1,
        *,
        init_payload_fn: Optional[Callable[[Dict[str, Any], Sequence[str], int], List[Dict[str, Any]]]] = None,
        timeout_s: Optional[float] = None,
    ) -> None:
        import requests  # local: Ray (a hard dep of every path reaching here) provides it.

        self._urls = list(server_urls)
        self._dp = max(1, int(data_parallel_size))
        self._init_payload_fn = init_payload_fn
        self._timeout_s = timeout_s
        self._start_endpoint = START_UPDATE_ENDPOINT
        if not self._urls:
            raise ValueError("SkyrlWeightSyncClient requires at least one server_url.")

        self._session = requests.Session()
        # See module docstring: fresh connection per call.
        self._session.headers["Connection"] = "close"
        # One worker per server so a fan-out call issues every POST concurrently.
        self._pool = ThreadPoolExecutor(max_workers=len(self._urls), thread_name_prefix="weight-sync-ctrl")

    # ---- VLLMWeightSyncClient protocol (the four methods vLLM's engines call) ----

    def init_weight_transfer_engine(self, init_info: Dict[str, Any]) -> None:
        if self._init_payload_fn is None:
            per_server = [dict(init_info) for _ in self._urls]
        else:
            per_server = self._init_payload_fn(init_info, self._urls, self._dp)
        if len(per_server) != len(self._urls):
            raise ValueError(
                f"init_payload_fn returned {len(per_server)} payloads for {len(self._urls)} servers; "
                "it must return exactly one per server, in server order."
            )
        self._fanout([(url, INIT_ENGINE_ENDPOINT, {"init_info": info}) for url, info in zip(self._urls, per_server)])

    def start_weight_update(self) -> None:
        self._fanout_uniform(self._start_endpoint, None)

    def update_weights(self, update_info: Dict[str, Any]) -> None:
        self._fanout_uniform(UPDATE_WEIGHTS_ENDPOINT, {"update_info": _json_safe(update_info)})

    def finish_weight_update(self, weight_version: Optional[str] = None) -> None:
        # Omit the key entirely when unset so the route's default applies.
        body = {"weight_version": weight_version} if weight_version is not None else None
        self._fanout_uniform(FINISH_UPDATE_ENDPOINT, body)

    @contextmanager
    def draft_session(self) -> Iterator[None]:
        """Open the sessions started inside this block on vLLM's spec-decode drafter.

        vLLM's trainer engines only call ``start_weight_update``. The draft route
        retargets the worker's receive engine at the drafter for one session, and
        ``/finish_weight_update`` restores the main model.
        """
        self._start_endpoint = START_DRAFT_UPDATE_ENDPOINT
        try:
            yield
        finally:
            self._start_endpoint = START_UPDATE_ENDPOINT

    # ---- extras: the checkpoint-delta lifecycle ----
    #
    # Not the transport contract, and here anyway, because they have to be driven
    # from inside `send_weights`: `fetch_weights` needs the `target_version` the
    # publish produces mid-send, and must run before the pause. `send_weights` is
    # sync and runs off the event loop, so `RemoteInferenceClient` -- where these
    # routes otherwise live -- is reachable only via `run_coroutine_threadsafe`,
    # the loop coupling this client exists to avoid (see module docstring).
    #
    # The cost: an engine calling these needs a SkyRL client, not any object
    # satisfying VLLMWeightSyncClient. Only the delta engine does.

    def fetch_weights(self, target_version: int, sync_dir: Optional[str] = None, uri: Optional[str] = None) -> None:
        body: Dict[str, Any] = {"target_version": int(target_version)}
        if sync_dir is not None:
            body["sync_dir"] = sync_dir
        if uri is not None:
            body["uri"] = uri
        self._fanout_uniform(FETCH_WEIGHTS_ENDPOINT, body)

    def reset_prefix_cache(self, reset_running_requests: bool = True) -> None:
        """Reset every server's cache; raise if any server does not confirm it."""
        endpoint = f"{RESET_PREFIX_CACHE_ENDPOINT}?reset_running_requests={str(reset_running_requests).lower()}"
        responses = self._fanout_uniform(endpoint, None)
        for url, response in zip(self._urls, responses):
            try:
                body = response.json()
            except ValueError as exc:
                raise RuntimeError(f"Prefix cache reset failed on {url}: invalid JSON response") from exc
            if not isinstance(body, dict) or body.get("success") is not True:
                raise RuntimeError(f"Prefix cache reset failed on {url}")

    def pause_generation(self, clear_cache: bool = False) -> None:
        # /pause takes query params, not a body (mirrors RemoteInferenceClient.pause).
        self._fanout(
            [(url, f"{PAUSE_ENDPOINT}?mode=keep&clear_cache={str(clear_cache).lower()}", None) for url in self._urls]
        )

    def resume_generation(self) -> None:
        """Resume generation on every server, even if a subset has a stuck in-flight RPC.

        Healthy servers are resumed immediately within the active deadline. Any
        server whose prior request is still running past the drain window gets a
        best-effort deferred ``/resume`` callback when that future settles, so it
        is not left permanently paused once its stuck RPC finishes.
        """
        with _CONTROL_PLANE_LOCK:
            _ACTIVE_CLIENTS.add(self)

        deadline_at = self._effective_deadline()
        if deadline_at is not None:
            remaining = max(0.0, deadline_at - time.monotonic())
            drain_deadline: Optional[float] = time.monotonic() + min(remaining * 0.5, 5.0)
        else:
            drain_deadline = None
        self.drain_inflight(deadline_at=drain_deadline, include_unconfirmed=False)

        with _CONTROL_PLANE_LOCK:
            stuck_entries = list(self._inflight_entries())
        stuck_urls = {url for url, _, _ in stuck_entries}
        ready_urls = [url for url in self._urls if url not in stuck_urls]

        for url, _ep, fut in stuck_entries:
            if _ep.split("?", 1)[0] != RESUME_ENDPOINT:
                self._schedule_deferred_resume(url, fut)

        resume_exc: Optional[BaseException] = None
        if ready_urls:
            try:
                self._fanout(
                    [(url, RESUME_ENDPOINT, None) for url in ready_urls],
                    allow_unconfirmed=True,
                    skip_drain=True,
                )
            except Exception as exc:  # noqa: BLE001
                resume_exc = exc

        if stuck_entries:
            stuck_desc = ", ".join(f"{url}{ep}" for url, ep, _ in stuck_entries)
            stuck_exc = TimeoutError(f"Deferred /resume for server(s) with request(s) still in flight: {stuck_desc}")
            if resume_exc is not None:
                resume_exc.add_note(str(stuck_exc))
                raise resume_exc
            raise stuck_exc

        if resume_exc is not None:
            raise resume_exc

    def close(self) -> None:
        """Release the HTTP session + fan-out pool. Idempotent.

        Closes the HTTP session first so any socket waiting on a hung server is
        aborted rather than blocking teardown forever. If requests remain
        unresolved after a short drain, returns without hanging in
        ``_pool.shutdown(wait=True)`` and keeps the client registered in
        ``_ACTIVE_CLIENTS`` until those requests settle.
        """
        self._closed = True
        self._prune_done_inflight()
        with _CONTROL_PLANE_LOCK:
            has_inflight = bool(self._inflight_entries())
            has_unconfirmed = bool(self._unconfirmed_entries())

        try:
            self._session.close()
        except Exception:  # noqa: BLE001
            pass

        if has_inflight:
            self.drain_inflight(deadline_at=time.monotonic() + 0.5, include_unconfirmed=False)
            self._prune_done_inflight()
            with _CONTROL_PLANE_LOCK:
                still_inflight = bool(self._inflight_entries())
                still_unconfirmed = bool(self._unconfirmed_entries())
            self._pool.shutdown(wait=not still_inflight, cancel_futures=True)
            if not still_inflight and not still_unconfirmed:
                with _CONTROL_PLANE_LOCK:
                    _ACTIVE_CLIENTS.discard(self)
        else:
            self._pool.shutdown(wait=True)
            if not has_unconfirmed:
                with _CONTROL_PLANE_LOCK:
                    _ACTIVE_CLIENTS.discard(self)

    # ---- internals ----

    def _fanout_uniform(self, endpoint: str, body: Optional[Dict[str, Any]]) -> List["Response"]:
        return self._fanout([(url, endpoint, body) for url in self._urls])

    def _inflight_entries(self) -> List[Tuple[str, str, Future]]:
        inflight = getattr(self, "_inflight_futures", None)
        if inflight is None:
            inflight = []
            self._inflight_futures = inflight
        return inflight

    def _unconfirmed_entries(self) -> List[str]:
        unconfirmed = getattr(self, "_unconfirmed_lifecycle_requests", None)
        if unconfirmed is None:
            unconfirmed = []
            self._unconfirmed_lifecycle_requests = unconfirmed
        return unconfirmed

    def _deferred_resume_set(self) -> "weakref.WeakSet[Future]":
        deferred = getattr(self, "_deferred_resume_futures", None)
        if deferred is None:
            deferred = weakref.WeakSet()
            self._deferred_resume_futures = deferred
        return deferred

    def _record_unconfirmed_lifecycle(self, url: str, endpoint: str, exc: Optional[BaseException] = None) -> None:
        reason = f"unconfirmed {type(exc).__name__}" if exc is not None else "unconfirmed transport error"
        entry = f"{url}{endpoint} ({reason})"
        with _CONTROL_PLANE_LOCK:
            unconfirmed = self._unconfirmed_entries()
            if entry not in unconfirmed:
                unconfirmed.append(entry)

    def _prune_done_inflight(self) -> List[Tuple[str, str, Future]]:
        """Atomically remove completed futures and record any unconfirmed failures under the lock."""
        active: List[Tuple[str, str, Future]] = []
        with _CONTROL_PLANE_LOCK:
            remaining_entries: List[Tuple[str, str, Future]] = []
            for entry in self._inflight_entries():
                url, endpoint, fut = entry
                if fut.done():
                    try:
                        fut.result()
                    except Exception as exc:  # noqa: BLE001
                        if _is_stateful_endpoint(endpoint) and _is_unconfirmed_lifecycle_error(exc):
                            reason = f"unconfirmed {type(exc).__name__}"
                            unconfirmed_entry = f"{url}{endpoint} ({reason})"
                            unconfirmed = self._unconfirmed_entries()
                            if unconfirmed_entry not in unconfirmed:
                                unconfirmed.append(unconfirmed_entry)
                else:
                    remaining_entries.append(entry)
                    active.append(entry)
            self._inflight_entries()[:] = remaining_entries
            if getattr(self, "_closed", False) and not remaining_entries and not self._unconfirmed_entries():
                _ACTIVE_CLIENTS.discard(self)
        return active

    def _schedule_deferred_resume(self, url: str, fut: Future) -> None:
        deferred_fut: Future = Future()
        with _CONTROL_PLANE_LOCK:
            deferred = self._deferred_resume_set()
            if fut in deferred:
                return
            deferred.add(fut)
            # Track the deferred /resume itself in _inflight_entries BEFORE `fut`
            # completes so no subsequent round can pass drain_inflight() and start
            # a new pause/update while this older deferred /resume is still pending.
            self._inflight_entries().append((url, RESUME_ENDPOINT, deferred_fut))

        def _on_settled(_done_fut: Future) -> None:
            try:
                self._prune_done_inflight()
            except Exception:  # noqa: BLE001
                pass
            if getattr(self, "_closed", False):
                if not deferred_fut.done():
                    deferred_fut.set_result(None)
                return

            def _send_resume() -> None:
                try:
                    if getattr(self, "_closed", False):
                        return
                    # Connect-only timeout so we wait for server-confirmed /resume completion;
                    # record ambiguous post-connect transport errors as unconfirmed so a delayed
                    # /resume cannot overlap a subsequent pause/reload.
                    self._post(url, RESUME_ENDPOINT, None, timeout_s=(10.0, None))
                except Exception as exc:  # noqa: BLE001
                    if _is_unconfirmed_lifecycle_error(exc):
                        self._record_unconfirmed_lifecycle(url, RESUME_ENDPOINT, exc)
                    logger.warning("Deferred /resume to %s failed after stuck RPC settled: %s", url, exc)
                finally:
                    if not deferred_fut.done():
                        deferred_fut.set_result(None)

            threading.Thread(target=_send_resume, name="weight-sync-deferred-resume", daemon=True).start()

        fut.add_done_callback(_on_settled)

    def _effective_deadline(self) -> Optional[float]:
        proc_deadline = _current_process_deadline()
        if proc_deadline is not None:
            return proc_deadline
        inst_timeout = getattr(self, "_timeout_s", None)
        if inst_timeout is not None and inst_timeout > 0:
            return time.monotonic() + inst_timeout
        return None

    def drain_inflight(
        self,
        deadline_at: Optional[float] = None,
        *,
        include_unconfirmed: bool = True,
        cancel_event: Optional[threading.Event] = None,
    ) -> List[str]:
        """Drain previously timed-out POST futures on this client up to ``deadline_at``.

        Unfinished futures stay registered in ``_inflight_futures`` throughout the
        wait so cancellation or a concurrent caller can never observe an empty
        registry while a prior lifecycle request is still running.
        """
        while True:
            active = self._prune_done_inflight()
            if not active:
                break
            if cancel_event is not None and cancel_event.is_set():
                break
            remaining = None if deadline_at is None else max(0.0, deadline_at - time.monotonic())
            if remaining is not None and remaining <= 0.0:
                break
            wait_s = remaining if cancel_event is None else (0.05 if remaining is None else min(0.05, remaining))
            # Use futures_wait rather than fut.result(timeout=...) because in
            # Python 3.11+ concurrent.futures.TimeoutError is builtin
            # TimeoutError, so a completed future that raised TimeoutError
            # (e.g. socket.timeout) would be mistaken for a wait timeout.
            futures_wait([fut for _, _, fut in active], timeout=wait_s)

        self._prune_done_inflight()
        with _CONTROL_PLANE_LOCK:
            still_running = [f"{url}{endpoint}" for url, endpoint, _ in self._inflight_entries()]
            unconfirmed = list(self._unconfirmed_entries()) if include_unconfirmed else []
        return still_running + unconfirmed

    def _fanout(
        self,
        calls: Sequence[Tuple[str, str, Optional[Dict[str, Any]]]],
        *,
        allow_unconfirmed: bool = False,
        skip_drain: bool = False,
    ) -> List["Response"]:
        """POST to every server concurrently; raise the first failure after all
        return (or when the active absolute deadline expires).

        Any previously timed-out requests must settle with server confirmation
        before new lifecycle requests are submitted so stale lifecycle RPCs
        cannot overlap a subsequent round.
        """
        with _CONTROL_PLANE_LOCK:
            _ACTIVE_CLIENTS.add(self)

        deadline_at = self._effective_deadline()
        if not skip_drain:
            still_running = self.drain_inflight(deadline_at=deadline_at, include_unconfirmed=not allow_unconfirmed)
            if still_running:
                raise TimeoutError(
                    f"Cannot issue weight-sync POSTs while prior timed-out request(s) are still in flight "
                    f"or unconfirmed on the server: {', '.join(still_running)}"
                )

        is_stateful = any(_is_stateful_endpoint(endpoint) for _, endpoint, _ in calls)
        if deadline_at is not None:
            remaining_before_submit = deadline_at - time.monotonic()
            if remaining_before_submit <= 0.0:
                first_url, first_ep, _ = calls[0]
                raise TimeoutError(f"POST {first_url}{first_ep} skipped: weight-sync phase deadline already expired")
            if is_stateful:
                # Use a connect timeout only (read_timeout=None) for stateful
                # weight-sync lifecycle/pause RPCs so the HTTP socket stays open
                # until the server actually finishes executing the request. The
                # phase deadline is enforced below via futures_wait(timeout=...),
                # keeping unfinished futures in _inflight_futures until
                # server-confirmed settlement.
                req_timeout: Any = (max(0.05, min(10.0, remaining_before_submit)), None)
            else:
                req_timeout = max(0.05, remaining_before_submit)
        else:
            req_timeout = None

        futures = [
            (url, endpoint, self._pool.submit(self._post, url, endpoint, body, req_timeout))
            for url, endpoint, body in calls
        ]
        # Wait with futures_wait rather than fut.result(timeout=...) so a wait
        # timeout is never conflated with a future's own TimeoutError or a
        # successful completion arriving right after the deadline.
        if deadline_at is None:
            futures_wait([fut for _, _, fut in futures])
        else:
            remaining = max(0.0, deadline_at - time.monotonic())
            futures_wait([fut for _, _, fut in futures], timeout=remaining)

        responses = []
        first_exc: Optional[BaseException] = None
        timed_out: List[Tuple[str, str, Future]] = []
        for url, endpoint, fut in futures:
            if not fut.done():
                fut.cancel()
            if fut.cancelled():
                first_exc = first_exc or TimeoutError(f"POST {url}{endpoint} exceeded weight-sync phase deadline")
            elif fut.done():
                try:
                    responses.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    if _is_stateful_endpoint(endpoint) and _is_unconfirmed_lifecycle_error(exc):
                        self._record_unconfirmed_lifecycle(url, endpoint, exc)
                    first_exc = first_exc or exc
            else:
                # Register unconditionally (never re-check fut.done() here); if `fut`
                # completes concurrently right after this check, _prune_done_inflight()
                # will inspect its actual result and record any unconfirmed error.
                timed_out.append((url, endpoint, fut))
                first_exc = first_exc or TimeoutError(f"POST {url}{endpoint} exceeded weight-sync phase deadline")
        if timed_out:
            with _CONTROL_PLANE_LOCK:
                self._inflight_entries().extend(timed_out)
        if first_exc is not None:
            raise first_exc
        return responses

    def _post(
        self,
        url: str,
        endpoint: str,
        body: Optional[Dict[str, Any]],
        timeout_s: Any = None,
    ) -> "Response":
        # No retry; timeout defaults to None unless bounded by caller deadline (see module docstring).
        resp = self._session.post(f"{url}{endpoint}", json=body, timeout=timeout_s)
        if resp.status_code >= 400:
            raise RuntimeError(_error_message(url, endpoint, resp))
        return resp


def _json_safe(update_info: Dict[str, Any]) -> Dict[str, Any]:
    """Make an update_info dict JSON-serializable for HTTP transport.

    ``IPCTrainerWeightTransferEngine`` emits raw CUDA IPC handles -- tuples
    containing storage types and handle bytes -- because a Ray transport carries
    them natively. Over HTTP they must be pickled into ``ipc_handles_pickled``,
    which the worker auto-deserializes when
    ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` (set by ``vllm_server_actor``).

    Other backends carry only JSON-native metadata and pass through unchanged.
    """
    handles = update_info.get("ipc_handles")
    if handles is None:
        return update_info

    import base64
    import pickle

    out = {k: v for k, v in update_info.items() if k != "ipc_handles"}
    out["ipc_handles_pickled"] = base64.b64encode(pickle.dumps(handles)).decode("utf-8")
    return out


def nccl_init_payloads(
    init_info: Dict[str, Any],
    server_urls: Sequence[str],
    data_parallel_size: int,
) -> List[Dict[str, Any]]:
    """Per-server NCCL init payloads with cumulative ``rank_offset``.

    ``worker_init_process_group`` derives a worker's rank as
    ``dp_rank * world_size_per_dp + rank_within_dp + rank_offset``, and each
    deployment's indices restart at 0, so every deployment's offset must advance
    past the previous one's workers or their ranges collide. The engine builds
    one init info with ``rank_offset=1``, which is only correct for a single
    deployment.

    DP servers within one deployment share an offset -- vLLM's
    ``data_parallel_index`` already separates them, so advancing per DP server
    would double-count. The offset advances only at deployment boundaries.

    A wrong offset mis-maps ranks and hangs in the NCCL rendezvous rather than
    failing, so ``world_size`` is cross-checked against what the offsets imply.
    """
    dp = max(1, int(data_parallel_size))
    num_servers = len(server_urls)
    if num_servers % dp != 0:
        raise ValueError(f"Number of servers ({num_servers}) must be divisible by data_parallel_size ({dp}).")
    base_offset = int(init_info.get("rank_offset", 1))
    world_size = int(init_info["world_size"])
    num_deployments = max(1, num_servers // dp)
    # world_size counts the trainer sender (rank 0) plus every inference worker,
    # so the per-deployment worker count follows from it.
    workers_total = world_size - base_offset
    if workers_total <= 0 or workers_total % num_deployments != 0:
        raise ValueError(
            f"NCCL weight sync world_size={world_size} with rank_offset={base_offset} implies "
            f"{workers_total} inference workers, which does not divide across {num_deployments} "
            f"deployment(s) ({num_servers} servers / dp={dp})."
        )
    world_size_per_deployment = workers_total // num_deployments

    payloads: List[Dict[str, Any]] = []
    offset = base_offset
    for i in range(num_servers):
        payloads.append({**init_info, "rank_offset": offset})
        if (i + 1) % dp == 0:
            offset += world_size_per_deployment
    return payloads


def rdt_init_payloads(
    init_info: Dict[str, Any],
    server_urls: Sequence[str],
    data_parallel_size: int,
) -> List[Dict[str, Any]]:
    """Per-server sharded-RDT init payloads, stamped with the deployment ordinal.

    Each deployment has its own self-contained parallel config, so its internal
    worker index restarts at 0 and would collide under the M:N block assignment.
    Stamping each server with its ordinal (``server_index // data_parallel_size``)
    as ``replica_rank``, plus the deployment count as ``num_replicas``, lets the
    engine offset its consumers into a globally distinct range.

    The ordinal divides by ``data_parallel_size`` because the DP servers of one
    deployment share a parallel config, so they must share one ``replica_rank``.
    """
    dp = max(1, int(data_parallel_size))
    num_replicas = max(1, len(server_urls) // dp)
    return [{**init_info, "replica_rank": i // dp, "num_replicas": num_replicas} for i in range(len(server_urls))]


def _error_message(url: str, endpoint: str, resp: Any) -> str:
    """Surface the response body's error detail (``{"error": {"message": ...}}``
    or FastAPI's ``{"detail": ...}``) rather than the bare HTTP reason phrase."""
    detail = resp.reason
    try:
        body = resp.json()
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                detail = err.get("message", detail)
            elif isinstance(err, str):
                detail = err
            elif body.get("detail") is not None:
                detail = body["detail"]
    except Exception:  # noqa: BLE001
        detail = (resp.text or resp.reason)[:1000]
    return f"Weight-sync control-plane call {endpoint} to {url} failed [{resp.status_code}]: {detail}"
