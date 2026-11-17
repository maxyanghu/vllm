# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Overlap of the sparse-attention indexer with independent same-layer work.

Enabled by `VLLM_INDEXER_OVERLAP`, and only while a FULL CUDA graph is
captured for a decode batch: an eager segment of a breakable (piecewise)
graph cannot wait on an event recorded inside a captured segment, and eager
steps keep the stock order. The moved kernels and their inputs are
unchanged, so the outputs are bit-identical to the stock order.

Deferred top-k (Kimi K3.1, through `MultiHeadLatentAttentionWrapper`): the
wrapper asks for it right before calling the indexer. The indexer op then
runs its scan and top-k on other streams (the stock indexer on one
high-priority stream, the die-local indexer on the locality domains'
streams) and returns without joining. The wrapper issues the work that does
not read the kept tokens, and records the index group's ready event on the
stream that writes them; the index group's remap is their first reader.

Deferred candidates (DeepSeek-V4.1): the candidate source layer selects its
candidate blocks, and expands them into the sparse block ids its first
consumer would build first, on a side stream; that consumer joins.
"""

import threading
from collections.abc import Callable
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def full_cuda_graph_capture() -> bool:
    """Whether a plain (FULL) CUDA graph is being captured."""
    if not torch.cuda.is_current_stream_capturing():
        return False
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.forward_context import in_piecewise_cudagraph

    return not BreakableCUDAGraphCapture.is_active() and not in_piecewise_cudagraph()


def decode_only(metadata: Any) -> bool:
    return (
        metadata is not None
        and getattr(metadata, "num_prefills", 1) == 0
        and getattr(metadata, "num_decodes", 0) > 0
    )


def index_group_joins_topk(impl: Any, metadata: Any, num_tokens: int) -> bool:
    """Whether sparse attention reads the top-k only through its index group's
    conversion, which waits for the top-k's ready event (`_convert_once`):
    the shared sparse MLA `forward_mqa`, a non-HiSparse group, no DCP, no
    token offset, a group that converts decode tokens, and a batch that fits
    the group's buffer. Any other reader would not wait for a deferred top-k."""
    from vllm.model_executor.layers.attention.sparse_mla_attention import (
        SparseMLACommonImpl,
    )
    from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup

    group = getattr(impl, "index_group", None)
    return (
        isinstance(impl, SparseMLACommonImpl)
        and type(impl).forward_mqa is SparseMLACommonImpl.forward_mqa
        and group is not None
        and not isinstance(group, HiSparseMLAIndexGroup)
        and impl.dcp_world_size == 1
        and not getattr(metadata, "topk_start", 0)
        and not (impl.group_conversion_decode_only and group.num_layers <= 1)
        and num_tokens <= group.physical_topk_indices.shape[0]
    )


# ---------------------------------------------------------------------------
# Deferred top-k
# ---------------------------------------------------------------------------
_deferred = threading.local()
_stock_stream: torch.cuda.Stream | None = None


def init_stock_topk_stream() -> None:
    """Create the stock indexer's stream (at model construction).

    High priority: its top-k is on the critical path, and the work moved
    beside it competes for the SMs the scan leaves free.
    """
    global _stock_stream
    if _stock_stream is None and torch.cuda.is_available():
        _stock_stream = torch.cuda.Stream(priority=-1)


def request_deferred_topk() -> None:
    """Called right before an indexer call whose top-k may land late."""
    _deferred.requested = True
    _deferred.stream = None
    _deferred.keep = None


def consume_deferred_topk_request() -> bool:
    """Called by the indexer op: whether this call may return before its
    top-k is written (`set_deferred_topk` then says where it lands)."""
    requested = getattr(_deferred, "requested", False)
    _deferred.requested = False
    return requested and full_cuda_graph_capture()


def set_deferred_topk(stream: torch.cuda.Stream, keep: tuple) -> None:
    """The indexer op deferred its top-k: it is written on `stream`, which
    reads `keep`, tensors allocated on the caller's stream."""
    _deferred.stream = stream
    _deferred.keep = keep
    logger.info_once("Sparse indexer overlap: deferred top-k active.")


def defer_stock_topk(work: Callable[[], None], keep: tuple) -> None:
    """Run the stock scan and top-k `work` on the stock indexer's stream,
    after the caller's work so far, without joining."""
    assert _stock_stream is not None
    start = torch.cuda.Event()
    start.record(torch.cuda.current_stream())
    _stock_stream.wait_event(start)
    with torch.cuda.stream(_stock_stream):
        work()
    set_deferred_topk(_stock_stream, keep)


def take_deferred_topk_stream() -> torch.cuda.Stream | None:
    """After the indexer call: the stream that writes its top-k, or None if
    the call joined. The caller must call `release_deferred_topk` once its
    stream has waited for the top-k."""
    stream = getattr(_deferred, "stream", None)
    _deferred.requested = False
    _deferred.stream = None
    return stream


def release_deferred_topk() -> None:
    """The caller's stream has waited for the top-k: until then the tensors
    the deferred work reads stay referenced, so the caching allocator cannot
    hand their memory to the caller's stream while that work still runs."""
    _deferred.keep = None


# ---------------------------------------------------------------------------
# Deferred candidates (DeepSeek-V4.1)
# ---------------------------------------------------------------------------
class _Candidates:
    stream: torch.cuda.Stream | None = None
    # Recorded on the candidate stream after the deferred work.
    done: torch.cuda.Event | None = None
    # Tensors the deferred work reads that the caller's stream allocated.
    keep: tuple | None = None
    # The sparse decode rows whose block ids the deferred work expanded.
    expanded: Any = None


_candidates = _Candidates()


def init_candidate_stream(use_ubatching: bool) -> None:
    """Create the candidate stream (at model construction). Not with
    micro-batching: the deferred candidate state is one per process."""
    if (
        envs.VLLM_INDEXER_OVERLAP
        and not use_ubatching
        and _candidates.stream is None
        and torch.cuda.is_available()
    ):
        _candidates.stream = torch.cuda.Stream()


def candidate_defer_allowed(metadata: Any) -> bool:
    """Whether the candidate source may defer: a FULL CUDA graph capture of a
    decode-only batch whose later layers read the candidates through the
    sparse decode rows (their first reader joins)."""
    return (
        _candidates.stream is not None
        and decode_only(metadata)
        and getattr(metadata, "sparse_decode", None) is not None
        and full_cuda_graph_capture()
    )


def defer_candidates(work: Callable[[], None], keep: tuple) -> None:
    """Run `work()` on the candidate stream once the caller's stream reaches
    this point, without joining. `keep` holds the caller-allocated tensors
    `work` reads until `join_candidates`."""
    assert _candidates.done is None, "deferred candidates were never joined"
    stream = _candidates.stream
    assert stream is not None
    start = torch.cuda.Event()
    start.record(torch.cuda.current_stream())
    stream.wait_event(start)
    with torch.cuda.stream(stream):
        work()
        done = torch.cuda.Event()
        done.record(stream)
    _candidates.done = done
    _candidates.keep = keep
    logger.info_once("Sparse indexer overlap: deferred candidate selection active.")


def join_candidates() -> None:
    """Called by the first reader of the candidates: wait for the deferred
    work, if any."""
    if _candidates.done is None:
        return
    torch.cuda.current_stream().wait_event(_candidates.done)
    _candidates.done = None
    _candidates.keep = None


def take_expanded(rows: Any) -> bool:
    """True once, for the rows whose sparse block ids the deferred work
    already expanded."""
    if _candidates.expanded is not None and _candidates.expanded is rows:
        _candidates.expanded = None
        return True
    return False


def expand_sparse_decode_rows(
    metadata: Any, candidate_blocks: torch.Tensor, candidate_block_size: int
) -> None:
    """Expand the candidates into the sparse decode rows' block ids.

    The same call, with the same arguments, as the first consumer's
    `sparse_mqa_logits_paged_decode` makes before building its DeepGEMM
    schedule; that consumer then only builds the schedule (`take_expanded`).
    """
    from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
        candidate_blocks_to_sparse_indices,
    )

    rows = metadata.sparse_decode
    num_rows = rows.row_ke.shape[0]
    if num_rows == 0 or rows.kernel_metadata is not None:
        return
    candidate_blocks_to_sparse_indices(
        candidate_blocks[:num_rows],
        rows.row_ks,
        rows.row_ke,
        candidate_block_size,
        metadata.sparse_block_kv,
        out=(rows.sparse_indices, rows.end),
    )
    _candidates.expanded = rows
