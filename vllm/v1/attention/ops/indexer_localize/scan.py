# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-local decode scan, top-k and merge of the sparse-attention indexer.

Each locality domain's green-context stream runs the unchanged DeepGEMM
`fp8_fp4_paged_mqa_logits` over the indexer pages in its own memory (the
compact block table, context lengths and schedule `build.py` made for the
domain), with DeepGEMM's programmatic dependent launch off for these
launches. Then either:

* `localized_paged_mqa_topk`: each domain runs the stock `cooperative_topk`
  on its compact scores, and one kernel (`merge.cu`; a Triton kernel for
  shapes it does not cover) merges the 2 x K candidates into the K
  request-local token positions: the same selected score multiset as the
  single-launch scan and top-k (ties may pick other tokens); or
* `localized_paged_mqa_logits`: one kernel joins both domains' compact scores
  into the full-length score array with the stock layout, bit-identical to
  the single-launch scan, for consumers that need the whole array.
"""

import ctypes
from collections.abc import Callable
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers import indexer_overlap
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.indexer_localize.build import (
    LocalizedDecodeMetadata,
    ensure_built,
)
from vllm.v1.attention.ops.indexer_localize.kernels import CudaKernel, get_kernel
from vllm.v1.attention.ops.indexer_localize.placement import find_buffer

logger = init_logger(__name__)


def _view_matches(loc: LocalizedDecodeMetadata, kv_cache: torch.Tensor) -> bool:
    """Whether the layer's cache view has the group's page geometry.

    The page count may differ from the builder's view (the extensible KV
    cache grows after capture and some models keep the view they first saw);
    page `p` starts at `data_ptr + p * stride` either way and block ids stay
    below the live page count, so only the rows, the stride and the
    allocation matter.
    """
    return (
        kv_cache.shape[1] == loc.page_rows
        and kv_cache.stride(0) * kv_cache.element_size() == loc.page_stride
        and find_buffer(kv_cache.data_ptr()) is not None
    )


def _scan(
    loc: LocalizedDecodeMetadata,
    domain: int,
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    max_model_len: int,
    logits_dtype: torch.dtype,
    indices: torch.Tensor | None,
) -> torch.Tensor:
    """One domain's DeepGEMM scan, on the current (the domain's) stream."""
    from vllm.utils.deep_gemm import (
        fp8_fp4_paged_mqa_logits,
        get_num_sms,
        get_pdl,
        set_num_sms,
        set_pdl,
    )

    num_sms = get_num_sms()
    set_num_sms(loc.resources.sms[domain])
    pdl = get_pdl()
    if pdl:
        set_pdl(False)
    try:
        return fp8_fp4_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            loc.seq_lens[domain],
            loc.block_tables[domain],
            loc.schedules[domain],
            max_model_len=max_model_len,
            logits_dtype=logits_dtype,
            indices=indices,
        )
    finally:
        set_num_sms(num_sms)
        if pdl:
            set_pdl(True)


# ---------------------------------------------------------------------------
# Join of both domains' compact scores
# ---------------------------------------------------------------------------
@triton.jit
def _join_fallback_kernel(
    src0_ptr,
    src1_ptr,
    src_stride,
    dst_ptr,
    dst_stride,
    row_ends_ptr,
    jmap_ptr,
    jmap_stride,
    R: tl.constexpr,
    PB: tl.constexpr,
):
    """dst[row, p*R + i] = src_{dom(p)}[row, local_col(p) + i] for columns
    below the row end, in [PB pages x R rows] tiles contiguous along R."""
    row = tl.program_id(0)
    end = tl.load(row_ends_ptr + row)
    npages = (end + R - 1) // R
    r64 = row.to(tl.int64)
    src0 = src0_ptr + r64 * src_stride
    src1 = src1_ptr + r64 * src_stride
    dst = dst_ptr + r64 * dst_stride
    jm = jmap_ptr + r64 * jmap_stride
    i = tl.arange(0, R)
    for p0 in range(tl.program_id(1) * PB, npages, tl.num_programs(1) * PB):
        p = p0 + tl.arange(0, PB)
        mp = p < npages
        e = tl.load(jm + p, mask=mp, other=0)
        col = (e >> 1)[:, None] + i[None, :]
        gcol = p[:, None] * R + i[None, :]
        on1 = ((e & 1) != 0)[:, None]
        m = mp[:, None] & (gcol < end)
        v = tl.load(tl.where(on1, src1, src0) + col, mask=m)
        tl.store(dst + gcol, v.to(dst_ptr.dtype.element_ty), mask=m)


@triton.jit
def _copy_pages(s0, off, m, drow, p, i, R: tl.constexpr):
    """out[p * R + i] = src0[off + i] for the pages of a tile; off is the
    per-page column, plus src1 - src0 for pages of domain 1. (Triton loses
    the 16-byte alignment through a per-page select, of pointers or of
    offsets, and keeps it through (e & 1) * delta.)"""
    v = tl.load(s0 + off[:, None] + i[None, :], mask=m[:, None])
    tl.store(
        drow + p[:, None] * R + i[None, :],
        v.to(drow.dtype.element_ty),
        mask=m[:, None],
    )


@triton.jit
def _join_kernel(
    src0_ptr,
    src1_ptr,
    src_stride,
    dst_ptr,
    dst_stride,
    row_ends_ptr,
    jmap_ptr,
    jmap_stride,
    num_rows,
    delta,  # src1 - src0 in elements
    R: tl.constexpr,
    PB: tl.constexpr,
    LV: tl.constexpr,  # log2 of the vector width in source elements
    ROWS_P2: tl.constexpr,
):
    """The fallback kernel's bytes with 16-byte copies of whole pages.

    Program `pid` copies an equal share of all rows' full pages laid end to
    end; the partial last page of each row is copied element by element. A
    tile whose source columns are not vector-aligned is copied element by
    element too.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    i = tl.arange(0, R)
    pb = tl.arange(0, PB)

    # Full pages: program pid copies global full pages [lo, hi).
    rows = tl.arange(0, ROWS_P2)
    ends = tl.load(row_ends_ptr + rows, mask=rows < num_rows, other=0)
    nfull = tl.maximum(ends, 0) // R
    cum = tl.cumsum(nfull, axis=0)  # inclusive prefix of full pages
    total = tl.sum(nfull, axis=0).to(tl.int64)
    lo = total * pid // nprog
    hi = total * (pid + 1) // nprog
    r = tl.sum((cum.to(tl.int64) <= lo).to(tl.int32), axis=0)  # row of page lo
    base = tl.sum(tl.where(rows < r, nfull, 0), axis=0).to(tl.int64)
    gp = lo
    while gp < hi:
        end_r = tl.load(row_ends_ptr + r)
        nf = (tl.maximum(end_r, 0) // R).to(tl.int64)
        p_lo = (gp - base).to(tl.int32)
        p_hi = tl.minimum(hi - base, nf).to(tl.int32)
        r64 = r.to(tl.int64)
        s0 = src0_ptr + r64 * src_stride
        drow = dst_ptr + r64 * dst_stride
        jrow = jmap_ptr + r64 * jmap_stride
        e_next = tl.load(jrow + p_lo + pb, mask=(p_lo + pb) < p_hi, other=0)
        for p0 in range(p_lo, p_hi, PB):
            p = p0 + pb
            mp = p < p_hi
            e = e_next
            e_next = tl.load(jrow + p + PB, mask=(p + PB) < p_hi, other=0)
            m = mp
            die = (e & 1) * delta  # keeps delta's divisibility, tl.where not
            # (e >> (LV + 1)) << LV equals e >> 1 when that is a multiple of
            # the vector, and the compiler can see it.
            bad = tl.max(tl.where(m, (e >> 1) & ((1 << LV) - 1), 0), axis=0)
            if bad == 0:
                _copy_pages(s0, die + ((e >> (LV + 1)) << LV), m, drow, p, i, R)
            else:
                _copy_pages(s0, die + (e >> 1), m, drow, p, i, R)
        gp = base + p_hi
        base = base + nf
        r = r + 1

    # The partial last page of rows pid, pid + nprog, ...
    for rr in range(pid, num_rows, nprog):
        end_r = tl.load(row_ends_ptr + rr)
        nf = tl.maximum(end_r, 0) // R
        rem = end_r - nf * R
        if rem > 0:
            r64 = rr.to(tl.int64)  # type: ignore[attr-defined]
            e = tl.load(jmap_ptr + r64 * jmap_stride + nf)
            m = i < rem
            sp = tl.where((e & 1) != 0, src1_ptr, src0_ptr) + r64 * src_stride
            v = tl.load(sp + (e >> 1) + i, mask=m)
            tl.store(
                dst_ptr + r64 * dst_stride + nf * R + i,
                v.to(dst_ptr.dtype.element_ty),
                mask=m,
            )


# Join geometry (tuned on Rubin): 6 programs per SM, 32 pages x 4 warps.
_JOIN_PROGRAMS_PER_SM = 6
_JOIN_PAGES_PER_TILE = 32
_JOIN_WARPS = 4


def _join_fast_path_ok(
    o0: torch.Tensor, o1: torch.Tensor, out: torch.Tensor, page_rows: int
) -> bool:
    """Whether 16-byte copies apply: unit column strides, row strides a
    multiple of 16 elements, 16-byte aligned bases, whole vectors per page."""
    vec = 16 // o0.element_size()
    return (
        o0.dtype == o1.dtype
        and o0.element_size() in (2, 4)
        and out.element_size() in (2, 4)
        and o0.stride(1) == 1
        and o1.stride(1) == 1
        and out.stride(1) == 1
        and o0.stride(0) % 16 == 0
        and out.stride(0) % 16 == 0
        and o0.data_ptr() % 16 == 0
        and o1.data_ptr() % 16 == 0
        and out.data_ptr() % 16 == 0
        and page_rows % vec == 0
        and page_rows >= 8
    )


def localized_join_alloc(
    o0: torch.Tensor, want_fp32: Callable[[torch.Tensor], bool]
) -> torch.Tensor:
    """The full-length score array for these compact scores: DeepGEMM's
    shape and row stride, with storage for whole rows; FP32 when
    `want_fp32(o0)` says the consumer would upcast, else `o0`'s dtype."""
    dtype = torch.float32 if want_fp32(o0) else o0.dtype
    rows, width = o0.shape
    full = torch.empty((rows, max(o0.stride(0), width)), dtype=dtype, device=o0.device)
    return full[:, :width]


def localized_join_scores(
    loc: LocalizedDecodeMetadata,
    o0: torch.Tensor,
    o1: torch.Tensor,
    out: torch.Tensor,
    row_ends: torch.Tensor,
) -> None:
    """Join both domains' compact scores into `out`, on the current stream.

    Writes `out[r, c]` for `c < row_ends[r]` only (bit-identical to the
    single-launch scan, converted to `out`'s dtype); columns at or past a
    row's end stay unwritten, as DeepGEMM leaves them.
    """
    num_rows, width = o0.shape
    if num_rows == 0 or width == 0:
        return
    assert o0.stride() == o1.stride()
    page_rows = loc.page_rows
    if _join_fast_path_ok(o0, o1, out, page_rows):
        _join_kernel[(_JOIN_PROGRAMS_PER_SM * loc.resources.total_sms,)](
            o0,
            o1,
            o0.stride(0),
            out,
            out.stride(0),
            row_ends,
            loc.join_map,
            loc.join_map.stride(0),
            num_rows,
            (o1.data_ptr() - o0.data_ptr()) // o0.element_size(),
            R=page_rows,
            PB=_JOIN_PAGES_PER_TILE,
            LV=3 if o0.element_size() == 2 else 2,
            ROWS_P2=max(16, triton.next_power_of_2(num_rows)),
            num_warps=_JOIN_WARPS,
        )
        return
    logger.warning_once(
        "Die-local indexer: score arrays outside the join's 16-byte path; "
        "using its element-wise kernel."
    )
    pages_per_program = max(1, 2048 // page_rows)
    splits = min(
        triton.cdiv(width, pages_per_program * page_rows),
        max(1, 2 * sum(loc.resources.sms) // num_rows),
    )
    _join_fallback_kernel[(num_rows, max(1, splits))](
        o0,
        o1,
        o0.stride(0),
        out,
        out.stride(0),
        row_ends,
        loc.join_map,
        loc.join_map.stride(0),
        R=page_rows,
        PB=pages_per_program,
        num_warps=8,
    )


def localized_paged_mqa_logits(
    loc: LocalizedDecodeMetadata,
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    max_model_len: int,
    logits_dtype: torch.dtype,
    indices: torch.Tensor | None,
    want_fp32: Callable[[torch.Tensor], bool],
) -> torch.Tensor | None:
    """Die-split decode scan and join.

    Returns the scores with the stock layout (FP32 if `want_fp32(bf16)` says
    the consumer upcasts them), or None when this call cannot be localized
    (the caller runs the stock scan).
    """
    rows = q[0].shape[0] * q[0].shape[1]
    if not _view_matches(loc, kv_cache) or rows != loc.seq_lens[0].shape[0]:
        logger.warning_once(
            "Die-local indexer: a decode scan does not match its localized "
            "metadata; stock scan for it."
        )
        return None
    ensure_built(loc)
    o0, o1 = torch.cuda.execute_on_streams(
        loc.resources.streams,
        lambda d: _scan(
            loc, d, q, kv_cache, weights, max_model_len, logits_dtype, indices
        ),
        range(len(loc.resources.streams)),
    )
    out = localized_join_alloc(o0, want_fp32)
    localized_join_scores(loc, o0, o1, out, seq_lens.reshape(-1))
    return out


# ---------------------------------------------------------------------------
# Per-domain top-k and merge
# ---------------------------------------------------------------------------
@triton.jit
def _merge_topk_kernel(
    l0_ptr,
    l1_ptr,
    l_stride,
    i0_ptr,
    i1_ptr,
    i_stride,
    inv0_ptr,
    inv1_ptr,
    inv_stride,
    out_ptr,
    out_stride,
    K: tl.constexpr,
    R: tl.constexpr,
    KEY32: tl.constexpr = False,
):
    """Top-K of the 2K per-domain candidates of a row. Scores are gathered
    from each domain's compact logits, candidate columns mapped to global
    positions through the domain's local->global page map. Writes K global
    positions (-1 when the row has fewer than K valid keys)."""
    row = tl.program_id(0)
    r64 = row.to(tl.int64)
    j = tl.arange(0, 2 * K)
    first = j < K
    jj = tl.where(first, j, j - K)
    c0 = tl.load(i0_ptr + r64 * i_stride + jj, mask=first, other=-1)
    c1 = tl.load(i1_ptr + r64 * i_stride + jj, mask=~first, other=-1)
    col = tl.where(first, c0, c1)
    valid = col >= 0
    colc = tl.maximum(col, 0)
    lp = colc // R
    g0 = tl.load(inv0_ptr + r64 * inv_stride + lp, mask=valid & first, other=0)
    g1 = tl.load(inv1_ptr + r64 * inv_stride + lp, mask=valid & ~first, other=0)
    gpos = tl.where(first, g0, g1) * R + colc % R
    s0 = tl.load(l0_ptr + r64 * l_stride + colc, mask=valid & first, other=0.0)
    s1 = tl.load(l1_ptr + r64 * l_stride + colc, mask=valid & ~first, other=0.0)
    s = tl.where(first, s0.to(tl.float32), s1.to(tl.float32))
    nvalid = tl.sum(valid.to(tl.int32), axis=0)
    bins = tl.arange(0, 256)
    bits = s.to(tl.int32, bitcast=True)
    if KEY32:
        # FP32 scores: exact order-preserving 32-bit key (signed-int order ==
        # float order after flipping the magnitude bits of negatives), shifted
        # to 0..2^32-1; invalid -> below every valid key. K-th largest key by
        # a four-pass (8 + 8 + 8 + 8 bit) radix select.
        key32 = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
        u32 = tl.where(valid, key32.to(tl.int64) + 2147483648, 0)
        need_r = K - 0 * nvalid
        live = valid
        acc = 0 * nvalid.to(tl.int64)
        for ps in tl.static_range(4):
            dg = ((u32 >> (24 - 8 * ps)) & 0xFF).to(tl.int32)
            hh = tl.histogram(dg, 256, mask=live)
            suf = tl.sum(hh, axis=0) - tl.cumsum(hh, axis=0) + hh  # digit >= b
            bb = tl.max(tl.where(suf >= need_r, bins, 0), axis=0)
            need_r = need_r - tl.sum(tl.where(bins > bb, hh, 0), axis=0)
            live = live & (dg == bb)
            acc = acc * 256 + bb.to(tl.int64)
        thr = tl.where(nvalid >= K, acc, -1)
        k64 = tl.where(valid, u32, -2)
    else:
        # Order-preserving key of the BF16 score (its fp32 bits >> 16, exact);
        # invalid -> below every valid key.
        key = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF) >> 16
        u = tl.where(valid, key + 32768, 0)  # 0..65535
        # K-th largest 16-bit key by a two-pass (8 + 8 bit) radix select.
        hb = u >> 8
        h1 = tl.histogram(hb, 256, mask=valid)
        suf1 = tl.sum(h1, axis=0) - tl.cumsum(h1, axis=0) + h1  # bin >= b
        b1 = tl.max(tl.where(suf1 >= K, bins, 0), axis=0)
        above1 = tl.sum(tl.where(bins > b1, h1, 0), axis=0)
        in1 = valid & (hb == b1)
        h2 = tl.histogram(u & 0xFF, 256, mask=in1)
        suf2 = tl.sum(h2, axis=0) - tl.cumsum(h2, axis=0) + h2
        b2 = tl.max(tl.where(suf2 >= K - above1, bins, 0), axis=0)
        thr_u = tl.where(nvalid >= K, b1 * 256 + b2, -1)
        k64 = tl.where(valid, u, -2).to(tl.int64)
        thr = thr_u.to(tl.int64)
    gt = valid & (k64 > thr)
    eq = valid & (k64 == thr)
    need = K - tl.sum(gt.to(tl.int32), axis=0)
    eqc = tl.cumsum(eq.to(tl.int32), axis=0)
    sel = gt | (eq & (eqc <= need))
    pos = tl.cumsum(sel.to(tl.int32), axis=0) - 1
    nsel = tl.sum(sel.to(tl.int32), axis=0)
    orow = out_ptr + r64 * out_stride
    tl.store(orow + pos, gpos, mask=sel)
    tl.store(orow + j, tl.full((2 * K,), -1, tl.int32), mask=(j >= nsel) & (j < K))


# CTA size of the CUDA merge per top-k.
_MERGE_THREADS = {512: 512, 1024: 512, 2048: 1024}
_MERGE_PAGE_ROWS = (64, 128)


def _merge_smem_bytes(topk: int, threads: int, score_f32: bool) -> int:
    """`SMEM_TOTAL` of merge.cu."""
    hist = 4 * (1024 if score_f32 else 512)
    end = hist + 32 + 4 * (threads // 32) + 4 * topk
    return -(-end // 128) * 128


def _merge_kernel(
    topk: int, score_f32: bool, page_rows: int, device_index: int
) -> CudaKernel | None:
    """The CUDA merge for this shape, compiled at its first use outside a
    CUDA graph capture (a first use inside one takes the Triton merge)."""
    if topk not in _MERGE_THREADS or page_rows not in _MERGE_PAGE_ROWS:
        return None
    defines = (
        ("TOP_K", topk),
        ("THREADS", _MERGE_THREADS[topk]),
        ("SCORE_F32", int(score_f32)),
        ("LOG2_PAGE_ROWS", page_rows.bit_length() - 1),
    )
    return get_kernel(
        "merge.cu",
        "indexer_localize_merge",
        defines,
        device_index,
        build=not torch.cuda.is_current_stream_capturing(),
    )


def prepare_merge(
    topk: int, logits_dtype: torch.dtype, page_rows: int, device_index: int
) -> None:
    """Compile the CUDA merge for a shape ahead of its first use, outside any
    CUDA graph capture."""
    if not torch.cuda.is_current_stream_capturing():
        _merge_kernel(topk, logits_dtype == torch.float32, page_rows, device_index)


def _merge_fits(
    o0: torch.Tensor,
    o1: torch.Tensor,
    idx: list[torch.Tensor],
    inv: list[torch.Tensor],
    out: torch.Tensor,
) -> bool:
    """The CUDA merge's buffer requirements: one row stride per pair, unit
    inner strides, 16-byte aligned rows for its vector loads (candidate
    columns) and stores (output), 32-bit strides."""
    return (
        o0.stride(0) == o1.stride(0)
        and o0.stride(1) == 1
        and o1.stride(1) == 1
        and idx[0].stride(0) == idx[1].stride(0)
        and inv[0].stride(0) == inv[1].stride(0)
        and all(t.stride(-1) == 1 for t in (idx[0], idx[1], inv[0], inv[1], out))
        and idx[0].stride(0) % 4 == 0
        and out.stride(0) % 4 == 0
        and all(t.data_ptr() % 16 == 0 for t in (idx[0], idx[1], out))
        and max(o0.stride(0), inv[0].stride(0), out.stride(0)) < 2**31
    )


def merge_candidates(
    loc: LocalizedDecodeMetadata,
    o0: torch.Tensor,
    o1: torch.Tensor,
    idx: list[torch.Tensor],
    topk_tokens: int,
    topk_out: torch.Tensor,
) -> None:
    """Merge both domains' top-k candidates into `topk_out`, on the current
    stream."""
    rows = topk_out.shape[0]
    score_f32 = o0.dtype == torch.float32
    kernel = _merge_kernel(topk_tokens, score_f32, loc.page_rows, loc.resources.device)
    if kernel is not None and _merge_fits(o0, o1, idx, loc.inv_maps, topk_out):
        threads = _MERGE_THREADS[topk_tokens]

        def ptr(t: torch.Tensor) -> ctypes.c_uint64:
            return ctypes.c_uint64(t.data_ptr())

        kernel.launch(
            rows,
            threads,
            _merge_smem_bytes(topk_tokens, threads, score_f32),
            [
                ptr(o0),
                ptr(o1),
                ctypes.c_int32(o0.stride(0)),
                ptr(idx[0]),
                ptr(idx[1]),
                ctypes.c_int32(idx[0].stride(0)),
                ptr(loc.inv_maps[0]),
                ptr(loc.inv_maps[1]),
                ctypes.c_int32(loc.inv_maps[0].stride(0)),
                ptr(topk_out),
                ctypes.c_int32(topk_out.stride(0)),
            ],
            torch.cuda.current_stream(),
        )
        return
    logger.warning_once(
        "Die-local indexer: no CUDA merge for top-k %d, %d-row pages and these "
        "buffers; using the Triton merge.",
        topk_tokens,
        loc.page_rows,
    )
    _merge_topk_kernel[(rows,)](
        o0,
        o1,
        o0.stride(0),
        idx[0],
        idx[1],
        idx[0].stride(0),
        loc.inv_maps[0],
        loc.inv_maps[1],
        loc.inv_maps[0].stride(0),
        topk_out,
        topk_out.stride(0),
        K=topk_tokens,
        R=loc.page_rows,
        KEY32=score_f32,
        num_warps=8 if topk_tokens >= 2048 else 4,
    )


def localized_topk_supported(
    selector: Any, rows: int, max_model_len: int, topk_tokens: int
) -> bool:
    """Whether the stock selector would run `cooperative_topk` on these BF16
    scores (the per-domain path uses that kernel: clusters, no grid sync, so
    it runs on a domain's SMs)."""
    probe = torch.empty_strided(
        (rows, max_model_len),
        (-(-max_model_len // 64) * 64, 1),
        dtype=torch.bfloat16,
        device="meta",
    )
    try:
        return selector.resolve_backend(probe, topk_tokens, rows) == "cooperative"
    except Exception:  # noqa: BLE001 - an unsupported shape
        return False


# Per-domain top-k scratch, per (device, domain) and (device, domain, topk).
_TOPK_WORKSPACES: dict[tuple[int, int], torch.Tensor] = {}
_TOPK_INDICES: dict[tuple[int, int, int], torch.Tensor] = {}


def _topk_scratch(
    loc: LocalizedDecodeMetadata, topk_tokens: int, rows: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    from vllm.model_executor.layers.indexer_topk import RADIX_TOPK_WORKSPACE_SIZE

    device = loc.resources.device
    workspaces, indices = [], []
    for d in range(len(loc.resources.streams)):
        if (device, d) not in _TOPK_WORKSPACES:
            _TOPK_WORKSPACES[device, d] = torch.empty(
                RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=device
            )
        workspaces.append(_TOPK_WORKSPACES[device, d])
        key = (device, d, topk_tokens)
        if key not in _TOPK_INDICES or _TOPK_INDICES[key].shape[0] < rows:
            _TOPK_INDICES[key] = torch.empty(
                (max(rows, loc.capacity), topk_tokens), dtype=torch.int32, device=device
            )
        indices.append(_TOPK_INDICES[key][:rows])
    return workspaces, indices


def localized_paged_mqa_topk(
    loc: LocalizedDecodeMetadata,
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    max_model_len: int,
    logits_dtype: torch.dtype,
    indices: torch.Tensor | None,
    topk_tokens: int,
    max_seq_len: int,
    topk_out: torch.Tensor,
    keep: list[torch.Tensor] | None = None,
) -> bool:
    """Per-domain scan and top-k, then the merge, into `topk_out`.

    `topk_out` is the `[rows, K]` int32 view of the top-k buffer. With
    `keep`, both domains' compact scores are appended to it (for a join the
    caller runs later). When the caller requested a deferred top-k
    (`indexer_overlap.request_deferred_topk`), the call returns without
    joining: domain 0's stream waits for domain 1 and merges. Returns False,
    without launching anything, when this call cannot be localized.
    """
    rows = q[0].shape[0] * q[0].shape[1]
    if (
        not _view_matches(loc, kv_cache)
        or logits_dtype not in (torch.bfloat16, torch.float32)
        or rows != loc.seq_lens[0].shape[0]
        or topk_out.shape[0] != rows
    ):
        logger.warning_once(
            "Die-local indexer: a decode top-k does not match its localized "
            "metadata; stock scan and top-k for it."
        )
        return False
    ensure_built(loc)
    streams = loc.resources.streams
    workspaces, idx = _topk_scratch(loc, topk_tokens, rows)

    def scan_topk(d: int) -> torch.Tensor:
        o = _scan(loc, d, q, kv_cache, weights, max_model_len, logits_dtype, indices)
        torch.ops._C.cooperative_topk(
            o, loc.seq_lens[d], idx[d], workspaces[d], topk_tokens, max_seq_len
        )
        return o

    if indexer_overlap.consume_deferred_topk_request():
        # Fork without joining: each domain stream waits for the caller's
        # work so far, scans and selects; domain 0 then waits for domain 1
        # and merges. The caller joins on domain 0's stream.
        start = torch.cuda.Event()
        start.record(torch.cuda.current_stream())
        outs, done = [], []
        for d, stream in enumerate(streams):
            stream.wait_event(start)
            with torch.cuda.stream(stream):
                outs.append(scan_topk(d))
                event = torch.cuda.Event()
                event.record(stream)
                done.append(event)
        merge_stream = streams[0]
        for event in done[1:]:
            merge_stream.wait_event(event)
        with torch.cuda.stream(merge_stream):
            merge_candidates(loc, outs[0], outs[1], idx, topk_tokens, topk_out)
        # The domain streams read q, weights and indices, allocated on the
        # caller's stream, and domain 0 reads domain 1's scores.
        indexer_overlap.set_deferred_topk(merge_stream, (q, weights, indices, outs))
        return True
    o0, o1 = torch.cuda.execute_on_streams(streams, scan_topk, range(len(streams)))
    if keep is not None:
        keep.extend((o0, o1))
    merge_candidates(loc, o0, o1, idx, topk_tokens, topk_out)
    return True
