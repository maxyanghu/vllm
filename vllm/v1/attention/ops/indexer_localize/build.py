# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step decode metadata of the die-local indexer.

Every decode step, each indexer cache group needs its rows' pages split by
locality domain (compact per-domain block tables, per-domain context lengths,
a join map and local-to-row page maps) and both domains' DeepGEMM schedules.
The indexer metadata builder only records the inputs (its persistent block
table, lengths and request indices); the first localized scan of the forward
launches one kernel (`build.cu`) that builds every recorded group, so FULL
CUDA graphs capture it and nothing runs between graphs.

The first build of every group checks the kernel against `reference_build`
(two Triton kernels and DeepGEMM's own schedule) bit for bit; a mismatch
disables the group (stock indexer).
"""

import ctypes
import weakref
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.indexer_localize.kernels import get_kernel
from vllm.v1.attention.ops.indexer_localize.placement import (
    NUM_DOMAINS,
    DomainResources,
    get_domain_resources,
    page_domain_plan,
    page_domain_table,
    unsupported_config_reason,
)

logger = init_logger(__name__)

# Reference split: pages per Triton program.
_CHUNK = 512
# Build kernel geometry (build.cu): 256 threads x 4 pages per split block.
_THREADS = 256
_BUILD_CHUNK = 1024
_RESET_ROWS = 8
_MAX_GROUPS = 4
# The schedule block keeps 16 bytes per row in shared memory.
_MAX_ROWS = 2048


# ---------------------------------------------------------------------------
# Reference build
# ---------------------------------------------------------------------------
@triton.jit
def _split_count_kernel(
    bt_ptr,
    bt_stride,
    lens_ptr,
    dom_ptr,
    cnt_ptr,
    cnt_stride,
    num_rows,
    R: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """cnt[row, c] = pages of chunk c of the row that live on domain 1."""
    row = tl.program_id(0)
    c = tl.program_id(1)
    L = tl.load(lens_ptr + row, mask=row < num_rows, other=0)
    npages = (L + R - 1) // R
    p = c * CHUNK + tl.arange(0, CHUNK)
    m = p < npages
    b = tl.load(bt_ptr + row.to(tl.int64) * bt_stride + p, mask=m, other=0)
    d = tl.load(dom_ptr + b, mask=m, other=0)
    n1 = tl.sum(((d != 0) & m).to(tl.int32), axis=0)
    tl.store(cnt_ptr + row * cnt_stride + c, n1)


@triton.jit
def _split_scatter_kernel(
    bt_ptr,
    bt_stride,
    lens_ptr,
    dom_ptr,
    cnt_ptr,
    cnt_stride,
    num_chunks,
    num_rows,
    lbt0_ptr,
    lbt1_ptr,
    lbt_stride,
    llen0_ptr,
    llen1_ptr,
    jmap_ptr,
    jmap_stride,
    inv0_ptr,
    inv1_ptr,
    width,
    R: tl.constexpr,
    CHUNK: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,
    ZPAD: tl.constexpr,
):
    """Compact per-domain block tables + join map for chunk c of a row;
    chunk 0 also writes the per-domain lengths and zero-pads the tables.
    Rows from num_rows up to the buffer capacity get zero lengths (padded
    rows of a FULL CUDA graph), as the stock decode seq_lens do."""
    row = tl.program_id(0)
    c = tl.program_id(1)
    L = tl.load(lens_ptr + row, mask=row < num_rows, other=0)
    npages = (L + R - 1) // R
    r64 = row.to(tl.int64)
    bt_row = bt_ptr + r64 * bt_stride
    lbt0 = lbt0_ptr + r64 * lbt_stride
    lbt1 = lbt1_ptr + r64 * lbt_stride
    ci = tl.arange(0, MAX_CHUNKS)
    cnt = tl.load(cnt_ptr + row * cnt_stride + ci, mask=ci < num_chunks, other=0)
    if c * CHUNK < npages:
        pre1 = tl.sum(tl.where(ci < c, cnt, 0), axis=0)
        pre0 = c * CHUNK - pre1  # earlier chunks are full
        p = c * CHUNK + tl.arange(0, CHUNK)
        m = p < npages
        b = tl.load(bt_row + p, mask=m, other=0)
        d = tl.load(dom_ptr + b, mask=m, other=0)
        is1 = (d != 0) & m
        is0 = (d == 0) & m
        i0 = is0.to(tl.int32)
        i1 = is1.to(tl.int32)
        pos0 = pre0 + tl.cumsum(i0, axis=0) - i0
        pos1 = pre1 + tl.cumsum(i1, axis=0) - i1
        tl.store(lbt0 + pos0, b, mask=is0)
        tl.store(lbt1 + pos1, b, mask=is1)
        # local page -> global page (same row stride as the local tables)
        tl.store(inv0_ptr + r64 * lbt_stride + pos0, p, mask=is0)
        tl.store(inv1_ptr + r64 * lbt_stride + pos1, p, mask=is1)
        e = tl.where(is1, pos1 * (2 * R) + 1, pos0 * (2 * R))
        tl.store(jmap_ptr + r64 * jmap_stride + p, e, mask=m)
    if c == 0:
        n1 = tl.sum(cnt, axis=0)
        n0 = npages - n1
        last_ok = npages > 0
        last_b = tl.load(bt_row + tl.maximum(npages - 1, 0), mask=last_ok, other=0)
        d_last = tl.load(dom_ptr + last_b, mask=last_ok, other=0)
        short = R * npages - L  # rows missing from the last page
        len0 = n0 * R - tl.where(last_ok & (d_last == 0), short, 0)
        len1 = n1 * R - tl.where(last_ok & (d_last != 0), short, 0)
        tl.store(llen0_ptr + row, len0)
        tl.store(llen1_ptr + row, len1)
        # Valid ids past each local page list (any DeepGEMM prefetch stays in
        # the cache allocation).
        z = tl.arange(0, ZPAD)
        zero = tl.zeros((ZPAD,), dtype=tl.int32)
        tl.store(lbt0 + n0 + z, zero, mask=(n0 + z) < width)
        tl.store(lbt1 + n1 + z, zero, mask=(n1 + z) < width)


def reference_build(
    block_table: torch.Tensor,
    lens: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    dom_table: torch.Tensor,
    page_rows: int,
    sms: list[int],
    out: dict[str, Any],
) -> None:
    """The per-step build with Triton and DeepGEMM, into the buffers `out`.

    Splits each row's pages by domain (two launches, capacity rows x chunks;
    rows past `num_rows` are reset) and builds each domain's DeepGEMM
    schedule for its SM count. `out` holds `bt`, `lens`, `inv` and `sched`
    (one tensor per domain), `jmap` and the scratch `cnt`.
    """
    from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata

    local_bt, local_lens, inv = out["bt"], out["lens"], out["inv"]
    cap, width = local_bt[0].shape
    n_chunks = triton.cdiv(width, _CHUNK)
    cnt = out["cnt"]
    _split_count_kernel[(cap, n_chunks)](
        block_table,
        block_table.stride(0),
        lens,
        dom_table,
        cnt,
        cnt.stride(0),
        num_rows,
        R=page_rows,
        CHUNK=_CHUNK,
        num_warps=4,
    )
    _split_scatter_kernel[(cap, n_chunks)](
        block_table,
        block_table.stride(0),
        lens,
        dom_table,
        cnt,
        cnt.stride(0),
        n_chunks,
        num_rows,
        local_bt[0],
        local_bt[1],
        local_bt[0].stride(0),
        local_lens[0],
        local_lens[1],
        out["jmap"],
        out["jmap"].stride(0),
        inv[0],
        inv[1],
        width,
        R=page_rows,
        CHUNK=_CHUNK,
        MAX_CHUNKS=triton.next_power_of_2(n_chunks),
        ZPAD=64,
        num_warps=4,
    )
    for d, num_sms in enumerate(sms):
        meta = get_paged_mqa_logits_metadata(
            local_lens[d][:num_rows], page_rows, num_sms, indices=indices
        )
        out["sched"][d][: meta.shape[0]].copy_(meta)


# ---------------------------------------------------------------------------
# Build kernel launcher
# ---------------------------------------------------------------------------
_PTRS = (
    "bt",
    "lens",
    "dom",
    "indices",
    "lbt0",
    "lbt1",
    "inv0",
    "inv1",
    "jmap",
    "llen0",
    "llen1",
    "sched0",
    "sched1",
    "flags",
    "state",
)
_I64 = ("bt_stride", "out_stride", "jmap_stride", "dom_q0")
_I32 = (
    "R",
    "width",
    "cap",
    "num_rows",
    "nch",
    "split_begin",
    "reset_begin",
    "reset_end",
    "dom_k",
    "pad0",
)


class _Group(ctypes.Structure):
    _fields_ = (
        [(n, ctypes.c_uint64) for n in _PTRS]
        + [(n, ctypes.c_int64) for n in _I64]
        + [(n, ctypes.c_int32) for n in _I32]
    )


class _Params(ctypes.Structure):
    _fields_ = [
        ("g", _Group * _MAX_GROUPS),
        ("num_groups", ctypes.c_int32),
        ("sms0", ctypes.c_int32),
        ("sms1", ctypes.c_int32),
        ("rows_max", ctypes.c_int32),
    ]


assert ctypes.sizeof(_Group) == 192 and ctypes.sizeof(_Params) == 784


@dataclass
class _GroupArgs:
    """One group's tensors for a launch: the inputs (block table, lengths,
    request indices, the page-to-domain table), the persistent outputs and the
    group's zero-initialized flags and epoch."""

    bt: torch.Tensor
    lens: torch.Tensor
    dom: torch.Tensor
    indices: torch.Tensor
    lbt: list[torch.Tensor]
    inv: list[torch.Tensor]
    jmap: torch.Tensor
    llen: list[torch.Tensor]
    sched: list[torch.Tensor]
    flags: torch.Tensor
    state: torch.Tensor
    R: int
    width: int
    cap: int
    num_rows: int
    # dom_k > 0: domain(b) = ((dom_q0 + b) // dom_k) % 2, no table reads.
    dom_q0: int = 0
    dom_k: int = 0


def _num_chunks(width: int) -> int:
    return max(1, -(-width // _BUILD_CHUNK))


def _pack(groups: list[_GroupArgs], sms: list[int]) -> tuple[_Params, int, int]:
    """The parameter block, grid size and dynamic shared memory of a launch."""
    p = _Params()
    blocks = 0
    for i, g in enumerate(groups):
        p.g[i].split_begin = blocks  # the schedule block; split blocks follow
        blocks += 1 + g.num_rows * _num_chunks(g.width)
    for i, g in enumerate(groups):
        G = p.g[i]
        num_reset = -(-max(0, g.cap - g.num_rows) // _RESET_ROWS)
        G.reset_begin, G.reset_end = blocks, blocks + num_reset
        blocks += num_reset
        tensors = (
            g.bt,
            g.lens,
            g.dom,
            g.indices,
            g.lbt[0],
            g.lbt[1],
            g.inv[0],
            g.inv[1],
            g.jmap,
            g.llen[0],
            g.llen[1],
            g.sched[0],
            g.sched[1],
            g.flags,
            g.state,
        )
        for name, t in zip(_PTRS, tensors):
            setattr(G, name, t.data_ptr())
        G.bt_stride = g.bt.stride(0)
        G.out_stride = g.lbt[0].stride(0)
        G.jmap_stride = g.jmap.stride(0)
        G.R, G.width, G.cap, G.num_rows = g.R, g.width, g.cap, g.num_rows
        G.nch = _num_chunks(g.width)
        G.dom_q0, G.dom_k = g.dom_q0, g.dom_k
    p.num_groups = len(groups)
    p.sms0, p.sms1 = sms
    p.rows_max = max(g.num_rows for g in groups)
    return p, blocks, 16 * p.rows_max


class _BuildKernel:
    """Launches `build.cu` for up to `_MAX_GROUPS` groups at once."""

    def __init__(self, device_index: int):
        kernel = get_kernel("build.cu", "indexer_localize_build", (), device_index)
        if kernel is None:
            raise RuntimeError("the build kernel is unavailable")
        self._kernel = kernel
        # Packed parameters per list of _GroupArgs, checked by identity: the
        # localizers keep one _GroupArgs per set of tensors.
        self._packed: dict[tuple, tuple] = {}

    def launch(self, groups: list[_GroupArgs], sms: list[int]) -> None:
        key = tuple(id(g) for g in groups) + tuple(sms)
        entry = self._packed.get(key)
        if entry is None or any(r() is not g for r, g in zip(entry[0], groups)):
            params, grid, smem = _pack(groups, sms)
            entry = ([weakref.ref(g) for g in groups], params, grid, smem)
            self._packed[key] = entry
            if len(self._packed) > 1024:
                self._packed = {
                    k: e
                    for k, e in self._packed.items()
                    if all(r() is not None for r in e[0])
                }
        _, params, grid, smem = entry
        self._kernel.launch(grid, _THREADS, smem, [params], torch.cuda.current_stream())


_BUILD_KERNELS: dict[int, _BuildKernel | None] = {}


def _build_kernel(device_index: int) -> _BuildKernel | None:
    if device_index not in _BUILD_KERNELS:
        try:
            _BUILD_KERNELS[device_index] = _BuildKernel(device_index)
        except RuntimeError:
            _BUILD_KERNELS[device_index] = None
    return _BUILD_KERNELS[device_index]


# ---------------------------------------------------------------------------
# Localizer
# ---------------------------------------------------------------------------
class _PendingBuild:
    """A group's build, recorded by its metadata builder for the forward's
    first localized scan."""

    __slots__ = ("args", "done")

    def __init__(self, args: _GroupArgs):
        self.args = args
        self.done = False


# Builds recorded since the last launch, per device and localizer.
_PENDING: dict[int, dict[int, _PendingBuild]] = {}


@dataclass
class LocalizedDecodeMetadata:
    """Per-step, per-domain decode scan inputs (views of persistent buffers),
    shared by every indexer layer of one cache group."""

    page_rows: int
    capacity: int
    block_tables: list[torch.Tensor]  # [T, W] int32 per domain
    seq_lens: list[torch.Tensor]  # [T, 1] int32 per domain
    schedules: list[torch.Tensor]  # DeepGEMM schedule per domain
    join_map: torch.Tensor  # [T, W] int32: (local_col << 1) | domain
    inv_maps: list[torch.Tensor]  # [T, W] int32 per domain: local -> row page
    page_stride: int  # bytes between pages of the indexer cache view
    resources: DomainResources = field(repr=False)
    pending: _PendingBuild | None = field(default=None, repr=False)


def ensure_built(loc: LocalizedDecodeMetadata) -> None:
    """Launch every build recorded since the last launch, on this stream.

    Called by each localized scan before its fork; only the forward's first
    call launches (inside the CUDA graph while one is captured).
    """
    pending = loc.pending
    if pending is None or pending.done:
        return
    device = loc.resources.device
    todo = [p for p in _PENDING.get(device, {}).values() if not p.done]
    if pending not in todo:
        todo.append(pending)
    kernel = _build_kernel(device)
    assert kernel is not None
    for i in range(0, len(todo), _MAX_GROUPS):
        kernel.launch([p.args for p in todo[i : i + _MAX_GROUPS]], loc.resources.sms)
    for p in todo:
        p.done = True


class IndexerDecodeLocalizer:
    """Builds `LocalizedDecodeMetadata` for one indexer cache group each
    decode step. Owned by the group's indexer metadata builder."""

    def __init__(
        self,
        layer_names: list[str],
        static_forward_context: dict[str, Any],
        device: torch.device,
        max_rows: int,
        width: int,
        page_rows: int,
    ):
        self.layer_names = list(layer_names)
        self.ctx = static_forward_context
        self.device = device
        self.max_rows = max_rows
        self.width = width
        self.page_rows = page_rows
        self.res = get_domain_resources(device)
        self.disabled = self.res is None
        self._key: Any = None
        self._dom_table: torch.Tensor | None = None
        self._dom_plan = (0, 0)
        self.page_stride = 0
        self._bufs: dict[str, Any] | None = None
        # Per schedule path (warp for <= 32 rows, block otherwise): absent if
        # not checked yet, True if the kernel matched the reference.
        self._checked: dict[bool, bool] = {}
        self._args: dict[tuple, tuple] = {}
        self._warned = False
        if self.res is not None and _build_kernel(self.res.device) is None:
            self.disabled = True

    def _disable(self, reason: str) -> None:
        self.disabled = True
        logger.warning(
            "Die-local sparse indexer disabled for %s.. (%s); using the stock "
            "indexer for it.",
            self.layer_names[0],
            reason,
        )

    def _layer_cache(self, name: str) -> torch.Tensor | None:
        kv = getattr(self.ctx.get(name), "kv_cache", None)
        if not isinstance(kv, torch.Tensor) or kv.numel() == 0:
            return None
        if kv.dim() == 4:
            kv = kv.squeeze(1) if kv.shape[1] == 1 else kv.squeeze(-2)
        return kv

    def _resolve(self) -> bool:
        """Page-to-domain table of the bound indexer caches, recomputed when
        they change (the builder first runs on the profiling-time cache)."""
        kv0 = self._layer_cache(self.layer_names[0]) if self.layer_names else None
        if kv0 is None:
            return False
        key = (kv0.data_ptr(), tuple(kv0.shape), tuple(kv0.stride()))
        if key == self._key:
            return self._dom_table is not None
        self._key = key
        self._dom_table = None
        tables = []
        for name in self.layer_names:
            kv = self._layer_cache(name)
            if kv is None:
                return False
            table = page_domain_table(kv)
            if table is None:
                # Not in a localized allocation (yet).
                return False
            if kv.shape[1] != self.page_rows:
                self._disable(f"{kv.shape[1]} page rows, expected {self.page_rows}")
                return False
            tables.append((table, kv.stride(0) * kv.element_size()))
        table0, stride0 = tables[0]
        if any(t.shape != table0.shape or s != stride0 for t, s in tables[1:]):
            self._disable("the group's layers have different page geometries")
            return False
        self._dom_table = table0
        self._dom_plan = page_domain_plan(kv0, table0)
        self.page_stride = stride0
        return True

    def _buffers(self) -> dict[str, Any]:
        if self._bufs is None:
            assert self.res is not None
            rows, width, device = self.max_rows, self.width, self.device

            def zeros(*shape: int, dtype: torch.dtype = torch.int32):
                return torch.zeros(shape, dtype=dtype, device=device)

            self._bufs = dict(
                bt=[zeros(rows, width) for _ in range(NUM_DOMAINS)],
                lens=[zeros(rows, 1) for _ in range(NUM_DOMAINS)],
                sched=[zeros(s + 1, 2) for s in self.res.sms],
                jmap=zeros(rows, width),
                inv=[zeros(rows, width) for _ in range(NUM_DOMAINS)],
                cnt=zeros(rows, triton.cdiv(width, _CHUNK)),
                flags=zeros(rows * _num_chunks(width), dtype=torch.int64),
                state=zeros(4),
                # The kernel's page-to-domain table: a captured launch keeps
                # its address, so it is updated in place.
                dom=None,
                dom_src=None,
            )
        return self._bufs

    def _kernel_dom(self, bufs: dict[str, Any]) -> torch.Tensor:
        table = self._dom_table
        assert table is not None
        if bufs["dom_src"] is not table:
            dom = bufs["dom"]
            if dom is None or dom.numel() < table.numel():
                if dom is not None:
                    # Captured builds keep the old table's address and size.
                    raise RuntimeError(
                        "Die-local indexer: the indexer cache grew from "
                        f"{dom.numel()} to {table.numel()} pages after its "
                        "page-to-domain table was first used."
                    )
                dom = bufs["dom"] = torch.empty(
                    max(1, table.numel()), dtype=torch.int8, device=table.device
                )
            dom[: table.numel()].copy_(table)
            bufs["dom_src"] = table
        return bufs["dom"]

    def build(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        indices: torch.Tensor,
        num_rows: int,
    ) -> LocalizedDecodeMetadata | None:
        """Record this step's inputs; None runs the stock indexer instead.

        The inputs must be the builder's persistent buffers: the build kernel
        reads them when the forward's first localized scan launches it.
        """
        if self.disabled or not self._resolve():
            return None
        assert self.res is not None
        lens = seq_lens.reshape(-1)
        if (
            not 0 < num_rows <= min(self.max_rows, _MAX_ROWS)
            or block_table.shape[1] > self.width
            or block_table.stride(-1) != 1
            or block_table.dtype != torch.int32
            or lens.numel() != num_rows
            or lens.dtype != torch.int32
            or not lens.is_contiguous()
            or indices.dtype != torch.int32
            or not indices.is_contiguous()
            or indices.numel() < num_rows
        ):
            if not self._warned:
                self._warned = True
                logger.warning(
                    "Die-local indexer: unsupported decode batch (%d rows of at "
                    "most %d, width %d of at most %d); stock indexer for it.",
                    num_rows,
                    min(self.max_rows, _MAX_ROWS),
                    block_table.shape[1],
                    self.width,
                )
            return None
        bufs = self._buffers()
        dom = self._kernel_dom(bufs)
        q0, k = self._dom_plan
        key = (
            num_rows,
            block_table.data_ptr(),
            block_table.stride(0),
            lens.data_ptr(),
            indices.data_ptr(),
            dom.data_ptr(),
            q0,
            k,
        )
        entry = self._args.get(key)
        if entry is None:
            args = _GroupArgs(
                bt=block_table,
                lens=lens,
                dom=dom,
                indices=indices[:num_rows],
                lbt=bufs["bt"],
                inv=bufs["inv"],
                jmap=bufs["jmap"],
                llen=bufs["lens"],
                sched=bufs["sched"],
                flags=bufs["flags"],
                state=bufs["state"],
                R=self.page_rows,
                width=self.width,
                cap=self.max_rows,
                num_rows=num_rows,
                dom_q0=q0,
                dom_k=k,
            )
            meta = LocalizedDecodeMetadata(
                page_rows=self.page_rows,
                capacity=self.max_rows,
                block_tables=[b[:num_rows] for b in bufs["bt"]],
                seq_lens=[x[:num_rows] for x in bufs["lens"]],
                schedules=list(bufs["sched"]),
                join_map=bufs["jmap"][:num_rows],
                inv_maps=[x[:num_rows] for x in bufs["inv"]],
                page_stride=self.page_stride,
                resources=self.res,
            )
            entry = self._args[key] = (args, meta)
        args, meta = entry
        path = num_rows <= 32
        if path not in self._checked:
            if torch.cuda.is_current_stream_capturing():
                return None
            self._checked[path] = self._self_check(
                args, block_table, lens, indices, bufs
            )
        if not self._checked[path]:
            return None
        pending = _PendingBuild(args)
        _PENDING.setdefault(self.res.device, {})[id(self)] = pending
        return replace(meta, pending=pending)

    def _self_check(
        self,
        args: _GroupArgs,
        block_table: torch.Tensor,
        lens: torch.Tensor,
        indices: torch.Tensor,
        bufs: dict[str, Any],
    ) -> bool:
        """The first build: the reference into the real buffers, the kernel
        into copies of them; every output must match bit for bit."""
        assert self.res is not None and self._dom_table is not None
        names = ("bt", "lens", "sched", "inv", "jmap")
        copies = {
            name: (
                [t.clone() for t in bufs[name]]
                if isinstance(bufs[name], list)
                else bufs[name].clone()
            )
            for name in names
        }
        reference_build(
            block_table,
            lens,
            indices[: args.num_rows],
            args.num_rows,
            self._dom_table,
            self.page_rows,
            self.res.sms,
            bufs,
        )
        kernel = _build_kernel(self.res.device)
        assert kernel is not None
        kernel.launch(
            [
                replace(
                    args,
                    lbt=copies["bt"],
                    llen=copies["lens"],
                    sched=copies["sched"],
                    inv=copies["inv"],
                    jmap=copies["jmap"],
                )
            ],
            self.res.sms,
        )
        for name in names:
            ref, got = bufs[name], copies[name]
            same = (
                all(torch.equal(a, b) for a, b in zip(ref, got))
                if isinstance(ref, list)
                else torch.equal(ref, got)
            )
            if not same:
                self._disable(f"the build kernel differs from the reference in {name}")
                return False
        table = self._dom_table
        logger.info(
            "Die-local sparse indexer active for %d layers (%s..): %d pages of "
            "%d rows, %.1f%% on locality domain 1, %s SMs per domain.",
            len(self.layer_names),
            self.layer_names[0],
            table.numel(),
            self.page_rows,
            100 * float(table.float().mean()),
            self.res.sms,
        )
        return True


def maybe_make_decode_localizer(
    builder: Any, block_table_width: int
) -> IndexerDecodeLocalizer | None:
    """The localizer of an indexer metadata builder, or None (stock).

    Called from the builder's constructor when `VLLM_INDEXER_LOCALIZE` is set.
    """
    vllm_config = builder.vllm_config
    reason = unsupported_config_reason(vllm_config)
    if reason is None and (builder.dcp_world_size > 1 or builder.use_pcp):
        reason = "decode or prefill context parallelism"
    if reason is not None:
        logger.warning_once(
            "VLLM_INDEXER_LOCALIZE does not support %s; using the stock sparse "
            "indexer.",
            reason,
        )
        return None
    scheduler_config = vllm_config.scheduler_config
    spec = builder.kv_cache_spec
    loc = IndexerDecodeLocalizer(
        builder.layer_names,
        vllm_config.compilation_config.static_forward_context,
        builder.device,
        min(scheduler_config.max_num_seqs, scheduler_config.max_num_batched_tokens),
        block_table_width,
        int(getattr(spec, "num_states", spec.block_size)),
    )
    return None if loc.disabled else loc
