# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from TensorRT-LLM's Muse Spark Open persistent expert op
# (tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open/
# muse_spark_open_816b_a42b_nvfp4__sm_103__tep/ops/moe/persistent_experts.py),
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
"""The routed NVFP4 experts of up to 8 decode rows in one persistent launch.

Replaces what ``trtllm_fp4_block_scale_routed_moe(do_finalize=False)`` launches
for a pre-routed decode step -- the routing-indices kernel, FC1 and FC2 -- with
one CuTe DSL kernel (``persistent_experts_kernel/kernel.py``): a static tile
queue over (expert, 128-row tile) of FC1 and then FC2 across every SM, where
FC2 starts per k-block as soon as the FC1 tiles that produce its intermediate
columns are published.

The kernel reads the same TRT-LLM-gen weight tensors as FlashInfer's routed
MoE (gate/up rows interleaved and 32-row shuffled, 128x4-interleaved block
scales) and runs the same block-scaled MMAs in the same K order with TRT-LLM-
gen's FC1 epilogue, so every routed FC2 row equals FlashInfer's bit for bit
(``tests/models/muse_spark/nvidia/specialized_decode/test_kernels.py``).

Output (route-major, ``UnfinalizedMoEOutput.route_major``):

* ``rows`` ``[M * 8, 4096]`` BF16: ``rows[t * 8 + k]`` is the FC2 output of
  local slot ``(t, k)``; other rows are not written.
* ``index`` ``[M, 8]`` int32: ``t * 8 + k`` for a local slot, else ``-1``.

The kernel reads the global expert ids the routing writes; this rank's window
is ``[offset, offset + L)`` with ``offset`` = EP rank x ``L``, a compile-time
constant of the rank's kernel. A slot is local when its id is in the window and
no earlier slot of the same token names the same id (a repeated expert is
routed once, through its first slot, as TRT-LLM-gen's routing does). ``L`` (the
rank's local experts) is 32 at EP8 and 64 at EP4.

Every CTA of a launch spins on FC1 tiles of other CTAs, so the whole grid (one
CTA per SM) must be resident at once: the kernel must not run beside a kernel
that waits on it, and MPS is refused. The workspace (per device) is created
and the kernel compiled before CUDA graph capture (``prepare``); launches on
it must be stream-ordered.
"""

from __future__ import annotations

import threading

import torch

HIDDEN = 4096
INTER = 4096
TOP_K = 8
# Rows of one launch: the MMA's N.
MAX_ROWS = 8
# Local expert counts the kernel is compiled for.
WINDOWS = (32, 64, 96)
SF_VEC = 16

# Compile-time options of the kernel (``persistent_experts_kernel/kernel.py``),
# as TensorRT-LLM ships them for SM103: six weight stages, four accumulator
# stages, no timeline stamps, balanced tile rounds, early slots slot-major,
# FC1 -> FC2 hand-off through epoch-tagged words.
OPTIONS: dict[str, int] = {
    "num_stages": 6,
    "acc_stages": 4,
    "delay": 0,
    "delay_ns": 2000,
    "trace": 0,
    "balance": 1,
    "order": 1,
    "publish": 0,
    "handoff": 1,
    "fastroute": 0,
}
SUPPORTED_CAPABILITIES = ((10, 3),)

_NUM_ADDRS = 12
_LOCK = threading.Lock()
_COMPILED: dict[tuple, object] = {}
_WORKSPACES: dict[tuple, _Workspace] = {}


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _resident_sms(device: torch.device) -> int:
    """SMs one grid can hold at once (the kernel runs one CTA per SM)."""
    from cuda.bindings import driver

    index = _device_index(device)
    ok = driver.CUresult.CUDA_SUCCESS
    mps = getattr(driver.CUdevice_attribute, "CU_DEVICE_ATTRIBUTE_MPS_ENABLED", None)
    if mps is not None:
        err, dev = driver.cuDeviceGet(index)
        if err == ok:
            err, enabled = driver.cuDeviceGetAttribute(mps, dev)
            if err == ok and enabled:
                raise RuntimeError(
                    "persistent experts: refusing to run under MPS; every CTA of "
                    "the grid must be resident at once"
                )
    return torch.cuda.get_device_properties(device).multi_processor_count


def _refuse_under_capture(what: str) -> None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"persistent experts: the {what} must exist before CUDA graph capture "
            "(call prepare() after loading the weights)"
        )


class _Workspace:
    """Graph-stable scratch shared by every launch on a device.

    ``epoch`` holds one launch counter per CTA; ``flags`` one word per (slot,
    FC1 tile) carrying the epoch of the launch that published the tile;
    ``images`` one FC2 operand image per slot. Nothing is cleared between
    launches or graph replays, so it is created once, outside capture.
    """

    def __init__(self, device: torch.device, num_ctas: int):
        from .persistent_experts_kernel.kernel import (
            FC1_MT,
            MAX_CTAS,
            MAX_SLOTS,
            TRACE_WORDS,
            WS_SLOT_BYTES,
        )

        assert num_ctas <= MAX_CTAS
        self.num_ctas = num_ctas
        self.epoch = torch.zeros(MAX_CTAS, dtype=torch.int32, device=device)
        self.flags = torch.zeros(MAX_SLOTS * FC1_MT, dtype=torch.int32, device=device)
        self.images = torch.zeros(
            MAX_SLOTS * WS_SLOT_BYTES, dtype=torch.uint8, device=device
        )
        self.trace = torch.zeros(
            num_ctas, TRACE_WORDS, dtype=torch.int64, device=device
        )


def _workspace(device: torch.device, num_ctas: int) -> _Workspace:
    key = (_device_index(device), num_ctas)
    with _LOCK:
        ws = _WORKSPACES.get(key)
        if ws is None:
            _refuse_under_capture("workspace")
            ws = _Workspace(device, num_ctas)
            _WORKSPACES[key] = ws
    return ws


def _pointers(w13, w13_sf, w2, w2_sf):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    gmem = cute.AddressSpace.gmem
    return [
        make_ptr(cutlass.Float4E2M1FN, w13.data_ptr(), gmem, assumed_align=32),
        make_ptr(cutlass.Float8E4M3FN, w13_sf.data_ptr(), gmem, assumed_align=16),
        make_ptr(cutlass.Float4E2M1FN, w2.data_ptr(), gmem, assumed_align=32),
        make_ptr(cutlass.Float8E4M3FN, w2_sf.data_ptr(), gmem, assumed_align=16),
    ]


def _compiled(
    device: torch.device, num_local: int, id_offset: int, w13, w13_sf, w2, w2_sf
):
    """(compiled launcher, grid size) for this device and window."""
    num_ctas = _resident_sms(device)
    key = (_device_index(device), num_local, id_offset, num_ctas)
    with _LOCK:
        fn = _COMPILED.get(key)
        if fn is not None:
            return fn, num_ctas
        _refuse_under_capture("compiled kernel")
        import cutlass
        import cutlass.cute as cute
        from cuda.bindings import driver as cuda

        from .persistent_experts_kernel.kernel import PersistentExpertsKernel

        kernel = PersistentExpertsKernel(
            num_ctas, num_local, **OPTIONS, id_offset=id_offset
        )
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        fn = cute.compile(
            kernel.run,
            *_pointers(w13, w13_sf, w2, w2_sf),
            *([cutlass.Int64(0)] * _NUM_ADDRS),
            cutlass.Int32(1),
            stream,
        )
        _COMPILED[key] = fn
        return fn, num_ctas


def supported_device(device: torch.device) -> bool:
    return tuple(torch.cuda.get_device_capability(device)) in SUPPORTED_CAPABILITIES


def supports(
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    num_local: int,
) -> bool:
    """Whether a layer of this shape and step of ``num_tokens`` rows runs here."""
    return (
        1 <= num_tokens <= MAX_ROWS
        and top_k == TOP_K
        and hidden_size == HIDDEN
        and intermediate_size == INTER
        and num_local in WINDOWS
    )


def prepare(
    id_offset: int,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
) -> None:
    """Compile the kernel for this window and create the device's workspace.

    Call eagerly, before any CUDA graph capture (CuTe DSL JIT, about 1 s per
    window and process).
    """
    device = gemm1_weights.device
    _, num_ctas = _compiled(
        device,
        gemm1_weights.shape[0],
        id_offset,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
    )
    _workspace(device, num_ctas)
    torch.cuda.synchronize(device)


def persistent_experts(
    id_offset: int,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rows, index)`` of the routed experts; see the module docstring.

    ``hidden_states`` ``[M, 2048]`` uint8 packed NVFP4 with linear E4M3 block
    scales ``hidden_states_scale`` (``M * 256`` bytes); ``topk_ids``
    ``[M, 8]`` int32 global expert ids (local when ``offset <= id < offset +
    L``); the window's TRT-LLM-gen weight tensors as FlashInfer
    takes them; per-expert FP32 ``g1_scale_c`` / ``g1_alphas`` / ``g2_alphas``.
    """
    import cutlass
    from cuda.bindings import driver as cuda

    device = hidden_states.device
    num_tokens = hidden_states.shape[0]
    fn, num_ctas = _compiled(
        device,
        gemm1_weights.shape[0],
        id_offset,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
    )
    ws = _workspace(device, num_ctas)
    rows = torch.empty(num_tokens * TOP_K, HIDDEN, dtype=torch.bfloat16, device=device)
    index = torch.empty(num_tokens, TOP_K, dtype=torch.int32, device=device)
    addrs = (
        hidden_states.data_ptr(),
        hidden_states_scale.data_ptr(),
        topk_ids.data_ptr(),
        output1_scale_scalar.data_ptr(),
        output1_scale_gate_scalar.data_ptr(),
        output2_scale_scalar.data_ptr(),
        rows.data_ptr(),
        index.data_ptr(),
        ws.epoch.data_ptr(),
        ws.flags.data_ptr(),
        ws.images.data_ptr(),
        ws.trace.data_ptr(),
    )
    fn(
        *_pointers(
            gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale
        ),
        *[cutlass.Int64(a) for a in addrs],
        cutlass.Int32(num_tokens),
        cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )
    return rows, index


def check_inputs(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
) -> None:
    """Raises on inputs the kernel would misread without an error (the kernel
    reads raw addresses)."""
    m = hidden_states.shape[0]
    assert 1 <= m <= MAX_ROWS, f"one launch serves 1..{MAX_ROWS} rows, got {m}"
    assert hidden_states.dtype == torch.uint8 and hidden_states.is_contiguous()
    assert tuple(hidden_states.shape) == (m, HIDDEN // 2), tuple(hidden_states.shape)
    assert hidden_states_scale.is_contiguous()
    assert hidden_states_scale.numel() == m * (HIDDEN // SF_VEC)
    assert topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()
    assert topk_ids.numel() == m * TOP_K
    num_local = gemm1_weights.shape[0]
    assert num_local in WINDOWS, f"compiled for windows {WINDOWS}, got {num_local}"
    for name, t, shape in (
        ("gemm1_weights", gemm1_weights, (num_local, 2 * INTER, HIDDEN // 2)),
        ("gemm2_weights", gemm2_weights, (num_local, HIDDEN, INTER // 2)),
    ):
        assert t.is_contiguous() and t.element_size() == 1, name
        assert tuple(t.shape) == shape, f"{name} is {tuple(t.shape)}, not {shape}"
    for name, t, n in (
        ("gemm1_weights_scale", gemm1_weights_scale, num_local * 2 * INTER * 256),
        ("gemm2_weights_scale", gemm2_weights_scale, num_local * HIDDEN * 256),
    ):
        assert t.is_contiguous() and t.numel() == n, f"{name} must hold {n} scales"
    for name, t in (
        ("output1_scale_scalar", output1_scale_scalar),
        ("output1_scale_gate_scalar", output1_scale_gate_scalar),
        ("output2_scale_scalar", output2_scale_scalar),
    ):
        assert t.dtype == torch.float32 and t.is_contiguous(), name
        assert t.numel() == num_local, f"{name} must be [{num_local}] float32"
