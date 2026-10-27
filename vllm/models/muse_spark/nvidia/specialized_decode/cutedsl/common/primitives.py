# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
# The PTX primitives, launch helpers and Lamport stage constants are adapted
# from FlashInfer (flashinfer/comm/mnnvl_cutedsl/cute_dsl_primitives.py,
# runtime.py and kernel_ll/device_kernels.py,
# flashinfer-ai/flashinfer@14ba7aaa84718abe7a5648184fa9ad15362e2880,
# Apache-2.0), in the form of vLLM's DeepSeek V4.1 copy
# (vllm/models/deepseek_v41/nvidia/ops/cute_dsl/primitives.py).
# FlashInfer NOTICE: "FlashInfer / Copyright 2025-2026 NVIDIA / Copyright
# 2023-2026 FlashInfer community (https://flashinfer.ai/)".

"""PTX building blocks and launch helpers of the Muse Spark specialized decode.

Every Lamport kernel of the package (the boundary gathers, the attention
boundary, the MoE tail and the unpooler's zero-copy epilogue) is built from
these. BF16 negative zero (0x8000) marks a slot that has not been written yet;
a three-generation stage state (``NEXT_STAGE`` written by the consumer,
``ACTIVE_STAGE`` by the publisher) lets a fast rank publish the next call while
a slow one still reads the current one.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import (
    BFloat16,
    Float8E4M3FN,
    Float16,
    Float32,
    Int32,
    Int64,
    Uint16,
    Uint32,
)
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

WARP_SIZE = 32
VEC_BF16 = 8
QUAD_BF16 = 4
NEGATIVE_ZERO_BF16_BITS = 0x8000
NEGATIVE_ZERO_BF16_PAIR = 0x80008000
# CUTLASS cute::TMA::CacheHintSm100::EVICT_FIRST policy descriptor.
L2_EVICT_FIRST = 0x12F0000000000000

# Lamport stage protocol: [3, tp, capacity, width] mailboxes and an int32[2]
# stage state per mailbox.
LAMPORT_GENERATIONS = 3
NEXT_STAGE = 0
ACTIVE_STAGE = 1


def _asm(result, operands, text, constraints, *, side_effects, loc, ip):
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _words(loaded, count, *, loc, ip):
    packed = vector.from_elements(
        ir.VectorType.get([count], T.i32(), loc=loc),
        [
            llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip)
            for index in range(count)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, count, Uint32)


@dsl_user_op
def load_global_u32x4(
    pointer: cute.Pointer,
    predicate: Int32 | None = None,
    *,
    volatile: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
):
    """Load four words; an omitted predicate loads normally, false returns zeros."""
    predicate = Int32(1) if predicate is None else Int32(predicate)
    address = pointer.toint(loc=loc, ip=ip)
    opcode = "ld.volatile.global.v4.u32" if volatile else "ld.global.v4.u32"
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value(loc=loc, ip=ip), predicate.ir_value(loc=loc, ip=ip)],
        (
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.s32 p, $5, 0;\n\t"
            "@!p mov.u32 $0, 0;\n\t"
            "@!p mov.u32 $1, 0;\n\t"
            "@!p mov.u32 $2, 0;\n\t"
            "@!p mov.u32 $3, 0;\n\t"
            f"@p {opcode} {{$0, $1, $2, $3}}, [$4];\n\t"
            "}"
        ),
        "=r,=r,=r,=r,l,r",
        side_effects=volatile,
        loc=loc,
        ip=ip,
    )
    return _words(loaded, 4, loc=loc, ip=ip)


@dsl_user_op
def load_global_u32(pointer: cute.Pointer, *, loc=None, ip=None) -> Uint32:
    address = pointer.toint(loc=loc, ip=ip)
    return Uint32(
        _asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.global.u32 $0, [$1];",
            "=r,l",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_global_u32_predicated(
    pointer: cute.Pointer, predicate: Int32, *, loc=None, ip=None
) -> Uint32:
    address = pointer.toint(loc=loc, ip=ip)
    return Uint32(
        _asm(
            T.i32(),
            [
                address.ir_value(loc=loc, ip=ip),
                Int32(predicate).ir_value(loc=loc, ip=ip),
            ],
            (
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.s32 p, $2, 0;\n\t"
                "@!p mov.u32 $0, 0;\n\t"
                "@p ld.global.u32 $0, [$1];\n\t"
                "}"
            ),
            "=r,l,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_global_u32x2(pointer: cute.Pointer, *, loc=None, ip=None):
    address = pointer.toint(loc=loc, ip=ip)
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [address.ir_value(loc=loc, ip=ip)],
        "ld.global.v2.u32 {$0, $1}, [$2];",
        "=r,=r,l",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _words(loaded, 2, loc=loc, ip=ip)


@dsl_user_op
def load_global_u32x2_predicated(
    pointer: cute.Pointer, predicate: Int32, *, loc=None, ip=None
):
    address = pointer.toint(loc=loc, ip=ip)
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [
            address.ir_value(loc=loc, ip=ip),
            Int32(predicate).ir_value(loc=loc, ip=ip),
        ],
        (
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.s32 p, $3, 0;\n\t"
            "@!p mov.u32 $0, 0;\n\t"
            "@!p mov.u32 $1, 0;\n\t"
            "@p ld.global.v2.u32 {$0, $1}, [$2];\n\t"
            "}"
        ),
        "=r,=r,l,r",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _words(loaded, 2, loc=loc, ip=ip)


@dsl_user_op
def load_global_u32x4_address(
    address: Int64,
    *,
    volatile: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
):
    opcode = "ld.volatile.global.v4.u32" if volatile else "ld.global.v4.u32"
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        f"{opcode} {{$0, $1, $2, $3}}, [$4];",
        "=r,=r,=r,=r,l",
        side_effects=volatile,
        loc=loc,
        ip=ip,
    )
    return _words(loaded, 4, loc=loc, ip=ip)


@dsl_user_op
def load_global_bf16_as_f32(address: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [address.ir_value(loc=loc, ip=ip)],
            (
                "{\n\t"
                ".reg .b16 bits;\n\t"
                "ld.global.b16 bits, [$1];\n\t"
                "cvt.f32.bf16 $0, bits;\n\t"
                "}"
            ),
            "=f,l",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_global_bf16_as_f32_predicated(
    address: Int64, predicate: Int32, *, loc=None, ip=None
) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [
                address.ir_value(loc=loc, ip=ip),
                Int32(predicate).ir_value(loc=loc, ip=ip),
            ],
            (
                "{\n\t"
                ".reg .pred p;\n\t"
                ".reg .b16 bits;\n\t"
                "setp.ne.s32 p, $2, 0;\n\t"
                "@!p mov.b16 bits, 0;\n\t"
                "@p ld.global.b16 bits, [$1];\n\t"
                "cvt.f32.bf16 $0, bits;\n\t"
                "}"
            ),
            "=f,l,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_volatile_u32(pointer: cute.Pointer, *, loc=None, ip=None) -> Uint32:
    address = pointer.toint(loc=loc, ip=ip)
    return Uint32(
        _asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.u32 $0, [$1];",
            "=r,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_global_u32x4(address: Int64, packed, *, loc=None, ip=None) -> None:
    words = [packed[index].ir_value(loc=loc, ip=ip) for index in range(4)]
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_global_u32x2(address: Int64, packed, *, loc=None, ip=None) -> None:
    words = [packed[index].ir_value(loc=loc, ip=ip) for index in range(2)]
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        "st.global.v2.u32 [$0], {$1, $2};",
        "l,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_global_u32(
    pointer: cute.Pointer, value: Uint32, *, loc=None, ip=None
) -> None:
    address = pointer.toint(loc=loc, ip=ip)
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.global.u32 [$0], $1;",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_global_u32_address(
    address: Int64, value: Uint32, *, loc=None, ip=None
) -> None:
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.global.u32 [$0], $1;",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_global_u16_bits(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        (
            "{\n\t"
            ".reg .b16 bits;\n\t"
            "cvt.u16.u32 bits, $1;\n\t"
            "st.global.u16 [$0], bits;\n\t"
            "}"
        ),
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_lamport_sentinel_u32x4(address: Int64, *, loc=None, ip=None) -> None:
    sentinel = Uint32(NEGATIVE_ZERO_BF16_PAIR).ir_value(loc=loc, ip=ip)
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), sentinel, sentinel, sentinel, sentinel],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def f32_to_bf16_bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [value.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 bits;\n\tcvt.rn.bf16.f32 bits, $1;\n\t"
            "cvt.u32.u16 $0, bits;\n\t}",
            "=r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def pack_bf16x2_rn(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """Two FP32 rounded to nearest-even BF16, lo in bits 0-15, hi in bits 16-31."""
    return Uint32(
        _asm(
            T.i32(),
            [hi.ir_value(loc=loc, ip=ip), lo.ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf16x2_to_f32x2(word: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    """The BF16 pair in ``word`` widened to FP32 (exact): (bits 0-15, 16-31)."""
    pair = _asm(
        ir.Type.parse("!llvm.struct<(f32, f32)>"),
        [Uint32(word).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 hl, hh;\n"
        "  mov.b32 {hl, hh}, $2;\n"
        "  cvt.f32.bf16 $0, hl;\n"
        "  cvt.f32.bf16 $1, hh; }",
        "=f,=f,r",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), pair, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), pair, [1], loc=loc, ip=ip)),
    )


# FP32 arithmetic as explicit PTX: the rounding of each op is the one named, and
# nothing contracts across them (inline asm results are opaque to LLVM).
def _f32_op(text: str, constraints: str, operands, loc, ip) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip) for x in operands],
            text,
            constraints,
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """A + b, round to nearest, subnormals kept."""
    return _f32_op("add.rn.f32 $0, $1, $2;", "=f,f,f", (a, b), loc, ip)


@dsl_user_op
def add_ftz_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """A + b, round to nearest, subnormal inputs and result flushed to a zero
    of the same sign (the add of code built with fast math)."""
    return _f32_op("add.rn.ftz.f32 $0, $1, $2;", "=f,f,f", (a, b), loc, ip)


@dsl_user_op
def mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """A * b, round to nearest, subnormals kept; never contracted."""
    return _f32_op("mul.rn.f32 $0, $1, $2;", "=f,f,f", (a, b), loc, ip)


@dsl_user_op
def fma_rn(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """A * b + c with one rounding."""
    return _f32_op("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", (a, b, c), loc, ip)


@dsl_user_op
def rsqrt_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """rsqrt.approx with subnormals flushed (libdevice's rsqrt under FTZ)."""
    return _f32_op("rsqrt.approx.ftz.f32 $0, $1;", "=f,f", (a,), loc, ip)


@dsl_user_op
def round_bf16(a: Float32, *, loc=None, ip=None) -> Float32:
    """A rounded to BF16 (nearest even), widened back."""
    return _f32_op(
        "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tcvt.f32.bf16 $0, h;\n\t}",
        "=f,f",
        (a,),
        loc,
        ip,
    )


@dsl_user_op
def shuffle_sync_idx_u32(
    value: Uint32, source_lane: Int32, *, loc=None, ip=None
) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [value.ir_value(loc=loc, ip=ip), source_lane.ir_value(loc=loc, ip=ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            # Preserve full-warp execution across later divergent consumers.
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def packed_u32x4_to_bf16x8(packed, *, loc=None, ip=None):
    values = llvm.bitcast(
        ir.VectorType.get([VEC_BF16], BFloat16.mlir_type, loc=loc),
        packed.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(values, VEC_BF16, BFloat16)


@dsl_user_op
def packed_u32x4_to_f16x8(packed, *, loc=None, ip=None):
    values = llvm.bitcast(
        ir.VectorType.get([VEC_BF16], Float16.mlir_type, loc=loc),
        packed.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(values, VEC_BF16, Float16)


@dsl_user_op
def packed_u32_to_bf16x2(packed: Uint32, *, loc=None, ip=None):
    values = llvm.bitcast(
        ir.VectorType.get([2], BFloat16.mlir_type, loc=loc),
        packed.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(values, 2, BFloat16)


@dsl_user_op
def packed_u32x2_to_bf16x4(packed, *, loc=None, ip=None):
    values = llvm.bitcast(
        ir.VectorType.get([QUAD_BF16], BFloat16.mlir_type, loc=loc),
        packed.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(values, QUAD_BF16, BFloat16)


@dsl_user_op
def bf16x8_to_packed_u32x4(values, *, loc=None, ip=None):
    packed = llvm.bitcast(
        ir.VectorType.get([4], T.i32(), loc=loc),
        values.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def bf16x2_to_packed_u32(values, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.bitcast(
            T.i32(),
            values.ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf16x4_to_packed_u32x2(values, *, loc=None, ip=None):
    packed = llvm.bitcast(
        ir.VectorType.get([2], T.i32(), loc=loc),
        values.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 2, Uint32)


@dsl_user_op
def packed_negative_zero_bf16x8(*, loc=None, ip=None):
    word = Uint32(NEGATIVE_ZERO_BF16_PAIR).ir_value(loc=loc, ip=ip)
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [word, word, word, word],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@cute.jit
def sanitize_negative_zero_u32x4(packed):
    sanitized = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
    for index in cutlass.range_constexpr(4):
        sanitized[index] = sanitize_negative_zero_u32(packed[index])
    return sanitized.load()


@cute.jit
def sanitize_negative_zero_u32(word: Uint32) -> Uint32:
    low = Uint16(word & Uint32(0xFFFF))
    high = Uint16(word >> Uint32(16))
    if low == Uint16(NEGATIVE_ZERO_BF16_BITS):
        word = word & Uint32(0xFFFF0000)
    if high == Uint16(NEGATIVE_ZERO_BF16_BITS):
        word = word & Uint32(0x0000FFFF)
    return word


@cute.jit
def sanitize_negative_zero_u32x2(packed):
    sanitized = cute.make_rmem_tensor(cute.make_layout((2,)), Uint32)
    for index in cutlass.range_constexpr(2):
        sanitized[index] = sanitize_negative_zero_u32(packed[index])
    return sanitized.load()


@cute.jit
def fragment_has_negative_zero(packed):
    dirty = False
    for index in cutlass.range_constexpr(4):
        word = packed[index]
        dirty = (
            dirty
            | (Uint16(word & Uint32(0xFFFF)) == Uint16(NEGATIVE_ZERO_BF16_BITS))
            | (Uint16(word >> Uint32(16)) == Uint16(NEGATIVE_ZERO_BF16_BITS))
        )
    return dirty


@dsl_user_op
def map_shared_to_peer(
    smem_pointer: cute.Pointer, peer_rank: Int32, *, loc=None, ip=None
) -> Int32:
    address = smem_pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    return Int32(
        _asm(
            T.i32(),
            [address, peer_rank.ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_shared_cluster_f32(
    remote_address: Int32, value: Float32, *, loc=None, ip=None
) -> None:
    _asm(
        None,
        [remote_address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.shared::cluster.f32 [$0], $1;",
        "r,f",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_u32x4(pointer: cute.Pointer, *, loc=None, ip=None):
    address = pointer.toint(loc=loc, ip=ip)
    # Prevent motion across the named-barrier pipeline protocol.
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(address).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _words(loaded, 4, loc=loc, ip=ip)


@dsl_user_op
def store_shared_u32x4(pointer: cute.Pointer, packed, *, loc=None, ip=None) -> None:
    address = pointer.toint(loc=loc, ip=ip)
    words = [packed[index].ir_value(loc=loc, ip=ip) for index in range(4)]
    _asm(
        None,
        [Int32(address).ir_value(loc=loc, ip=ip), *words],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cpasync_bulk_g2s(
    gmem_ptr: cute.Pointer,
    smem_ptr: cute.Pointer,
    barrier_ptr: cute.Pointer,
    size_bytes: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    operands = [
        gmem_ptr.toint(loc=loc, ip=ip).ir_value(),
        smem_ptr.toint(loc=loc, ip=ip).ir_value(),
        barrier_ptr.toint(loc=loc, ip=ip).ir_value(),
        size_bytes.ir_value(loc=loc, ip=ip),
        Int64(L2_EVICT_FIRST).ir_value(),
    ]
    _asm(
        None,
        operands,
        (
            "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
            ".L2::cache_hint [$1], [$0], $3, [$2], $4;"
        ),
        "l,r,r,r,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fence_proxy_async_shared_cta(*, loc=None, ip=None) -> None:
    _asm(
        None,
        [],
        "fence.proxy.async.shared::cta;",
        "",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def remote_release_add1_u32(address: Int64, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip)],
        "red.release.sys.global.add.u32 [$0], 1;",
        "l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atomic_add1_acq_rel_u32(pointer: cute.Pointer, *, loc=None, ip=None) -> Uint32:
    """Add 1 to the global word at ``pointer`` and return its value before
    (acquire-release, GPU scope): a ticket."""
    address = pointer.toint(loc=loc, ip=ip)
    return Uint32(
        _asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], 1;",
            "=r,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ldmc_bf16x8(address: Int64, predicate: Int32 | None = None, *, loc=None, ip=None):
    """Reduce-load eight BF16s; an omitted predicate loads, false returns zeros."""
    predicate = Int32(1) if predicate is None else Int32(predicate)
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value(loc=loc, ip=ip), predicate.ir_value(loc=loc, ip=ip)],
        (
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.s32 p, $5, 0;\n\t"
            "@!p mov.u32 $0, 0;\n\t"
            "@!p mov.u32 $1, 0;\n\t"
            "@!p mov.u32 $2, 0;\n\t"
            "@!p mov.u32 $3, 0;\n\t"
            "@p multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 "
            "{$0, $1, $2, $3}, [$4];\n\t"
            "}"
        ),
        "=r,=r,=r,=r,l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _words(loaded, 4, loc=loc, ip=ip)


@dsl_user_op
def stmc_bf16x2(address: Int64, packed: Uint32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), packed.ir_value(loc=loc, ip=ip)],
        "multimem.st.relaxed.sys.global.bf16x2 [$0], $1;",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stmc_bf16x4(address: Int64, values, *, loc=None, ip=None) -> None:
    words = [values[index].ir_value(loc=loc, ip=ip) for index in range(2)]
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        "multimem.st.relaxed.sys.global.v2.bf16x2 [$0], {$1, $2};",
        "l,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stmc_bf16x8(
    address: Int64, values, predicate: Int32 | None = None, *, loc=None, ip=None
) -> None:
    """Multicast-store eight BF16s; an omitted predicate stores, false skips."""
    predicate = Int32(1) if predicate is None else Int32(predicate)
    words = [values[index].ir_value(loc=loc, ip=ip) for index in range(4)]
    _asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words, predicate.ir_value(loc=loc, ip=ip)],
        (
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.s32 p, $5, 0;\n\t"
            "@p multimem.st.relaxed.sys.global.v4.bf16x2 "
            "[$0], {$1, $2, $3, $4};\n\t"
            "}"
        ),
        "l,r,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def global_timer_ns(*, loc=None, ip=None) -> Int64:
    """The GPU's global nanosecond timer (bounds the Lamport spins)."""
    return Int64(
        _asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def trap(*, loc=None, ip=None) -> None:
    """Abort the kernel: the launch fails instead of hanging."""
    _asm(None, [], "trap;", "", side_effects=True, loc=loc, ip=ip)


@cute.jit
def load_f32x8(base: cute.Tensor, offset: Int64):
    pointer = cute.make_ptr(
        Float32,
        (base.iterator + offset).llvm_ptr,
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    fragment = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
    cute.autovec_copy(
        cute.make_tensor(pointer, cute.make_layout((VEC_BF16,))), fragment
    )
    return fragment.load()


@cute.jit
def store_f32x8(base: cute.Tensor, offset: Int64, values) -> None:
    pointer = cute.make_ptr(
        Float32,
        (base.iterator + offset).llvm_ptr,
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    fragment = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
    fragment.store(values)
    cute.autovec_copy(
        fragment, cute.make_tensor(pointer, cute.make_layout((VEC_BF16,)))
    )


@cute.jit
def load_bf16x8_as_f32(base: cute.Tensor, offset: Int64):
    pointer = cute.make_ptr(
        BFloat16,
        (base.iterator + offset).llvm_ptr,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return packed_u32x4_to_bf16x8(load_global_u32x4(pointer)).to(Float32)


@cute.jit
def load_f16x8_as_f32(base: cute.Tensor, offset: Int64):
    """8 FP16 elements from element ``offset`` of ``base`` (one 16-byte
    access), converted exactly to FP32."""
    pointer = cute.make_ptr(
        Float16,
        (base.iterator + offset).llvm_ptr,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return packed_u32x4_to_f16x8(load_global_u32x4(pointer)).to(Float32)


def regs(values) -> list:
    """An 8-wide FP32 TensorSSA (``load_f32x8``, ``load_bf16x8_as_f32``) as
    its 8 scalars (traced inline)."""
    fragment = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
    fragment.store(values)
    return [fragment[i] for i in range(VEC_BF16)]


def pack_bf16x8_rn(values: list):
    """8 FP32 scalars rounded to nearest-even BF16, as 4 packed words (an
    rmem tensor; traced inline)."""
    words = cute.make_rmem_tensor(cute.make_layout((VEC_BF16 // 2,)), Uint32)
    for q in range(VEC_BF16 // 2):
        words[q] = pack_bf16x2_rn(values[2 * q], values[2 * q + 1])
    return words


def store_bf16x8_rn(base: cute.Tensor, offset: Int64, values: list) -> None:
    """8 FP32 scalars rounded to nearest-even BF16 and stored as one 16-byte
    access at element ``offset`` of the BF16 ``base`` (traced inline)."""
    store_global_u32x4(
        Int64((base.iterator + offset).toint()), pack_bf16x8_rn(values).load()
    )


class _GraphSafeDLPack:
    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def __dlpack__(self, stream=None):
        # stream=-1 skips producer sync; CuTe launches on the current stream,
        # which is the capturing one under CUDA graphs.
        return self.tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


def to_cute(tensor: torch.Tensor, alignment: int) -> cute.Tensor:
    if tensor.dtype == torch.float8_e4m3fn:
        # from_dlpack has no Float8E4M3FN entry: export the bytes and restore
        # the element type.
        result = from_dlpack(
            _GraphSafeDLPack(tensor.view(torch.uint8).detach()),
            assumed_align=alignment,
        )
        result.element_type = Float8E4M3FN
        return result
    return from_dlpack(_GraphSafeDLPack(tensor.detach()), assumed_align=alignment)


def to_cute_dynamic(
    tensor: torch.Tensor, alignment: int, *, divisibility: int
) -> cute.Tensor:
    return to_cute(tensor, alignment).mark_compact_shape_dynamic(
        mode=0, divisibility=divisibility
    )


def make_fake_dynamic_compact_tensor(dtype, *, alignment: int, divisibility: int):
    return make_fake_compact_tensor(
        dtype, (cute.sym_int32(divisibility=divisibility),), assumed_align=alignment
    )


def current_cu_stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


# Global memory by byte address (an Int64). An ``ordered`` load is
# side-effecting asm, kept in program order with the other side-effecting
# operations (stores, atomics, barriers, the PDL wait); an unordered load may be
# moved by the compiler. With a ``predicate``, a false one loads zero.


def _load_global_address(
    kind, zero, result, constraint, address, predicate, ordered, loc, ip
):
    operands = [address.ir_value(loc=loc, ip=ip)]
    if predicate is None:
        text = f"ld.global.{kind} $0, [$1];"
        constraints = f"={constraint},l"
    else:
        operands.append(Int32(predicate).ir_value(loc=loc, ip=ip))
        text = (
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t"
            f"mov.{zero} $0, 0;\n\t@p ld.global.{kind} $0, [$1];\n\t}}"
        )
        constraints = f"={constraint},l,r"
    return _asm(
        result, operands, text, constraints, side_effects=ordered, loc=loc, ip=ip
    )


@dsl_user_op
def load_global_u32_address(
    address: Int64,
    predicate: Int32 | None = None,
    *,
    ordered: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        _load_global_address(
            "u32", "u32", T.i32(), "r", address, predicate, ordered, loc, ip
        )
    )


@dsl_user_op
def load_global_s64_address(
    address: Int64,
    predicate: Int32 | None = None,
    *,
    ordered: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
) -> Int64:
    return Int64(
        _load_global_address(
            "s64", "u64", T.i64(), "l", address, predicate, ordered, loc, ip
        )
    )


@dsl_user_op
def atomic_add1_acq_rel_u32_address(address: Int64, *, loc=None, ip=None) -> Uint32:
    """``atomic_add1_acq_rel_u32`` of the global word at ``address``."""
    return Uint32(
        _asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], 1;",
            "=r,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )
