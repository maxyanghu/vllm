# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Muse Spark csrc ops (csrc/muse_spark/) the specialized decode and the
DSpark draft call, with the checks that decide where each one runs.

``muse_spark_attn2``
    Paged decode attention of the 8-token verify step (one request, 1 + 7 draft
    tokens) fused with the attention output's weightless RMS norm and sigmoid
    gate: the o_proj input of a decoder layer. Contract: head_dim 64, 8 query
    heads per KV head, bf16 [pages, num_kv_heads, 16, 128] cache (K | V per
    slot), 8 query rows per request, any context length L >= 8, and the
    layer's sliding window if it has one. One launch covers the KV heads of
    the rank. With max_model_len past 2048 (or a window) a split kernel
    spreads a request's pages over ``attn2_splits`` z slices, merged through a
    workspace. The launch shape depends on max_model_len only, not on the
    step, so it is graph-capturable.

    The op has no fallback of its own: the executor's plan() leaves every step
    it does not cover (prefill, other query lengths) to the stock forward.

``muse_spark_routing_shorttail``
    Muse Spark routing for decode batches of at most 8 tokens. It writes, bit
    for bit, the expert ids and weights ``fused_musespark1x_routing`` writes
    for Muse Spark's router: sigmoid scores, the correction bias for the
    selection only, top-8 of 256 experts, weights
    ``score * route_scale / (sum + 1e-15)``. The router's output multiplier must
    already be folded into its weight, as ``fold_router_output_multiplier``
    does after loading. Optionally the same launch writes the TRT-LLM-gen
    routing metadata of this rank's experts for one GEMM tile (``RoutingMetadata``):
    the CTA map, the permutation and the padded size the routed experts' GEMMs
    read, as that op's own routing kernel would write them for these ids.

``muse_spark_qkv_front``
    For 1 to 16 tokens on SM103 the op replaces the QKV + o-gate projection
    GEMM and ``fused_muse_spark_qknorm_rope_kv_insert``. Called eagerly (Muse
    Spark runs uncompiled), so the ``data_ptr()`` checks are not traced.
"""

import struct
from typing import NamedTuple

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.muse_spark.common.moe import MuseSpark1xRouter

# Both attn2 and the QKV front read head_dim 64 and 16-slot KV cache pages.
HEAD_DIM = 64
BLOCK_SIZE = 16

# attn2
ATTN2_OP = "muse_spark_attn2"
HEADS_PER_KV_HEAD = 8
QUERY_LEN = 8
# Up to this max_model_len without a sliding window, the one-cluster kernel;
# beyond, the split kernel.
SHORT_MAX_LEN = 2048
# 16-row groups of a KV head's 8 query rows x 8 heads: clusters per z slice.
ROW_GROUPS = QUERY_LEN * HEADS_PER_KV_HEAD // 16
# Clusters a long-context launch aims for per request. A constant, so the split
# (and the result) does not depend on the device.
SPLIT_TARGET_CLUSTERS = 16
MAX_SPLITS = 64
# Workspace bytes per row: a uint32 counter, and per cluster a partial of 68
# floats (64 o, m, l and 2 pad).
_COUNTER_BYTES = 4
_PARTIAL_BYTES = 68 * 4
# The kernel's softmax scale (head_dim ** -0.5 * qk_scale_factor) and o-norm
# eps, kSmScale and kEps in csrc/muse_spark/attn2_kernel.cu: the op has no
# arguments for them, so a layer with other values is refused.
SM_SCALE = 0.6841258108
ATTN2_EPS = 1e-5

# Routing short tail
ROUTING_OP = "muse_spark_routing_shorttail"
NUM_EXPERTS = 256
TOP_K = 8
ROUTING_MAX_TOKENS = 8
# The division epsilon the kernel adds to the sum of the selected scores; Muse
# Spark builds its router with this value.
DIVISION_EPS = 1e-15
# The routing metadata: int32 offsets of its fields (kMeta* in the kernel). A
# GEMM CTA holds at least one route, so 8 tokens take at most 64 CTAs.
ROUTES = ROUTING_MAX_TOKENS * TOP_K
METADATA_MAX_CTAS = ROUTES
META_PADDED_TOKENS = 0
META_NUM_CTAS = 1
META_EXPANDED_TO_PERMUTED = 4
META_CTA_BATCH = META_EXPANDED_TO_PERMUTED + ROUTES
META_CTA_LIMIT = META_CTA_BATCH + METADATA_MAX_CTAS
META_PERMUTED_TO_TOKEN = META_CTA_LIMIT + METADATA_MAX_CTAS
METADATA_MAX_TILE = 256

# QKV front
QKV_FRONT_OP = "muse_spark_qkv_front"
HIDDEN_SIZE = 8192
QKV_FRONT_MAX_TOKENS = 16
# K split per output width; the rule's SM count is fixed so the reduction order
# (and the result) does not depend on the device.
_SPLIT_SMS = 152
_SPLITS = (2, 4, 8)


def _fp32(x: float) -> float:
    """Round x to fp32, the precision the kernel takes both constants in."""
    return struct.unpack("f", struct.pack("f", x))[0]


def attn2_available() -> bool:
    """Whether this build has the attn2 op (CUDA >= 13 builds for the SM100
    family)."""
    return hasattr(torch.ops._C, ATTN2_OP)


def unsupported_attn2(layer: torch.nn.Module) -> str | None:
    """Why the kernel cannot compute this MuseSpark1xAttention layer, or None."""
    if not (layer.use_o_norm and layer.use_o_gate):
        return "the layer has no o-norm with a sigmoid o-gate"
    if layer.head_dim != HEAD_DIM:
        return f"head_dim {layer.head_dim} (the kernel has {HEAD_DIM})"
    if layer.num_heads != HEADS_PER_KV_HEAD * layer.num_kv_heads:
        return (
            f"{layer.num_heads} query heads for {layer.num_kv_heads} KV heads "
            f"(the kernel has {HEADS_PER_KV_HEAD} per KV head)"
        )
    if _fp32(layer.scaling) != _fp32(SM_SCALE):
        return f"softmax scale {layer.scaling} (the kernel is built for {SM_SCALE})"
    if _fp32(layer.rms_norm_eps) != _fp32(ATTN2_EPS):
        return f"o-norm eps {layer.rms_norm_eps} (the kernel is built for {ATTN2_EPS})"
    return None


def attn2_window(sliding_window: int | None, max_model_len: int) -> int:
    """The window attn2 applies for a layer: 0 (none) when the layer has none
    or the window covers max_model_len (no key is then outside a row's
    window)."""
    if sliding_window is None or sliding_window >= max_model_len:
        return 0
    if sliding_window < 1:
        raise ValueError(f"sliding window {sliding_window}")
    return sliding_window


def attn2_splits(max_model_len: int, window: int, num_kv_heads: int) -> int:
    """The z slices of an attn2 launch (ROW_GROUPS clusters each per request
    and KV head) for a deployment: 1 for the one-cluster kernel and for a
    window (it spans at most window + 7 keys), else about
    SPLIT_TARGET_CLUSTERS clusters per request over the rank's KV heads."""
    if window or max_model_len <= SHORT_MAX_LEN:
        return 1
    fill = SPLIT_TARGET_CLUSTERS // (ROW_GROUPS * num_kv_heads)
    return max(1, min(fill, MAX_SPLITS))


def attn2_workspace_bytes(num_reqs: int, num_kv_heads: int, num_splits: int) -> int:
    """Bytes of the workspace the split kernel needs: the counters (16-byte
    padded) and a partial per row and cluster."""
    rows = num_reqs * QUERY_LEN * HEADS_PER_KV_HEAD * num_kv_heads
    counters = -(-rows * _COUNTER_BYTES // 16) * 16
    return counters + rows * num_splits * ROW_GROUPS * _PARTIAL_BYTES


def attn2_split_kernel(max_model_len: int, window: int) -> bool:
    """Whether attn2 launches the split kernel (which needs a workspace)."""
    return max_model_len > SHORT_MAX_LEN or window > 0


def split_qkv(
    qkv: torch.Tensor, num_kv_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The q and gate column views of the attention front's output rows
    ``[q | k | v | gate]`` (q and gate: 8 heads of 64 per KV head)."""
    q_width = HEADS_PER_KV_HEAD * num_kv_heads * HEAD_DIM
    kv_width = num_kv_heads * HEAD_DIM
    if qkv.dim() != 2 or qkv.shape[1] != 2 * q_width + 2 * kv_width:
        raise ValueError(
            f"qkv must be [tokens, {2 * q_width + 2 * kv_width}] for "
            f"{num_kv_heads} KV heads, got {tuple(qkv.shape)}"
        )
    return qkv[:, :q_width], qkv[:, q_width + 2 * kv_width :]


def attn2(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    gate: torch.Tensor,
    out: torch.Tensor,
    num_kv_heads: int,
    enable_pdl: bool = True,
    max_seq_len: int = SHORT_MAX_LEN,
    window: int = 0,
    num_splits: int = 1,
    workspace: torch.Tensor | None = None,
) -> None:
    """``out [8 R, 512 num_kv_heads] = sigmoid(gate) * rmsnorm(attention)`` per
    query head, for R = q.shape[0] / 8 requests whose 8 query rows are the last
    8 of their ``seq_lens`` cache entries, each row seeing at most ``window``
    keys (0: all). ``max_seq_len`` is the deployment's maximum length: past
    SHORT_MAX_LEN or with a window the split kernel runs, over ``num_splits``
    (``attn2_splits``) z slices, and needs ``workspace``, uint8 of
    ``attn2_workspace_bytes`` zeroed once before the first call (each call
    leaves it reusable). The op checks shapes, dtypes, strides and devices;
    lengths (8 to max_seq_len, covered by the block table) are the caller's.
    Writes into ``out`` and allocates nothing, so it is graph-capturable."""
    ops.muse_spark_attn2(
        q,
        kv_cache,
        block_table,
        seq_lens,
        None,
        gate,
        out,
        num_kv_heads,
        enable_pdl,
        max_seq_len,
        window,
        num_splits,
        workspace,
    )


def routing_available() -> bool:
    """Whether this build has the routing op (CUDA 13 builds for the SM100
    family)."""
    return hasattr(torch.ops._C, ROUTING_OP)


def unsupported_router(router: MuseSpark1xRouter) -> str | None:
    """Why the op cannot reproduce ``router``'s routing, or None if it can."""
    if router.router_score_normalization != "sigmoid":
        return f"{router.router_score_normalization} router scores"
    if router.top_k != TOP_K or router.global_num_experts != NUM_EXPERTS:
        return (
            f"top-{router.top_k} of {router.global_num_experts} experts "
            f"(the op routes top-{TOP_K} of {NUM_EXPERTS})"
        )
    if router.router_output_multiplier is not None:
        return "a router output multiplier not folded into the router weight"
    if router.router_score_division_eps != DIVISION_EPS:
        return f"router_score_division_eps {router.router_score_division_eps}"
    bias = router.expert_bias
    if bias.dtype != torch.float32 or not bias.is_contiguous():
        return f"a {bias.dtype} expert bias (the op reads contiguous fp32)"
    return None


def metadata_size(tile_n: int) -> int:
    """int32 elements of the metadata for GEMM tile ``tile_n``: the fields
    above and the permuted-to-token map (64 CTAs of tile_n rows, plus a guard
    element)."""
    return META_PERMUTED_TO_TOKEN + METADATA_MAX_CTAS * tile_n + 1


class RoutingMetadata(NamedTuple):
    """Where the routing writes the routing metadata of the local experts
    ``[local_expert_offset, local_expert_offset + num_local_experts)`` for GEMM
    tile ``tile_n``: ``buffer``, int32 of at least ``metadata_size(tile_n)``
    elements, not aliasing the ids or weights."""

    buffer: torch.Tensor
    local_expert_offset: int
    num_local_experts: int
    tile_n: int

    def fields(self, num_tokens: int) -> dict[str, torch.Tensor]:
        """The fields of ``num_tokens`` tokens, as 1-D int32 views of
        ``buffer`` at the extents the routed MoE op allocates for them (with
        8 num_tokens <= 256 routes, at most one CTA per route):
        total_num_padded_tokens [1], num_non_exiting_ctas [1],
        expanded_idx_to_permuted_idx [8 num_tokens] (-1 for another rank's
        expert), cta_idx_xy_to_batch_idx and cta_idx_xy_to_mn_limit
        [8 num_tokens], and permuted_idx_to_token_idx [8 num_tokens tile_n + 1]
        (-1 for a padding row; the last element is the guard the GEMMs may
        read, never written)."""
        if not 1 <= num_tokens <= ROUTING_MAX_TOKENS:
            raise ValueError(f"1 to {ROUTING_MAX_TOKENS} tokens, got {num_tokens}")
        m = self.buffer
        routes = num_tokens * TOP_K
        return dict(
            total_num_padded_tokens=m[META_PADDED_TOKENS : META_PADDED_TOKENS + 1],
            num_non_exiting_ctas=m[META_NUM_CTAS : META_NUM_CTAS + 1],
            expanded_idx_to_permuted_idx=m[
                META_EXPANDED_TO_PERMUTED : META_EXPANDED_TO_PERMUTED + routes
            ],
            cta_idx_xy_to_batch_idx=m[META_CTA_BATCH : META_CTA_BATCH + routes],
            cta_idx_xy_to_mn_limit=m[META_CTA_LIMIT : META_CTA_LIMIT + routes],
            permuted_idx_to_token_idx=m[
                META_PERMUTED_TO_TOKEN : META_PERMUTED_TO_TOKEN
                + routes * self.tile_n
                + 1
            ],
        )


class MuseRouting:
    """The routing of one Muse Spark MoE layer for 1 to 8 tokens, bound to its
    router's bias and route scale at build time. A call allocates nothing, so
    it can be captured in a CUDA graph."""

    def __init__(self, router: MuseSpark1xRouter):
        reason = unsupported_router(router)
        if reason is not None:
            raise ValueError(f"Muse Spark routing does not support {reason}")
        self.bias = router.expert_bias
        self.route_scale = float(router.route_scale)

    def __call__(
        self,
        logits: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        metadata: RoutingMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route fp32 router logits [b, 256] (1 <= b <= 8, 16-byte aligned
        rows) into ``ids`` (int32 [b, 8], global expert ids) and ``weights``
        (fp32 [b, 8]), and with ``metadata`` the routing metadata of its local
        experts; returns (ids, weights)."""
        if metadata is None:
            ops.muse_spark_routing_shorttail(
                logits, self.bias, ids, weights, self.route_scale, True
            )
        else:
            ops.muse_spark_routing_shorttail(
                logits,
                self.bias,
                ids,
                weights,
                self.route_scale,
                True,
                metadata.buffer,
                metadata.local_expert_offset,
                metadata.num_local_experts,
                metadata.tile_n,
            )
        return ids, weights


def k_split(n: int) -> int:
    """The kernel's K split: one CTA per SM when a split of at least 4 allows
    it (largest power of two <= 8 with (n / 64) * split <= 152 CTAs), else two
    per SM (the same with 2 * 152 CTAs)."""
    heads, one = n // HEAD_DIM, 1
    while one < 8 and heads * one * 2 <= _SPLIT_SMS:
        one *= 2
    if one >= 4:
        return one
    split = 1
    while split < 8 and heads * split * 2 <= 2 * _SPLIT_SMS:
        split *= 2
    return split


def supports_shape(n: int, num_kv_heads: int) -> bool:
    """Whether the kernel has an instantiation for this projection width."""
    return (
        n % (2 * HEAD_DIM) == 0
        and 0 < num_kv_heads < n // (2 * HEAD_DIM)
        and k_split(n) in _SPLITS
    )


def qkv_front_available() -> bool:
    """Whether this build has the QKV front op (CUDA 13 builds for the SM100
    family)."""
    return hasattr(torch.ops._C, QKV_FRONT_OP)


def unsupported_projection(attn: torch.nn.Module) -> str | None:
    """Why the op cannot read this Muse Spark attention layer's loaded QKV
    projection and rotary table, or None."""
    proj = attn.qkv_proj
    weight = getattr(proj, "weight", None)
    n = 2 * (attn.num_heads + attn.num_kv_heads) * attn.head_dim
    if not (
        type(proj.quant_method) is UnquantizedLinearMethod
        and proj.bias is None
        and not (proj.gather_output and proj.tp_size > 1)
        and supports_shape(n, attn.num_kv_heads)
        and weight is not None
        and _matrix_ok(weight, n)
    ):
        return f"a QKV projection other than an unquantized bf16 [N, {HIDDEN_SIZE}]"
    emb = attn.rotary_emb
    if not attn.use_rope:
        return None if emb is None else "a rotary table on a layer without RoPE"
    cos_sin = getattr(emb, "cos_sin_cache", None)
    if not (
        emb is not None
        and emb.rotary_dim == attn.head_dim
        and not emb.is_neox_style
        and cos_sin is not None
        and cos_sin.dtype == torch.float32
        and cos_sin.dim() == 2
        and cos_sin.shape[1] == attn.head_dim
        and cos_sin.is_contiguous()
    ):
        return "a rotary table other than an interleaved fp32 one"
    return None


def _matrix_ok(t: torch.Tensor, rows: int) -> bool:
    return (
        t.dim() == 2
        and tuple(t.shape) == (rows, HIDDEN_SIZE)
        and t.dtype == torch.bfloat16
        and t.is_contiguous()
        and t.data_ptr() % 16 == 0
    )


# The KV cache dtypes of an fp8 cache as vLLM stores it (uint8) and as the
# kernels take it.
FP8_KV_CACHE_DTYPES = ("fp8", "fp8_e4m3")


def kv_cache_view(kv_cache: torch.Tensor, cache_dtype: str) -> torch.Tensor:
    """The layer's KV cache as the kernels take it: an fp8 cache (vLLM's uint8
    storage) as float8_e4m3fn, a bf16 one as is."""
    if cache_dtype in FP8_KV_CACHE_DTYPES and kv_cache.dtype == torch.uint8:
        return kv_cache.view(torch.float8_e4m3fn)
    return kv_cache


def kv_cache_layout_ok(kv_cache: torch.Tensor, num_kv_heads: int) -> bool:
    """Whether a KV cache has the layout the QKV front and attn2 read:
    ``[pages, num_kv_heads, 16, 128]`` (K | V per slot), dense pages of dense
    heads, 16-byte aligned; bf16 (both) or float8_e4m3fn (the QKV front
    only; attn2 refuses it), whose pages may hold 32 tokens (vLLM aligns an
    fp8 page to the bf16 skip layers' pages)."""
    row = 2 * HEAD_DIM
    fp8 = kv_cache.dtype == torch.float8_e4m3fn
    if (
        kv_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
        or kv_cache.dim() != 4
        or tuple(kv_cache.shape[1:2]) != (num_kv_heads,)
        or kv_cache.shape[2]
        not in ((BLOCK_SIZE, 2 * BLOCK_SIZE) if fp8 else (BLOCK_SIZE,))
        or kv_cache.shape[3] != row
    ):
        return False
    page = kv_cache.shape[2]
    stride = kv_cache.stride()
    return (
        stride[0] == num_kv_heads * page * row
        and (num_kv_heads == 1 or stride[1] == page * row)
        and stride[2] == row
        and stride[3] == 1
        and kv_cache.data_ptr() % 16 == 0
    )


def supports_call(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    num_kv_heads: int,
) -> bool:
    """Whether a call is inside what the kernel computes; otherwise the caller
    keeps the unfused path."""
    m = x.shape[0] if x.dim() >= 1 else 0
    n = weight.shape[0] if weight.dim() == 2 else 0
    if not (1 <= m <= QKV_FRONT_MAX_TOKENS and supports_shape(n, num_kv_heads)):
        return False
    if not (_matrix_ok(x, m) and _matrix_ok(weight, n)):
        return False
    for idx in (positions, slot_mapping):
        if (
            idx.dtype != torch.int64
            or idx.dim() != 1
            or idx.shape[0] < m
            or idx.stride(0) != 1
        ):
            return False
    if cos_sin_cache is not None and (
        cos_sin_cache.dtype != torch.float32
        or cos_sin_cache.dim() != 2
        or cos_sin_cache.shape[1] != HEAD_DIM
        or not cos_sin_cache.is_contiguous()
    ):
        return False
    if not kv_cache_layout_ok(kv_cache, num_kv_heads):
        return False
    tensors = [x, weight, positions, slot_mapping, kv_cache]
    if cos_sin_cache is not None:
        tensors.append(cos_sin_cache)
    device = x.device
    return device.type == "cuda" and all(t.device == device for t in tensors)
