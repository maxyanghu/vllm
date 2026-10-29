# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fp8 (e4m3) KV cache of the Muse Spark specialized decode's
full-attention layers: the QKV front's fp8 K/V store (bitwise the bf16 store's
values times 1 / scale, saturated to e4m3, with the projection output
unchanged), and the stock TRT-LLM-gen decode on the fp8 cache with an fp8
query (``QkvAttention.stock_decode_attention``) against an FP64 reference of
the dequantized cache, eagerly and replayed. Needs one GB300 and a build with
the Muse Spark csrc ops.

Run: pytest -v tests/models/muse_spark/nvidia/specialized_decode/test_fp8_kv.py
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_device_capability(103):
    pytest.skip("needs a GB300 (SM103) GPU", allow_module_level=True)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.models.muse_spark.nvidia.specialized_decode.runtime import (  # noqa: E402
    csrc_ops,
)

if not csrc_ops.qkv_front_available():
    pytest.skip("needs a build with the Muse Spark csrc ops", allow_module_level=True)

from vllm.models.muse_spark.nvidia.specialized_decode.runtime.providers import (  # noqa: E402,E501
    QkvAttention,
)

from .test_attention import (  # noqa: E402
    EPS,
    HEAD_DIM,
    MAX_LEN,
    ROWS,
    SCALE,
    _reference,
    _rel,
)

HIDDEN = 8192


def _front_inputs(kv_heads: int, m: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    heads = 8 * kv_heads
    n = (2 * heads + 2 * kv_heads) * HEAD_DIM
    x = (torch.randn(m, HIDDEN, generator=gen, device="cuda") * 0.5).bfloat16()
    w = (torch.randn(n, HIDDEN, generator=gen, device="cuda") * 0.02).bfloat16()
    positions = torch.randint(0, 4096, (m,), generator=gen, device="cuda")
    angles = torch.rand(8192, HEAD_DIM // 2, generator=gen, device="cuda") * 6.28
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=1).contiguous()
    pages = 8
    slots = torch.randperm(pages * 16, generator=gen, device="cuda")[:m]
    return x, w, positions, cos_sin, slots, pages


@pytest.mark.parametrize("kv_heads", [4, 2], ids=["tp4", "tp8"])
@pytest.mark.parametrize("m", [8, 3])
@pytest.mark.parametrize("page", [16, 32])
# (k, v): powers of two, others, and one that saturates e4m3.
@pytest.mark.parametrize("scales", [(0.5, 2.0), (0.3, 1.7), (1 / 512, 0.11)])
def test_qkv_front_fp8_store(kv_heads, m, page, scales):
    """The fp8 store (16- or 32-token pages) against the bf16 store's values
    quantized in torch; the projection output is the bf16 call's."""
    x, w, positions, cos_sin, slots, pages = _front_inputs(kv_heads, m, m + kv_heads)
    dev = x.device
    pages = pages * 16 // page
    cache16 = torch.zeros(pages, kv_heads, page, 2 * HEAD_DIM, dtype=torch.bfloat16)
    cache16 = cache16.to(dev)
    cache8 = torch.zeros_like(cache16, dtype=torch.float8_e4m3fn)
    k_scale = torch.tensor([scales[0]], device=dev)
    v_scale = torch.tensor([scales[1]], device=dev)
    out16 = torch.empty(m, w.shape[0], dtype=torch.bfloat16, device=dev)
    out8 = torch.empty_like(out16)
    if page == 32:
        # bf16 pages hold 16 tokens: the reference writes a 16-token view of
        # the same slots (slot s is token s % 16 of page s // 16 there).
        ref16 = torch.zeros(pages * 2, kv_heads, 16, 2 * HEAD_DIM)
        ref16 = ref16.to(dev, torch.bfloat16)
        ops.muse_spark_qkv_front(x, w, positions, cos_sin, slots, ref16, out16, EPS)
        # Back to 32-token pages: page p holds tokens of 16-pages 2p, 2p + 1.
        cache16.copy_(
            ref16.view(pages, 2, kv_heads, 16, 2 * HEAD_DIM)
            .transpose(1, 2)
            .reshape(pages, kv_heads, 32, 2 * HEAD_DIM)
        )
    else:
        ops.muse_spark_qkv_front(x, w, positions, cos_sin, slots, cache16, out16, EPS)
    ops.muse_spark_qkv_front(
        x, w, positions, cos_sin, slots, cache8, out8, EPS, k_scale, v_scale
    )
    assert torch.equal(out16.view(torch.int16), out8.view(torch.int16))
    # The kernel multiplies by the fp32 reciprocal, as the stock insert does.
    k16 = cache16[..., :HEAD_DIM].float() * (1.0 / k_scale)
    v16 = cache16[..., HEAD_DIM:].float() * (1.0 / v_scale)
    want = torch.cat([k16, v16], dim=-1).clamp(-448.0, 448.0)
    want = want.to(torch.float8_e4m3fn)
    assert torch.equal(want.view(torch.uint8), cache8.view(torch.uint8))
    assert int((cache16 != 0).any(-1).sum()) == m * kv_heads


def _provider(kv_heads: int, cache: torch.Tensor, scales, dev):
    from vllm.v1.attention.backends.flashinfer import (
        _get_multi_ctas_kv_counter_buffer,
        _get_trtllm_workspace_buffer,
    )

    q_scale, k_scale, v_scale = scales
    heads = kv_heads * 8
    impl = SimpleNamespace(
        bmm1_scale=None,
        bmm2_scale=None,
        scale=SCALE,
        sinks=None,
        cache_dtype="fp8_e4m3",
    )
    layer = SimpleNamespace(
        impl=impl,
        _q_scale=torch.tensor([q_scale], device=dev),
        _q_scale_float=q_scale,
        _k_scale_float=k_scale,
        _v_scale_float=v_scale,
    )
    provider = object.__new__(QkvAttention)
    provider.attn = SimpleNamespace(
        num_heads=heads, head_dim=HEAD_DIM, rms_norm_eps=EPS, attn=layer
    )
    provider.kv_fp8 = True
    provider.max_context = MAX_LEN
    provider.query_fp8 = torch.empty(
        16, heads, HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev
    )
    provider.out = torch.empty(16, heads * HEAD_DIM, dtype=torch.bfloat16, device=dev)
    provider.fi_workspace = _get_trtllm_workspace_buffer()
    provider.kv_counters = _get_multi_ctas_kv_counter_buffer(1 << 20, dev)
    provider.warm_up_stock_attention(cache)
    return provider


@pytest.mark.parametrize("kv_heads", [4, 2], ids=["tp4", "tp8"])
@pytest.mark.parametrize("length", [100, 5000, 100_000])
@pytest.mark.parametrize("page", [16, 32])
def test_stock_decode_fp8(kv_heads, length, page):
    torch.manual_seed(length + kv_heads)
    dev = torch.device("cuda")
    scales = (0.5, 0.5, 0.25)
    pages = (length + page - 1) // page
    cache = (torch.randn(pages + 3, kv_heads, page, 128, device=dev) * 0.25).to(
        torch.float8_e4m3fn
    )
    block_table = torch.randperm(pages + 3, device=dev).to(torch.int32)[None, :pages]
    block_table = block_table.contiguous()
    seq_lens = torch.tensor([length], dtype=torch.int32, device=dev)
    width = kv_heads * 8 * HEAD_DIM
    front = torch.randn(ROWS, 3 * width, device=dev).to(torch.bfloat16)
    q, gate = front[:, :width], front[:, 2 * width :]
    # The reference on the dequantized cache, with the query as the fp8 path
    # quantizes it (q / q_scale in e4m3, times q_scale).
    deq = cache.float()
    deq[..., :HEAD_DIM] *= scales[1]
    deq[..., HEAD_DIM:] *= scales[2]
    table16 = block_table
    if page == 32:
        # The reference reads 16-token pages: page p is 16-pages 2p, 2p + 1.
        deq = (
            deq.view(-1, kv_heads, 2, 16, 128)
            .transpose(1, 2)
            .reshape(-1, kv_heads, 16, 128)
        )
        table16 = torch.stack([2 * block_table, 2 * block_table + 1], dim=-1)
        table16 = table16.reshape(1, -1)
    q8 = (q.float() / scales[0]).to(torch.float8_e4m3fn).float() * scales[0]
    ref = _reference(q8, gate, deq, table16, length, kv_heads)
    provider = _provider(kv_heads, cache, scales, dev)
    decode = SimpleNamespace(
        block_tables=block_table, seq_lens=seq_lens, max_seq_len=MAX_LEN
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = provider.stock_decode_attention(q, gate, cache, decode)
    graph.replay()
    torch.cuda.synchronize()
    replayed = captured.clone()
    eager = provider.stock_decode_attention(q, gate, cache, decode).clone()
    assert torch.equal(eager.view(torch.int16), replayed.view(torch.int16))
    err = _rel(replayed, ref)
    print(f"kv_heads {kv_heads} L {length} page {page}: fp8 decode rel L2 {err:.2e}")
    assert err < 3e-2, err
