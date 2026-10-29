# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The full-attention decode of the Muse Spark specialized decode: attn2 (the
csrc op) and the stock path's TRT-LLM-gen decode + o-norm/gate
(``QkvAttention.stock_decode_attention``), each against an FP64 reference of
the 8-token verify step, at short and long contexts, eagerly and replayed from
a CUDA graph. Needs one GB300 and a build with the Muse Spark csrc ops.

Run: pytest -v tests/models/muse_spark/nvidia/specialized_decode/test_attention.py
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_device_capability(103):
    pytest.skip("needs a GB300 (SM103) GPU", allow_module_level=True)

from vllm.models.muse_spark.nvidia.specialized_decode.runtime import (  # noqa: E402
    csrc_ops,
)

if not csrc_ops.attn2_available():
    pytest.skip("needs a build with the Muse Spark csrc ops", allow_module_level=True)

from vllm.models.muse_spark.nvidia.specialized_decode.runtime.providers import (  # noqa: E402,E501
    QkvAttention,
)

HEAD_DIM = 64
ROWS = 8
# head_dim ** -0.5 * qk_scale_factor, attn2's kSmScale.
SCALE = 0.6841258108
EPS = 1e-5
MAX_LEN = 1048576


def _reference(q, gate, cache, block_table, length, kv_heads):
    """FP64 attention of the last 8 cache entries (causal among them), RMS
    norm without weight, sigmoid gate: ``[8, heads * 64]``."""
    pages = block_table[0, : (length + 15) // 16].long()
    kv = cache[pages].double()  # [p, H, 16, 128]
    kv = kv.permute(1, 0, 2, 3).reshape(kv_heads, -1, 2 * HEAD_DIM)[:, :length]
    k, v = kv[..., :HEAD_DIM], kv[..., HEAD_DIM:]
    heads = kv_heads * 8
    qh = q.double().view(ROWS, heads, HEAD_DIM)
    out = torch.empty(ROWS, heads, HEAD_DIM, dtype=torch.float64, device=q.device)
    for t in range(ROWS):
        visible = length - ROWS + t + 1
        for h in range(heads):
            kh = h // 8
            s = (k[kh, :visible] @ qh[t, h]) * SCALE
            p = torch.softmax(s, dim=0)
            out[t, h] = p @ v[kh, :visible]
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + EPS)
    out = out * torch.sigmoid(gate.double().view(ROWS, heads, HEAD_DIM))
    return out.view(ROWS, heads * HEAD_DIM)


def _rel(a, b):
    return ((a.double() - b).norm() / b.norm()).item()


@pytest.mark.parametrize("kv_heads", [4, 2], ids=["tp4", "tp8"])
@pytest.mark.parametrize("length", [100, 5000, 100_000])
def test_full_attention_decode(kv_heads, length):
    torch.manual_seed(length + kv_heads)
    dev = torch.device("cuda")
    pages = (length + 15) // 16
    cache = (torch.randn(pages + 3, kv_heads, 16, 128, device=dev) * 0.5).to(
        torch.bfloat16
    )
    block_table = torch.randperm(pages + 3, device=dev).to(torch.int32)[None, :pages]
    block_table = block_table.contiguous()
    seq_lens = torch.tensor([length], dtype=torch.int32, device=dev)
    heads = kv_heads * 8
    width = heads * HEAD_DIM
    # q and gate as column views of one front output, as the QKV front leaves them.
    front = torch.randn(ROWS, 3 * width, device=dev).to(torch.bfloat16)
    q, gate = front[:, :width], front[:, 2 * width :]
    ref = _reference(q, gate, cache, block_table, length, kv_heads)

    # attn2
    out2 = torch.empty(ROWS, width, dtype=torch.bfloat16, device=dev)
    splits = csrc_ops.attn2_splits(MAX_LEN, 0, kv_heads)
    ws = torch.zeros(
        csrc_ops.attn2_workspace_bytes(1, kv_heads, splits),
        dtype=torch.uint8,
        device=dev,
    )
    csrc_ops.attn2(
        q.contiguous(), cache, block_table, seq_lens, gate.contiguous(), out2,
        kv_heads, True, MAX_LEN, 0, splits, ws,
    )  # fmt: skip

    # The stock path, through the provider on a stand-in layer: the
    # pre-capture warm-up, then the capture with no eager call at this shape.
    from vllm.v1.attention.backends.flashinfer import (
        _get_multi_ctas_kv_counter_buffer,
        _get_trtllm_workspace_buffer,
    )

    impl = SimpleNamespace(
        bmm1_scale=None, bmm2_scale=None, scale=SCALE, sinks=None, cache_dtype="auto"
    )
    provider = object.__new__(QkvAttention)
    provider.attn = SimpleNamespace(
        num_heads=heads,
        head_dim=HEAD_DIM,
        rms_norm_eps=EPS,
        attn=SimpleNamespace(impl=impl),
    )
    provider.max_context = MAX_LEN
    provider.kv_fp8 = False
    provider.query = torch.empty(16, heads, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    provider.out = torch.empty(16, width, dtype=torch.bfloat16, device=dev)
    provider.fi_workspace = _get_trtllm_workspace_buffer()
    provider.kv_counters = _get_multi_ctas_kv_counter_buffer(1 << 20, dev)
    provider.warm_up_stock_attention(cache)
    decode = SimpleNamespace(
        block_tables=block_table, seq_lens=seq_lens, max_seq_len=MAX_LEN
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = provider.stock_decode_attention(q, gate, cache, decode)
    graph.replay()
    torch.cuda.synchronize()
    stock = captured.clone()
    eager = provider.stock_decode_attention(q, gate, cache, decode).clone()
    assert torch.equal(eager.view(torch.int16), stock.view(torch.int16))

    e2, es = _rel(out2, ref), _rel(stock, ref)
    print(f"kv_heads {kv_heads} L {length}: attn2 {e2:.2e} stock {es:.2e}")
    assert es < 6e-3, es
    assert e2 < 6e-3, e2

    # The replay reads the step's inputs: new queries, cache, pages and a
    # shorter length, in place.
    shorter = max(ROWS, length // 2 + 3)
    front.copy_(torch.randn_like(front, dtype=torch.float32).to(torch.bfloat16))
    cache.copy_((torch.randn_like(cache, dtype=torch.float32) * 0.5).to(cache.dtype))
    block_table.copy_(torch.randperm(pages + 3, device=dev).to(torch.int32)[:pages])
    seq_lens.fill_(shorter)
    graph.replay()
    torch.cuda.synchronize()
    replayed = captured.clone()  # the eager call below writes the same scratch
    ref2 = _reference(q, gate, cache, block_table, shorter, kv_heads)
    eager2 = provider.stock_decode_attention(q, gate, cache, decode).clone()
    assert torch.equal(replayed.view(torch.int16), eager2.view(torch.int16))
    assert _rel(replayed, ref2) < 6e-3
