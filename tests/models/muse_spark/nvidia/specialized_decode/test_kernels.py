# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU kernels of the Muse Spark specialized decode
(``vllm/models/muse_spark/nvidia/specialized_decode``), each against the stock
op it replaces, eagerly and replayed from a CUDA graph. Needs one GB300
(SM103); about two minutes, mostly kernel compilation.

Run: pytest -v tests/models/muse_spark/nvidia/specialized_decode/test_kernels.py
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import get_routing_method_type
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
)
from vllm.platforms import current_platform

if not current_platform.is_device_capability(103):
    pytest.skip("needs a GB300 (SM103) GPU", allow_module_level=True)

from vllm.models.muse_spark.common.ops.fused_finalize import (  # noqa: E402
    fused_post_expert_finalize,
)
from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.moe import (  # noqa: E402
    persistent_experts as pe,
)

HIDDEN = pe.HIDDEN
INTER = pe.INTER
TOP_K = pe.TOP_K
NUM_EXPERTS = 256


def _fp8_scales(n: int, gen: torch.Generator, lo: float, hi: float) -> torch.Tensor:
    values = torch.rand(n, generator=gen, device="cuda") * (hi - lo) + lo
    return values.to(torch.float8_e4m3fn)


def _codes(shape: tuple[int, ...], gen: torch.Generator) -> torch.Tensor:
    return torch.randint(0, 256, shape, generator=gen, device="cuda").to(torch.uint8)


def _window(num_local: int, seed: int) -> dict[str, torch.Tensor]:
    """Random weights already in the TRT-LLM-gen layout both kernels read: any
    byte is a valid E2M1 pair, and the block scales are positive E4M3, so the
    layout transform of a checkpoint does not matter for a bitwise comparison."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = {
        "w13": _codes((num_local, 2 * INTER, HIDDEN // 2), gen),
        "w13_sf": _fp8_scales(num_local * 2 * INTER * HIDDEN // 16, gen, 0.25, 1.0),
        "w2": _codes((num_local, HIDDEN, INTER // 2), gen),
        "w2_sf": _fp8_scales(num_local * HIDDEN * INTER // 16, gen, 0.25, 1.0),
    }
    for name in ("g1_scale_c", "g1_alphas", "g2_alphas"):
        w[name] = (
            torch.rand(num_local, generator=gen, device="cuda") * 2e-3 + 1e-3
        ).float()
    w["w13_sf"] = w["w13_sf"].view(num_local, 2 * INTER, HIDDEN // 16)
    w["w2_sf"] = w["w2_sf"].view(num_local, HIDDEN, INTER // 16)
    return w


def _inputs(rows: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(1000 + seed)
    x = _codes((rows, HIDDEN // 2), gen)
    x_sf = _fp8_scales(rows * HIDDEN // 16, gen, 0.5, 2.0).view(torch.uint8)
    x_sf = x_sf.view(rows, HIDDEN // 16)
    # Distinct experts per token, as the top-k router produces them.
    ids = torch.stack(
        [
            torch.randperm(NUM_EXPERTS, generator=gen, device="cuda")[:TOP_K]
            for _ in range(rows)
        ]
    ).to(torch.int32)
    weights = torch.rand(rows, TOP_K, generator=gen, device="cuda").float()
    return x, x_sf, ids, weights


def _stock(w, x, x_sf, ids, weights, num_local: int, ep_rank: int):
    """The stock op: FlashInfer's routed TRT-LLM-gen MoE, unfinalized, as
    ``MuseSparkTrtLlmNvFp4Experts.apply_routed`` calls it."""
    import flashinfer

    rows = x.shape[0]
    out = flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(
        topk_ids=(ids, weights),
        routing_bias=None,
        hidden_states=x,
        hidden_states_scale=x_sf.view(torch.float8_e4m3fn).reshape(rows, -1),
        gemm1_weights=w["w13"],
        gemm1_weights_scale=w["w13_sf"],
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=w["w2"],
        gemm2_weights_scale=w["w2_sf"],
        gemm2_bias=None,
        output1_scale_scalar=w["g1_scale_c"],
        output1_scale_gate_scalar=w["g1_alphas"],
        output2_scale_scalar=w["g2_alphas"],
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=0,
        topk_group=0,
        intermediate_size=INTER,
        local_expert_offset=ep_rank * num_local,
        local_num_experts=num_local,
        routed_scaling_factor=None,
        routing_method_type=get_routing_method_type(
            scoring_func="sigmoid",
            top_k=TOP_K,
            renormalize=True,
            num_expert_group=None,
            has_e_score_bias=True,
        ),
        do_finalize=False,
        activation_type=activation_to_flashinfer_int(MoEActivation.SILU),
    )
    gemm2_rows, expert_weights, index = out
    return gemm2_rows, expert_weights, index.view(rows, TOP_K)


def _persistent(w, x, x_sf, ids, num_local: int, ep_rank: int):
    args = (
        x,
        x_sf,
        ids,
        w["w13"],
        w["w13_sf"],
        w["w2"],
        w["w2_sf"],
        w["g1_scale_c"],
        w["g1_alphas"],
        w["g2_alphas"],
    )
    pe.check_inputs(*args)
    return pe.persistent_experts(ep_rank * num_local, *args)


def _assert_same_routes(stock, ours, rows: int):
    stock_rows, _, stock_index = stock
    our_rows, our_index = ours
    local = stock_index >= 0
    assert torch.equal(local, our_index >= 0), "local slots differ"
    expected = torch.arange(rows * TOP_K, device="cuda", dtype=torch.int32)
    assert torch.equal(our_index[local], expected.view(rows, TOP_K)[local])
    picked = stock_rows[stock_index[local].long()]
    mine = our_rows[our_index[local].long()]
    assert torch.equal(picked.view(torch.int16), mine.view(torch.int16)), (
        f"{int((picked != mine).any(-1).sum())} of {int(local.sum())} routed rows "
        "differ"
    )
    return int(local.sum())


@pytest.fixture(scope="module", params=[(32, 3), (64, 1)], ids=["ep8", "ep4"])
def window(request):
    num_local, ep_rank = request.param
    w = _window(num_local, seed=num_local)
    pe.prepare(ep_rank * num_local, w["w13"], w["w13_sf"], w["w2"], w["w2_sf"])
    return num_local, ep_rank, w


@pytest.mark.parametrize("rows", range(1, pe.MAX_ROWS + 1))
def test_persistent_experts_eager(window, rows):
    """Every routed FC2 row equals FlashInfer's bit for bit, the index maps
    each local route to its own row, and a repeat gives the same bits."""
    num_local, ep_rank, w = window
    for seed in range(3):
        x, x_sf, ids, weights = _inputs(rows, seed)
        stock = _stock(w, x, x_sf, ids, weights, num_local, ep_rank)
        # The stock op passes the router's weights through untouched.
        assert stock[1].dtype == weights.dtype
        assert torch.equal(stock[1].view(rows, TOP_K), weights)
        ours = _persistent(w, x, x_sf, ids, num_local, ep_rank)
        _assert_same_routes(stock, ours, rows)
        again = _persistent(w, x, x_sf, ids, num_local, ep_rank)
        _assert_same_routes(stock, again, rows)


@pytest.mark.parametrize("rows", [1, 8])
def test_persistent_experts_finalized(window, rows):
    """The post-expert finalize gives the same bits on either op's output."""
    num_local, ep_rank, w = window
    x, x_sf, ids, weights = _inputs(rows, 7)
    stock_rows, stock_weights, stock_index = _stock(
        w, x, x_sf, ids, weights, num_local, ep_rank
    )
    our_rows, our_index = _persistent(w, x, x_sf, ids, num_local, ep_rank)
    norm_weight = torch.rand(HIDDEN, device="cuda").to(torch.bfloat16)
    outs = []
    for rows_all, index in ((stock_rows, stock_index), (our_rows, our_index)):
        out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
        fused_post_expert_finalize(
            rows_all,
            norm_weight,
            stock_weights.view(rows, TOP_K),
            1e-6,
            0.0,
            False,
            out,
            permuted_idx=index,
        )
        outs.append(out)
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))


@pytest.mark.parametrize("rows", [1, 4, 8])
def test_persistent_experts_graph_replay(window, rows):
    """Captured once and replayed on new inputs, as the decode graphs run it:
    the workspace's epochs advance across replays without clearing."""
    num_local, ep_rank, w = window
    x, x_sf, ids, weights = _inputs(rows, 0)
    static = [x.clone(), x_sf.clone(), ids.clone()]

    def run():
        return pe.persistent_experts(
            ep_rank * num_local,
            static[0],
            static[1],
            static[2],
            w["w13"],
            w["w13_sf"],
            w["w2"],
            w["w2_sf"],
            w["g1_scale_c"],
            w["g1_alphas"],
            w["g2_alphas"],
        )

    run()  # warm up outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run()
    for replay in range(20):
        x, x_sf, ids, weights = _inputs(rows, 100 + replay % 5)
        static[0].copy_(x)
        static[1].copy_(x_sf)
        static[2].copy_(ids)
        graph.replay()
        torch.cuda.synchronize()
        stock = _stock(w, x, x_sf, ids, weights, num_local, ep_rank)
        _assert_same_routes(stock, graph_out, rows)
