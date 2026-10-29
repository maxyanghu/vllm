# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert replicas of the Muse Spark specialized decode
(``runtime/replicas.py``): the per-step assignment kernel against its
host reference on every rank, and the persistent expert kernel on each rank's
window (own experts + the next rank's first replicas) with the assigned ids,
whose every route row must equal FlashInfer's on the expert's owner, bit for
bit, with every route served by exactly one rank. Needs one GB300 (SM103).

Run: pytest -v tests/models/muse_spark/nvidia/specialized_decode/test_expert_replicas.py
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_device_capability(103):
    pytest.skip("needs a GB300 (SM103) GPU", allow_module_level=True)

from vllm.models.muse_spark.nvidia.specialized_decode.cutedsl.moe import (  # noqa: E402
    persistent_experts as pe,
)
from vllm.models.muse_spark.nvidia.specialized_decode.runtime import (  # noqa: E402
    replicas as er,
)

from .test_kernels import (  # noqa: E402
    NUM_EXPERTS,
    TOP_K,
    _inputs,
    _stock,
    _window,
)


def _ids(rows: int, gen: torch.Generator, skew: int | None) -> torch.Tensor:
    """Distinct experts per token; with ``skew`` all from the 64 experts
    starting there (a hot rank)."""
    pool = NUM_EXPERTS if skew is None else 64
    picks = [torch.randperm(pool, generator=gen)[:TOP_K] for _ in range(rows)]
    ids = torch.stack(picks).to(torch.int32)
    if skew is not None:
        ids = (ids + skew) % NUM_EXPERTS
    return ids.cuda()


def _ranks(ids: torch.Tensor, tp: int) -> torch.Tensor:
    """Every route's rank from the kernel run for every rank, checking that
    each route is local to exactly one rank and its local id names its expert
    in that rank's window."""
    local = NUM_EXPERTS // tp
    dest = torch.full(ids.shape, -1, dtype=torch.int32)
    for rank in range(tp):
        out = torch.full_like(ids, -7)
        er.assign(ids, out, rank, tp, local)
        out = out.cpu()
        mine = out >= 0
        assert bool(((out == -1) | mine).all())
        assert not bool((mine & (dest >= 0)).any()), "a route on two ranks"
        dest[mine] = rank
        e = ids.cpu()[mine]
        lid = out[mine]
        nxt = (rank + 1) % tp
        expected = torch.where(
            e // local == rank, e - rank * local, local + e % local
        ).to(torch.int32)
        assert torch.equal(lid, expected)
        owner = e // local
        assert bool(
            ((owner == rank) | ((owner == nxt) & (e % local < er.REPLICAS))).all()
        ), "a route on a rank without the expert"
    assert bool((dest >= 0).all()), "a route on no rank"
    return dest


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("skew", [None, 0, 96, 200])
def test_assignment(tp, skew):
    gen = torch.Generator().manual_seed(tp * 1000 + (skew or 1))
    for trial in range(40):
        rows = 1 + trial % 8
        ids = _ids(rows, gen, skew)
        dest = _ranks(ids, tp)
        ref = er.assign_reference(ids.cpu(), tp, NUM_EXPERTS // tp).to(torch.int32)
        assert torch.equal(dest, ref), (ids, dest, ref)
        # Each rank's distinct experts: never above the no-replica placement.
        local = NUM_EXPERTS // tp
        uniq = {int(e): int(d) for e, d in zip(ids.view(-1), dest.view(-1))}
        loads = [sum(1 for d in uniq.values() if d == r) for r in range(tp)]
        stock = [sum(1 for e in uniq if e // local == r) for r in range(tp)]
        assert max(loads) <= max(stock)


@pytest.fixture(scope="module")
def experts():
    """All 256 experts (EP4 windows are slices of these)."""
    return _window(NUM_EXPERTS, seed=7)


def _rank_window(w, rank: int, tp: int):
    local = NUM_EXPERTS // tp
    q = (rank + 1) % tp
    own = slice(rank * local, (rank + 1) * local)
    rep = slice(q * local, q * local + er.REPLICAS)
    return {k: torch.cat([v[own], v[rep]]).contiguous() for k, v in w.items()}


def _owner_window(w, owner: int, tp: int):
    local = NUM_EXPERTS // tp
    return {k: v[owner * local : (owner + 1) * local] for k, v in w.items()}


@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("skew", [None, 0])
def test_window_rows(experts, rows, skew):
    """EP4 with replicas: every route row, on whichever rank serves it, equals
    FlashInfer's row on the expert's owner (stock EP4) bit for bit."""
    tp, local = 4, NUM_EXPERTS // 4
    x, x_sf, _, weights = _inputs(rows, seed=rows)
    ids = _ids(rows, torch.Generator().manual_seed(rows), skew)
    dest = _ranks(ids, tp)
    served = torch.zeros(rows, TOP_K, dtype=torch.int32)
    for rank in range(tp):
        w = _rank_window(experts, rank, tp)
        pe.prepare(0, w["w13"], w["w13_sf"], w["w2"], w["w2_sf"])
        lids = torch.empty_like(ids)
        er.assign(ids, lids, rank, tp, local)
        rows_out, index = pe.persistent_experts(
            0, x, x_sf, lids, w["w13"], w["w13_sf"], w["w2"], w["w2_sf"],
            w["g1_scale_c"], w["g1_alphas"], w["g2_alphas"],
        )  # fmt: skip
        mine = index.cpu() >= 0
        assert torch.equal(mine, dest == rank)
        served += mine.int()
        for owner in range(tp):
            stock_rows, _, stock_index = _stock(
                _owner_window(experts, owner, tp), x, x_sf, ids, weights, local, owner
            )
            sel = mine & (ids.cpu() // local == owner)
            if not bool(sel.any()):
                continue
            want = stock_rows[stock_index.cpu()[sel].long().cuda()]
            got = rows_out[index.cpu()[sel].long().cuda()]
            assert torch.equal(want.view(torch.int16), got.view(torch.int16)), (
                f"rank {rank}, owner {owner}: route rows differ"
            )
    assert bool((served == 1).all())


def test_assign_graph_replay():
    """The assignment captured once, replayed on new routes in place."""
    tp, local = 4, NUM_EXPERTS // 4
    gen = torch.Generator().manual_seed(3)
    ids = _ids(8, gen, None)
    out = torch.empty_like(ids)
    er.assign(ids, out, 1, tp, local)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        er.assign(ids, out, 1, tp, local)
    for skew in (None, 0, 64, 128):
        ids.copy_(_ids(8, gen, skew))
        graph.replay()
        want = torch.empty_like(ids)
        er.assign(ids, want, 1, tp, local)
        assert torch.equal(out, want)
        ref = er.assign_reference(ids.cpu(), tp, local)
        assert torch.equal((out.cpu() >= 0), ref == 1)


def _extend_worker(rank: int, world: int, init: str, queue) -> None:
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=init, rank=rank, world_size=world)
    try:
        gen = torch.Generator(device="cuda").manual_seed(11)
        local = NUM_EXPERTS // world
        full_w = torch.randint(
            0, 256, (NUM_EXPERTS, 3, 5), generator=gen, device="cuda"
        ).to(torch.uint8)
        full_s = torch.rand(NUM_EXPERTS, generator=gen, device="cuda")
        full_f8 = full_s.view(NUM_EXPERTS, 1).expand(-1, 4).contiguous()
        full_f8 = full_f8.to(torch.float8_e4m3fn)
        own = slice(rank * local, (rank + 1) * local)
        q = (rank + 1) % world
        rep = slice(q * local, q * local + er.REPLICAS)
        mine = (full_w[own].contiguous(), full_s[own].contiguous(),
                full_f8[own].contiguous())  # fmt: skip
        out = er.extend_window(mine, dist.group.WORLD)
        for got, full in zip(out, (full_w, full_s, full_f8)):
            want = torch.cat([full[own], full[rep]])
            assert got.dtype == full.dtype and got.shape == want.shape
            assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
        queue.put((rank, "ok"))
    except Exception as e:  # noqa: BLE001
        queue.put((rank, repr(e)))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [4])
def test_extend_window(world, tmp_path):
    """The NCCL ring: each rank's window is its own experts, then the next
    rank's first REPLICAS, byte for byte, for uint8 / fp32 / fp8 tensors."""
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    init = f"file://{tmp_path}/rdzv"
    procs = [
        ctx.Process(target=_extend_worker, args=(r, world, init, queue))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    results = dict(queue.get(timeout=300) for _ in procs)
    for p in procs:
        p.join(timeout=60)
    assert results == {r: "ok" for r in range(world)}, results
