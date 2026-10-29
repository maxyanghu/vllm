# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert replicas for the persistent expert kernel: every EP rank holds its
``L`` stock experts plus ``R`` replicas of the next rank's first ``R`` experts,
and each decode step routes every expert it uses to the less loaded of the
ranks holding it.

Placement (a ring, fixed at build): rank ``r`` holds window
``[own experts [rL, (r + 1)L) | experts [qL, qL + R)]`` with ``q = (r + 1) %
tp``, ``L + R`` local experts in one contiguous tensor per weight kind, which
is what the persistent kernel reads (``WINDOWS``). The stock layer's
parameters are re-pointed at the first ``L`` experts of the window tensors (the
same bytes, layout and strides), so the window costs only the replicas.

Assignment (per step, on the device, identical on every rank since every rank
routes the same replicated tokens): an expert's cost is one weight stream per
rank, however many routes use it. Experts without a replica stay on their
owner. The replicated ones start on their owner; then, one at a time, an
expert moves from the rank ``p`` with the largest load lead over ``p - 1``
(at least 2, with a replicated expert of ``p`` left) to ``p - 1``, which
holds its replica. The experts that move are the owner's replicated experts
in slot order. Every route of an expert goes where the expert goes. The
kernel writes this rank's window-local ids (``-1`` for a route served
elsewhere), which the persistent kernel reads with no id offset.

The routes move between ranks, so each rank's BF16 partial of the EP combine
changes: the bits differ from the stock EP placement (not the arithmetic of a
route's row, which is the same kernel on the same weights).
"""

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, triton

# Replicas per rank: the window of an EP4 rank is 64 + 32 = 96 experts and of
# an EP8 rank 32 + 32 = 64 (persistent_experts.WINDOWS).
REPLICAS = 32
# Routes per step the assignment serves: 8 tokens x top-8.
MAX_SLOTS = 64


@triton.jit
def _assign_kernel(
    ids_ptr,
    out_ptr,
    num_slots,
    rank,
    TP: tl.constexpr,
    LOCAL: tl.constexpr,
    REP: tl.constexpr,
    SLOTS: tl.constexpr,
):
    j = tl.arange(0, SLOTS)
    valid = j < num_slots
    e = tl.load(ids_ptr + j, mask=valid, other=-1)
    valid = valid & (e >= 0)
    ei = e[:, None]
    ej = e[None, :]
    ii = j[:, None]
    jj = j[None, :]
    same = (ei == ej) & valid[:, None] & valid[None, :]
    # First slot of each expert: no earlier slot names it.
    earlier = tl.sum((same & (ii < jj)).to(tl.int32), axis=0)
    first = valid & (earlier == 0)
    owner = tl.where(valid, e // LOCAL, -1)
    rep = (e % LOCAL) < REP
    cand = first & rep
    r = tl.arange(0, TP)
    on = owner[:, None] == r[None, :]  # [SLOTS, TP]
    fixed = tl.sum((on & (first & ~rep)[:, None]).to(tl.int32), axis=0)
    movable = tl.sum((on & cand[:, None]).to(tl.int32), axis=0)
    # prev[a, b]: b is a's predecessor on the ring.
    prev = r[None, :] == ((r[:, None] + TP - 1) % TP)
    nxt = r[None, :] == ((r[:, None] + 1) % TP)
    moved = tl.zeros([TP], dtype=tl.int32)
    go = tl.sum(movable) > 0
    while go:
        load = fixed + movable - moved + tl.sum(tl.where(nxt, moved[None, :], 0), 1)
        lead = load - tl.sum(tl.where(prev, load[None, :], 0), axis=1)
        lead = tl.where(moved < movable, lead, -SLOTS)
        best = tl.max(lead)
        pick = tl.min(tl.where(lead == best, r, TP))
        go = best >= 2
        moved = moved + tl.where((r == pick) & go, 1, 0)
    # The first moved[p] replicated experts of owner p (in slot order) move to
    # p - 1.
    order = tl.sum(
        ((ii < jj) & cand[:, None] & (owner[:, None] == owner[None, :])).to(tl.int32),
        axis=0,
    )
    quota = tl.sum(tl.where(on, moved[None, :], 0), axis=1)
    goes = cand & (order < quota)
    home = tl.where(goes, (owner + TP - 1) % TP, owner)
    # Every route goes where its expert's first slot goes.
    dest = tl.max(tl.where(same & first[:, None], home[:, None], -1), axis=0)
    local = tl.where(owner == rank, e - rank * LOCAL, LOCAL + e % LOCAL)
    out = tl.where(valid & (dest == rank), local, -1)
    tl.store(out_ptr + j, out, mask=j < num_slots)


def assign(ids: torch.Tensor, out: torch.Tensor, rank: int, tp: int, local: int):
    """This rank's window-local ids (``out``, int32 like ``ids``) of the routes
    ``ids`` (int32 ``[b, top_k]`` global expert ids, ``b * top_k <=
    MAX_SLOTS``); graph-capturable, allocates nothing."""
    n = ids.numel()
    assert n <= MAX_SLOTS and ids.is_contiguous() and ids.dtype == torch.int32
    assert out.numel() >= n and out.is_contiguous() and out.dtype == torch.int32
    _assign_kernel[(1,)](
        ids, out, n, rank, TP=tp, LOCAL=local, REP=REPLICAS, SLOTS=MAX_SLOTS,
        num_warps=4,
    )  # fmt: skip


def assign_reference(ids: torch.Tensor, tp: int, local: int) -> torch.Tensor:
    """The rank of every route under the assignment above (host, for tests)."""
    flat = [int(x) for x in ids.reshape(-1).tolist()]
    seen: dict[int, int] = {}
    for slot, e in enumerate(flat):
        if e >= 0 and e not in seen:
            seen[e] = slot
    fixed, movable = [0] * tp, [0] * tp
    for e in seen:
        if e % local < REPLICAS:
            movable[e // local] += 1
        else:
            fixed[e // local] += 1
    moved = [0] * tp
    while True:
        load = [
            fixed[p] + movable[p] - moved[p] + moved[(p + 1) % tp] for p in range(tp)
        ]
        leads = [
            (load[p] - load[(p - 1) % tp]) if moved[p] < movable[p] else -MAX_SLOTS
            for p in range(tp)
        ]
        best = max(leads)
        if best < 2:
            break
        moved[leads.index(best)] += 1
    home: dict[int, int] = {}
    count = [0] * tp
    for e, _ in sorted(seen.items(), key=lambda kv: kv[1]):
        p = e // local
        if e % local < REPLICAS and count[p] < moved[p]:
            count[p] += 1
            home[e] = (p - 1) % tp
        else:
            home[e] = p
    return torch.tensor([home.get(e, -1) for e in flat]).view(ids.shape)


def extend_window(
    tensors: tuple[torch.Tensor, ...], group: dist.ProcessGroup
) -> tuple[torch.Tensor, ...]:
    """Each of this rank's per-expert tensors ``[L, ...]`` (contiguous) as its
    window ``[L + REPLICAS, ...]``: its own ``L`` experts, then the next rank's
    first ``REPLICAS`` (received over ``group`` with NCCL send/recv on a ring).
    Collective; run on every rank before capture, on the current stream."""
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    src = dist.get_global_rank(group, (rank + 1) % world)
    dst = dist.get_global_rank(group, (rank - 1) % world)
    out = []
    for t in tensors:
        assert t.is_contiguous() and t.shape[0] >= REPLICAS
        window = torch.empty(
            (t.shape[0] + REPLICAS, *t.shape[1:]), dtype=t.dtype, device=t.device
        )
        window[: t.shape[0]].copy_(t)
        # Bytes on the wire: NCCL takes no float8 / packed dtypes everywhere.
        send = t[:REPLICAS].view(torch.uint8)
        recv = window[t.shape[0] :].view(torch.uint8)
        for req in dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, send, dst, group),
                dist.P2POp(dist.irecv, recv, src, group),
            ]
        ):
            req.wait()
        out.append(window)
    return tuple(out)
