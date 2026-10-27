// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Muse Spark attention front for decode batches (1 <= M <= 16) in one launch:
// the QKV + o-gate projection (split-K bf16 GEMM, mma.sync, fp32 accumulate),
// then per-head weightless RMS norm, interleaved RoPE, the y store and the
// paged K/V insert. It writes what the projection GEMM followed by
// fused_muse_spark_qknorm_rope_kv_insert writes for a bf16 KV cache.
//
// Structure: a thread-block cluster computes one head (64 output columns), its
// S CTAs splitting K. In each CTA a producer thread streams the W and x tiles
// into a shared-memory ring with TMA, four consumer warps run mma.sync on them
// and an aux warp fetches the epilogue inputs. Each CTA pushes its partial of
// token t into the owner CTA (rank t % S) with st.async; the owner sums the S
// partials in fixed rank order (bitwise deterministic), rounds to bf16 and runs
// the epilogue. No global scratch, no atomics, no state across calls.
//
// PDL: the kernel may start while its predecessor still runs. Before
// griddepcontrol.wait it reads only the weights, which no kernel writes inside
// a graph; every thread waits before it reads x, positions, slots or cos|sin
// and before any global store. y and the cache are bitwise those of a launch
// without PDL.
//
// Host rules: tm_w = 2D bf16 map over w [N, 8192] (box 64 x 64 rows), tm_x over
// x [M, 8192] (box 64 x 8*NT rows), both 128B swizzle, L2 promotion 256B, zero
// OOB fill; grid (N/64)*S CTAs of NTHREADS threads, cluster (S,1,1),
// programmatic stream serialization, dynamic shared memory
// Cfg<NT,STAGES>::SMEM; NT = 1 for M <= 8, NT = 2 for 9 <= M <= 16; S and
// STAGES from k_split and ring_stages below.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include <map>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_set>

#include <torch/headeronly/core/ScalarType.h>

#include "torch_utils.h"

namespace vllm {
namespace muse_spark_qkv {

constexpr int D = 64;       // head width
constexpr int KDIM = 8192;  // reduction length (constant axis of the problem)
constexpr int KT = 128;     // k per pipeline stage (2 swizzled 64-wide boxes)
constexpr int KTILES = KDIM / KT;
constexpr int KSTRIPE =
    2;  // adjacent K128 tiles per rank before cycling across the cluster
constexpr int BOXK = 64;  // TMA box inner extent (64 bf16 = 128 B)
constexpr int NBOX = KT / BOXK;
constexpr int W_BOX_BYTES = 64 * 128;  // one [64 rows x 64 k] box
constexpr int NCW = 4;                 // consumer warps (32 k each per stage)
constexpr int PRODW = NCW;             // producer warp
constexpr int AUXW = NCW + 1;          // aux warp
constexpr int NWARPS = NCW + 2;
constexpr int NTHREADS = NWARPS * 32;

template <int NT, int STAGES>
struct Cfg {
  static constexpr int X_BOX_BYTES =
      NT * 8 * 128;  // one [8*NT rows x 64 k] box
  static constexpr int W_BYTES = NBOX * W_BOX_BYTES;
  static constexpr int STAGE_BYTES = W_BYTES + NBOX * X_BOX_BYTES;
  static constexpr int SMEM = STAGES * STAGE_BYTES + 1024;  // + alignment slack
};

__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count)
               : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
      "r"(bytes)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar)
               : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_try(bar, parity)) {
  }
}
// Wait with cluster-scope acquire: the phase is completed by peers' st.async
// complete_tx (release.cluster).
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar,
                                                  uint32_t parity) {
  uint32_t ok = 0;
  while (!ok) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], "
        "%2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(ok)
        : "r"(bar), "r"(parity)
        : "memory");
  }
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst,
                                            const CUtensorMap* map, int c0,
                                            int c1, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
      "l"(map), "r"(c0), "r"(c1), "r"(bar)
      : "memory");
}
__device__ __forceinline__ uint4 lds128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(addr));
  return v;
}
__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1,
                                         uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t dsmem_addr(uint32_t local, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(r)
               : "r"(local), "r"(rank));
  return r;
}
// 16-byte remote store into a cluster peer's smem, completing 16 bytes on the
// peer's mbarrier.
__device__ __forceinline__ void st_async_v4(uint32_t raddr, float4 v,
                                            uint32_t rbar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], "
      "{%1,%2,%3,%4}, [%5];" ::"r"(raddr),
      "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar)
      : "memory");
}
__device__ __forceinline__ void st_async_v2(uint32_t raddr, float2 v,
                                            uint32_t rbar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], "
      "{%1,%2}, [%3];" ::"r"(raddr),
      "f"(v.x), "f"(v.y), "r"(rbar)
      : "memory");
}
__device__ __forceinline__ void prefetch_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}
// L2 prefetch of one box of `map` at (c0, c1) (cp.async.bulk.prefetch.tensor,
// sm_90+); nothing to wait for.
__device__ __forceinline__ void prefetch_tile_l2(const CUtensorMap* map, int c0,
                                                 int c1) {
  asm volatile(
      "cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" ::"l"(
          map),
      "r"(c0), "r"(c1)
      : "memory");
}
// griddepcontrol.wait (sm_90+): blocks until the PDL predecessor has completed
// and its writes are visible; a no-op without a programmatic dependency.
__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

// Byte offset of 16-byte chunk `c` (0..7) of row `r` inside a 128B-swizzled
// box.
__device__ __forceinline__ uint32_t swz(int r, int c) {
  return (uint32_t)(r * 128 + ((c ^ (r & 7)) << 4));
}

// First k of the i-th k-tile that cluster rank `rank` reduces (S-way split in
// KSTRIPE-tile cyclic stripes).
template <int S>
__device__ __forceinline__ int k_tile(int i, int rank) {
  return ((i / KSTRIPE) * (S * KSTRIPE) + rank * KSTRIPE + i % KSTRIPE) * KT;
}

// kv_cache row of `slot` for kv head `kvh` (64-bit index math: cache offsets
// exceed 2^31).
__device__ __forceinline__ __nv_bfloat16* cache_row(__nv_bfloat16* cache,
                                                    long long slot, int HKV,
                                                    int kvh) {
  return cache +
         (((slot >> 4) * HKV + kvh) * 16 + (slot & 15)) * (int64_t)(2 * D);
}

// Epilogue for one (token, head): sum the S split-K partials in rank order,
// round to bf16, then norm / rope / stores.  cv/sn are this lane's cos/sin
// values (prefetched).
template <int S>
__device__ __forceinline__ void finish_token(const float* rp, int rstride,
                                             float cv, float sn, long long slot,
                                             int rope, int head, int HQ,
                                             int HKV, float eps,
                                             __nv_bfloat16* yrow,
                                             __nv_bfloat16* cache) {
  const int lane = threadIdx.x & 31;
  const int grpB = HQ, grpC = HQ + HKV, grpD = HQ + 2 * HKV;
  float2 p[S];
#pragma unroll
  for (int s = 0; s < S; ++s)
    p[s] = reinterpret_cast<const float2*>(rp + s * rstride)[lane];
  float vx = 0.f, vy = 0.f;
#pragma unroll
  for (int s = 0; s < S; ++s) {  // fixed rank order: bitwise deterministic
    vx += p[s].x;
    vy += p[s].y;
  }
  const __nv_bfloat162 yb = __floats2bfloat162_rn(vx, vy);
  __nv_bfloat162 outv = yb;
  if (head < grpC) {
    const float h0 = __low2float(yb), h1 = __high2float(yb);
    float ss = h0 * h0 + h1 * h1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float rn = rsqrtf(ss * (1.0f / 64.0f) + eps);
    float o0 = h0 * rn, o1 = h1 * rn;
    if (rope) {
      const float r0 = o0 * cv - o1 * sn;
      const float r1 = o1 * cv + o0 * sn;
      o0 = r0;
      o1 = r1;
    }
    outv = __floats2bfloat162_rn(o0, o1);
    if (head >= grpB && slot >= 0)
      reinterpret_cast<__nv_bfloat162*>(
          cache_row(cache, slot, HKV, head - grpB))[lane] = outv;
  } else if (head < grpD && slot >= 0) {
    reinterpret_cast<__nv_bfloat162*>(cache_row(cache, slot, HKV, head - grpC) +
                                      D)[lane] = yb;
  }
  reinterpret_cast<__nv_bfloat162*>(yrow)[lane] = outv;
}

// NT = number of 8-token mma column tiles (1 for M <= 8, 2 for M <= 16); STAGES
// = ring depth; S = K split = cluster size (power of two <= 8). FIXED_M = 8
// specializes M == 8, FIXED_M = 0 serves every other M. Cluster dims (S,1,1):
// blockIdx.x / S = head, cluster rank = K split.
template <int NT, int STAGES, int S, int FIXED_M>
__global__ void __launch_bounds__(NTHREADS, 2)
    qkv_fused_kernel(const __grid_constant__ CUtensorMap tm_w,
                     const __grid_constant__ CUtensorMap tm_x,
                     const int64_t* __restrict__ positions,
                     const float* __restrict__ cos_sin,
                     const int64_t* __restrict__ slots,
                     __nv_bfloat16* __restrict__ cache,
                     __nv_bfloat16* __restrict__ y, int M, int N, int HQ,
                     int HKV, int rope, float eps) {
  using C = Cfg<NT, STAGES>;
  constexpr int MAX_OWNED = (8 * NT + S - 1) / S;
  constexpr int MAX_RECV = S * MAX_OWNED;
  extern __shared__ unsigned char smem_raw[];
  __shared__ __align__(8) uint64_t full_bar[STAGES];
  __shared__ __align__(8) uint64_t empty_bar[STAGES];
  __shared__ __align__(8) uint64_t recv_bar;
  __shared__ __align__(16) float
      s_cs[MAX_OWNED][D];  // cos|sin row of each token this CTA finishes
  __shared__ __align__(16) float
      s_recv[MAX_RECV][D];  // [src rank][local token] split-K partials
  __shared__ long long s_slot[MAX_OWNED];

  // The first TMA needs only special registers and compile-time constants, so
  // the ring starts filling before any other kernel parameter is read.
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rank = (int)cluster_rank(), head = (int)(blockIdx.x / S);
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase =
      (sraw + 1023u) & ~1023u;  // 128B swizzle needs 1 KB aligned boxes
  const uint32_t fb = (uint32_t)__cvta_generic_to_shared(full_bar);
  const uint32_t eb = (uint32_t)__cvta_generic_to_shared(empty_bar);
  const uint32_t rb = (uint32_t)__cvta_generic_to_shared(&recv_bar);
  constexpr int nk = KTILES / S;  // S is a power of two <= 8: equal K splits
  const int row0 = head * D;
  constexpr int npre = nk < STAGES ? nk : STAGES;

  if (warp == PRODW && lane == 0) {
    // Stage 0's weights go first; the rest of the ring is initialized while
    // they are in flight. The full barrier counts bytes, so these W boxes and
    // the x boxes issued after griddepcontrol.wait complete one phase.
    mbar_init(fb, 1);
    mbar_init(eb, NCW);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(fb, (uint32_t)C::STAGE_BYTES);
#pragma unroll
    for (int j = 0; j < NBOX; ++j)
      tma_load_2d(sbase + j * W_BOX_BYTES, &tm_w, k_tile<S>(0, rank) + j * BOXK,
                  row0, fb);
#pragma unroll
    for (int s = 1; s < STAGES; ++s) {
      mbar_init(fb + 8 * s, 1);
      mbar_init(eb + 8 * s, NCW);
    }
    mbar_init(rb, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  const int active_m = FIXED_M == 0 ? M : FIXED_M;
  const int ntok_max = FIXED_M == 0 ? (M + S - 1) / S : FIXED_M / S;
  const int ntok = FIXED_M == 0 ? (M - rank + S - 1) / S : FIXED_M / S;
  if (warp == PRODW && lane == 0) {
    mbar_expect_tx(
        rb, (uint32_t)(S * ntok * D *
                       4));  // all S partials of my tokens (incl. my own)
  }
  __syncthreads();
  cluster_arrive_relaxed();  // receive barriers initialised (producer fenced
                             // them); peers wait before pushing

  float acc[4][NT][4];  // [k-block set][token tile][fragment]
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < NT; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;

  const int grpB = HQ, grpC = HQ + HKV, grpD = HQ + 2 * HKV;
  if (warp == AUXW) {
    // ---- aux warp: epilogue inputs, and L2 prefetch of the destinations
    // slots and positions can be written by the kernels before this one in a
    // graph: read them after the wait.
    pdl_wait();
    // local token u <-> global token rank + u*S
    long long sl = -1, pos = 0;
    if (lane < ntok) {
      sl = slots[rank + lane * S];
      s_slot[lane] = sl;
      prefetch_l2(y + (size_t)(rank + lane * S) * N + row0);
      if (sl >= 0 && head >= grpB && head < grpD)
        prefetch_l2(
            cache_row(cache, sl, HKV, head < grpC ? head - grpB : head - grpC) +
            (head < grpC ? 0 : D));
    }
    if (rope && head < grpC) {
      if (lane < ntok) pos = positions[rank + lane * S];
      for (int u = 0; u < ntok; ++u) {
        const long long p = __shfl_sync(0xffffffffu, pos, u);
        const float* cs = cos_sin + p * (int64_t)D;
        s_cs[u][lane] = __ldg(cs + lane);
        s_cs[u][32 + lane] = __ldg(cs + 32 + lane);
      }
    }

  } else if (warp == PRODW) {
    // ---- producer (one thread): the weights of stages 1..npre-1, then the
    // steady state. Weights only before griddepcontrol.wait, activations after.
    if (lane == 0) {
#pragma unroll
      for (int i = 1; i < npre; ++i) {
        const uint32_t sw = sbase + i * C::STAGE_BYTES;
        const uint32_t bar = fb + 8 * i;
        const int k0 = k_tile<S>(i, rank);
        mbar_expect_tx(bar, (uint32_t)C::STAGE_BYTES);
#pragma unroll
        for (int j = 0; j < NBOX; ++j)
          tma_load_2d(sw + j * W_BOX_BYTES, &tm_w, k0 + j * BOXK, row0, bar);
      }
      // The next ring's weights into L2.
      for (int i = npre; i < nk && i < npre + STAGES; ++i) {
        const int k0 = k_tile<S>(i, rank);
#pragma unroll
        for (int j = 0; j < NBOX; ++j)
          prefetch_tile_l2(&tm_w, k0 + j * BOXK, row0);
      }
    }
    pdl_wait();
    if (lane == 0) {
#pragma unroll
      for (int i = 0; i < npre; ++i) {
        const uint32_t sw = sbase + i * C::STAGE_BYTES;
        const int k0 = k_tile<S>(i, rank);
#pragma unroll
        for (int j = 0; j < NBOX; ++j)
          tma_load_2d(sw + C::W_BYTES + j * C::X_BOX_BYTES, &tm_x,
                      k0 + j * BOXK, 0, fb + 8 * i);
      }
      for (int i = npre; i < nk; ++i) {
        const int st = i % STAGES;
        const uint32_t ph = (uint32_t)(i / STAGES) & 1u;
        mbar_wait(eb + 8 * st, ph ^ 1u);
        const uint32_t sw = sbase + st * C::STAGE_BYTES;
        const uint32_t bar = fb + 8 * st;
        const int k0 = k_tile<S>(i, rank);
        mbar_expect_tx(bar, (uint32_t)C::STAGE_BYTES);
#pragma unroll
        for (int j = 0; j < NBOX; ++j)
          tma_load_2d(sw + j * W_BOX_BYTES, &tm_w, k0 + j * BOXK, row0, bar);
#pragma unroll
        for (int j = 0; j < NBOX; ++j)
          tma_load_2d(sw + C::W_BYTES + j * C::X_BOX_BYTES, &tm_x,
                      k0 + j * BOXK, 0, bar);
      }
    }
  } else {
    // ---- consumer warps: warp w owns W rows 16w..16w+15 over all 128 k of
    // every stage, so it ends with the CTA partial of its rows. The 32-k block
    // b lives in box b/2 at 16B chunks c = 2t + (b&1); A and B use the same k
    // permutation (any order is a valid fp32 reduction). One accumulator set
    // per k-block, summed in fixed order at the end.
    const int g = lane >> 2, t = lane & 3;
    uint32_t aoff[4][2], xoff[4][NT];
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const int j = b >> 1, c = 2 * t + (b & 1);
      aoff[b][0] = j * W_BOX_BYTES + swz(16 * warp + g, c);
      aoff[b][1] = j * W_BOX_BYTES + swz(16 * warp + g + 8, c);
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
        xoff[b][nt] = C::W_BYTES + j * C::X_BOX_BYTES + swz(8 * nt + g, c);
    }
    // The epilogue stores of this warp come after the wait; the ring itself is
    // gated by full_bar.
    pdl_wait();
    for (int i = 0; i < nk; ++i) {
      const int st = i % STAGES;
      const uint32_t ph = (uint32_t)(i / STAGES) & 1u;
      mbar_wait(fb + 8 * st, ph);
      const uint32_t sw = sbase + st * C::STAGE_BYTES;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        uint4 bx[NT];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
          bx[nt] = lds128(sw + xoff[b][nt]);  // rows >= M are zero-filled
        const uint4 a0 = lds128(sw + aoff[b][0]);
        const uint4 a1 = lds128(sw + aoff[b][1]);
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          mma_bf16(acc[b][nt], a0.x, a1.x, a0.y, a1.y, bx[nt].x, bx[nt].y);
          mma_bf16(acc[b][nt], a0.z, a1.z, a0.w, a1.w, bx[nt].z, bx[nt].w);
        }
      }
      if (lane == 0) mbar_arrive(eb + 8 * st);
    }
    // CTA partial of rows 16w+g (+8) for tokens 8nt+2t (+1): fixed-order sum of
    // the k-block sets.
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e)
        acc[0][nt][e] =
            ((acc[0][nt][e] + acc[1][nt][e]) + acc[2][nt][e]) + acc[3][nt][e];
    cluster_wait();  // every peer's receive barrier is initialised
    // Pair rows (r, r+1) of one token per lane with one xor-4 shuffle (lanes g
    // and g^1 share t), then push 8 bytes straight from registers into the
    // owner rank's receive slot [rank][token / S][row].
    const uint32_t recv_local =
        (uint32_t)__cvta_generic_to_shared(&s_recv[0][0]);
    const bool odd = (g & 1) != 0;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float a = acc[0][nt][2 * h],
                    bb = acc[0][nt][2 * h + 1];  // tokens 2t, 2t+1 of row r
        const float other = __shfl_xor_sync(0xffffffffu, odd ? a : bb, 4);
        const int r = 16 * warp + 8 * h + (g & ~1);  // even row of the pair
        const int tk = 8 * nt + 2 * t + (odd ? 1 : 0);
        const float2 v = odd ? make_float2(other, bb) : make_float2(a, other);
        if (tk < active_m) {
          const uint32_t owner = (uint32_t)(tk % S);
          const uint32_t off =
              (uint32_t)(((rank * ntok_max + tk / S) * D + r) * 4);
          st_async_v2(dsmem_addr(recv_local + off, owner), v,
                      dsmem_addr(rb, owner));
        }
      }
  }
  if (warp >= NCW)
    cluster_wait();  // producer / aux: matching wait of the start-of-kernel
                     // arrive
  __syncthreads();   // aux-warp epilogue inputs (s_slot, s_cs) visible to every
                     // warp
  // Epilogue inputs of this warp's first token, loaded before the receive wait.
  float cv0 = 0.f, sn0 = 0.f;
  long long slot0 = -1;
  if (warp < ntok) {
    cv0 = s_cs[warp][lane];
    sn0 = s_cs[warp][32 + lane];
    slot0 = s_slot[warp];
  }
  mbar_wait_cluster(rb, 0u);  // all S partials of my tokens have landed
  cluster_arrive_relaxed();   // my receive is complete; the wait before exit
                              // keeps every CTA alive until all peers have
                              // received (so no push is in flight to/from an
                              // exited CTA)

  // ---- epilogue: rank r owns tokens r, r+S, ... (local index u)
  for (int u = warp; u < ntok; u += NWARPS) {
    const int tk = rank + u * S;
    const bool first = u == warp;
    finish_token<S>(&s_recv[u][0], ntok_max * D, first ? cv0 : s_cs[u][lane],
                    first ? sn0 : s_cs[u][32 + lane], first ? slot0 : s_slot[u],
                    rope, head, HQ, HKV, eps, y + (size_t)tk * N + row0, cache);
  }
  cluster_wait();
}

constexpr int kStages = 5;  // a five-stage ring keeps two CTAs resident per SM
// The ring at S = 4 with one CTA per SM. The ring depth does not change the
// summation order.
constexpr int kStagesDeep = 8;
static_assert(
    Cfg<2, kStagesDeep>::SMEM + 8 * 1024 <= 227 * 1024,
    "the deep ring and the static buffers fit one CTA's shared memory");
constexpr int kMaxTokens = 16;
// SM count of the K-split rule. Fixed, so the reduction order (and the result)
// does not depend on the device.
constexpr int kSplitSms = 152;

// K split S (the cluster size) for N output columns. One CTA per SM when a
// split of at least 4 allows it: the largest power of two <= 8 with
// (N / 64) * S <= kSplitSms CTAs. Otherwise two CTAs per SM: the largest power
// of two <= 8 with (N / 64) * S <= 2 * kSplitSms. 1 means no instantiation
// fits.
// NUMERICS: S sets the summation order of the fp32 partials. Against the
// unfused GEMM + norm/rope the relative L2 error of y is at most 2.83e-4, and
// 1.89e-3 against an fp64 reference (measured at M 1, 5, 8, 13 and 16).
inline int k_split(int n) {
  const int heads = n / D;
  int one = 1;
  while (one < 8 && heads * one * 2 <= kSplitSms) one *= 2;
  if (one >= 4) return one;
  int s = 1;
  while (s < 8 && heads * s * 2 <= 2 * kSplitSms) s *= 2;
  return s;
}

// Ring depth for N output columns at split s (see kStagesDeep).
inline int ring_stages(int n, int s) {
  return (n / D) * s <= kSplitSms && s == 4 ? kStagesDeep : kStages;
}

inline void check_cuda(cudaError_t err, const char* what) {
  STD_TORCH_CHECK(err == cudaSuccess, what,
                  " failed: ", cudaGetErrorString(err));
}

using EncodeTiledFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType,
                                   cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*,
                                   const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion,
                                   CUtensorMapFloatOOBfill);

// The driver entry point, resolved through the runtime so the extension does
// not link libcuda.
EncodeTiledFn encode_fn() {
  static const EncodeTiledFn fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    check_cuda(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p,
                                                12000, cudaEnableDefault, &q),
               "cudaGetDriverEntryPointByVersion(cuTensorMapEncodeTiled)");
    STD_TORCH_CHECK(q == cudaDriverEntryPointSuccess && p != nullptr,
                    "cuTensorMapEncodeTiled is not available from the driver");
    return reinterpret_cast<EncodeTiledFn>(p);
  }();
  return fn;
}

// 2D bf16 map over a row-major [rows, 8192] matrix: box 64 x box_rows, 128B
// swizzle, L2 promotion 256B, zero fill outside the matrix.
CUtensorMap make_map(const void* base, int rows, int box_rows) {
  CUtensorMap map;
  cuuint64_t dims[2] = {(cuuint64_t)KDIM, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)KDIM * 2};
  cuuint32_t box[2] = {(cuuint32_t)BOXK, (cuuint32_t)box_rows};
  cuuint32_t estr[2] = {1, 1};
  const CUresult res = encode_fn()(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims,
      strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  STD_TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed (",
                  static_cast<int>(res), ") for rows=", rows,
                  " box_rows=", box_rows);
  return map;
}

// Weight maps are encoded once per (device, address, rows): weights do not move
// after loading.
CUtensorMap weight_map(int device, const void* w, int rows) {
  static std::mutex mu;
  static std::map<std::tuple<int, uintptr_t, int>, CUtensorMap> maps;
  const auto key =
      std::make_tuple(device, reinterpret_cast<uintptr_t>(w), rows);
  std::lock_guard<std::mutex> lock(mu);
  auto it = maps.find(key);
  if (it == maps.end()) it = maps.emplace(key, make_map(w, rows, 64)).first;
  return it->second;
}

template <int NT, int STAGES, int S, int FIXED_M>
void launch_cfg(int device, const CUtensorMap& tm_w, const void* x,
                const int64_t* pos, const float* cs, const int64_t* slots,
                void* cache, void* y, int M, int N, int HQ, int HKV, int rope,
                float eps, cudaStream_t stream) {
  using C = Cfg<NT, STAGES>;
  auto kern = qkv_fused_kernel<NT, STAGES, S, FIXED_M>;
  {
    static std::mutex mu;
    static std::unordered_set<int> configured;
    std::lock_guard<std::mutex> lock(mu);
    if (configured.count(device) == 0) {
      check_cuda(
          cudaFuncSetAttribute(
              kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM),
          "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");
      configured.insert(device);
    }
  }
  const CUtensorMap tm_x = make_map(x, M, 8 * NT);

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)((N / D) * S), 1, 1);
  cfg.blockDim = dim3(NTHREADS, 1, 1);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = stream;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = (unsigned)S;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  // PDL: the kernel waits (griddepcontrol.wait) before it touches anything
  // but the weights; the SM100 family this file is built for supports it.
  attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 2;
  check_cuda(cudaLaunchKernelEx(&cfg, kern, tm_w, tm_x, pos, cs, slots,
                                reinterpret_cast<__nv_bfloat16*>(cache),
                                reinterpret_cast<__nv_bfloat16*>(y), M, N, HQ,
                                HKV, rope, eps),
             "fused_muse_spark_qkv_qknorm_rope_kv_insert launch");
}

template <int NT, int FIXED_M>
void launch_nt(int S, int stages, int device, const CUtensorMap& tm_w,
               const void* x, const int64_t* pos, const float* cs,
               const int64_t* slots, void* cache, void* y, int M, int N, int HQ,
               int HKV, int rope, float eps, cudaStream_t stream) {
  if (S == 2 && stages == kStages) {
    launch_cfg<NT, kStages, 2, FIXED_M>(device, tm_w, x, pos, cs, slots, cache,
                                        y, M, N, HQ, HKV, rope, eps, stream);
  } else if (S == 4 && stages == kStages) {
    launch_cfg<NT, kStages, 4, FIXED_M>(device, tm_w, x, pos, cs, slots, cache,
                                        y, M, N, HQ, HKV, rope, eps, stream);
  } else if (S == 4 && stages == kStagesDeep) {
    launch_cfg<NT, kStagesDeep, 4, FIXED_M>(device, tm_w, x, pos, cs, slots,
                                            cache, y, M, N, HQ, HKV, rope, eps,
                                            stream);
  } else if (S == 8 && stages == kStages) {
    launch_cfg<NT, kStages, 8, FIXED_M>(device, tm_w, x, pos, cs, slots, cache,
                                        y, M, N, HQ, HKV, rope, eps, stream);
  } else {
    STD_TORCH_CHECK(false, "no kernel for N = ", N, " (split ", S, ", ", stages,
                    " stages)");
  }
}

bool aligned16(const torch::stable::Tensor& t) {
  return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}

bool on_device(const torch::stable::Tensor& t, int device) {
  return t.is_cuda() && t.get_device_index() == device;
}

}  // namespace muse_spark_qkv
}  // namespace vllm

// out [M, N] = x [M, 8192] @ weight [N, 8192]^T with the q and k heads
// RMS-normalized (and rotated when cos_sin_cache is given), and the normalized
// k and raw v heads written to the kv_cache rows of slot_mapping (-1: none).
// N = (2 * HQ + 2 * HKV) * 64 with HKV = kv_cache.size(1).
void fused_muse_spark_qkv_qknorm_rope_kv_insert(
    torch::stable::Tensor const& x, torch::stable::Tensor const& weight,
    torch::stable::Tensor const& positions,
    std::optional<torch::stable::Tensor> cos_sin_cache,
    torch::stable::Tensor const& slot_mapping, torch::stable::Tensor& kv_cache,
    torch::stable::Tensor& out, double eps) {
  using namespace vllm::muse_spark_qkv;
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  const int dev = x.get_device_index();
  STD_TORCH_CHECK(x.dim() == 2 && x.scalar_type() == ScalarType::BFloat16 &&
                      x.size(1) == KDIM && x.is_contiguous() && aligned16(x),
                  "x must be a contiguous, 16-byte aligned bf16 [M, ", KDIM,
                  "] tensor");
  const int M = static_cast<int>(x.size(0));
  STD_TORCH_CHECK(M >= 1 && M <= kMaxTokens, "M = ", M, " is outside 1..",
                  kMaxTokens);
  STD_TORCH_CHECK(
      weight.dim() == 2 && weight.scalar_type() == ScalarType::BFloat16 &&
          weight.size(1) == KDIM && weight.is_contiguous() && aligned16(weight),
      "weight must be a contiguous, 16-byte aligned bf16 [N, ", KDIM,
      "] tensor");
  const int N = static_cast<int>(weight.size(0));
  STD_TORCH_CHECK(kv_cache.dim() == 4, "kv_cache must be 4-D");
  const int HKV = static_cast<int>(kv_cache.size(1));
  const int S = k_split(N);
  STD_TORCH_CHECK(N % (2 * D) == 0 && HKV > 0 && HKV < N / (2 * D) &&
                      (S == 2 || S == 4 || S == 8),
                  "no kernel for N = ", N, " with ", HKV, " KV heads");
  const int HQ = N / (2 * D) - HKV;
  STD_TORCH_CHECK(positions.scalar_type() == ScalarType::Long &&
                      positions.dim() == 1 && positions.size(0) >= M &&
                      positions.stride(0) == 1,
                  "positions must be a dense int64 vector of at least M "
                  "entries");
  STD_TORCH_CHECK(slot_mapping.scalar_type() == ScalarType::Long &&
                      slot_mapping.dim() == 1 && slot_mapping.size(0) >= M &&
                      slot_mapping.stride(0) == 1,
                  "slot_mapping must be a dense int64 vector of at least M "
                  "entries");
  const bool rope = cos_sin_cache.has_value();
  if (rope) {
    STD_TORCH_CHECK(
        cos_sin_cache->scalar_type() == ScalarType::Float &&
            cos_sin_cache->dim() == 2 && cos_sin_cache->size(1) == D &&
            cos_sin_cache->is_contiguous() && on_device(*cos_sin_cache, dev),
        "cos_sin_cache must be a contiguous fp32 [P, ", D,
        "] tensor on x's device");
  }
  STD_TORCH_CHECK(kv_cache.scalar_type() == ScalarType::BFloat16 &&
                      kv_cache.size(2) == 16 && kv_cache.size(3) == 2 * D,
                  "kv_cache must be a bf16 [pages, HKV, 16, ", 2 * D,
                  "] tensor");
  STD_TORCH_CHECK(kv_cache.stride(0) == (int64_t)HKV * 16 * 2 * D &&
                      (HKV == 1 || kv_cache.stride(1) == 16 * 2 * D) &&
                      kv_cache.stride(2) == 2 * D && kv_cache.stride(3) == 1 &&
                      aligned16(kv_cache),
                  "kv_cache rows must be at (page * HKV + head) * 16 + offset");
  STD_TORCH_CHECK(out.scalar_type() == ScalarType::BFloat16 && out.dim() == 2 &&
                      out.size(0) == M && out.size(1) == N &&
                      out.is_contiguous(),
                  "out must be a contiguous bf16 [M, N] tensor");
  STD_TORCH_CHECK(on_device(weight, dev) && on_device(positions, dev) &&
                      on_device(slot_mapping, dev) &&
                      on_device(kv_cache, dev) && on_device(out, dev),
                  "all tensors must be on x's device");

  const torch::stable::accelerator::DeviceGuard device_guard(dev);
  const cudaStream_t stream = get_current_cuda_stream(dev);
  const CUtensorMap tm_w = weight_map(dev, weight.data_ptr(), N);
  const auto* pos = static_cast<const int64_t*>(positions.data_ptr());
  const float* cs =
      rope ? static_cast<const float*>(cos_sin_cache->data_ptr()) : nullptr;
  const auto* slots = static_cast<const int64_t*>(slot_mapping.data_ptr());
  void* cache = kv_cache.mutable_data_ptr();
  void* y = out.mutable_data_ptr();
  const float e = static_cast<float>(eps);
  const int stages = ring_stages(N, S);
  if (M == 8) {
    launch_nt<1, 8>(S, stages, dev, tm_w, x.data_ptr(), pos, cs, slots, cache,
                    y, M, N, HQ, HKV, rope, e, stream);
  } else if (M < 8) {
    launch_nt<1, 0>(S, stages, dev, tm_w, x.data_ptr(), pos, cs, slots, cache,
                    y, M, N, HQ, HKV, rope, e, stream);
  } else {
    launch_nt<2, 0>(S, stages, dev, tm_w, x.data_ptr(), pos, cs, slots, cache,
                    y, M, N, HQ, HKV, rope, e, stream);
  }
}
