// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Muse Spark paged decode attention for the 8-token verify step, fused with
// the per-head output RMS norm (no weight) and the sigmoid gate: head_dim 64,
// 8 query heads per KV head, bf16 paged cache [pages, HKV, 16, 128] (K in
// columns 0:64, V in 64:128), 8 query rows per request (the last 8 cache
// entries of the request), any context length L >= 8, optionally with a
// sliding window. One launch covers kv_heads_per_rank KV heads (blockIdx.y);
// the heads are independent.
//
// Two instantiations, picked on the host from the deployment's maximum length:
//   * One-cluster kernel (max_seq_len <= 2048, no window): a cluster of CL
//     CTAs owns 16 query-head rows of one request and one KV head. Each warp
//     attends a share of the pages with an online softmax (mma.sync bf16 ->
//     fp32); the CTAs merge their partials over distributed shared memory
//     (st.async + mbarrier) into the CTA that owns each row, which applies the
//     norm and gate. No global scratch, no atomics.
//   * Split kernel (longer contexts or a sliding window): short requests take
//     the path above; long ones are spread over num_splits z slices of
//     clusters, whose partials merge through a global workspace in cluster
//     order (independent of arrival order). The launch shape depends only on
//     the deployment's maximum length, so a captured graph serves any step.
//     With window W, row t sees keys [L - 8 + t - W + 1, L - 8 + t].
//
// NUMERICS: not bitwise to the stock trtllm-gen decode + rmsnorm_sigmoid_gate
// pair: the softmax weights are rounded to bf16 before P.V, partials merge in
// fp32 and the file is built with --use_fast_math. Against an fp64 reference
// its relative L2 error is at most 2.75e-3 and never above the stock pair's
// (tested up to L = 1048576 and with windows).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include <mutex>
#include <optional>
#include <unordered_set>

#include <torch/headeronly/core/ScalarType.h>

#include "torch_utils.h"

// The device code uses thread-block clusters, st.async with mbarrier
// transaction counts and griddepcontrol: SM100 family only (CMake 10.0f).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
  #error "muse_spark_attn2 needs the SM100 family"
#endif

namespace vllm {
namespace muse_spark_attn2 {

constexpr int kPage = 16;
constexpr int kRows = 16;  // query-head rows per cluster
constexpr int kSlot = 68;  // floats per exchanged partial row: 64 O + m + l + 2
                           // pad (16B multiple)
constexpr int kPartialBytes = 64 * 4 + 8;
#define kNegInf __int_as_float(0xff800000)

struct Params {
  const __nv_bfloat16* q;  // KV head 0's query heads; row stride q_stride
  long long q_stride;
  const __nv_bfloat16* gate;  // KV head 0's gate heads; row stride gate_stride
  long long gate_stride;
  const __nv_bfloat16* kv;  // KV head 0 of page 0; pages at kv_page_stride
  long long kv_page_stride;
  const int* bt;
  long long bt_stride;
  int max_pages;
  const int* seq_lens;
  __nv_bfloat16* out;  // KV head 0's output heads; row stride out_stride
  long long out_stride;
  float scale_log2;
  float eps;
  // kSplit only: keys visible to a row including its own (0 = all), and the
  // long-context workspace: per row (request, KV head, query-head row) one
  // counter (zero between launches) and a partial of kSlot floats per cluster
  // of the (request, KV head).
  int window;
  unsigned int* counters;
  float* partials;
};

__device__ __forceinline__ uint4 ldg16(const __nv_bfloat16* p) {
  return __ldg(reinterpret_cast<const uint4*>(p));
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4],
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t movT(uint32_t x) {
  uint32_t y;
  asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(y) : "r"(x));
  return y;
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(r)
               : "r"(addr), "r"(rank));
  return r;
}

__device__ __forceinline__ void st_async_v4(uint32_t addr, float a, float b,
                                            float c, float d, uint32_t mbar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], "
      "{%1,%2,%3,%4}, [%5];" ::"r"(addr),
      "f"(a), "f"(b), "f"(c), "f"(d), "r"(mbar)
      : "memory");
}

__device__ __forceinline__ void st_async_v2(uint32_t addr, float a, float b,
                                            uint32_t mbar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], "
      "{%1,%2}, [%3];" ::"r"(addr),
      "f"(a), "f"(b), "r"(mbar)
      : "memory");
}

__device__ __forceinline__ void words8(const uint4& a, const uint4& b,
                                       uint32_t (&w)[8]) {
  w[0] = a.x;
  w[1] = a.y;
  w[2] = a.z;
  w[3] = a.w;
  w[4] = b.x;
  w[5] = b.y;
  w[6] = b.z;
  w[7] = b.w;
}

// One KV page as held by a lane (g = lane / 4, t = lane % 4):
//   k[j][0..1]: key   slot 8j+g, head dims [16t, 16t+16)
//   v[h][0..1]: value slot 8h+g, head dims [16t, 16t+16)
struct PageRegs {
  uint4 k[2][2];
  uint4 v[2][2];
};

__device__ __forceinline__ void load_page(PageRegs& r, const __nv_bfloat16* kv,
                                          long long page_stride, int entry,
                                          int g, int t) {
  const __nv_bfloat16* base =
      kv + static_cast<long long>(entry) * page_stride + 16 * t;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const __nv_bfloat16* kp = base + (8 * j + g) * 128;
    r.k[j][0] = ldg16(kp);
    r.k[j][1] = ldg16(kp + 8);
    r.v[j][0] = ldg16(kp + 64);
    r.v[j][1] = ldg16(kp + 72);
  }
}

// One page (16 keys) of online-softmax attention for the warp's rows g and
// g + 8, with the causal mask (and with kLow the window). pos is the position
// of the lane's first key; a row sees keys low <= key <= lim.
template <bool kLow>
__device__ __forceinline__ void attend_page(const PageRegs& cur,
                                            const uint32_t (&qf)[4][4], int pos,
                                            int lim0, int lim1, int low0,
                                            int low1, float sl2, float& m0,
                                            float& m1, float& l0, float& l1,
                                            float (&o)[8][4]) {
  float s[2][4];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    uint32_t kw[8];
    words8(cur.k[j][0], cur.k[j][1], kw);
    float sa[4] = {0.f, 0.f, 0.f, 0.f}, sb[4] = {0.f, 0.f, 0.f, 0.f};
    mma16816(sa, qf[0], kw[0], kw[1]);
    mma16816(sb, qf[2], kw[4], kw[5]);
    mma16816(sa, qf[1], kw[2], kw[3]);
    mma16816(sb, qf[3], kw[6], kw[7]);
#pragma unroll
    for (int e = 0; e < 4; ++e) s[j][e] = sa[e] + sb[e];
  }

  float mx0 = kNegInf, mx1 = kNegInf;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int q = pos + 8 * j + e;
      if constexpr (kLow) {
        s[j][e] = (q <= lim0 && q >= low0) ? s[j][e] * sl2 : kNegInf;
        s[j][2 + e] = (q <= lim1 && q >= low1) ? s[j][2 + e] * sl2 : kNegInf;
      } else {
        s[j][e] = (q <= lim0) ? s[j][e] * sl2 : kNegInf;
        s[j][2 + e] = (q <= lim1) ? s[j][2 + e] * sl2 : kNegInf;
      }
      mx0 = fmaxf(mx0, s[j][e]);
      mx1 = fmaxf(mx1, s[j][2 + e]);
    }
  }
  mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
  mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
  mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
  mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
  const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
  const float a0 = ex2(m0 - mn0), a1 = ex2(m1 - mn1);
  m0 = mn0;
  m1 = mn1;
  float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      s[j][e] = ex2(s[j][e] - mn0);
      s[j][2 + e] = ex2(s[j][2 + e] - mn1);
      ps0 += s[j][e];
      ps1 += s[j][2 + e];
    }
  }
  l0 = l0 * a0 + ps0;
  l1 = l1 * a1 + ps1;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    o[j][0] *= a0;
    o[j][1] *= a0;
    o[j][2] *= a1;
    o[j][3] *= a1;
  }
  uint32_t pa[4];
  pa[0] = pack_bf16(s[0][0], s[0][1]);
  pa[1] = pack_bf16(s[0][2], s[0][3]);
  pa[2] = pack_bf16(s[1][0], s[1][1]);
  pa[3] = pack_bf16(s[1][2], s[1][3]);

  // O += P V; V fragments transposed in registers (head dim permuted: logical
  // (j, 2t+e) = 16t + 2j + e)
  uint32_t vlo[8], vhi[8];
  words8(cur.v[0][0], cur.v[0][1], vlo);
  words8(cur.v[1][0], cur.v[1][1], vhi);
#pragma unroll
  for (int j = 0; j < 8; ++j) mma16816(o[j], pa, movT(vlo[j]), movT(vhi[j]));
}

// attend_page for up to NP pages staged in the shared-memory ring (slots
// slot0 + p, page indices page0 + p) in one online-softmax step; a row sees
// keys low <= key <= lim.
template <int NP, int ST>
__device__ __forceinline__ void attend_group(const __nv_bfloat16* ring,
                                             int slot0, int npages, int page0,
                                             const uint32_t (&qf)[4][4], int g,
                                             int t, int lim0, int lim1,
                                             int low0, int low1, float sl2,
                                             float& m0, float& m1, float& l0,
                                             float& l1, float (&o)[8][4]) {
  float s[NP][2][4];
#pragma unroll
  for (int pi = 0; pi < NP; ++pi) {
    if (pi < npages) {
      const __nv_bfloat16* page = ring + ((slot0 + pi) % ST) * (kPage * 128);
      const int pos = (page0 + pi) * kPage + 2 * t;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const uint4* kp =
            reinterpret_cast<const uint4*>(page + (8 * j + g) * 128 + 16 * t);
        uint32_t kw[8];
        words8(kp[0], kp[1], kw);
        float sa[4] = {0.f, 0.f, 0.f, 0.f}, sb[4] = {0.f, 0.f, 0.f, 0.f};
        mma16816(sa, qf[0], kw[0], kw[1]);
        mma16816(sb, qf[2], kw[4], kw[5]);
        mma16816(sa, qf[1], kw[2], kw[3]);
        mma16816(sb, qf[3], kw[6], kw[7]);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int q = pos + 8 * j + e;
          s[pi][j][e] =
              (q <= lim0 && q >= low0) ? (sa[e] + sb[e]) * sl2 : kNegInf;
          s[pi][j][2 + e] = (q <= lim1 && q >= low1)
                                ? (sa[2 + e] + sb[2 + e]) * sl2
                                : kNegInf;
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) s[pi][j][e] = kNegInf;
    }
  }

  float mx0 = kNegInf, mx1 = kNegInf;
#pragma unroll
  for (int pi = 0; pi < NP; ++pi)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        mx0 = fmaxf(mx0, s[pi][j][e]);
        mx1 = fmaxf(mx1, s[pi][j][2 + e]);
      }
  mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
  mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
  mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
  mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
  const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
  const float a0 = ex2(m0 - mn0), a1 = ex2(m1 - mn1);
  m0 = mn0;
  m1 = mn1;
  float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
  for (int pi = 0; pi < NP; ++pi)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        s[pi][j][e] = ex2(s[pi][j][e] - mn0);
        s[pi][j][2 + e] = ex2(s[pi][j][2 + e] - mn1);
        ps0 += s[pi][j][e];
        ps1 += s[pi][j][2 + e];
      }
  l0 = l0 * a0 + ps0;
  l1 = l1 * a1 + ps1;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    o[j][0] *= a0;
    o[j][1] *= a0;
    o[j][2] *= a1;
    o[j][3] *= a1;
  }
#pragma unroll
  for (int pi = 0; pi < NP; ++pi) {
    if (pi < npages) {
      const __nv_bfloat16* page = ring + ((slot0 + pi) % ST) * (kPage * 128);
      uint32_t pa[4];
      pa[0] = pack_bf16(s[pi][0][0], s[pi][0][1]);
      pa[1] = pack_bf16(s[pi][0][2], s[pi][0][3]);
      pa[2] = pack_bf16(s[pi][1][0], s[pi][1][1]);
      pa[3] = pack_bf16(s[pi][1][2], s[pi][1][3]);
      const uint4* v0 =
          reinterpret_cast<const uint4*>(page + g * 128 + 64 + 16 * t);
      const uint4* v1 =
          reinterpret_cast<const uint4*>(page + (8 + g) * 128 + 64 + 16 * t);
      uint32_t vlo[8], vhi[8];
      words8(v0[0], v0[1], vlo);
      words8(v1[0], v1[1], vhi);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        mma16816(o[j], pa, movT(vlo[j]), movT(vhi[j]));
    }
  }
}

// A warp's softmax state (lane holds dims [16t, 16t+16) of rows g and g+8)
// as rows of logical dims in s_o, with the row maxima and sums.
__device__ __forceinline__ void stash_partial(float (&so)[kRows][kSlot],
                                              float (&sm)[kRows],
                                              float (&sl)[kRows], int g, int t,
                                              const float (&o)[8][4], float m0,
                                              float m1, float l0, float l1) {
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  float4* r0 = reinterpret_cast<float4*>(&so[g][16 * t]);
  float4* r1 = reinterpret_cast<float4*>(&so[g + 8][16 * t]);
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    r0[q] =
        make_float4(o[2 * q][0], o[2 * q][1], o[2 * q + 1][0], o[2 * q + 1][1]);
    r1[q] =
        make_float4(o[2 * q][2], o[2 * q][3], o[2 * q + 1][2], o[2 * q + 1][3]);
  }
  if (t == 0) {
    sm[g] = m0;
    sl[g] = l0;
    sm[g + 8] = m1;
    sl[g + 8] = l1;
  }
}

// Spins until the mbarrier's phase 0 completes (cluster-scope acquire).
__device__ __forceinline__ void wait_phase0(uint32_t mbar) {
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{\n .reg .pred P1;\n"
        " mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%1], "
        "0;\n"
        " selp.b32 %0, 1, 0, P1;\n}\n"
        : "=r"(done)
        : "r"(mbar)
        : "memory");
  }
}

// cp.async.cg.shared.global (sm_80+): 16 bytes from global to this CTA's
// shared memory, bypassing L1.
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src)
               : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// The epilogue of one row (lane: dims 2 lane, 2 lane + 1): o / l rounded to
// bf16 (the op's first rounding point), RMS norm without weight, sigmoid gate,
// bf16 store.
__device__ __forceinline__ void finish_row(float ox, float oy, float Ls,
                                           __nv_bfloat162 gate2, float eps,
                                           __nv_bfloat16* out, int lane) {
  const float sgx = 1.f / (1.f + __expf(-__bfloat162float(gate2.x)));
  const float sgy = 1.f / (1.f + __expf(-__bfloat162float(gate2.y)));
  const float inv = 1.f / Ls;
  ox = __bfloat162float(__float2bfloat16_rn(ox * inv));
  oy = __bfloat162float(__float2bfloat16_rn(oy * inv));
  float ss = ox * ox + oy * oy;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    ss += __shfl_xor_sync(0xffffffffu, ss, off);
  const float rinv = rsqrtf(ss * (1.f / 64.f) + eps);
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  *reinterpret_cast<__nv_bfloat162*>(out + 2 * lane) =
      __floats2bfloat162_rn(ox * rinv * sgx, oy * rinv * sgy);
}

// Long contexts: the cluster's (o, m, l) of a row (lane: dims 2 lane,
// 2 lane + 1) is one of `active` partials of the row; publish it, and return
// true in the warp that completes the row's count with (Ls, ox, oy) the merge
// of every partial in cluster order (the same result whichever warp is last),
// after resetting the counter for the next launch.
__device__ __forceinline__ bool merge_partials(const Params& p,
                                               long long row_id, int slots,
                                               int split, int active, int lane,
                                               float M, float& Ls, float& ox,
                                               float& oy) {
  float* all = p.partials + row_id * slots * kSlot;
  float* mine = all + split * kSlot;
  reinterpret_cast<float2*>(mine)[lane] = make_float2(ox, oy);
  if (lane == 0) {
    mine[64] = M;
    mine[65] = Ls;
  }
  __threadfence();
  __syncwarp();
  unsigned int arrived = 0;
  if (lane == 0) arrived = atomicAdd(p.counters + row_id, 1u);
  arrived = __shfl_sync(0xffffffffu, arrived, 0);
  if (arrived != static_cast<unsigned int>(active - 1)) return false;
  __threadfence();
  float Mz = -1e30f;
  for (int z = 0; z < active; ++z) Mz = fmaxf(Mz, __ldcg(all + z * kSlot + 64));
  Ls = ox = oy = 0.f;
  for (int z = 0; z < active; ++z) {
    const float sc = ex2(__ldcg(all + z * kSlot + 64) - Mz);
    Ls += sc * __ldcg(all + z * kSlot + 65);
    const float2 v =
        __ldcg(reinterpret_cast<const float2*>(all + z * kSlot) + lane);
    ox += sc * v.x;
    oy += sc * v.y;
  }
  if (lane == 0) atomicExch(p.counters + row_id, 0u);
  return true;
}

// Pages of the long-context path per CTA at least (below, fewer clusters).
constexpr int kMinCtaPages = 4;
// Short-path limit of the split kernel in rounds of a cluster's warps (512
// pages, 8192 tokens); a 2048-token window (at most 130 pages) stays below it.
constexpr int kShortRounds = 8;
// The long-context path's online-softmax step (pages) and the steps its page
// ring holds (one in flight while one is attended).
constexpr int kGroupPages = 4;
constexpr int kRingGroups = 2;
constexpr int kRingStages = kGroupPages * kRingGroups;
constexpr int kPageElems = kPage * 128;
constexpr int kPageBytes = kPageElems * 2;

// Dynamic shared memory of the long-context path: the page ring, then the
// partials pushed to this CTA's rows.
template <int CL, int NW, int ST>
constexpr int long_smem_bytes() {
  return ST * kPageBytes + CL * NW * kSlot * 4;
}

// Long contexts (kSplit, more than kShortRounds * CL * NW pages from row 0's
// first key): the GROUPS * gridDim.z clusters of a (request, KV head) each take
// an equal contiguous page range. CTA r of a cluster streams its sixteenth of
// the range through an ST-deep cp.async ring and warp w attends the 16 rows of
// group w, so each page is read once per KV head. CTA r owns row w * 16 + r,
// gathers the cluster's partials of it over DSMEM and merges the clusters
// through the workspace.
template <int H, int CL, int NW, int ST>
__device__ __forceinline__ void long_context(const Params& p, int L, int first,
                                             int n, int b, int rg_c,
                                             float (*s_o)[kRows][kSlot],
                                             float (*s_m)[kRows],
                                             float (*s_l)[kRows]) {
  constexpr int GROUPS = H / 2;
  static_assert(NW == GROUPS, "one warp per 16-row group of a KV head");
  static_assert(CL == kRows, "CTA r owns row r of every group");
  static_assert(NW * 32 * 2 * 16 == kPageBytes, "two 16-byte copies a thread");
  static_assert(ST >= 2, "page ring depth");
  extern __shared__ __align__(128) unsigned char s_dyn[];
  // The ring: ST pages of [16, 128] bf16; then [CL][NW][kSlot] partials.
  auto* s_kv = reinterpret_cast<__nv_bfloat16*>(s_dyn);
  auto* s_recv =
      reinterpret_cast<float (*)[NW][kSlot]>(s_dyn + ST * kPageBytes);
  __shared__ __align__(8) uint64_t s_bar;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const uint32_t rank = cluster_ctarank();
  const int kvh = blockIdx.y;
  const __nv_bfloat16* q_base = p.q + kvh * (64 * H);
  const __nv_bfloat16* gate_base = p.gate + kvh * (64 * H);
  const __nv_bfloat16* kv_base = p.kv + kvh * kPageElems;
  __nv_bfloat16* out_base = p.out + kvh * (64 * H);
  const int* bt_row = p.bt + static_cast<long long>(b) * p.bt_stride;

  // ---- this cluster's pages [sbeg, sbeg + sn), this CTA's [cbeg, cbeg + cnt)
  const int slots = static_cast<int>(gridDim.z) * GROUPS;
  const int split = static_cast<int>(blockIdx.z) * GROUPS + rg_c;
  int active = min(slots, (n + CL * kMinCtaPages - 1) / (CL * kMinCtaPages));
  const int per = (n + active - 1) / active;
  active = (n + per - 1) / per;
  if (split >= active) return;  // the whole cluster, before any barrier
  const int sbeg = first + split * per;
  const int sn = min(per, first + n - sbeg);
  const int cper = (sn + CL - 1) / CL;
  const int n_active = (sn + cper - 1) / cper;  // CTAs holding pages
  const int cbeg = sbeg + static_cast<int>(rank) * cper;
  const int cnt =
      static_cast<int>(rank) < n_active ? min(cper, sbeg + sn - cbeg) : 0;

  // ---- queries of rows (g, g + 8) of group `warp`; the gate of owned row R
  const int R0 = warp * kRows + g, R1 = R0 + 8;
  const int tok0 = R0 / H, h0 = R0 % H;
  const int tok1 = R1 / H, h1 = R1 % H;
  uint4 qr[4];
  {
    const __nv_bfloat16* q0 =
        q_base + static_cast<long long>(b * 8 + tok0) * p.q_stride + h0 * 64 +
        16 * t;
    const __nv_bfloat16* q1 =
        q_base + static_cast<long long>(b * 8 + tok1) * p.q_stride + h1 * 64 +
        16 * t;
    qr[0] = ldg16(q0);
    qr[1] = ldg16(q0 + 8);
    qr[2] = ldg16(q1);
    qr[3] = ldg16(q1 + 8);
  }
  const int oR = warp * kRows + static_cast<int>(rank);
  const int otok = oR / H, oh = oR % H;
  const long long otok_row = static_cast<long long>(b * 8 + otok);
  const __nv_bfloat162 gate2 = *reinterpret_cast<const __nv_bfloat162*>(
      gate_base + otok_row * p.gate_stride + 64 * oh + 2 * lane);

  // ---- page ring of ST / NP steps of NP pages: thread i copies 16-byte
  // chunks i and i + 128 of each page of a step (one commit group per step)
  constexpr int NP = kGroupPages;
  static_assert(ST % NP == 0, "whole steps in the ring");
  constexpr int D = ST / NP;
  const int steps = (cnt + NP - 1) / NP;
  const uint32_t ring = smem_u32(s_kv);
  auto issue = [&](int k) {
#pragma unroll
    for (int pi = 0; pi < NP; ++pi) {
      const int i = k * NP + pi;
      if (i < cnt) {
        const int entry = __ldg(bt_row + cbeg + i);
        const __nv_bfloat16* src =
            kv_base + static_cast<long long>(entry) * p.kv_page_stride;
        const uint32_t dst = ring + ((k % D) * NP + pi) * kPageBytes;
        cp_async16(dst + tid * 16, src + tid * 8);
        cp_async16(dst + (tid + NW * 32) * 16, src + (tid + NW * 32) * 8);
      }
    }
  };
#pragma unroll
  for (int k = 0; k < D - 1; ++k) {
    if (k < steps) issue(k);
    cp_async_commit();
  }

  // ---- owner mbarrier (as the short path): the owner's arrive.expect_tx plus
  // the pushed bytes of the active CTAs
  const uint32_t mbar = smem_u32(&s_bar);
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mbar)
                 : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  uint32_t qf[4][4];
  {
    uint32_t w0[8], w1[8];
    words8(qr[0], qr[1], w0);
    words8(qr[2], qr[3], w1);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      qf[s][0] = w0[2 * s];
      qf[s][1] = w1[2 * s];
      qf[s][2] = w0[2 * s + 1];
      qf[s][3] = w1[2 * s + 1];
    }
  }
  if (tid == 0)
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mbar),
        "r"(n_active * NW * kPartialBytes)
        : "memory");
  const int lim0 = L - 8 + tok0, lim1 = L - 8 + tok1;
  const int low0 = p.window > 0 ? lim0 - p.window + 1 : 0;
  const int low1 = p.window > 0 ? lim1 - p.window + 1 : 0;
  const float sl2 = p.scale_log2;

  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  float o[8][4];
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;

  for (int k = 0; k < steps; ++k) {
    // step k landed (at most D - 2 younger groups pending), visible to every
    // thread, and every warp is done with step k - 1, whose slots are refilled
    cp_async_wait<D - 2>();
    __syncthreads();
    if (k + D - 1 < steps) issue(k + D - 1);
    cp_async_commit();
    attend_group<NP, ST>(s_kv, (k % D) * NP, min(NP, cnt - k * NP),
                         cbeg + k * NP, qf, g, t, lim0, lim1, low0, low1, sl2,
                         m0, m1, l0, l1, o);
  }
  cp_async_wait<0>();

  if (cnt > 0)
    stash_partial(s_o[warp], s_m[warp], s_l[warp], g, t, o, m0, m1, l0, l1);
  __syncwarp();
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");

  // ---- push: warp w's rows r (8 lanes each, 4 rows per lane group) to CTA r
  if (cnt > 0) {
    const int i8 = lane & 7;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int r = (lane >> 3) + 4 * k;
      const float4 xa = *reinterpret_cast<const float4*>(&s_o[warp][r][4 * i8]);
      const float4 xb =
          *reinterpret_cast<const float4*>(&s_o[warp][r][32 + 4 * i8]);
      const uint32_t owner = static_cast<uint32_t>(r);
      const uint32_t dst = mapa(smem_u32(&s_recv[rank][warp][4 * i8]), owner);
      const uint32_t rbar = mapa(mbar, owner);
      st_async_v4(dst, xa.x, xa.y, xa.z, xa.w, rbar);
      st_async_v4(dst + 128, xb.x, xb.y, xb.z, xb.w, rbar);
      if (i8 == 0) st_async_v2(dst + 256, s_m[warp][r], s_l[warp][r], rbar);
    }
  }

  // ---- owner of row oR: merge the cluster's CTAs, then the clusters
  wait_phase0(mbar);
  float M = -1e30f;
  for (int k = 0; k < n_active; ++k) M = fmaxf(M, s_recv[k][warp][64]);
  float Ls = 0.f, ox = 0.f, oy = 0.f;
  for (int k = 0; k < n_active; ++k) {
    const float sc = ex2(s_recv[k][warp][64] - M);
    Ls += sc * s_recv[k][warp][65];
    const float2 v =
        *reinterpret_cast<const float2*>(&s_recv[k][warp][2 * lane]);
    ox += sc * v.x;
    oy += sc * v.y;
  }
  if (active > 1) {
    const long long row_id =
        (static_cast<long long>(b) * gridDim.y + kvh) * (8 * H) + oR;
    if (!merge_partials(p, row_id, slots, split, active, lane, M, Ls, ox, oy))
      return;
  }
  finish_row(ox, oy, Ls, gate2, p.eps,
             out_base + otok_row * p.out_stride + 64 * oh, lane);
}

template <int H, int CL, int NW, bool kSplit>
__global__ void __launch_bounds__(NW * 32, 1)
    attn_norm_gate_kernel(const Params p) {
  constexpr int RPC = kRows / CL;  // final rows owned by each CTA
  constexpr int NWG = CL * NW;     // warps per cluster
  constexpr int GROUPS = H / 2;    // 16-row groups per request (8H rows)
  static_assert(
      NW == 4,
      "merge/push mapping assumes 4 warps (4 rows per warp, 8 lanes per row)");
  static_assert(RPC >= 1 && RPC <= NW, "cluster configuration");

  __shared__ __align__(16) float s_o[NW][kRows][kSlot];
  __shared__ float s_m[NW][kRows];
  __shared__ float s_l[NW][kRows];
  __shared__ __align__(16) float s_recv[CL][RPC][kSlot];
  __shared__ __align__(8) uint64_t s_mbar;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const uint32_t rank = cluster_ctarank();
  const int cluster_id = blockIdx.x / CL;
  const int b = cluster_id / GROUPS, rg = cluster_id % GROUPS;
  // KV head of this cluster: its query, gate and output heads and its cache
  // rows.
  const int kvh = blockIdx.y;
  const __nv_bfloat16* q_base = p.q + kvh * (64 * H);
  const __nv_bfloat16* gate_base = p.gate + kvh * (64 * H);
  const __nv_bfloat16* kv_base = p.kv + kvh * (kPage * 128);
  __nv_bfloat16* out_base = p.out + kvh * (64 * H);

  asm volatile("griddepcontrol.wait;" ::: "memory");

  // ---- issue every load that does not depend on the sequence length first:
  //      length, block-table entries (read only; dereferenced below only if <
  //      ceil(L/16)), q, gate
  const int L = __ldg(p.seq_lens + b);
  const int gw =
      static_cast<int>(rank) * NW + warp;  // CTA-major page assignment
  const int* bt_row = p.bt + static_cast<long long>(b) * p.bt_stride;
  int e_first = 0, e_second = 0;
  if (gw < p.max_pages) e_first = __ldg(bt_row + gw);
  if (gw + NWG < p.max_pages) e_second = __ldg(bt_row + gw + NWG);
  // kSplit: the pages from row 0's first visible key, pbeg, to the last. Past
  // kShortRounds rounds of the cluster's warps the long-context path takes the
  // request; otherwise z slice 0 runs the short path and the other slices exit
  // at once (whole clusters, before any barrier).
  int pbeg = 0;
  [[maybe_unused]] int pages = 0;
  if constexpr (kSplit) {
    const int last = (L + kPage - 1) / kPage;
    pbeg = p.window > 0 ? max(0, L - 8 - p.window + 1) / kPage : 0;
    pages = last - pbeg;
    if (pages > kShortRounds * NWG) {
      long_context<H, CL, NW, kRingStages>(p, L, pbeg, pages, b, rg, s_o, s_m,
                                           s_l);
      return;
    }
    if (blockIdx.z != 0) return;
    if (pbeg != 0) {  // the entries above are those of pages from 0
      e_first = gw < pages ? __ldg(bt_row + pbeg + gw) : 0;
      e_second = gw + NWG < pages ? __ldg(bt_row + pbeg + gw + NWG) : 0;
    }
  }

  const int R0 = rg * kRows + g, R1 = R0 + 8;
  const int tok0 = R0 / H, h0 = R0 % H;
  const int tok1 = R1 / H, h1 = R1 % H;
  uint4 qr[4];
  {
    const __nv_bfloat16* q0 =
        q_base + static_cast<long long>(b * 8 + tok0) * p.q_stride + h0 * 64 +
        16 * t;
    const __nv_bfloat16* q1 =
        q_base + static_cast<long long>(b * 8 + tok1) * p.q_stride + h1 * 64 +
        16 * t;
    qr[0] = ldg16(q0);
    qr[1] = ldg16(q0 + 8);
    qr[2] = ldg16(q1);
    qr[3] = ldg16(q1 + 8);
  }
  // owner rows: row = slot * CL + rank, slot = warp (< RPC)
  const int orow = warp * CL + static_cast<int>(rank);
  const int oR = rg * kRows + orow;
  const int otok = oR / H, oh = oR % H;
  const long long otok_row = static_cast<long long>(b * 8 + otok);
  __nv_bfloat162 gate2 = __floats2bfloat162_rn(0.f, 0.f);
  if (warp < RPC)
    gate2 = *reinterpret_cast<const __nv_bfloat162*>(
        gate_base + otok_row * p.gate_stride + 64 * oh + 2 * lane);

  // ---- owner mbarrier: completed by the owner's own arrive.expect_tx plus the
  // pushed transaction bytes
  const uint32_t mbar = smem_u32(&s_mbar);
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mbar)
                 : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  uint32_t qf[4][4];
  {
    uint32_t w0[8], w1[8];
    words8(qr[0], qr[1], w0);
    words8(qr[2], qr[3], w1);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      qf[s][0] = w0[2 * s];
      qf[s][1] = w1[2 * s];
      qf[s][2] = w0[2 * s + 1];
      qf[s][3] = w1[2 * s + 1];
    }
  }

  // pages of this request (kSplit: from pbeg); only these table entries are
  // dereferenced
  const int num_pages = kSplit ? pages : (L + kPage - 1) / kPage;
  const int n_active =
      min(CL, (num_pages + NW - 1) / NW);  // CTAs holding at least one page
  if (tid == 0)  // the pushes of the active CTAs complete this many bytes
                 // (tx-count may transiently go negative)
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mbar),
        "r"(n_active * RPC * kPartialBytes)
        : "memory");
  const int lim0 = L - 8 + tok0,
            lim1 = L - 8 + tok1;  // causal limit (inclusive) of rows g and g+8
  // kSplit: first visible key (inclusive) of rows g and g+8
  const int low0 = kSplit && p.window > 0 ? lim0 - p.window + 1 : 0;
  const int low1 = kSplit && p.window > 0 ? lim1 - p.window + 1 : 0;
  const float sl2 = p.scale_log2;

  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
  float o[8][4];
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;

  PageRegs cur, nxt;
  int pg = gw;
  if (pg < num_pages) load_page(cur, kv_base, p.kv_page_stride, e_first, g, t);
  for (int it = 0; pg < num_pages; ++it) {
    const int pn = pg + NWG;
    if (pn < num_pages)
      load_page(nxt, kv_base, p.kv_page_stride,
                it == 0 ? e_second : __ldg(bt_row + pbeg + pn), g, t);
    attend_page<kSplit>(cur, qf, (pbeg + pg) * kPage + 2 * t, lim0, lim1, low0,
                        low1, sl2, m0, m1, l0, l1, o);
    cur = nxt;
    pg = pn;
  }

  if (static_cast<int>(rank) < n_active) {
    // ---- warp partial -> shared memory
    stash_partial(s_o[warp], s_m[warp], s_l[warp], g, t, o, m0, m1, l0, l1);
    __syncthreads();
    asm volatile("barrier.cluster.wait.aligned;" ::
                     : "memory");  // peers' mbarriers are initialised

    // ---- merge the NW warp partials: warp w handles rows 4w..4w+3, 8 lanes
    // per row; lane i owns dims [4i, 4i+4) and [32+4i, 32+4i+4), so each
    // st.async covers 128 contiguous bytes
    const int row = tid >> 3, i8 = tid & 7;
    float M = -1e30f;
#pragma unroll
    for (int w = 0; w < NW; ++w) M = fmaxf(M, s_m[w][row]);
    float Ls = 0.f;
    float4 xa = make_float4(0.f, 0.f, 0.f, 0.f), xb = xa;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float sc = ex2(s_m[w][row] - M);
      Ls += sc * s_l[w][row];
      const float4 a = *reinterpret_cast<const float4*>(&s_o[w][row][4 * i8]);
      const float4 c =
          *reinterpret_cast<const float4*>(&s_o[w][row][32 + 4 * i8]);
      xa.x += sc * a.x;
      xa.y += sc * a.y;
      xa.z += sc * a.z;
      xa.w += sc * a.w;
      xb.x += sc * c.x;
      xb.y += sc * c.y;
      xb.z += sc * c.z;
      xb.w += sc * c.w;
    }
    const uint32_t owner = static_cast<uint32_t>(row % CL);
    const int slot = row / CL;
    const uint32_t dst = mapa(smem_u32(&s_recv[rank][slot][4 * i8]), owner);
    const uint32_t rbar = mapa(mbar, owner);
    st_async_v4(dst, xa.x, xa.y, xa.z, xa.w, rbar);
    st_async_v4(dst + 128, xb.x, xb.y, xb.z, xb.w, rbar);
    if (i8 == 0) st_async_v2(dst + 256, M, Ls, rbar);
  } else {
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  }

  // ---- owner: merge CL partials of its rows, bf16 o, RMS norm, sigmoid gate,
  // store
  if (warp < RPC) {
    const int slot = warp;
    wait_phase0(mbar);

    float mk[CL];
    float M = -1e30f;
#pragma unroll
    for (int k = 0; k < CL; ++k) {
      mk[k] = (k < n_active) ? s_recv[k][slot][64] : -1e30f;
      M = fmaxf(M, mk[k]);
    }
    float Ls = 0.f, ox = 0.f, oy = 0.f;
#pragma unroll
    for (int k = 0; k < CL; ++k) {
      if (k < n_active) {
        const float sc = ex2(mk[k] - M);
        Ls += sc * s_recv[k][slot][65];
        const float2 v =
            *reinterpret_cast<const float2*>(&s_recv[k][slot][2 * lane]);
        ox += sc * v.x;
        oy += sc * v.y;
      }
    }
    finish_row(ox, oy, Ls, gate2, p.eps,
               out_base + otok_row * p.out_stride + 64 * oh, lane);
  }
}

#undef kNegInf

// ---------------------------------------------------------------- host side

constexpr int kHeadDim = 64;
constexpr int kQueryRows = 8;   // query tokens per request (1 + 7 drafts)
constexpr int kHeadsPerKv = 8;  // query heads per KV head (H)
constexpr int kCluster = 16;    // CTAs per cluster (CL), above the portable 8
constexpr int kWarps = 4;       // warps per CTA (NW)
constexpr int kMaxSplits = 64;  // z slices of a long-context launch
// Contexts one cluster takes on the short path (two rounds of its warps).
constexpr int kShortMaxLen = 2 * kCluster * kWarps * kPage;
// The layer's softmax scale (head_dim ** -0.5 * qk_scale_factor) and RMS-norm
// eps: constants of the one Muse Spark config served. unsupported_attn2 in the
// specialized decode's runtime/csrc_ops.py refuses a layer with other values.
constexpr double kSmScale = 0.6841258108;
constexpr double kEps = 1e-5;

inline void check_cuda(cudaError_t err, const char* what) {
  STD_TORCH_CHECK(err == cudaSuccess, what,
                  " failed: ", cudaGetErrorString(err));
}

bool aligned(const torch::stable::Tensor& t, uintptr_t bytes) {
  return reinterpret_cast<uintptr_t>(t.data_ptr()) % bytes == 0;
}

bool on_device(const torch::stable::Tensor& t, int device) {
  return t.is_cuda() && t.get_device_index() == device;
}

// Workspace bytes of a long-context launch: one counter per row (16-byte
// padded), then a partial of kSlot floats per row and cluster of its
// (request, KV head): (8 H / 16) clusters per z slice.
int64_t counters_bytes(int64_t rows) { return (rows * 4 + 15) / 16 * 16; }

int64_t workspace_bytes(int64_t rows, int64_t num_splits) {
  return counters_bytes(rows) +
         rows * num_splits * (kHeadsPerKv / 2) * kSlot * 4;
}

template <int H, int CL, int NW, bool kSplit>
void launch_cfg(int device, const Params& prm, int num_reqs, int kv_heads,
                int num_splits, bool enable_pdl, cudaStream_t stream) {
  auto kern = attn_norm_gate_kernel<H, CL, NW, kSplit>;
  if constexpr (CL > 8) {
    static std::mutex mu;
    static std::unordered_set<int> configured;
    std::lock_guard<std::mutex> lock(mu);
    if (configured.count(device) == 0) {
      check_cuda(cudaFuncSetAttribute(
                     kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1),
                 "cudaFuncSetAttribute(NonPortableClusterSizeAllowed)");
      if constexpr (kSplit)
        check_cuda(cudaFuncSetAttribute(
                       kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       long_smem_bytes<CL, NW, kRingStages>()),
                   "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");
      configured.insert(device);
    }
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim =
      dim3(static_cast<unsigned>(CL * num_reqs * (H / 2)),
           static_cast<unsigned>(kv_heads), static_cast<unsigned>(num_splits));
  cfg.blockDim = dim3(NW * 32, 1, 1);
  cfg.dynamicSmemBytes = kSplit ? long_smem_bytes<CL, NW, kRingStages>() : 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[2] = {};
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = CL;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = enable_pdl ? 2 : 1;
  check_cuda(cudaLaunchKernelEx(&cfg, kern, prm), "muse_spark_attn2 launch");
}

}  // namespace muse_spark_attn2
}  // namespace vllm

// out [8 R, 512 kv_heads_per_rank] = sigmoid(gate) * rmsnorm(attention(q,
// kv_cache)) per query head, for R = q.size(0) / 8 requests whose 8 query rows
// are the last 8 of their seq_lens cache entries. q, gate and out hold the
// heads of each KV head contiguously (8 x 64 columns per KV head); their rows
// may be strided (q and gate are column views of the attention front's output
// [q | k | v | gate]). kv_cache is the layer's bf16 [pages, kv_heads_per_rank,
// 16, 128] cache. Lengths must be 8..max_seq_len and covered by the first
// ceil(L / 16) block_table entries (device values, not checked here).
// max_seq_len is the deployment's maximum length (host-known, fixed for a
// captured graph): above 2048, or with sliding_window W > 0 (row t of a
// request sees the keys [L - 8 + t - W + 1, L - 8 + t]; 0: every key up to
// its own), the split kernel runs over num_splits z slices and needs
// workspace, a uint8 tensor of at least 64 R kv_heads_per_rank * (4 + 1088
// num_splits) bytes (16-byte padded counters first) whose counters are zero
// before the first call: every call leaves them zero.
void muse_spark_attn2(torch::stable::Tensor const& q,
                      torch::stable::Tensor const& kv_cache,
                      torch::stable::Tensor const& block_table,
                      torch::stable::Tensor const& seq_lens,
                      std::optional<torch::stable::Tensor> o_norm_weight,
                      torch::stable::Tensor const& gate,
                      torch::stable::Tensor& out, int64_t kv_heads_per_rank,
                      bool enable_pdl, int64_t max_seq_len,
                      int64_t sliding_window, int64_t num_splits,
                      std::optional<torch::stable::Tensor> workspace) {
  using namespace vllm::muse_spark_attn2;
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(!o_norm_weight.has_value(),
                  "muse_spark_attn2: the Muse Spark o-norm has no weight; "
                  "o_norm_weight must be None");
  STD_TORCH_CHECK(kv_heads_per_rank >= 1 && kv_heads_per_rank <= 65535,
                  "muse_spark_attn2: kv_heads_per_rank = ", kv_heads_per_rank,
                  " is outside 1..65535");
  const int kv_heads = static_cast<int>(kv_heads_per_rank);
  const int64_t width = static_cast<int64_t>(kv_heads) * kHeadsPerKv * kHeadDim;
  STD_TORCH_CHECK(q.is_cuda(), "muse_spark_attn2: q must be a CUDA tensor");
  const int dev = q.get_device_index();
  STD_TORCH_CHECK(q.scalar_type() == ScalarType::BFloat16 && q.dim() == 2 &&
                      q.size(1) == width && q.stride(1) == 1 &&
                      q.size(0) >= kQueryRows && q.size(0) % kQueryRows == 0,
                  "muse_spark_attn2: q must be a bf16 [8 R, ", width,
                  "] tensor with dense rows (8 query heads of 64 per KV head)");
  STD_TORCH_CHECK(aligned(q, 16) && (q.stride(0) * 2) % 16 == 0,
                  "muse_spark_attn2: q rows must be 16-byte aligned");
  const int64_t rows = q.size(0);
  const int64_t num_reqs = rows / kQueryRows;
  STD_TORCH_CHECK(gate.scalar_type() == ScalarType::BFloat16 &&
                      gate.dim() == 2 && gate.size(0) == rows &&
                      gate.size(1) == width && gate.stride(1) == 1,
                  "muse_spark_attn2: gate must be a bf16 [", rows, ", ", width,
                  "] tensor with dense rows");
  STD_TORCH_CHECK(aligned(gate, 4) && gate.stride(0) % 2 == 0,
                  "muse_spark_attn2: gate rows must be 4-byte aligned");
  STD_TORCH_CHECK(out.scalar_type() == ScalarType::BFloat16 && out.dim() == 2 &&
                      out.size(0) == rows && out.size(1) == width &&
                      out.stride(1) == 1,
                  "muse_spark_attn2: out must be a bf16 [", rows, ", ", width,
                  "] tensor with dense rows");
  STD_TORCH_CHECK(aligned(out, 4) && out.stride(0) % 2 == 0,
                  "muse_spark_attn2: out rows must be 4-byte aligned");
  STD_TORCH_CHECK(kv_cache.scalar_type() == ScalarType::BFloat16 &&
                      kv_cache.dim() == 4 && kv_cache.size(1) == kv_heads &&
                      kv_cache.size(2) == kPage &&
                      kv_cache.size(3) == 2 * kHeadDim,
                  "muse_spark_attn2: kv_cache must be a bf16 [pages, ",
                  kv_heads, ", ", kPage, ", ", 2 * kHeadDim, "] tensor");
  STD_TORCH_CHECK(
      kv_cache.stride(3) == 1 && kv_cache.stride(2) == 2 * kHeadDim &&
          (kv_heads == 1 || kv_cache.stride(1) == kPage * 2 * kHeadDim) &&
          kv_cache.stride(0) >=
              static_cast<int64_t>(kv_heads) * kPage * 2 * kHeadDim &&
          kv_cache.stride(0) % 8 == 0 && aligned(kv_cache, 16),
      "muse_spark_attn2: kv_cache pages must hold dense 16-byte aligned "
      "[heads, 16, 128] blocks");
  STD_TORCH_CHECK(
      block_table.scalar_type() == ScalarType::Int && block_table.dim() == 2 &&
          block_table.size(0) >= num_reqs && block_table.size(1) >= 1 &&
          block_table.stride(1) == 1,
      "muse_spark_attn2: block_table must be an int32 [>= ", num_reqs,
      ", pages] tensor with dense rows");
  STD_TORCH_CHECK(block_table.size(1) <= INT32_MAX,
                  "muse_spark_attn2: block_table is too wide");
  STD_TORCH_CHECK(seq_lens.scalar_type() == ScalarType::Int &&
                      seq_lens.dim() == 1 && seq_lens.size(0) >= num_reqs &&
                      (seq_lens.size(0) == 1 || seq_lens.stride(0) == 1),
                  "muse_spark_attn2: seq_lens must be a dense int32 vector of "
                  "at least ",
                  num_reqs, " entries");
  STD_TORCH_CHECK(on_device(kv_cache, dev) && on_device(block_table, dev) &&
                      on_device(seq_lens, dev) && on_device(gate, dev) &&
                      on_device(out, dev),
                  "muse_spark_attn2: all tensors must be on q's device");
  STD_TORCH_CHECK(num_reqs * kCluster * (kHeadsPerKv / 2) <= INT32_MAX,
                  "muse_spark_attn2: too many requests for one launch");
  STD_TORCH_CHECK(max_seq_len >= kQueryRows && max_seq_len <= INT32_MAX,
                  "muse_spark_attn2: max_seq_len = ", max_seq_len,
                  " is outside 8..", INT32_MAX);
  STD_TORCH_CHECK(sliding_window >= 0 && sliding_window <= INT32_MAX,
                  "muse_spark_attn2: sliding_window = ", sliding_window,
                  " is outside 0..", INT32_MAX);
  STD_TORCH_CHECK(num_splits >= 1 && num_splits <= kMaxSplits,
                  "muse_spark_attn2: num_splits = ", num_splits,
                  " is outside 1..", kMaxSplits);
  // The deployment's maximum length (host-known, fixed for a captured graph)
  // picks the kernel: the one-cluster kernel serves 2048 tokens without a
  // window, the split kernel any length and window.
  const bool split = max_seq_len > kShortMaxLen || sliding_window > 0;
  const int64_t partial_rows = rows * kHeadsPerKv * kv_heads;
  unsigned int* counters = nullptr;
  float* partials = nullptr;
  if (split) {
    STD_TORCH_CHECK(workspace.has_value(), "muse_spark_attn2: max_seq_len > ",
                    kShortMaxLen, " or a sliding window needs a workspace");
    const torch::stable::Tensor& ws = *workspace;
    const int64_t need = workspace_bytes(partial_rows, num_splits);
    STD_TORCH_CHECK(ws.scalar_type() == ScalarType::Byte && ws.dim() == 1 &&
                        ws.stride(0) == 1 && ws.size(0) >= need &&
                        aligned(ws, 16) && on_device(ws, dev),
                    "muse_spark_attn2: workspace must be a dense uint8 vector "
                    "of at least ",
                    need, " bytes, 16-byte aligned, on q's device");
    auto* base = static_cast<uint8_t*>(ws.mutable_data_ptr());
    // The byte buffer holds the uint32 counters, then the fp32 partials at a
    // 16-byte aligned offset.
    counters = reinterpret_cast<unsigned int*>(base);
    partials = reinterpret_cast<float*>(base + counters_bytes(partial_rows));
  }

  Params prm{};
  prm.q = static_cast<const __nv_bfloat16*>(q.data_ptr());
  prm.q_stride = q.stride(0);
  prm.gate = static_cast<const __nv_bfloat16*>(gate.data_ptr());
  prm.gate_stride = gate.stride(0);
  prm.kv = static_cast<const __nv_bfloat16*>(kv_cache.data_ptr());
  prm.kv_page_stride = kv_cache.stride(0);
  prm.bt = static_cast<const int*>(block_table.data_ptr());
  prm.bt_stride = block_table.stride(0);
  prm.max_pages = static_cast<int>(block_table.size(1));
  prm.seq_lens = static_cast<const int*>(seq_lens.data_ptr());
  prm.out = static_cast<__nv_bfloat16*>(out.mutable_data_ptr());
  prm.out_stride = out.stride(0);
  prm.scale_log2 = static_cast<float>(kSmScale) * 1.4426950408889634f;
  prm.eps = static_cast<float>(kEps);
  prm.window = static_cast<int>(sliding_window);
  prm.counters = counters;
  prm.partials = partials;

  const torch::stable::accelerator::DeviceGuard device_guard(dev);
  const cudaStream_t stream = get_current_cuda_stream(dev);
  if (!split) {
    launch_cfg<kHeadsPerKv, kCluster, kWarps, false>(
        dev, prm, static_cast<int>(num_reqs), kv_heads, 1, enable_pdl, stream);
  } else {
    launch_cfg<kHeadsPerKv, kCluster, kWarps, true>(
        dev, prm, static_cast<int>(num_reqs), kv_heads,
        static_cast<int>(num_splits), enable_pdl, stream);
  }
}
