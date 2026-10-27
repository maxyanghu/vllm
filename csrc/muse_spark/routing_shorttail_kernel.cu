// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Muse Spark routing for decode batches of at most 8 tokens: sigmoid of the
// fp32 router logits plus the correction bias, top-8 of 256 experts, the
// selected sigmoid scores renormalized and scaled. Writes the global expert
// ids (int32 [b, 8]) and the routing weights (fp32 [b, 8]) that the routed
// MoE op takes.
//
// The result is bitwise equal to fused_musespark1x_routing (the Triton kernel
// vllm/models/muse_spark/common/ops/fused_routing.py) as Muse Spark calls it:
// sigmoid scores, no output multiplier (it is folded into the router weight
// after loading) and router_score_division_eps = 1e-15. With renormalize the
// weights are score * (scale / (sum + 1e-15)), the sum taken in top-k order
// from 0.0; without it they are score * scale.
//
// Optionally, the same launch also writes the TRT-LLM-gen routing metadata of
// this rank's experts for one GEMM tile, as FlashInfer's
// routingIndicesDynBlockKernel writes it, so the routed MoE GEMMs need no
// routing kernel of their own: local expert e takes ceil(count_e / tile) CTAs
// at the exclusive prefix sum of the CTA counts in expert order, and its routes
// take the permuted rows from that CTA offset times tile, in token order. A
// route to another rank's expert and the padding rows of each expert's last
// CTA map to -1. The histogram and per-expert token counts are not written.
//
// Launch: one CTA of 256 threads, one warp per token row, with PDL: the bias (a
// weight) is read before griddepcontrol.wait, the logits after it.
//
// Selection: lane l holds experts 8l .. 8l + 7 and sorts them once by an int32
// key that orders as score + bias (NaN is outside the domain). Each of the 8
// rounds takes the warp maximum of the lanes' head keys and the lowest lane
// holding it, matching the Triton argmax (the lowest expert among equal
// values); the winner pops its head.
//
// Arithmetic: every float operation is the inline-PTX instruction Triton emits
// for the same expression of _routing_kernel, so nvcc can neither contract nor
// reorder them and no result depends on the compiler flags.

#include <cuda_runtime.h>
#include <stdint.h>

#include <optional>

#include <torch/headeronly/core/ScalarType.h>

#include "torch_utils.h"

namespace vllm {
namespace muse_spark_routing {

constexpr int kExperts = 256;
constexpr int kTopK = 8;
constexpr int kMaxTokens = 8;
constexpr int kThreads = 32 * kMaxTokens;
constexpr unsigned kFull = 0xffffffffu;
// Muse Spark's router_score_division_eps (MuseSpark1xMoE builds its router
// with 1e-15), rounded to fp32 as the Triton kernel receives it.
constexpr float kDivisionEps = 1e-15f;
static_assert(kExperts == 32 * 8, "each lane holds 8 consecutive experts");
static_assert(kThreads == 256, "one warp per token row in a 256-thread block");
constexpr int kWarps = kThreads / 32;
constexpr int kRoutes = kMaxTokens * kTopK;
static_assert(kRoutes <= kThreads, "one thread per route");
static_assert(kExperts <= kThreads, "one thread per local expert");

// The int32 metadata buffer: offsets of its fields. A CTA holds at least one
// route, so b <= 8 tokens take at most kRoutes CTAs. The offsets keep every
// field 16-byte aligned.
constexpr int kMaxCtas = kRoutes;
constexpr int kMetaPaddedTokens = 0;        // [1] CTAs * tile
constexpr int kMetaNumCtas = 1;             // [1] CTAs
constexpr int kMetaExpandedToPermuted = 4;  // [b * 8]
constexpr int kMetaCtaBatch = kMetaExpandedToPermuted + kRoutes;  // [CTAs]
constexpr int kMetaCtaLimit = kMetaCtaBatch + kMaxCtas;           // [CTAs]
constexpr int kMetaPermutedToToken = kMetaCtaLimit + kMaxCtas;    // [CTAs*tile]
static_assert(kMetaPermutedToToken % 4 == 0, "16-byte aligned fields");
constexpr int kMaxTile = 256;

// The int32 elements the metadata of one tile needs: the fields above and the
// permuted-to-token map, kMaxCtas * tile rows plus one guard element.
constexpr int64_t metadata_size(int64_t tile) {
  return kMetaPermutedToToken + kMaxCtas * tile + 1;
}

// Where the routing metadata goes, or data == nullptr for none. Local experts
// are the global ids [local_offset, local_offset + num_local).
struct MetadataArgs {
  int* data;
  int local_offset;
  int num_local;
  int tile;
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
  #error "routing_shorttail_kernel needs griddepcontrol (sm_90+)"
#endif

// griddepcontrol.wait (sm_90+): waits until the PDL primary grid completed and
// its writes are visible; a no-op without a PDL edge. The "memory" clobber
// keeps later loads after it.
__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

// The float wrappers below are one PTX instruction each (round to nearest
// even, no .ftz) for the bitwise contract with the Triton kernel: C++
// arithmetic would let nvcc contract a * b + c into an FMA or pick another
// division sequence.

// tl.sigmoid(x) for fp32 as Triton lowers it: x * -log2(e), ex2.approx, + 1,
// then 1 / that with div.full.
__device__ __forceinline__ float sigmoid_f32(float x) {
  float a, e, d, r;
  asm("mul.f32 %0, %1, 0fBFB8AA3B;" : "=f"(a) : "f"(x));
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(a));
  asm("add.f32 %0, %1, 0f3F800000;" : "=f"(d) : "f"(e));
  asm("div.full.f32 %0, 0f3F800000, %1;" : "=f"(r) : "f"(d));
  return r;
}

// Triton's fp32 a + b.
__device__ __forceinline__ float add_f32(float a, float b) {
  float r;
  asm("add.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// Triton's fp32 a * b.
__device__ __forceinline__ float mul_f32(float a, float b) {
  float r;
  asm("mul.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// Triton's fp32 a / b.
__device__ __forceinline__ float div_f32(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// An int32 that orders as the float does for non-NaN values: the bits for
// v >= 0, the bits with the 31 magnitude bits flipped for v < 0 (a larger
// magnitude is smaller); -0.0 maps to 0, the key of +0.0, as the two compare
// equal. Integer operations only, so denormals keep their order whatever the
// flush-to-zero setting.
__device__ __forceinline__ int order_key(float v) {
  const int b = __float_as_int(v);
  const int k = b < 0 ? (b ^ 0x7fffffff) : b;
  return (b & 0x7fffffff) == 0 ? 0 : k;
}

// Sort words: (key + 2^31) in bits 3..34, (7 - local index) in bits 0..2, so a
// larger word is a larger key, then a lower index. A popped slot becomes 0,
// whose key is INT32_MIN, below the key of any non-NaN value (-inf included).
__device__ __forceinline__ unsigned long long sort_word(int key, int j) {
  return (static_cast<unsigned long long>(static_cast<unsigned>(key) ^
                                          0x80000000u)
          << 3) |
         static_cast<unsigned long long>(7 - j);
}

__device__ __forceinline__ int word_key(unsigned long long w) {
  return static_cast<int>(static_cast<unsigned>(w >> 3) ^ 0x80000000u);
}

__device__ __forceinline__ int word_index(unsigned long long w) {
  return 7 - static_cast<int>(w & 7ull);
}

__device__ __forceinline__ void cas_desc(unsigned long long& a,
                                         unsigned long long& b) {
  const unsigned long long hi = a > b ? a : b;
  const unsigned long long lo = a > b ? b : a;
  a = hi;
  b = lo;
}

// Batcher's 19-comparator network for 8 values, descending.
__device__ __forceinline__ void sort8_desc(unsigned long long (&c)[8]) {
  cas_desc(c[0], c[1]);
  cas_desc(c[2], c[3]);
  cas_desc(c[4], c[5]);
  cas_desc(c[6], c[7]);
  cas_desc(c[0], c[2]);
  cas_desc(c[1], c[3]);
  cas_desc(c[4], c[6]);
  cas_desc(c[5], c[7]);
  cas_desc(c[1], c[2]);
  cas_desc(c[5], c[6]);
  cas_desc(c[0], c[4]);
  cas_desc(c[1], c[5]);
  cas_desc(c[2], c[6]);
  cas_desc(c[3], c[7]);
  cas_desc(c[2], c[4]);
  cas_desc(c[3], c[5]);
  cas_desc(c[1], c[2]);
  cas_desc(c[3], c[4]);
  cas_desc(c[5], c[6]);
}

// One warp, one token row: the top-8 selection. Returns, in lanes 0..7, the
// global expert id of pick `lane` and its weight; other lanes return (0, 0).
struct Pick {
  int id;
  float weight;
};

__device__ __forceinline__ Pick select_top8(const float* __restrict__ row,
                                            float4 bA, float4 bB, int lane,
                                            float scale, bool renormalize) {
  const float4* g4 = reinterpret_cast<const float4*>(row) + lane * 2;
  const float4 gA = g4[0], gB = g4[1];
  const float g[8] = {gA.x, gA.y, gA.z, gA.w, gB.x, gB.y, gB.z, gB.w};
  const float b[8] = {bA.x, bA.y, bA.z, bA.w, bB.x, bB.y, bB.z, bB.w};
  float s[8];
  unsigned long long c[8];
#pragma unroll
  for (int j = 0; j < 8; j++) {
    s[j] = sigmoid_f32(g[j]);
    c[j] = sort_word(order_key(add_f32(s[j], b[j])), j);
  }
  // Each lane's 8 (key, local index) pairs sorted once, best first: a round
  // reads the heads, and only the winning lane pops its head.
  sort8_desc(c);

  int sel_id[kTopK];
  float sel_s[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; k++) {
    const int best = word_key(c[0]);
    const int bj = word_index(c[0]);
    float bs = s[0];
#pragma unroll
    for (int j = 1; j < 8; j++) {
      bs = (bj == j) ? s[j] : bs;
    }
    const int kmax = __reduce_max_sync(kFull, best);
    const unsigned holders = __ballot_sync(kFull, best == kmax);
    const int wl = __ffs(holders) - 1;
    const int wj = __shfl_sync(kFull, bj, wl);
    sel_s[k] = __shfl_sync(kFull, bs, wl);
    sel_id[k] = wl * 8 + wj;
    if (lane == wl) {
#pragma unroll
      for (int j = 0; j < 7; j++) {
        c[j] = c[j + 1];
      }
      c[7] = 0ull;
    }
  }

  float inv = scale;
  if (renormalize) {
    float sum = add_f32(0.0f, sel_s[0]);
#pragma unroll
    for (int k = 1; k < kTopK; k++) {
      sum = add_f32(sum, sel_s[k]);
    }
    inv = div_f32(scale, add_f32(sum, kDivisionEps));
  }
  Pick pick = {0, 0.0f};
#pragma unroll
  for (int k = 0; k < kTopK; k++) {
    if (lane == k) {
      pick = {sel_id[k], mul_f32(sel_s[k], inv)};
    }
  }
  return pick;
}

// The local expert of a global id, or -1 for another rank's expert.
__device__ __forceinline__ int local_expert(int id, const MetadataArgs& meta) {
  const int e = id - meta.local_offset;
  return (e >= 0 && e < meta.num_local) ? e : -1;
}

// Shared state of the metadata pass. tokens[e]: bit t set when token t routes
// to local expert e (atomicOr by the selecting warps); cta_offset[e]: its
// first CTA; warp_ctas[w]: the CTAs of warp w's experts; ids: the routes.
struct MetadataSmem {
  unsigned tokens[kExperts];
  int cta_offset[kExperts];
  int warp_ctas[kWarps];
  int ids[kRoutes];
};

// Thread e (e < num_local): local expert e's CTAs (batch index and row limit)
// and its permuted rows (token ids in token order, then -1 to the end of its
// last CTA). Every thread of the block calls it (it holds a barrier).
__device__ void write_expert_metadata(MetadataSmem& smem,
                                      const MetadataArgs& meta) {
  const int e = threadIdx.x;
  const int lane = e & 31;
  const int warp = e >> 5;
  const bool local = e < meta.num_local;
  const unsigned tokens = local ? smem.tokens[e] : 0u;
  const int count = __popc(tokens);
  const int ctas = (count + meta.tile - 1) / meta.tile;
  // Block-wide exclusive prefix sum of ctas over the experts.
  int inclusive = ctas;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int v = __shfl_up_sync(kFull, inclusive, d);
    inclusive += lane >= d ? v : 0;
  }
  if (lane == 31) {
    smem.warp_ctas[warp] = inclusive;
  }
  __syncthreads();
  int before = 0, total = 0;
#pragma unroll
  for (int w = 0; w < kWarps; w++) {
    const int c = smem.warp_ctas[w];
    before += w < warp ? c : 0;
    total += c;
  }
  const int cta0 = before + inclusive - ctas;
  int* batch = meta.data + kMetaCtaBatch;
  int* limit = meta.data + kMetaCtaLimit;
  int* to_token = meta.data + kMetaPermutedToToken;
  if (local) {
    smem.cta_offset[e] = cta0;
    const int row0 = cta0 * meta.tile;
    for (int j = 0; j < ctas; j++) {
      batch[cta0 + j] = e;
      limit[cta0 + j] = min((cta0 + j + 1) * meta.tile, row0 + count);
    }
    unsigned rest = tokens;
    for (int r = 0; r < count; r++) {
      to_token[row0 + r] = __ffs(rest) - 1;
      rest &= rest - 1;
    }
    for (int row = row0 + count; row < (cta0 + ctas) * meta.tile; row++) {
      to_token[row] = -1;
    }
  }
  if (threadIdx.x == 0) {
    meta.data[kMetaPaddedTokens] = total * meta.tile;
    meta.data[kMetaNumCtas] = total;
  }
}

// Thread r (r < 8 b): route r's permuted row, or -1 for another rank's
// expert. Reads cta_offset after the barrier that follows its writes.
__device__ void write_route_metadata(const MetadataSmem& smem,
                                     const MetadataArgs& meta, int num_tokens) {
  const int r = threadIdx.x;
  if (r >= num_tokens * kTopK) {
    return;
  }
  const int e = local_expert(smem.ids[r], meta);
  int permuted = -1;
  if (e >= 0) {
    const unsigned earlier = (1u << (r / kTopK)) - 1u;
    permuted =
        smem.cta_offset[e] * meta.tile + __popc(smem.tokens[e] & earlier);
  }
  meta.data[kMetaExpandedToPermuted + r] = permuted;
}

__global__ void __launch_bounds__(kThreads)
    routing_shorttail_kernel(const float* __restrict__ logits,
                             int64_t logits_stride,
                             const float* __restrict__ bias,
                             int* __restrict__ ids, float* __restrict__ weights,
                             int num_tokens, float scale, bool renormalize,
                             MetadataArgs meta) {
  __shared__ MetadataSmem smem;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const bool with_meta = meta.data != nullptr;
  if (with_meta && threadIdx.x < meta.num_local) {
    smem.tokens[threadIdx.x] = 0u;
  }
  // float4 views of fp32 rows: the launcher requires 16-byte aligned bias and
  // logits rows.
  const float4* b4 = reinterpret_cast<const float4*>(bias) + lane * 2;
  const float4 bA = b4[0], bB = b4[1];
  pdl_wait();
  // tokens[] is zero before any warp sets a bit.
  __syncthreads();

  if (warp < num_tokens) {
    const Pick pick = select_top8(logits + warp * logits_stride, bA, bB, lane,
                                  scale, renormalize);
    if (lane < kTopK) {
      ids[warp * kTopK + lane] = pick.id;
      weights[warp * kTopK + lane] = pick.weight;
      if (with_meta) {
        smem.ids[warp * kTopK + lane] = pick.id;
        const int e = local_expert(pick.id, meta);
        if (e >= 0) {
          atomicOr(&smem.tokens[e], 1u << warp);
        }
      }
    }
  }
  // with_meta is a kernel argument: the same in every thread, so every thread
  // reaches each barrier below or none does.
  if (!with_meta) {
    return;
  }
  // The selections (tokens[], ids[]) are complete.
  __syncthreads();
  write_expert_metadata(smem, meta);
  // cta_offset[] is complete.
  __syncthreads();
  write_route_metadata(smem, meta, num_tokens);
}

inline void check_cuda(cudaError_t err, const char* what) {
  STD_TORCH_CHECK(err == cudaSuccess, what,
                  " failed: ", cudaGetErrorString(err));
}

inline bool aligned16(const torch::stable::Tensor& t) {
  return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}

inline bool on_device(const torch::stable::Tensor& t, int device) {
  return t.is_cuda() && t.get_device_index() == device;
}

// The checked metadata arguments, or {nullptr, ...} when metadata is absent.
inline MetadataArgs metadata_args(
    std::optional<torch::stable::Tensor> const& metadata, int device,
    int64_t local_expert_offset, int64_t num_local_experts, int64_t tile_n) {
  if (!metadata.has_value()) {
    return {nullptr, 0, 0, 0};
  }
  using torch::headeronly::ScalarType;
  const torch::stable::Tensor& m = metadata.value();
  STD_TORCH_CHECK(tile_n >= 1 && tile_n <= kMaxTile, "tile_n must be 1 to ",
                  kMaxTile, ", got ", tile_n);
  STD_TORCH_CHECK(num_local_experts >= 1 && local_expert_offset >= 0 &&
                      local_expert_offset + num_local_experts <= kExperts,
                  "local experts [", local_expert_offset, ", ",
                  local_expert_offset + num_local_experts, ") outside the ",
                  kExperts, " experts");
  STD_TORCH_CHECK(
      m.dim() == 1 && m.scalar_type() == ScalarType::Int && m.is_contiguous() &&
          aligned16(m),
      "metadata must be a contiguous, 16-byte aligned int32 vector");
  STD_TORCH_CHECK(m.size(0) >= metadata_size(tile_n), "metadata needs ",
                  metadata_size(tile_n), " elements for tile_n ", tile_n,
                  ", got ", m.size(0));
  STD_TORCH_CHECK(on_device(m, device),
                  "metadata must be on the logits' CUDA device");
  return {static_cast<int*>(m.mutable_data_ptr()),
          static_cast<int>(local_expert_offset),
          static_cast<int>(num_local_experts), static_cast<int>(tile_n)};
}

}  // namespace muse_spark_routing
}  // namespace vllm

// ids [b, 8] (global expert ids) and weights [b, 8] of the top-8 of
// sigmoid(logits) + bias for logits fp32 [b, 256], 1 <= b <= 8. With
// `metadata` (int32, at least metadata_size(tile_n) elements, not aliasing
// ids or weights), also the routing metadata of the local experts
// [local_expert_offset, local_expert_offset + num_local_experts) for GEMM
// tile tile_n, at the offsets kMeta* above.
void muse_spark_routing_shorttail(
    torch::stable::Tensor const& logits, torch::stable::Tensor const& bias,
    torch::stable::Tensor& ids, torch::stable::Tensor& weights, double scale,
    bool renormalize, std::optional<torch::stable::Tensor> const& metadata,
    int64_t local_expert_offset, int64_t num_local_experts, int64_t tile_n) {
  using namespace vllm::muse_spark_routing;
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
  const int dev = logits.get_device_index();
  STD_TORCH_CHECK(
      logits.dim() == 2 && logits.scalar_type() == ScalarType::Float &&
          logits.size(1) == kExperts && logits.stride(1) == 1,
      "logits must be fp32 [b, ", kExperts, "] with unit column stride");
  STD_TORCH_CHECK(aligned16(logits) && logits.stride(0) % 4 == 0,
                  "logits rows must be 16-byte aligned");
  const int64_t num_tokens = logits.size(0);
  STD_TORCH_CHECK(num_tokens >= 1 && num_tokens <= kMaxTokens,
                  "the routing short tail takes 1 to ", kMaxTokens,
                  " tokens, got ", num_tokens);
  STD_TORCH_CHECK(
      bias.dim() == 1 && bias.scalar_type() == ScalarType::Float &&
          bias.size(0) == kExperts && bias.is_contiguous() && aligned16(bias),
      "bias must be fp32 [", kExperts, "], contiguous and 16-byte aligned");
  STD_TORCH_CHECK(ids.dim() == 2 && ids.scalar_type() == ScalarType::Int &&
                      ids.size(0) == num_tokens && ids.size(1) == kTopK &&
                      ids.is_contiguous(),
                  "ids must be int32 [b, ", kTopK, "] and contiguous");
  STD_TORCH_CHECK(weights.dim() == 2 &&
                      weights.scalar_type() == ScalarType::Float &&
                      weights.size(0) == num_tokens &&
                      weights.size(1) == kTopK && weights.is_contiguous(),
                  "weights must be fp32 [b, ", kTopK, "] and contiguous");
  STD_TORCH_CHECK(
      on_device(bias, dev) && on_device(ids, dev) && on_device(weights, dev),
      "logits, bias, ids and weights must be on one CUDA device");
  const MetadataArgs meta = metadata_args(metadata, dev, local_expert_offset,
                                          num_local_experts, tile_n);

  const torch::stable::accelerator::DeviceGuard device_guard(dev);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(1, 1, 1);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = get_current_cuda_stream(dev);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  check_cuda(cudaLaunchKernelEx(&cfg, routing_shorttail_kernel,
                                static_cast<const float*>(logits.data_ptr()),
                                logits.stride(0),
                                static_cast<const float*>(bias.data_ptr()),
                                static_cast<int*>(ids.mutable_data_ptr()),
                                static_cast<float*>(weights.mutable_data_ptr()),
                                static_cast<int>(num_tokens),
                                static_cast<float>(scale), renormalize, meta),
             "muse_spark_routing_shorttail launch");
}
