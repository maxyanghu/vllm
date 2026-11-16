# Die-Local Sparse Indexer

An opt-in decode optimization for the sparse-attention indexer of Kimi K3.1 and
DeepSeek-V4.1 (DeepSeek Sparse Attention), off by default; when unset, vLLM runs
the stock indexer.

| Environment variable | Default | What it does |
| -------------------- | ------- | ------------ |
| `VLLM_INDEXER_LOCALIZE` | `0` | Die-local indexer decode on GPUs with two memory locality domains |

```bash
VLLM_INDEXER_LOCALIZE=1 vllm serve <Kimi K3.1 or DeepSeek-V4.1 checkpoint> ...
```

## Die-local indexer (`VLLM_INDEXER_LOCALIZE`)

On GPUs whose memory and SMs form two locality domains (dies), such as NVIDIA
Rubin, a die reads its own memory faster than the other die's. In decode, the
indexer scans every cached index key of every request, so it is bound by memory
bandwidth. With `VLLM_INDEXER_LOCALIZE=1`:

- The KV cache is placed in runs of whole KV blocks, close to 32 MiB each, that
  alternate between the two locality domains. Block ids and the cache-insert
  kernels are unchanged; this applies to the extensible KV cache (the default)
  and to the plain allocation.
- Every decode step, one kernel (captured in the decode CUDA graph) splits each
  request's index-key pages by domain and builds both domains' DeepGEMM
  schedules. The first build of each indexer cache group is checked against a
  reference build, bit for bit.
- Each domain's SMs (a green context per domain) scan the pages in their own
  memory with DeepGEMM's `fp8_fp4_paged_mqa_logits`. For batches of at most 64
  rows with top-k 512, 1024 or 2048, each domain also runs the top-k on its own
  scores and one kernel merges the two candidate lists; otherwise the two score
  arrays are joined into the stock full-length array and the stock top-k runs.

The scores are bit-identical to the single-launch scan, and the kept tokens have
the same score multiset as the stock top-k (ties may keep other tokens). The CUDA
kernels ship as source and are compiled with NVRTC at their first use, in well
under a second.

Requirements: a GPU with two locality domains, DeepGEMM, `cuda-bindings` 13.4 or
later (driver API and NVRTC), and a PyTorch build with locality-domain green
contexts and `torch.cuda.execute_on_streams` (nightly from 2026-10-08, 2.16).
Where any of these is missing, vLLM logs why and runs the stock indexer. The die
split is also off with speculative decoding, decode or prefill context
parallelism, a KV connector, and for other models.

When active, each rank logs `KV cache placed in ... runs alternating between 2
locality domains` and, once per indexer cache group, `Die-local sparse indexer
active for N layers`. Each fallback is logged once as a warning.
