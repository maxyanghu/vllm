# Die-Local Sparse Indexer and Indexer Overlap

Two opt-in decode optimizations for the sparse-attention indexer of Kimi K3.1 and
DeepSeek-V4.1 (DeepSeek Sparse Attention). Both are off by default; with both
unset, vLLM runs the stock indexer.

| Environment variable | Default | What it does |
| -------------------- | ------- | ------------ |
| `VLLM_INDEXER_LOCALIZE` | `0` | Die-local indexer decode on GPUs with two memory locality domains |
| `VLLM_INDEXER_OVERLAP` | `0` | Overlap the indexer's top-k with independent work of the same layer |

```bash
VLLM_INDEXER_LOCALIZE=1 VLLM_INDEXER_OVERLAP=1 vllm serve <Kimi K3.1 or DeepSeek-V4.1 checkpoint> ...
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
  schedules. The first eager build of each indexer cache group, for at most 32
  rows and for more, is checked against a reference build, bit for bit; a batch
  size captured before its check runs the stock indexer.
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
split and the KV cache placement are also off for models other than
`KimiK31ForConditionalGeneration` and `DeepseekV41ForCausalLM`, with speculative
decoding, decode or prefill context parallelism, a KV connector, and with the
CuMem allocator on a plain (non-extensible) KV cache.

When active, each rank logs `KV cache placed in ... runs alternating between 2
locality domains` and, once per indexer cache group, `Die-local sparse indexer
active for N layers`. Each fallback is logged once as a warning.

## Indexer overlap (`VLLM_INDEXER_OVERLAP`)

Decode steps captured as FULL CUDA graphs reorder independent work of the same
layer around the indexer; the kernels and their inputs are unchanged, so the
outputs are bit-identical. Eager steps and piecewise CUDA graphs keep the stock
order.

- Kimi K3.1: the indexer's scan and top-k run on another stream (with the die
  split, on the domains' streams) and the layer issues the query projections,
  the output gate, the MLA cache insert and the query preparation meanwhile;
  sparse attention waits for the kept tokens. The indexer's key projection runs
  on a side stream beside the query chain.
- DeepSeek-V4.1: the candidate source layer selects its candidate blocks, and
  expands them into the sparse block ids of its first consumer layer, on a side
  stream that the consumer joins. The main query projection and cache inserts
  run beside the indexer, and the indexer query beside the index key. With the
  die split, the candidate source layer also keeps its top-k per domain and
  joins the two score arrays as part of the deferred candidate work.

The overlap works with and without `VLLM_INDEXER_LOCALIZE`. It is off with
micro-batching (`--enable-dbo` or `--ubatch-size`), and for Kimi K3.1 with an
attention backend whose top-k conversion does not go through the shared sparse
MLA index group.
