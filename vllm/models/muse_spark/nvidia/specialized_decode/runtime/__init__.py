# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark specialized decode: a decode executor over the stock backbone, enabled
by ``additional_config={"decode_backend": "latency_specialized"}``."""

from dataclasses import dataclass

# The one text config the providers are built for.
MUSE_SPARK_CONFIG = {
    "hidden_size": 8192,
    "num_attention_heads": 128,
    "num_key_value_heads": 16,
    "head_dim": 64,
    "num_local_experts": 256,
    "num_experts_per_tok": 8,
    "moe_hidden_size": 4096,
    "num_shared_experts": 0,
}
TP_SIZES = (4, 8, 16)


@dataclass(frozen=True)
class DecodeConfig:
    tp: int
    # Tokens per request in a verify step (1 + draft tokens).
    q: int
    max_context: int
    capture_sizes: tuple[int, ...]
    # The NVFP4 routed experts on the persistent expert kernel
    # (cutedsl/moe/persistent_experts.py) instead of TRT-LLM-gen's GEMMs, on
    # the stock EP experts (no expert TP). See persistent_experts().
    persistent_experts: bool = True
    # The full-attention layers' decode attention on the stock path's
    # TRT-LLM-gen decode + o-norm/gate instead of attn2. See
    # stock_full_attention().
    stock_full_attention: bool = False
    # REPLICAS replicas of the next EP rank's experts on every rank's
    # persistent expert kernel, each step routing an expert to the less loaded
    # of its ranks. See expert_replicas().
    expert_replicas: bool = False


def decode_backend(vllm_config) -> str:
    """The requested ``decode_backend`` (``additional_config``), "auto" if unset."""
    extra = vllm_config.additional_config
    return extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"


def specialized_decode_config(vllm_config) -> DecodeConfig | None:
    """The deployment's specialized decode settings, None for the stock decode.
    Raises when the specialized decode is requested for an unsupported deployment."""
    backend = decode_backend(vllm_config)
    if backend == "auto":
        return None
    if backend != "latency_specialized":
        raise ValueError("Supported decode_backend values are auto/latency_specialized")
    reason = unsupported(vllm_config)
    if reason is not None:
        raise ValueError(f"Muse Spark specialized decode does not support {reason}")
    return DecodeConfig(
        tp=vllm_config.parallel_config.tensor_parallel_size,
        q=1 + vllm_config.speculative_config.num_speculative_tokens,
        max_context=vllm_config.model_config.max_model_len,
        capture_sizes=tuple(
            vllm_config.compilation_config.cudagraph_capture_sizes or ()
        ),
        persistent_experts=persistent_experts(vllm_config),
        stock_full_attention=stock_full_attention(vllm_config),
        expert_replicas=expert_replicas(vllm_config),
    )


# max_model_len from which the full-attention layers run the stock path's
# TRT-LLM-gen decode by default, per TP size. The kernel is fixed per captured
# graph (the runner keys graphs by token count only; a per-step choice through
# CUDA graph conditional nodes costs ~9 us per node, ~0.28 ms per step, more
# than it saves). 16 full layers in one graph on GB300, us per step, attn2 vs
# TRT-LLM-gen: TP4 16k 275 / 292, 32k 435 / 327, 100k 1174 / 523; TP8 32k
# 278 / 358, 100k 646 / 427 (TRT-LLM-gen ~190 at 2k at both, attn2 61-71). So
# the switch sits above the crossover (TP4 ~17k keys, TP8 ~45k); TP16 (one KV
# head per rank) is unmeasured and keeps attn2.
STOCK_FULL_ATTENTION_MIN_CONTEXT = {4: 32768, 8: 65536}


def stock_full_attention(vllm_config) -> bool:
    """Whether the full-attention layers run the stock decode attention: when
    max_model_len is at least STOCK_FULL_ATTENTION_MIN_CONTEXT for the TP
    size, or as ``additional_config`` sets ``"stock_full_attention"`` (true
    or false). Reads the config only, so every rank agrees."""
    extra = vllm_config.additional_config
    tp = vllm_config.parallel_config.tensor_parallel_size
    threshold = STOCK_FULL_ATTENTION_MIN_CONTEXT.get(tp)
    default = (
        threshold is not None and vllm_config.model_config.max_model_len >= threshold
    )
    wanted = (
        extra.get("stock_full_attention", default)
        if isinstance(extra, dict)
        else default
    )
    if not isinstance(wanted, bool):
        raise ValueError("additional_config stock_full_attention must be a bool")
    return wanted


# The TP sizes whose NVFP4 experts run the persistent expert kernel by default.
# At TP8 the expert-TP experts (balanced over the ranks) measured as fast or
# faster on GB300 (c1 decode step 5.59 / 6.12 / 6.68 ms at 2k / 100k / 300k vs
# 5.85 / 6.23 / 6.59 with EP8 + the persistent kernel), so TP8 keeps them.
PERSISTENT_EXPERTS_TPS = (4,)


def persistent_experts(vllm_config) -> bool:
    """Whether the NVFP4 routed experts run on the persistent expert kernel:
    on SM103 at PERSISTENT_EXPERTS_TPS, or where ``additional_config`` sets
    ``"persistent_experts"`` (true or false). Reads the config only, so every
    rank agrees."""
    from vllm.platforms import current_platform

    tp = vllm_config.parallel_config.tensor_parallel_size
    extra = vllm_config.additional_config
    default = tp in PERSISTENT_EXPERTS_TPS
    wanted = (
        extra.get("persistent_experts", default) if isinstance(extra, dict) else default
    )
    if not isinstance(wanted, bool):
        raise ValueError("additional_config persistent_experts must be a bool")
    return wanted and current_platform.is_device_capability(103)


# The TP sizes whose persistent expert kernel runs with replicas by default
# (with the persistent experts on): TP4 c1 decode step -0.24 to -0.33 ms on
# GB300 at 2k-300k (DSpark K7, AL 4.05), the KV cache 5.77M -> 2.82M tokens.
EXPERT_REPLICAS_TPS = (4,)


def expert_replicas(vllm_config) -> bool:
    """Whether the persistent expert kernel runs a window of the rank's experts
    plus replicas of the next rank's (runtime/replicas.py): by default at
    EXPERT_REPLICAS_TPS when the persistent experts are on, or as
    ``additional_config`` ``"expert_replicas"`` sets it. Costs REPLICAS
    experts of memory per NVFP4 layer and rank (~51 GiB at TP4). Reads the
    config only, so every rank agrees."""
    extra = vllm_config.additional_config
    tp = vllm_config.parallel_config.tensor_parallel_size
    default = tp in EXPERT_REPLICAS_TPS and persistent_experts(vllm_config)
    wanted = (
        extra.get("expert_replicas", default) if isinstance(extra, dict) else default
    )
    if not isinstance(wanted, bool):
        raise ValueError("additional_config expert_replicas must be a bool")
    if wanted and not persistent_experts(vllm_config):
        raise ValueError("additional_config expert_replicas needs persistent_experts")
    if wanted and tp not in (4, 8):
        # The window (256 / tp + 32 experts) must be one the kernel compiles.
        raise ValueError("additional_config expert_replicas needs TP 4 or 8")
    return wanted


def unsupported(vllm_config) -> str | None:
    """The first reason this deployment cannot run the specialized decode, or None.

    Every line reads the config only, so every rank gets the same answer. The
    providers' own requirements (attention backend, context length, all-reduce
    backend) follow the deployment lines.
    """
    import torch

    from vllm import envs
    from vllm.config import CompilationMode, CUDAGraphMode
    from vllm.platforms import current_platform

    from .decode import provider_requirements

    parallel, spec = vllm_config.parallel_config, vllm_config.speculative_config
    model, cache = vllm_config.model_config, vllm_config.cache_config
    compilation = vllm_config.compilation_config
    graphs = compilation.cudagraph_mode
    tp = parallel.tensor_parallel_size
    kv_dtype = model.dtype if cache.cache_dtype == "auto" else cache.cache_dtype
    draft_model = None if spec is None else spec.draft_model_config
    draft = None if draft_model is None else draft_model.hf_config
    checks = (
        (not current_platform.is_device_capability_family(100), "this GPU (SM100/103)"),
        (
            any(
                getattr(model.hf_text_config, name, None) != value
                for name, value in MUSE_SPARK_CONFIG.items()
            ),
            "text configs but Muse Spark Open's",
        ),
        (tp not in TP_SIZES, "TP other than 4/8/16"),
        (parallel.pipeline_parallel_size != 1, "pipeline parallelism"),
        (
            not parallel.enable_expert_parallel
            or parallel.data_parallel_size * tp != tp,
            "expert parallelism other than EP = TP",
        ),
        (parallel.use_ubatching, "DBO"),
        (parallel.decode_context_parallel_size != 1, "decode context parallel"),
        (model.enable_sleep_mode, "sleep mode"),
        (vllm_config.lora_config is not None, "LoRA"),
        (
            vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe",
            "the MegaMoE backend",
        ),
        # The runner's ModelState hooks build the executor and bind the KV caches.
        (not vllm_config.use_v2_model_runner, "the V1 model runner"),
        # The decode loop runs in forward; a compiled forward would trace it.
        # The DSpark draft's fused layers are exact to its eager layers only,
        # and a compiled draft would trace their call checks.
        (
            compilation.mode != CompilationMode.NONE,
            "torch.compile (compilation modes but NONE)",
        ),
        # The loop runs only in FULL decode graphs; eager and PIECEWISE steps run stock.
        (
            graphs is None or graphs.decode_mode() != CUDAGraphMode.FULL,
            "deployments without FULL decode CUDA graphs (e.g. enforce-eager)",
        ),
        (
            spec is None or spec.method != "dspark",
            "speculative methods but DSpark",
        ),
        (
            spec is not None and spec.enable_adaptive_verification,
            "adaptive verification",
        ),
        # Verify steps are one request of 8 tokens.
        (
            spec is not None and spec.num_speculative_tokens != 7,
            "speculative token counts but 7",
        ),
        # The draft runs the fused Markov loop (markov.py), which the draft
        # model checks again at load against its weights.
        (
            spec is not None and spec.draft_sample_method != "probabilistic",
            "draft sampling methods but probabilistic",
        ),
        (
            spec is not None and model.head_dtype != torch.bfloat16,
            "draft logits but BF16 (head_dtype)",
        ),
        (
            spec is not None and model.use_fp64_gumbel,
            "FP64 Gumbel sampling",
        ),
        (getattr(vllm_config, "watermark_config", None) is not None, "watermarking"),
        (getattr(draft, "dspark_draft_topk", None) is not None, "a DSpark draft top-k"),
        (
            not getattr(draft, "sample_from_anchor", True),
            "DSpark drafts without sample_from_anchor",
        ),
        (
            getattr(draft, "draft_vocab_size", None)
            not in (None, getattr(draft, "vocab_size", None)),
            "a reduced DSpark draft vocabulary",
        ),
        # BF16, or an fp8 (e4m3) cache on the full-attention layers only: the
        # sliding layers (and the DSpark draft's, all sliding) stay BF16 for
        # attn2 and the draft's fused stores.
        (
            kv_dtype not in ("bfloat16", torch.bfloat16)
            and not (
                kv_dtype in ("fp8", "fp8_e4m3")
                and "sliding_window" in cache.kv_cache_dtype_skip_layers
            ),
            (
                "KV cache dtypes but BF16 or fp8_e4m3 with "
                "--kv-cache-dtype-skip-layers sliding_window"
            ),
        ),
        (cache.block_size != 16, "KV cache block sizes but 16"),
        (vllm_config.kv_transfer_config is not None, "KV transfer"),
        (envs.VLLM_BATCH_INVARIANT, "batch-invariant mode"),
    )
    reason = next((reason for failed, reason in checks if failed), None)
    return reason if reason is not None else provider_requirements(vllm_config)
