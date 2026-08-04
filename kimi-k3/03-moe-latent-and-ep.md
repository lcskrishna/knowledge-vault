# 03 — Latent MoE, Routing, and Expert Parallelism

`KimiK3MoE`, `python/sglang/srt/models/kimi_k3.py:354-1205`.
896 routed experts + 1 shared expert, top-16, SiTU activation, and — the K3 twist — the routed
experts run in a **3584-wide latent space** while the residual stream is 7168.

---

## 1. The MoE block

Compare against the DSV3 "MoELayer" box: same MoEGate → TopK → dispatch/experts/combine
skeleton, with a down/up projection wrapped around it.

```
   hidden_states [T, 7168]
        │
        ├──────────────────────────────► shared_experts (KimiK3MLP, 7168 space)
        │                                 SiTU, tp1-replicated under EP a2a
        │                                 issued on alt_stream (SBO) to overlap the routed a2a
        │
        ├── gate (MoEGate, fp32 logits) ──► TopK: grouped top-k + e_score_correction_bias
        │                                          (DSv3 "noaux_tc"), sigmoid router
        │
        └── routed_expert_down_proj [7168 -> 3584]   (ReplicatedLinear, bf16)
                    │
                    ▼
              [optional] routed_expert_norm (RMSNorm 3584, latent_moe_use_norm)
                    │
                    ▼
            ┌───────────────────────────────────────────┐
            │  Expert a2a + expert GEMMs                │
            │   MegaMoE (deep_gemm fused a2a+GEMM)      │
            │   | DeepEP dispatch/combine               │
            │   | plain TP fused MoE                    │
            │   experts: w1/w3 -> SiTU -> w2, in 3584   │
            └───────────────────┬───────────────────────┘
                                ▼
                    latent [T, 3584]  (reduced)
                                ▼
              routed_expert_up_proj [3584 -> 7168]  (ReplicatedLinear)
                                ▼
              _add3(routed_out, shared_out, prefix_sum)   ← attn-res prefix folded in here
```

**Why the latent MoE matters operationally:** the a2a payload is halved (3584 vs 7168 per
token), which is exactly why the DeepEP patch has to add `case 3584` to `SWITCH_HIDDEN` — stock
DeepEP has no template instantiation for that width. It also means the down/up projections are
replicated (`ReplicatedLinear`), so the routed output is *already fully reduced* when it leaves
`routed_expert_up_proj` (`kimi_k3.py:960`).

### 1.1 Router

`MoEGate` is imported from `deepseek_v2` and constructed with `quant_config=None` so router
logits stay fp32 through sigmoid + bias add + top-k (`kimi_k3.py:388-390`). `TopK`
(`424-450`) uses `use_grouped_topk=True`, `num_expert_group` / `topk_group`,
`correction_bias=self.gate.e_score_correction_bias`, `renormalize=moe_renormalize`, and
`routing_method_type=RoutingMethodType.DeepSeekV3` for the trtllm fused-routing backends that
route inside the kernel.

The `TopKOutputFormat` choice is backend-dependent (`439-449`): STANDARD when unquantized or on
the flashinfer-mxfp4 + SiTU path (which consumes precomputed packed routing), raw ids/weights
for MegaMoE pre-dispatch, otherwise runner-resolved.

### 1.2 SiTU experts

`hidden_act = "situ"`, `SituAndMul` (`srt/layers/activation.py:184-212`):

```
gate = beta * tanh(gate/beta) * sigmoid(gate)     # beta = 4.0
up   = linear_beta * tanh(up/linear_beta)         # linear_beta = 25.0  (soft clip)
out  = gate * up
```

The constants are asserted, not read, on the MegaMoE path (`kimi_k3.py:465-471`) because the
DeepGEMM mega kernel bakes them in — a checkpoint with different betas is rejected at init
rather than silently mis-activated.

CUDA implementation: `kernels/ops/kimi_k3/activation.py::situ_and_mul`, plus
`kernels/ops/kimi_k3/moe.py::situ_and_mul_masked_post_quant` which fuses SiTU with the masked
post-quantization for the mxfp8 grouped path.

### 1.3 Shared expert

One shared expert in the full 7168 space, `moe_intermediate_size * num_shared_experts` wide.
Under EP a2a it is built **tp1-replicated** (`_shared_experts_tp1`, `kimi_k3.py:505`): the MoE
region runs on a token shard or DP-local rows, and a TP-sharded partial sum could never be
reduced across ranks holding different tokens.

**SBO (single batch overlap)**, `kimi_k3.py:522-539`: the shared experts read a fixed ~264 MB/
layer/rank bf16 slab the routed path never touches, while the routed path is a2a-latency bound
in decode with HBM mostly idle. So the shared branch is issued on the alt stream and joined
just before the tail add. Measured on 2x4 GB300 (TP8/EP8 MegaMoE + SP-MoE): **+4–5% output
tok/s, −5% ITL** over bs 1–32, GSM8K unchanged — so it is unconditional, no flag. It is
deliberately issued *after* the front GEMMs so it overlaps the a2a rather than stealing
bandwidth from the critical path (`891-898`).

### 1.4 Fused front

Three GEMMs read the same `hidden_states`: shared `gate_up`, router `gate`, latent `down_proj`.
At decode each is a skinny memory-bound GEMV with its own splitK epilogue.
`_merge_front_weights` (`593`) merges them into one `[H, gu + E + latent]` GEMM — one read of
the input, two fewer launches and splitK tails per MoE layer. Only plain bf16/fp16 weights are
merged; quantized or mixed-dtype checkpoints keep the unfused path.

`_front_is_ep_pair` covers the narrower merge of just `[gate, routed_expert_down_proj]` (the EP
a2a pair) when the three-way merge does not apply.

---

## 2. Expert parallelism / all-to-all

`get_moe_a2a_backend()` selects one of three (`kimi_k3.py:460`, `478-479`):

| Backend | Path | Notes |
|---|---|---|
| **MegaMoE** | `_forward_mega_experts` (`657-742`), `deep_gemm.fp8_fp4_mega_moe` | fused a2a + grouped GEMM over the EP symmetric buffer. K3 routes **all** batches through it when enabled — the megamoe fallback is a `StandardDispatcher` with no a2a, which is wrong for scattered tokens — so `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` must cover the per-rank prefill chunk. SiTU is selected inside the kernel by a **sentinel `activation_clamp = 2^-5`**. Requires latent MoE + `situ` + exact betas. |
| **DeepEP** | standard SGLang DeepEP dispatch/combine | needs the K3 patch set (see below and doc 05) |
| **none** | plain TP fused MoE + `tensor_model_parallel_all_reduce` | `_routed_needs_reduce` |

Under either EP a2a backend the MoE region consumes **whatever rows the rank holds** — an
SP-MoE token shard or the DP-local batch — with every global token dispatched exactly once. No
DP gather and no TP reduce anywhere in the region (`473-477`).

### DeepEP K3 patches

`docker/kimi_k3/apply_deepep_k3_patch.sh` — stock DeepEP (`deepseek-ai@d28bd67`) cannot serve K3:

| Patch | Reason |
|---|---|
| `internode_ll.cu`: `kNumMaxTopK` 11 → 16 | K3 routes top-16 |
| `launch.cuh`: add `case 3584` to `SWITCH_HIDDEN` | latent-MoE dispatch width |
| `internode.cu`: field-wise SourceMeta scatter/gather | unaligned 64-bit access for packed FP8 scales when EP > 8 (crossing the 8-rank NVL domain) |
| `configs.cuh`: CPU timeout 100s → 1000s, cycles 2e11 → 2e12 | cross-node init headroom |
| `setup.py`: add `/usr/local/cuda/include/cccl` | CUDA 13 header relocation |

Rebuilt for `sm_90` / `sm_100a` / `sm_103a`.

---

## 3. SP-MoE — sequence-parallel MoE region

`kimi_k3.py:1904-1926` (the longest comment in the file, and the most load-bearing).

When an EP a2a backend is active **and** `attn_tp > 1`, the layer flips `o_proj.reduce_results`
off and completes the attention reduction as a **reduce-scatter** instead of an all-reduce. Then:

- the entire MoE region — aggregation 2, norms, gate, latent projections, tp1 shared experts,
  a2a dispatch — runs on `1/attn_tp` of the rows
- rows are all-gathered back after the MoE tail add

The accounting: RS + AG moves the same bytes the `o_proj` all-reduce did; the shared-expert
all-reduce disappears via tp1 weights; each rank dispatches only its shard through the a2a,
killing the `attn_tp`-fold dispatch redundancy. Net: **strictly less communication, and the
MoE front compute divided by `attn_tp`**.

Dense layers are excluded — a column-parallel MLP has no per-token decomposition that survives
a token shard.

`SGLANG_K3_SP_ATTN_RES` extends this by carrying the *raw attention-residual stream* as a token
shard across consecutive SP-MoE layers (`kimi_k3.py:2470-2476`), instead of gathering every
layer. Disabled under PP > 1 and under DSPARK capture, both of which need full tensors.

---

## 4. Quantization

| Component | Format |
|---|---|
| routed experts | checkpoint-dependent: mxfp4 (auto-swapped to `Mxfp4Config` when `quant_format` contains `mxfp4`, `kimi_k3.py:395-399`), nvfp4 w4a4, or marlin W4A16 on Hopper/ROCm. `gate_up_interleaved=False` — K3 stores per-expert w1/w3 non-interleaved |
| router gate | fp32 logits, never quantized |
| shared experts | bf16, unquantized in the checkpoint |
| latent down/up proj | bf16 `ReplicatedLinear`, `quant_config=None` |
| attention linears | resolve to `UnquantizedLinearMethod` — only the MoE experts are quantized (`kimi_k3.py:1260-1262`) |
| activations | mxfp8 group-quant, group size 32, JIT strided-input quant on the trtllm-gen path so the fused-front split view is consumed without a contiguous copy (`481-487`) |

**Deferred MoE finalize** (`_forward_routed_deferred`, `1002-1013`): on the
flashinfer-mxfp4 + SiTU path the top-k weighted unpermute is skipped inside the MoE op and
folded into the push all-reduce's staging pass (`k3_ar_fusion.finalize_all_reduce_push_norm`),
so the rank-local latent never materializes. Falls back to the in-op finalize when the batch
exceeds the push window (`finalize_push_fits`).
