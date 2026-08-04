# 02 — K3 Attention: KDA (linear) + MLA (NoPE)

93 layers, hybrid: **69 KDA** recurrent layers and **24 MLA** latent-attention layers.
The split is what makes K3 cheap to serve at long context — only 24 layers pay a per-token KV
cost; the other 69 carry a fixed-size recurrent state per sequence.

---

## 1. KDA — Kimi Delta Attention

`KimiK3DeltaAttention`, `python/sglang/srt/models/kimi_k3.py:1206-1700`.
Shapes: 96 heads, `head_dim` 128, short-conv kernel 4, `head_v_dim = config.v_head_dim`.

### 1.1 Projections

```
hidden_states [T, 7168]
   ├── q, k, v      : head_dim*num_heads each         (qkv_proj / fused_qkvg_proj)
   ├── beta  (b)    : num_heads   scalars per token   (b_proj)
   ├── forget (f)   : f_a_proj [H->128] -> f_b_proj [128 -> heads*head_dim]   (low-rank)
   └── gate  (g)    : full-rank g_proj [H -> heads*head_dim]     if use_full_rank_gate  (K3)
                      else low-rank g_a_proj -> g_b_proj                       (K2.5-style)
   plus: qkv_conv1d  [3*proj, kernel=4]  fp32   (short depthwise conv over q,k,v)
         A_log       [1,1,local_heads,1] fp32   (per-head decay)
         dt_bias     [proj/attn_tp]      fp32
```

`use_full_rank_gate` is K3-specific (`kimi_k3.py:1255`, listed in the file header as one of the
five K3 features). Three construction variants exist (`1263-1412`):

| Variant | Condition | Layout |
|---|---|---|
| fused q/k/v/g | `attn_tp == tp` and full-rank gate | one `MergedColumnParallelLinear` over `[q,k,v,g]` (6144/rank at TP8). `b` (12/rank) and `f_a` (128, replicated) deliberately stay separate — folding them in skews the output dim to 6284 and degrades GEMM kernel selection (`1267-1271`) |
| fused q/k/v/b + f/g | `attn_tp == tp`, low-rank gate | `MergedColumnParallelRepeatedLinear` + `ColumnParallelBatchedLinear` |
| unfused | otherwise | separate `qkv_proj`, `b_proj`, `f_a/f_b`, `g_a/g_b` |

Head sharding follows the **attention-TP group** (`attn_tp_size`), not global TP, so it matches
the state-cache sizing in `KimiLinearCacheParams` (`kimi_k3.py:1234-1251`).

### 1.2 The recurrence

Per head, state `S ∈ R^{128x128}`. Reading the fused decode kernel
(`python/sglang/kernels/ops/kimi_k3/kda_decode_mtp.py`) plus the FLA path:

```
q,k,v  = short_conv(q,k,v)                # 4-tap causal depthwise conv, then activation
beta   = sigmoid(b)                       # per-head scalar in (0,1)
decay  = exp(-softplus(f + dt_bias) * exp(A_log))   # per-head/per-channel gate, lower-bounded
                                                     # (gate_lower_bound = -5.0 in the ckpt)
# delta rule, per token:
S      = decay * S                        # gated forget
S      = S + beta * (v - S @ k) k^T       # delta correction  (write only the residual error)
o      = S @ q
o      = FusedRMSNormGated(o, g)          # o * rms(o) * sigmoid(g)
out    = o_proj(o)
```

The delta term `(v - S k)` is the "delta rule": instead of appending `v k^T` (plain linear
attention), it writes only the part of `v` the state does not already predict — which is what
gives the family its associative-recall behaviour.

`self.attn.lower_bound = linear_attn_config["gate_lower_bound"]` (`kimi_k3.py:1511`) — the
checkpoint was trained with `-5.0`, so the gate is clamped to keep the recurrence stable.

### 1.3 Prefill vs decode

Dispatch goes through `RadixLinearAttention` (`kimi_k3.py:1497`), which the hybrid backend
(`layers/attention/hybrid_linear_attn_backend.py`) routes per layer:

- **Extend/prefill** — chunked scan (`kernels/ops/attention/linear/kda_nvidia_prefill/*`:
  `chunk_fwd.py`, `fuse_kernel123_persistent.py`, `Akk_inverse_lower_triangle_bf16.py`).
  `beta` is pre-sigmoided on the host side (`kimi_k3.py:1665`), `forget_gate` is unflattened to
  `[.., heads, head_dim]`.
- **Decode / target-verify** — fused recurrent CuTe kernel
  (`kernels/ops/attention/kda_fused_decode.py`, `kernels/ops/kimi_k3/kda_decode_mtp.py`).
  `beta` is sigmoided *in-kernel*.

**Fused output norm handoff** (`kimi_k3.py:1669-1694`): the module stashes the gate on the
attention object (`attn._k3_onorm_gate`) and lets the kernel fold `FusedRMSNormGated` into the
recurrence epilogue. It is an attempt-and-verify protocol — if the backend leaves
`_k3_onorm_consumed` false (env off, or shape not covered), the module applies `o_norm` itself.
This pattern shows up repeatedly in K3: offer a fusion, check whether it was taken, otherwise
run the generic path.

### 1.4 State cache

`KimiLinearCacheParams` / `KimiLinearStateShape` (`srt/configs/mamba_utils.py`), built from
`linear_attn_config` and `get_parallel().attn_tp_size`:

- **conv state**: `[kernel-1 = 3, channels]` bf16 — sliced on axis 1, which disables dedup
- **temporal state**: `[num_heads/attn_tp, 128, 128]`, fp32 by default

These live in the hybrid (mamba-style) pool alongside the MLA KV pool. Sizing the two pools
against each other is the whole point of `--mamba-full-memory-ratio`; the docs snippet
`_kimi_k3_mamba_ratio_calculator.jsx:118-122` writes the balance as

```
r = (S + D) * state_bytes / (L * per_token_kv_bytes)
state_bytes  = 69 * ((96/attnTp) * 128 * 128 * ssmBytes + 3 * 3 * (96/attnTp) * 128 * 2)
kv_bytes/tok = 24 * (512 + 64) * kvBytes
```

The state term is **per-GPU and never DCP-sharded**; the KV term *is* DCP-shardable. That
asymmetry is why DCP changes the optimal ratio and why the calculator is coupled to the deploy
panel rather than being a constant.

### 1.5 Speculative decoding (DSPARK) hooks

`kda_decode_mtp.py` (~48 KB, CuTe DSL) implements `fused_kda_decode_mtp_dspark`:

- dense contract `T = num_requests * (1 + num_spec)` — one bonus token plus the draft block
  (`--speculative-dspark-block-size`, default 7)
- optional **ReplaySSM** ring buffers (`ring_rawv`, `ring_rawk`, `ring_g`, `ring_beta`) so the
  accepted prefix can be exactly re-folded into the state at commit time — enabled with
  `--enable-linear-replayssm-spec`
- phase 1 does the conv precompute (q/k/g on dedicated warps, v on the leftovers), phase 2 runs
  the state tile loop in streaming (single-token) or resident (multi-token verify) mode, state
  staged into smem in 2 chunks
- optional fused gated output norm (`onorm_gate` / `onorm_weight` / `onorm_eps`)

Recurrent state + speculation is the hard part of a hybrid model: you cannot just discard
rejected KV rows, you have to be able to roll the SSM state back. ReplaySSM is that mechanism.

---

## 2. MLA layers

`KimiK3MLAAttention(DeepseekV2AttentionMLA)`, `kimi_k3.py:1703-1873`. It reuses DeepSeek-V2/V3
MLA wholesale — `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`,
`v_head_dim`, the MQA radix attention over the latent, absorb/weight-absorption paths, DCP —
and changes exactly two things.

### 2.1 NoPE

```python
super().__init__(..., skip_rope=True, ...)     # kimi_k3.py:1730
```

`skip_rope=True` sets `self.rotary_emb = None` (`deepseek_v2.py:1882-1896`). **The MLA layers
apply no rotary embedding.** The `qk_rope_head_dim = 64` channels still exist (they are still
cached — the KV cost is `(512 + 64)` per token per layer) but carry no positional rotation.
Positional information in K3 comes from the KDA layers' causal conv + decay. `KimiLinearConfig`
exposes this as `mla_use_nope`.

This is a meaningful difference from DSV3 for anyone porting attention kernels: the MLA layers
are position-agnostic, so RoPE-fused MLA kernels have nothing to fuse.

### 2.2 Output gate

`mla_use_output_gate` (K3 feature #4). A `g_proj: [H -> num_heads*v_head_dim]`, sharded on
`attn_tp`, gates the attention output right before `o_proj`:

```python
x = x * sigmoid(g_proj(hidden_states))     # or the fused kernel below
```

Implemented by wrapping `o_proj.forward` at the instance level (`kimi_k3.py:1810-1836`) because
`o_proj` is called from deep inside the inherited MLA forward cores. The fused kernel
`kernels/ops/kimi_k3/mla_output_gate.py::kimi_k3_mla_output_gate` does `x * sigmoid(gate)` in
one pass and is documented to match the unfused pair **bit-for-bit** (double rounding
preserved); `mla_output_gate.covered(x, gate)` gates its use.

The gate GEMM is issued on an **alt stream** (`_precompute_output_gate`, `1838-1858`) so it
overlaps the attention core, but only under CUDA-graph capture, only when not in a breakable
graph (the wait would cross a segment boundary), and only up to 128 tokens on Blackwell / 64
elsewhere — above that the attention kernels already fill the SMs.

### 2.3 Attention-TP correctness note

K3 has no `LayerCommunicator`, so `o_proj` must reduce within the **attention-TP** group
itself (`use_dp_attention_reduce = True`, `kimi_k3.py:1771-1776`). The default full-TP
collective would be the wrong group at `attn_tp > 1` and would deadlock against idle DP ranks.
The same applies to the KDA `o_proj` (`1474-1484`). Worth knowing if you are debugging a hang
in a DP-attention K3 deployment.

---

## 3. Backend selection summary

| Layer type | Prefill | Decode | Notes |
|---|---|---|---|
| KDA | chunked scan (Triton / CuTe `kda_nvidia_prefill`) | fused recurrent CuTe (`cutedsl_kda`) | ROCm pins `--attention-backend triton` |
| MLA | ragged prefill (FlashInfer / FA) | `flashmla` (Hopper), `cutedsl_mla` (Blackwell) | DCP shards the latent KV |

The hybrid wrapper (`hybrid_linear_attn_backend.py:952-986`) holds both backends and dispatches
on `layer_id in full_attn_layers`.
