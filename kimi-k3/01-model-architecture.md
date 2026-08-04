# 01 — Kimi K3 Model Architecture (SGLang)

All paths relative to the `kimi-k3` branch of `sgl-project/sglang`.

Primary files:

| File | LOC | Role |
|---|---|---|
| `python/sglang/srt/models/kimi_k3.py` | 3203 | Text backbone, MoE, KDA/MLA attention, decoder layer, weight loading |
| `python/sglang/srt/models/kimi_k3_vl.py` | 937 | MoonViT3d vision tower + projector |
| `python/sglang/srt/configs/kimi_k3.py` | 124 | `KimiK3Config` = `text_config` (`KimiLinearConfig`) + `KimiK3VisionConfig` |
| `python/sglang/srt/configs/kimi_linear.py` | 180 | `KimiLinearConfig`, layer-type predicates, state-cache params |
| `python/sglang/srt/layers/attn_residual.py` | ~470 | Attention-residual bank + aggregation kernels |
| `python/sglang/kernels/ops/kimi_k3/*` | — | K3 fused kernels + collectives (also on `main`) |

---

## 1. Top-level block diagram

```
                                                    ┌──────────────────────────┐
   input_ids ──► VocabParallelEmbedding ─────────┐   │  Vision (if multimodal)  │
                 (embed_tokens)                  │   │  MoonViT3d tower         │
                                                 │   │   patch 14, 27 layers,   │
   pixel_values ─────────────────────────────────┼──►│   hidden 1024, ffn 4096  │
                                                 │   │  merge 2x2, sd2_tpool    │
                                                 │   │  mm_projector patchmergerv2
                                                 │   └────────────┬─────────────┘
                                                 │                │ image embeds (7168)
                                                 ▼                ▼
                                    ┌──────────────────────────────────────┐
                                    │  scatter into <|media_*|> placeholders│
                                    └──────────────────┬───────────────────┘
                                                       │  hidden_states [T, 7168]
                                                       ▼
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │            AttnResidual bank  (per forward pass, [T, NB, 7168])              │
    │            NB = ceil(num_layers / attn_res_block_size),  <= 8 rows           │
    └──────────────────────────────────────────────────────────────────────────────┘
                                                       │
                                     x 93   ┌──────────▼───────────┐
                                            │  KimiK3DecoderLayer  │
                                            └──────────┬───────────┘
                                                       │
                                                       ▼
                          final aggregation (output_attn_res_proj / _norm) ► RMSNorm
                                                       │
                                                       ▼
                                      LMHead / LogitsProcessor  (vocab 163840)
```

### Decoder layer (the K3 analogue of the DSV3 "DecodeLayer" box)

`KimiK3DecoderLayer.forward` → `_forward_attn_residual` (`kimi_k3.py:2205`) when
`attn_res_block_size` is set (it always is for K3); `forward` (`kimi_k3.py:2161`) is the
plain-residual fallback kept for non-K3 checkpoints.

```
   in: (hidden_states = previous layer's *un-added* MLP delta,  prefix_sum = the prefix it extends)

   ┌──────────────────────────────────────────────────────────────────────┐
   │ Aggregation 1 (attention side)                                       │
   │   prefix = prefix_sum + hidden_states                                │
   │   rows   = [bank[0..nvb-1], prefix]                                  │
   │   s_j    = <RMSNorm(row_j), self_attention_res_proj.w>               │
   │   mixed  = sum_j softmax(s)_j * row_j                                │
   │   x      = input_layernorm(mixed)          (RMSNorm, fused in kernel)│
   │   if layer_idx % attn_res_block_size == 0: bank[nvb++] = prefix      │
   └───────────────────────────────┬──────────────────────────────────────┘
                                   ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ Self-attention   (config.is_kda_layer(layer_idx) decides)            │
   │   KDA  : KimiK3DeltaAttention   (69 layers)                          │
   │   MLA  : KimiK3MLAAttention     (24 layers, NoPE, output-gated)      │
   │   -> o_proj may defer its TP reduction (SP-MoE / fused AR)           │
   └───────────────────────────────┬──────────────────────────────────────┘
                                   ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ Complete deferred o_proj reduce                                      │
   │   SP-MoE      : reduce-scatter to this rank's token shard            │
   │   AR fusion   : k3_ar_fusion.all_reduce(x, prefix_sum)  (folds add)  │
   │   otherwise   : o_proj already all-reduced itself                    │
   └───────────────────────────────┬──────────────────────────────────────┘
                                   ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ Aggregation 2 (MLP side) — same math, mlp_res_proj / mlp_res_norm,   │
   │   epilogue norm = post_attention_layernorm                           │
   └───────────────────────────────┬──────────────────────────────────────┘
                                   ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ MLP:  dense KimiK3MLP           (layer_idx < first_k_dense_replace)  │
   │       KimiK3MoE (Latent MoE)    (otherwise, on moe_layer_freq)       │
   │   consumes prefix_sum: MoE folds it into the 3-way tail add,         │
   │   dense adds it after down_proj                                      │
   └───────────────────────────────┬──────────────────────────────────────┘
                                   ▼
   out: (MLP delta, prefix_sum)   — the add is deferred into the next layer's Aggregation 1
```

Two things worth flagging, because they are what makes K3 *not* a DSV3 clone:

1. **There is no `LayerCommunicator`.** DSV3 in SGLang routes all TP/DP comm through
   `LayerCommunicator`; K3 hand-rolls it because the attention-residual stream and the
   SP-MoE token shard have to be threaded through the same reductions
   (`kimi_k3.py:1771-1776` explains why `o_proj.use_dp_attention_reduce` must be set manually).
2. **The residual add is deferred one layer.** `hidden_states` on the wire between layers is
   the *un-added* MLP delta; the add happens inside the next aggregation kernel. PP boundaries
   and DSPARK capture materialize it explicitly (`kimi_k3.py:2504-2512`, `2557-2584`).

---

## 2. Attention Residual (feature 1)

`python/sglang/srt/layers/attn_residual.py`

K3 replaces the plain residual stream with a **snapshot bank + learned softmax mixture**.

- Bank: `[T, NB, H]`, allocated once per forward (`AttnResidual.__init__`, line 343).
  `NB = ceil(end_layer / attn_res_block_size)`; `_MAX_ROWS = 16 = next_pow2(8+1)` — the comment
  states K3 has **≤ 8 snapshots**.
- Every `attn_res_block_size`-th layer snapshots the current prefix into the next bank row
  (`is_block_write_layer`, `kimi_k3.py:2001`).
- At each of the **2 aggregation points per layer** (attention side, MLP side) plus once at the
  model output, the stream value is recomputed as:

```python
rows   = cat([bank[:, :nvb, :], prefix.unsqueeze(1)], dim=1)   # [T, nvb+1, H]
scores = score_proj(score_norm(rows))                          # [T, nvb+1], proj is H -> 1
probs  = softmax(scores.float(), dim=-1)
mixed  = (probs.unsqueeze(-1) * rows.float()).sum(dim=1)
out    = out_norm(mixed)                                       # epilogue RMSNorm
```

(`aggregate_stream_torch`, `attn_residual.py:247-265` — the eager reference the kernels match.)

Per-layer weights: `self_attention_res_proj`, `self_attention_res_norm`,
`mlp_res_proj`, `mlp_res_norm` (all `[H] -> 1` `ReplicatedLinear` + `RMSNorm`), plus model-level
`output_attn_res_proj` / `output_attn_res_norm`.

Three implementations, dispatched by capability (`_aggregate`, line 313):

| Path | When | What |
|---|---|---|
| `fast` | SM100+ **and** `H == 7168` | Warp-specialized TMA kernel: `cp.async.bulk` producer + online-softmax consumers over a double-buffered chunk ring, output RMSNorm fused, one persistent CTA per SM, launch config tuned per `nvb`. Can also snapshot the bank row **in-kernel** (`write_prefix=True`) at zero extra reads. |
| `fused` | everywhere else | 2-kernel Triton pipeline (`_score_kernel` + mix), full H-parallelism |
| `torch` | `H % 1024 != 0` / tests | eager reference |

Cost note: the score pass is a scalar projection per row, so the whole mechanism is
memory-bound on the bank — hence the TMA/online-softmax treatment and the SP variants
(`forward_sp_all_gather`, `forward_sp_reduce_scatter`) that fuse the aggregation with the
collective instead of doing them back-to-back.

---

## 3. Layer typing: which of the 93 layers is KDA vs MLA

`KimiLinearConfig.is_kda_layer` (`configs/kimi_linear.py:154`):

```python
def is_kda_layer(self, layer_idx: int):
    return (self.linear_attn_config is not None
            and (layer_idx + 1) in self.linear_attn_config["kda_layers"])
```

Note the **1-based** membership test against the checkpoint's `kda_layers` list.
`linear_layer_ids` / `full_attention_layer_ids` derive the two index sets; the latter feeds the
hybrid attention backend (`layers/attention/attention_registry.py:482`).

Counts, from `docs/src/snippets/_kimi_k3_mamba_ratio_calculator.jsx:118-122`:

- **KDA: 69 layers**, 96 heads, `head_dim` 128, short-conv kernel 4 (conv state always bf16)
- **MLA: 24 layers**, `kv_lora_rank` 512, `qk_rope_head_dim` 64

The exact interleave order is checkpoint data, not repo data.

---

## 4. Dimensions actually pinned in the repo

| Quantity | Value | Where |
|---|---|---|
| hidden size `H` | 7168 | `attn_residual.py:26` (`H = 7168 = 7 x 1024`), `k3_ar_fusion.NORM_DIM * 2` |
| latent MoE width | 3584 | `kimi_k3.py:572-576` (`moe_hidden_size == NORM_DIM`), DeepEP patch adds `case 3584` |
| routed experts | 896 | `test/registered/kernels/ops/test_kimi_k3_prerequisite_ops.py:51` |
| shared experts | 1 | `num_shared_experts`, runs in full 7168 space |
| top-k | 16 | DeepEP patch raises `kNumMaxTopK` 11 → 16 for K3 |
| SiTU constants | `beta=4.0`, `linear_beta=25.0` | asserted at `kimi_k3.py:465-471` |
| vocab | 163840 | `KimiLinearConfig` default; media placeholder id 163605 |

---

## 5. Vision tower (multimodal path)

`KimiK3ForConditionalGeneration` (`kimi_k3.py:2933`):

- `vision_tower = KimiK3VisionTower(config.vision_config)` — MoonViT3d: patch 14,
  27 layers, hidden 1024, ffn 4096, 12 heads, `pos_emb_type="divided_fixed"`,
  `video_attn_type="spatial_temporal"`, `merge_type="sd2_tpool"`, `merge_kernel_size=(2,2)`.
- `mm_projector = KimiK3MultiModalProjector` — `patchmergerv2`, gelu, out = text hidden 7168.
- **Image-wise data parallel**: `use_data_parallel = True`; each image is encoded by exactly one
  TP rank via `run_dp_sharded_mrope_vision_model(..., rope_type="rope_2d",
  pool_temporal_dimension=True)`. One `MultimodalDataItem` must carry exactly one grid
  (`kimi_k3.py:3027-3035`) so CUDA-IPC lease accounting stays per-item.
- `precompile_kernels_after_loading()` warms the fused vision RoPE kernel and the FA4 vision
  attention kernel.
- Text-only serving: `KimiK3LinearForCausalLM` is usable standalone; `config.encoder_only`
  builds the tower alone (for a separate EPD encode server).

---

## 6. Weight loading and post-load fusion

`KimiK3LinearForCausalLM.load_weights` (`kimi_k3.py:2689`) → `post_load_weights` (`2857`).
After the checkpoint is in, K3 builds several **merged weight views** so the decode path issues
fewer skinny GEMVs:

| Merge | Built by | Merges |
|---|---|---|
| MoE fused front | `KimiK3MoE._merge_front_weights` (`593`) | shared `gate_up` + router `gate` + latent `down_proj` into one `[H, gu+E+latent]` GEMM (bf16/fp16 only; quantized checkpoints keep the unfused path) |
| KDA `[f_a | b]` | `KimiK3DeltaAttention._merge_bfa_weights` (`1525`) | `f_a_proj` (128 out) + `b_proj` (heads/tp out) into one skinny GEMM |
| KDA fused decode | `_prepare_fused_decode` (`1543`) | prepares the CuTe fused KDA decode/verify kernel handoff |

`_merge_weights_as_views` (`kimi_k3.py:176`) keeps the merged buffer and the original module
weights aliased, so nothing is duplicated in HBM.

The HF→SGLang weight mapper (`kimi_k3.py:2942`) rewrites `language_model.layers.` →
`language_model.model.layers.` and `block_sparse_moe` → `mlp`.
