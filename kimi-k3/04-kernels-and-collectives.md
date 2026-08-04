# 04 — The `kimi_k3` Kernel Package and Fused Collectives

`python/sglang/kernels/ops/kimi_k3/` (present on **both** `main` and the `kimi-k3` branch), with
Python-side dispatch glue in `python/sglang/srt/layers/k3_ar_fusion.py`, `k3_gemm_ar.py`,
`k3_sp_collective.py`.

This is where most of K3's decode performance lives. The theme throughout: at decode, K3's
per-layer work is a chain of skinny memory-bound GEMVs separated by collectives, so nearly
every kernel here fuses a GEMM with the collective that follows it, or a collective with the
norm that follows it.

---

## 1. Kernel inventory

| File | Fuses | Target | Selection gate | Replaces |
|---|---|---|---|---|
| `activation.py` | SiTU (`beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta)`) | SM90 fast-math / SM100+ precise `expf` | always (K3 default activation) | SiLU + scalar mul |
| `moe.py` | SiTU + masked post-quant | SM100+ with PDL | mxfp8 + masked dispatch at runtime | SiTU then separate quant |
| `mla_output_gate.py` | `x * sigmoid(gate)` | — | `covered(x, gate)` | mul + sigmoid pair (bit-exact match) |
| `attn_res.py` | attn-res score → online softmax → mix → output RMSNorm, optionally + RS or + AG | SM100a+ (TMA, `cp.async.bulk`, PDL) | SM100+ and `H == 7168` | Triton 2-kernel pipeline + separate norm |
| `gemm_ar.py` | `o_proj` GEMM **+ all-reduce** | SM100+, full NVLink P2P | `k3_gemm_ar.fits(x)`: bf16, contiguous, `M ∈ [1,512]` | GEMM + NCCL AR |
| `gemm_ag.py` | latent `up_proj` column-parallel GEMM + multicast all-gather + `add3` | SM100+ MNNVL, **TP8 only** | `gemm_ag_up_fits(M)` (~M ≤ 12) | replicated `[3584,7168]` GEMM + `_add3` |
| `all_reduce.py` | AR ± residual add ± RMSNorm ± deferred MoE finalize | SM100+ multicast | `k3_ar_fusion.enabled()` | AR + separate norm/add |
| `sp_collective.py` | SP reduce-scatter / all-gather over MNNVL push memory | SM100+ | JSON tuning tables | NCCL collectives |
| `kda_decode_mtp.py` | conv + KDA recurrence + optional gated output norm, MTP/DSPARK-aware | CuTe DSL | shape contract | unfused Triton chain |

Nearly all of it is **SM100+ (Blackwell)**. On Hopper and ROCm the generic paths run instead —
see doc 05.

---

## 2. `k3_ar_fusion` — the fused all-reduce family

`all_reduce.py` exposes four entry points, two algorithms × two epilogues:

| Entry | Algorithm | Epilogue |
|---|---|---|
| `push_res` | 1-shot multicast push (CustomAllReduceV2 workspace) | residual add |
| `push_norm` | 1-shot multicast push | residual add + RMSNorm |
| `pull_res` | low-SM NVLS 2-shot reduce-scatter + broadcast, in place | residual add |
| `pull_norm` | low-SM NVLS 2-shot | residual add + RMSNorm |
| `finalize_push_norm` | 1-shot push | deferred MoE finalize + norm |

Selection is by message size against tuning bands (RES_TUNING 128 KB–4 MB, NORM_TUNING
512 KB–2 MB); small messages take the push path, large ones the NVLS pull path.

Two call sites in the model:

1. **Attention output** (`kimi_k3.py:2313-2319`): `k3_ar_fusion.all_reduce(hidden_states,
   prefix_sum)` completes `o_proj`'s deferred reduce *and* folds the pending attention-residual
   prefix add into the same kernel.
2. **MoE tail** (`_forward_fused`): the `[latent | shared]` pair is laid out as one flat buffer
   viewed as `[3N, 3584]`, the kernel norms the first `N` rows. That layout requirement is why
   `fuse_ar_norm` statically checks `moe_hidden_size == 3584 and hidden_size == 7168`
   (`kimi_k3.py:566-576`) — decided once at init so the hot path only reads a bool.

Enabled by `SGLANG_K3_AR_FUSION`, and only when `attn_tp_size == tp_size` (the fused comm lives
in the full-TP group). When the fusion is on, `o_proj` must write into caller-owned storage (a
slice of the persistent symmetric buffer); if the linear method cannot take an `output_tensor`,
the layer silently reverts to `reduce_results=True` (`kimi_k3.py:1487-1492`, `1736-1741`).

---

## 3. `k3_gemm_ar` — GEMM + all-reduce in one launch

`maybe_wrap_o_proj(self.o_proj)` is applied to both KDA and MLA `o_proj`
(`kimi_k3.py:1493`, `1742`). Each rank computes its local `x_r @ W_r^T` partial and the
cross-rank sum happens inside the same kernel via a P2P flag ring, with the launch epoch held
in device memory so it is CUDA-graph safe. Output is fully reduced on every rank, one launch
instead of GEMM + NCCL AR.

## 4. `k3_sp_collective` — sequence-parallel collectives

Backs the SP-MoE path (doc 03 §3):

- `reduce_scatter_res(attn_out, residual)` — reduce-scatter with the residual add folded in
  (`kimi_k3.py:2066`)
- `attn_res_all_gather(...)` / `attn_res_fused_direct_ag` — fuse the attention-residual
  aggregation with the following row all-gather (`attn_residual.py:forward_sp_all_gather`)
- `forward_sp_reduce_scatter` — the MLP-side aggregation fused with the reduce-scatter

Strategy per token bucket comes from JSON tables in `kernels/ops/kimi_k3/configs/sp_collective/`
(`world=4,H=7168,device_name=NVIDIA_GB300.json`, `world=8,...`), each mapping a token bucket to
`{strategy: push|pull|nccl, num_blocks, block_size}`. Retuning is a JSON swap, no recompile.
Missing table → NCCL fallback.

The o_proj output only comes from the persistent symmetric buffer when the table selects NVLS
pull RS for that bucket; small push RS keeps the regular graph allocator
(`kimi_k3.py:2028-2047`).

---

## 5. Multi-stream overlap map

`KimiK3LinearModel.__init__` allocates 3 alt streams (`kimi_k3.py:2372-2380`), **disabled on
HIP**:

| Slot | Used by | Overlaps |
|---|---|---|
| 0 | MoE dual-stream shared-expert tail (SBO) | routed a2a |
| 1 | `DeepseekV2AttentionMLA` base internals | (forwarded, unused by K3) |
| 2 | MLA output-gate GEMM **and** KDA `[f_a|b]` GEMVs | the attention core |

Slot 2 is shared between MLA and KDA because the two never run concurrently within one forward
(`kimi_k3.py:1954-1956`). Both users are capped at 128 tokens on Blackwell / 64 elsewhere —
above that the core kernels already saturate the SMs and the overlap only adds sync overhead.

The attn-res bank write no longer needs a stream: it is fused into the aggregation-1 TMA kernel
(`write_prefix=True`).

---

## 6. Env vars

Grep `SGLANG_K3` in `python/sglang/srt/environ.py` (~lines 724-843). The ones that change
behaviour materially:

| Var | Effect |
|---|---|
| `SGLANG_K3_AR_FUSION` | enables the fused all-reduce family (§2) |
| `SGLANG_K3_SP_COLLECTIVE` | enables SP collectives → SP-MoE token sharding |
| `SGLANG_K3_SP_ATTN_RES` | carries the attn-res stream sharded across SP-MoE layers |
| `SGLANG_K3_GEMM_AR` | fused `o_proj` GEMM + all-reduce (§3) |
| `SGLANG_K3_FUSED_FRONT` | merged MoE front GEMM — **default on** |
| `SGLANG_K3_ATTN_RES_MODE=jit` | set on the H100/H200/GB cells in the jsx; not declared in `environ.py` on this branch, so it is read elsewhere (or stale) — verify before relying on it |
| `SGLANG_MOE_FUSED_GATE_RADIX` | radix router in the TopK layer |
| `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` | must cover the per-rank prefill chunk under MegaMoE |
| `SGLANG_USE_AITER`, `SGLANG_AITER_K3_OPT`, `AITER_FLYDSL_FORCE`, `AITER_SITUV2_A8W4` | ROCm/AITER K3 path |

---

## 7. Tests

- `test/registered/kernels/ops/kimi_k3/test_compute.py` — SiTU, tiny GEMMs, attn-res aggregation
- `test/registered/kernels/ops/kimi_k3/test_collectives.py` — fused AR / SP collectives
- `test/registered/kernels/ops/test_kimi_k3_prerequisite_ops.py` — the 896-expert / 7168 / 3584
  shape contracts
- `test/registered/kernels/ops/attention/test_kda_fused_decode.py` — KDA fused decode vs reference
- `test/registered/unit/parser/test_kimik3_reasoning_parser.py`,
  `test/registered/function_call/test_kimik3_detector.py` — serving-layer parsers
