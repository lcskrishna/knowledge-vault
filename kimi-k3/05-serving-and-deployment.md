# 05 — Serving Kimi K3: recipes, Docker, parsers, VL path

Authoritative public source for the recipes: `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`
(~2150 lines of per-hardware "cells"). The `benchmark/H200/script/v1/launch-k3.sh` and
`benchmark/B300/script/v1/launch-k3.sh` scripts that snippet cites are **not** in the public
repo — the jsx cells are the record.

Supported hardware declared in that file: `b300, gb300, b200, gb200, h200, h100, mi350x, mi355x`.

---

## 1. Per-hardware recipes (unified P+D)

Common to every cell: `--trust-remote-code --model-path moonshotai/Kimi-K3
--reasoning-parser kimi_k3 --tool-call-parser kimi_k3`.
`--nnodes/--node-rank/--dist-init-addr/--host/--port` are injected by the deploy panel.

| HW | Nodes | Strategy | Parallelism | Notable flags |
|---|---|---|---|---|
| **B300** | 1×8 | low-latency | TP8 | `--mem-fraction-static 0.85` |
| B300 | 1×8 | balanced / high-thrpt | TP8 + **DCP8** | `--disable-custom-all-reduce` |
| **GB300** | 2×4 | low-latency | TP8 | `--enable-symm-mem` (MNNVL) |
| GB300 | 2×4 | balanced | TP8 + DCP8 | |
| **B200** | 2×8 | low-latency | TP8 × **PP2** | `specCollapsePp: true` → flattens to TP16 under DSPARK |
| B200 | 2×8 | balanced | TP8 × PP2 + DCP8 + EP8 | `--moe-runner-backend flashinfer_mxfp4`, `--decode-attention-backend cutedsl_mla` |
| **GB200** | 4×4 | low-latency | TP16 | `--enable-symm-mem`, `NCCL_MNNVL_ENABLE=1` |
| GB200 | 4×4 | balanced | TP16 + DCP16 | |
| **H200** | 2×8 | low-lat / balanced | TP16 + EP16 | `--moe-runner-backend marlin`, `--decode-attention-backend flashmla` |
| H200 | 4×8 | high-throughput | TP32 + EP32 | `--mamba-radix-cache-strategy extra_buffer_lazy`, `--mem-fraction-static 0.90` |
| **H100** | 4×8 | all | TP32 + EP32 | `marlin` + `flashmla`, `--dist-timeout 3600`, env `SGLANG_K3_ATTN_RES_MODE=jit`, `SGLANG_MOE_FUSED_GATE_RADIX=1` |
| **MI350X / MI355X** | 1×8 | balanced | TP8 (no DCP, no EP) | `--attention-backend triton --dtype bfloat16 --cuda-graph-max-bs 256` |

PD-disaggregated cells also exist, e.g.:

- B300 prefill default: `TP8 --chunked-prefill-size 16384 --enable-symm-mem`
- B300 prefill long-context: **TP1 × PP8**, `--mem-fraction-static 0.90` — TP1 avoids
  replicating the MLA KV across ranks
- B200 prefill default: TP1 × PP16 (one pipeline stage per GPU)
- B200 decode balanced: flat TP16 + DCP16, `--disaggregation-mode decode
  --disaggregation-transfer-backend nixl --disaggregation-decode-extra-slots 16`

### Terminology in the cells

- **DCP** — decode context parallel; shards the TP-replicated MLA latent KV. Only the 24 MLA
  layers benefit; the 69 KDA state slabs are per-GPU and never DCP-sharded.
- **DCPEP** — DCP and EP composed on the same rank set (B200 balanced: DCP8 + EP8).
- **DSPARK** — K3's speculative decoding. Draft model `RadixArk/Kimi-K3-DSpark`,
  `--speculative-dspark-block-size` (default 7), plus `--enable-linear-replayssm-spec` for the
  KDA state rollback (doc 02 §1.5). A `pp_size == 1` speculative algorithm cannot run a
  pipelined recipe — cells either opt into `specCollapsePp` (re-lay PP2×TP8 as flat TP16) or are
  unavailable under speculation.

### Memory planning

`--mamba-full-memory-ratio` balances the KDA state pool against the MLA KV pool. Use the
calculator in `docs/src/snippets/_kimi_k3_mamba_ratio_calculator.jsx`; the formula and why DCP
shifts it are in doc 02 §1.4.

---

## 2. AMD / ROCm status

Relevant if you are running MI350X/MI355X.

- Both ROCm cells are marked **`verified: false, verificationStatus: "in-progress"`** in the
  jsx. Treat them as a starting point, not a validated recipe.
- Required env: `SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1
  AITER_SITUV2_A8W4=1`. `AITER_SITUV2_A8W4` implies the ROCm expert path is **A8W4 via AITER**,
  not mxfp4/marlin.
- `--attention-backend triton` — no `cutedsl_kda` / `cutedsl_mla` / `flashmla` on ROCm.
- No `--ep-size`, no `--dcp-size` in either ROCm cell: TP8 only, single node.
- `alt_streams` are **disabled on HIP** (`kimi_k3.py:2380`), so SBO shared-expert overlap, the
  MLA output-gate stream and the KDA `[f_a|b]` stream are all off.
- Everything in `kernels/ops/kimi_k3/` that requires SM100 (fused AR, gemm_ar, gemm_ag, SP
  collectives, TMA attn-res) is unavailable; attn-res falls back to the Triton 2-kernel
  pipeline, collectives to RCCL.
- There are ROCm nightly branches: `amd/mi355x-kimi-k3-nightly`, `kimi-k3-rocm720-nightly`.

Practical read: on ROCm today K3 runs the *generic* SGLang paths for everything K3-specific
except the Triton attn-res and AITER's SiTU/A8W4 MoE. The Blackwell-only fusion stack is the
bulk of the branch's performance work.

---

## 3. Docker / dependencies

`docker/kimi_k3/kimi_k3_cu12.Dockerfile`, `kimi_k3_cu13.Dockerfile`,
`docker/kimi_k3/apply_deepep_k3_patch.sh`, `docker/kimi_k3/flashinfer-perkz-dcp-0.6.15.txt`.

1. Base `lmsysorg/sglang:v0.5.16`, then an editable install of the `kimi-k3` branch.
2. **DeepEP patched** (5 changes; see doc 03 §2) and rebuilt for `sm_90 / sm_100a / sm_103a`.
3. **DeepGEMM** upgraded to `0.1.5.post1` for the K3 SiTU JIT headers used by MegaMoE.
4. **FlashInfer 0.6.15.post1** with a prebuilt trtllm-gen cubin pool (~1.5 GB) at
   `SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL`, plus CuTeDSL MLA DCP runtime patches. The build verifies
   cubins for all three archs.
5. CI mirror: `scripts/ci/cuda/ci_install_kimi_k3.sh` pins the same FlashInfer diffs and cubin
   pool SHA256.

If you are building K3 yourself, the DeepEP patch and the cubin pool are the two things most
likely to bite: an unpatched DeepEP fails at top-k 16 / hidden 3584, and a missing cubin pool
silently drops you off the `flashinfer_mxfp4` MoE runner.

---

## 4. Serving-layer integration

### Reasoning + tool calls (XTML)

K3 uses an XTML-style channel markup rather than JSON-in-tags:

```
<|open|>think<|sep|>   ... reasoning ...     <|close|>think<|sep|>
<|open|>response<|sep|> ... text ...         <|close|>response<|sep|>
<|open|>tools<|sep|>
  <|open|>call tool="name" index="1"<|sep|>
    <|open|>argument key="k" type="string"<|sep|>value<|close|>argument<|sep|>
  <|close|>call<|sep|>
<|close|>tools<|sep|>
```

- Tool-call detector: `python/sglang/srt/function_call/kimik3_detector.py`, format helpers in
  `kimik3_format.py`, xgrammar structural tags in `kimik3_structural_tag.py`
  (`get_kimik3_structural_tag`, `get_kimik3_auto_tool_call_structural_tag`).
  `type="string"` argument values are raw text; other types are JSON-decoded. Attribute
  escaping: `&quot;` / `&amp;`. Streaming holds back partial markers.
- Reasoning parser: `KimiK3Detector` in `python/sglang/srt/parser/reasoning_parser.py`,
  `<|open|>think<|sep|> ... <|close|>think<|sep|>`, with `force_reasoning` for resumption when
  the serving layer feeds the open marker as a generation prefix.
- Auto-detection: `python/sglang/srt/parser/template_detection.py` sets both parsers to
  `kimi_k3` on `model_type == "kimi_k3"` / `"KimiK3"` in the architecture string — so the
  explicit `--reasoning-parser` / `--tool-call-parser` flags in the recipes are belt-and-braces.

### Multimodal

`python/sglang/srt/multimodal/processors/kimi_k3.py` +
`python/sglang/srt/multimodal/kimi_k3_vit_cuda_graph_runner.py`.

- Each image is wrapped with dimension metadata:
  `<|media_begin|>image WIDTHxHEIGHT<|media_content|> [image tokens] <|media_end|>`
  (`_expand_k3_image_prompt_token_ids`); special tokens encoded with `allowed_special="all"` so
  BPE cannot split them. Placeholder id `163605`.
- GPU preprocessing (`KimiK3GPUProcessorWrapper`, extends the K2.5 wrapper): **preserves the
  alpha channel** through bicubic resize and composites onto a background configured in the
  checkpoint's `preprocessor_config.json` (`transparent_bg_config`: chessboard / white / black /
  gray). This is new in K3 relative to K2.5.
- Encoding is image-wise data-parallel across TP ranks (doc 01 §5), with CUDA-IPC handoff so an
  image crosses the tokenizer/scheduler boundary once rather than once per rank.

---

## 5. Quick start pointers

```bash
# clone the branch (the model is not on main)
gh repo clone sgl-project/sglang -- --depth 1
cd sglang && git fetch --depth 1 origin kimi-k3:kimi-k3 && git checkout kimi-k3

# key reading order
python/sglang/srt/models/kimi_k3.py            # 1-124 header, 354 MoE, 1206 KDA, 1703 MLA,
                                               # 1876 decoder layer, 2344 backbone
python/sglang/srt/layers/attn_residual.py      # the attention-residual mechanism
python/sglang/kernels/ops/kimi_k3/             # the fused kernels
docs/src/snippets/configs/moonshotai/kimi-k3.jsx   # every serving recipe
```
