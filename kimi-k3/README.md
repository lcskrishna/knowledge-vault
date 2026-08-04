# Kimi K3 in SGLang — Architecture Breakdown

Companion to the "DSV3 Architecture Design from SGLang" diagram, but for Moonshot **Kimi K3**.

**Source of truth:** `github.com/sgl-project/sglang`, branch **`kimi-k3`** (commit `36e31ee`, 2026-08-04).
K3 is *not* on `main` — `main` ships only the shared kernels (`python/sglang/kernels/ops/kimi_k3/`),
the K3 parsers, and the K3 Dockerfiles. The model itself
(`python/sglang/srt/models/kimi_k3.py`, 3203 LOC) lives on the branch.
Local clone used for this write-up: `~/github-workspace/sglang`.

## Documents

| Doc | Contents |
|---|---|
| [00-about-kimi-k3.md](00-about-kimi-k3.md) | **Start here / non-specialist.** What the model is and what the architecture contains, in plain English — no jargon, safe to repeat to anyone |
| [01-model-architecture.md](01-model-architecture.md) | Top-level block diagram, the five K3-specific features, decoder-layer dataflow, attention-residual bank |
| [02-attention-kda-mla.md](02-attention-kda-mla.md) | KDA linear attention (delta rule, short conv, state cache) and the NoPE MLA layers |
| [03-moe-latent-and-ep.md](03-moe-latent-and-ep.md) | Latent MoE, router, SiTU experts, MegaMoE/DeepEP a2a, SP-MoE token sharding |
| [04-kernels-and-collectives.md](04-kernels-and-collectives.md) | The `kimi_k3` kernel package: fused AR, GEMM+AR, GEMM+AG, SP collectives, TMA attn-res |
| [05-serving-and-deployment.md](05-serving-and-deployment.md) | Per-hardware launch recipes (incl. MI350X/MI355X), Docker/DeepEP patches, parsers, VL path |
| `kimi-k3-architecture.html` | All of the above as one self-contained browsable page with rendered block diagrams |

## The one-slide version

```
                    Kimi-K3 = MoonViT3d tower  +  KimiLinear hybrid MoE backbone

  KimiK3ForConditionalGeneration
  ├── vision_tower  : KimiK3VisionTower (MoonViT3d, 27 layers, 1024 hidden, patch 14, 2x2 merge)
  ├── mm_projector  : patchmergerv2  ->  text hidden 7168
  └── language_model: KimiK3LinearForCausalLM
        ├── embed_tokens (VocabParallelEmbedding)
        ├── 93 x KimiK3DecoderLayer
        │     ├── attention: 69 x KDA (linear)  |  24 x MLA (NoPE, output-gated)
        │     └── mlp      : dense MLP (first_k_dense_replace) | Latent MoE (896 routed + 1 shared)
        ├── attention-residual bank  (softmax mixture over <=8 snapshots, replaces the plain residual)
        └── final aggregation -> RMSNorm -> lm_head / LogitsProcessor
```

### K3-specific features vs. DeepSeek-V3

`kimi_k3.py:1-7` lists exactly what K3 adds on top of the `kimi_linear.py` backbone:

1. **Attention Residual** (`attn_res_block_size`) — the residual stream is a learned softmax
   mixture over banked snapshots, not a plain add.
2. **Latent MoE** (`routed_expert_hidden_size`) — routed experts run in a 3584-wide latent
   space, down/up projected around the expert call.
3. **SiTU activation** — `beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta)`.
4. **MLA output gate** (`mla_use_output_gate`) — `attn_out * sigmoid(g_proj(x))` before `o_proj`.
5. **Full-rank KDA gate** (`use_full_rank_gate`) — the KDA output gate is a full
   `[H, num_heads*head_dim]` projection instead of the low-rank `g_a/g_b` pair.

Plus the structural difference that dominates serving: **hybrid attention**. 69 of 93 layers are
recurrent (constant KV per sequence, `[128,128]` state per head), only 24 carry an MLA latent KV
cache. That is why K3 memory planning has its own `--mamba-full-memory-ratio` calculator.

## Confidence notes

- Everything in 01–04 was read directly out of the branch and is cited by file:line.
- The per-hardware tables in 05 come from `docs/src/snippets/configs/moonshotai/kimi-k3.jsx`.
  The `benchmark/*/script/v1/launch-k3.sh` files that snippet cites are **not** in the public
  repo, so the jsx cells are the authoritative public record.
- Exact `kda_layers` / `full_attn_layers` index lists live in the checkpoint's `config.json`
  (`moonshotai/Kimi-K3`), not in the repo. The repo only knows the counts (69 / 24) and the
  membership test.
