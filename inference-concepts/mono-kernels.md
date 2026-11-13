# Mono Kernels, explained from zero

> What a "mono kernel" (vLLM `MonoKernel`, SGLang `monokernel`) is, why it exists,
> how it works, and how it's wired into both vLLM and SGLang — written for someone
> who has never seen one. Grounded in the local `vllm-upstream` checkout
> (commit `a2bcb56`, Nov 5 2026) and upstream SGLang + the `sgl-amd-mono` plugin.

---

## TL;DR

A **mono kernel** is *one long-running GPU kernel that runs most or all of a
transformer decoder layer — including the tensor-parallel all-reduces — in a
single launch (sometimes two)*, instead of the dozens to hundreds of small
kernel launches the normal path uses. It targets **low-concurrency decode**
(1–16 rows) on **AMD Instinct MI350X/MI355X (gfx950)**, and it is written in
**FlyDSL**, AMD's Python kernel DSL that ships inside AITER.

- **vLLM** carries the kernels in-tree, behind the `--monokernel` flag (and, for
  some models, an env var or `--additional-config`). The code lives per-model
  under `vllm/models/<model>/amd/`.
- **SGLang** keeps only the *interface* in-tree and loads the actual kernels from
  an external plugin package (`sgl-amd-mono`) behind `--enable-amd-monokernel`.

Why it wins: on a single-token decode step the GPU is idle most of the time,
waiting on kernel-launch overhead and HBM reads. Fusing the layer into one
persistent launch removes the launch gaps, keeps data resident, and does the
cross-GPU all-reduce *inside* the kernel. On 8× MI350X, GLM-5.2 decode drops from
**~2,459 kernels/step to ~210**, and time-per-token at batch 1 goes **13.70 ms →
5.92 ms (2.31×)**.

---

## Part 1 — Background you need first

Skip this if you already know decode, kernel launches, tensor parallelism,
all-reduce, and CUDA graphs.

### Decode vs. prefill

An LLM serves a request in two phases:

- **Prefill**: process the whole prompt at once. Lots of tokens → big matrix
  multiplies → the GPU is *compute-bound* and busy. Life is good.
- **Decode**: generate one token at a time. Each step processes **one new token
  per sequence**. At low concurrency that's a *tiny* amount of work per layer —
  a few matrix-vector products. The GPU is *not* compute-bound; it's bound by
  overhead and memory latency.

Mono kernels are a **decode-only** optimization. They do nothing for prefill.

### What a "kernel launch" costs

A GPU "kernel" is one function the CPU launches onto the GPU. Each launch has
fixed overhead: the CPU builds and submits a command, the GPU scheduler picks it
up, reads its inputs from HBM (high-bandwidth memory), runs, writes results back
to HBM, and finishes. Between two launches the GPU often sits idle for a few
microseconds.

A single decoder layer is normally **many** kernels: RMSNorm, the QKV projection,
RoPE, attention, the output projection, an all-reduce, another norm, the MoE
router, the experts, another all-reduce, and so on. Multiply by ~40–90 layers.
At decode, each kernel does almost no work, so the **launch gaps and the
round-trips to HBM dominate**. You spend more time starting and stopping kernels
than computing.

> On 8× MI350X, a graph-mode GLM-5.2 decode step issues **~2,459 kernels**. Each
> one is mostly overhead at batch 1.

### Tensor parallelism (TP) and all-reduce

A model too big for one GPU is split across several. **Tensor parallelism** cuts
each weight matrix into shards, one per GPU. Each GPU computes a *partial* result
for its shard; the partials must be **summed across all GPUs** to get the real
answer. That sum is an **all-reduce** — a collective where every GPU sends its
partial and receives the total.

Normally the all-reduce is its own kernel (often a library call like RCCL or an
AITER quick-reduce). It needs all GPUs to synchronize, which is *more* overhead
on top of the launch gaps.

### CUDA graphs (HIP graphs)

To cut launch overhead, frameworks **record** a step's sequence of kernel
launches once, then **replay** the whole recording with a single submission.
This is a "CUDA graph" (HIP graph on AMD). It hides *per-launch CPU* cost, but
the kernels still run one after another on the GPU with their gaps, and the
*routing decisions* are frozen at capture time (a replay re-runs the exact same
launches it recorded).

vLLM has three graph modes you'll see referenced:

- **NONE / eager**: no graph, run kernels directly.
- **FULL**: the whole forward is captured as one graph and replayed.
- **PIECEWISE**: the forward is captured in pieces, keyed by token count; it
  replays mixed steps, so mono kernels stay off for it.

---

## Part 2 — What a mono kernel actually is

A mono kernel replaces the chain of per-op kernels in a layer with **one
persistent kernel launch** (two for models that also fuse attention). "Persistent"
and "mono" are the whole idea:

- **Mono** = one kernel does the work of the whole layer (monolithic), rather
  than one kernel per operation.
- **Persistent** = the kernel is launched *once* with as many workgroups as the
  GPU has compute units, and those workgroups **stay resident** for the entire
  layer, pulling work from a fixed internal schedule and handing off to each
  other through memory — never returning to the CPU mid-layer.

Here is the normal path vs. the mono path for one MoE decoder layer:

```
NORMAL DECODE (one MoE layer)                MONO DECODE (one MoE layer)
---------------------------------            ----------------------------------
  kernel: input RMSNorm                        kernel: input RMSNorm   (vLLM op)
  kernel: QKV projection                       kernel: attention       (vLLM op)
  kernel: RoPE                                 ┌──────────────────────────────┐
  kernel: attention                            │ ONE persistent FlyDSL launch │
  kernel: o_proj                               │   push partials to peers     │
  kernel: all-reduce (TP)       ───►           │   all-reduce + residual      │
  kernel: post-attn RMSNorm                    │   post-attn RMSNorm          │
  kernel: MoE router + top-k                   │   router + top-k             │
  kernel: shared expert                        │   shared + routed experts    │
  kernel: routed experts                       │   all-reduce (TP)            │
  kernel: all-reduce (TP)                      └──────────────────────────────┘
  ... (dozens of tiny launches)                (1–2 launches, all-reduce inside)
```

The exact split varies by model (some fuse attention too; see the K1/K2 split in
Part 4), but the shape is always: **collapse the launch chain into one resident
kernel, and do the collectives inside it.**

---

## Part 3 — The five core ideas

Every mono kernel is built from the same five mechanisms. Understand these and
you understand all of them.

### 1. A fully-resident grid, one workgroup per compute unit

The kernel launches a **fixed grid** sized to the GPU: `BLOCKS = 256` workgroups
(CTAs) with `THREADS = 512` each (`WAVES = 8`). On MI355X there are exactly 256
compute units (CUs), so there's **one workgroup per CU and all of them are
resident at the same time**.

This is non-negotiable, and here's why: inside the kernel, workgroups wait on
each other by spinning on memory flags. If one workgroup were *not* scheduled
(because the grid is bigger than the GPU can hold at once), another workgroup
spinning on it would wait forever → **deadlock**. So the grid must equal the CU
count, and the kernels **refuse to run** on partitioned GPUs or any GPU whose CU
count can't hold all 256 workgroups.

In the shared helpers (`vllm/models/deepseek_v41/amd/mono/common/plan.py`):

```python
BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64

def first_task(bid, base):
    """CTA bid's first task of a stage placed from CTA base, then every BLOCKS-th."""
    return (bid - base % BLOCKS + BLOCKS) % BLOCKS
```

### 2. Stages as "task tables"

The kernel body is a **fixed sequence of stages** (push, reduce, norm, router,
route, up, down, final, …). Each stage is a table of independent tasks, and the
256 workgroups split the table **strided**: workgroup `b` runs tasks
`b, b+256, b+512, …` (that's what `first_task` computes). Stages run in order,
and **a stage only ever waits on *earlier* stages**. Because every workgroup is
resident and nobody waits on a later stage, there's no deadlock.

The MiniMax-M3 FFN kernel's main loop makes this concrete
(`vllm/models/minimax_m3/amd/specialized_decode/mono/ffn.py`, `run_ffn`):

```python
@traced
def run_ffn(c, bid):
    s = c["S"]
    for task in range(first_task(bid, 0), SEAM_TASKS, BLOCKS):
        stage_push(c, task)        # push this rank's attn partial to every peer
    gpu.barrier()
    for task in range(first_task(bid, 0), SEAM_TASKS, BLOCKS):
        stage_reduce(c, task)      # sum peers' partials in rank order, + residual
    gpu.barrier()
    for t in range(first_task(bid, NORM0), s, BLOCKS):
        stage_norm(c, t)           # post-attention RMSNorm
    for task in range(first_task(bid, ROUTER0), ROUTER_TASKS, BLOCKS):
        stage_router(c, task)      # FP32 router GEMV
    for t in range(first_task(bid, NORM0), s, BLOCKS):
        stage_route(c, t)          # sigmoid, top-4, renormalize
    gpu.barrier()
    # ... shared expert, load_table, routed experts (up), down, final push+reduce ...
```

Note `gpu.barrier()` only synchronizes threads **within one workgroup**. The
waiting *between* workgroups, and between GPUs, happens through mailboxes (next).

### 3. The tagged mailbox (how workgroups and GPUs hand off)

Workgroups and GPUs coordinate by reading and writing **global memory flags**,
not by returning to the host. The clever bit is the **tagged pair**: every value
is stored next to a **tag** in one 8-byte store — `(value, tag)`. A consumer
polls the pair and only accepts the value once the tag matches the **current
step's epoch**.

Why that matters:

- **No clearing between steps.** A leftover value from a previous step carries an
  old tag, so it's ignored automatically. One round-trip per hand-off, no
  separate "flag cleared yet?" protocol.
- **CUDA-graph safe.** Every argument is a fixed device pointer and nothing needs
  host-side resetting, so the launch records cleanly into a FULL graph and
  replays correctly.
- **Two scopes.** `CM_DEV` (device scope) for hand-offs between workgroups on one
  GPU; `CM_SYS` (system scope) for hand-offs between GPUs over the interconnect.

From the shared `common/sync.py` (adapted from ROCm/ATOM's `atom/mono`):

```python
class Mailbox:
    """Every 32-bit value is stored next to this launch's tag; a consumer polls
    the payload until all tags match, so a hand-off costs one round trip
    (no store drain, no separate flag)."""
    def put(self, a, i, v): ...     # store (value, tag) as one vector
    def poll(self, specs, batch): ...  # spin until every tag == self.tag
```

One subtlety handled by `publish()`: before a workgroup raises a flag, its
earlier plain data stores must be **visible to other workgroups** first — a flag
must never overtake the data it announces. So `publish` drains outstanding stores
(`s_waitcnt(vmcnt=0)`) and does a workgroup barrier *before* writing the flag:

```python
@traced
def publish(put, a, i, v, who):
    rocdl.s_waitcnt(vmcnt=0)   # drain this CTA's plain stores
    gpu.barrier()
    if who:
        put(a, i, v)           # now the flag is safe to raise
```

### 4. The in-kernel all-reduce

Instead of calling a separate collective, the kernel does TP all-reduce itself
over **symmetric peer memory**. Each rank allocates one uncached IPC buffer
(AITER's `UncachedIpcHeap`), and every rank learns every peer's address by
exchanging IPC handles at startup (`common/peer_memory.py`, `PeerBuffer`).

During the layer, each GPU **writes its partial straight into every other GPU's
buffer** (system-scope `(value, tag)` stores), waits for all peers' tags, then
**sums the partials in rank order in fp32** and converts to bf16. Rank order +
fp32 accumulation makes the result **deterministic and bit-exact** — the same on
every rank, and matching what the normal path computes (at M ≤ 48 rows vLLM's own
decode all-reduce is below its INT4 quick-reduce threshold, so it's exact too).

### 5. Fail-stop (bounded waits + a sticky error word)

A spin-wait with no deadline is a hang waiting to happen. So **every in-kernel
wait is bounded**, and a timeout (or a host abort, or a failed epoch reset) sets
a **sticky error word** that is shared across all ranks. The guarantee is:

> No token is ever returned from a step any of whose launches failed on any
> rank, nor from any later step.

How each implementation enforces it:

- **vLLM MiniMax / Kimi**: waits bounded at **10 s**; after the layer loop, a
  device-side `torch._assert_async` checks the error word is clear (Kimi checks
  every 16 steps). A set word traps the GPU queue → sampling and the output copy
  never run → the worker's wait fails with `hipErrorLaunchFailure` → the engine
  stops. No per-step host sync; the check is two tiny kernels per forward (inside
  the FULL graph when one is replayed).
- **SGLang**: the deadline is **half of `--watchdog-timeout`, clamped to
  [5 s, 60 s]**. A failed wait faults the GPU at address `0x0`, which the
  watchdog turns into a clean stop.

---

## Part 4 — The K1 / K2 split

Models that also fuse **attention** use **two** launches per layer, called K1 and
K2. The boundaries always sit **at an all-reduce**, because that's the natural
cross-GPU synchronization point.

- **K1** — layer input through attention: the input norm / seam, QKV projection,
  q/k norms, RoPE, KV-cache write, and the attention itself.
- **K2** — the rest: `o_proj`, the **attention all-reduce**, the post-attention
  norm/seam, the MoE (router, top-k, shared + routed experts), and the **second
  all-reduce**.

Why this split is useful: **K2 doesn't care how its input was produced.** A model
that keeps stock attention can still reuse K2 by having vLLM's own attention op
feed it. That's exactly what the MiniMax-M3 "FFN launch" is — a K2-style kernel
with vLLM doing the norm and attention in front of it (see `executor.py` below).

DeepSeek-V4.1-Flash is the reference two-launch design (`deepseek_v41/amd/mono/`):
K1 covers the attention seam + attention, K2 covers the attention all-reduce + FFN
seam + MoE + MoE all-reduce.

---

## Part 5 — When does a mono kernel actually run?

A mono kernel is **not** a general-purpose path. Each build is compiled for **one
exact deployment**, and a step only runs on it if it matches. Think of it as two
gates:

### Startup gate (refuse loudly)

When you pass `--monokernel`, the model checks the *whole deployment* and
**refuses to start** if anything is wrong — it does not silently fall back. The
MiniMax-M3 `spec.py` `refusal_reasons()` checks 20+ conditions; a mismatch raises
a `ValueError` that lists **every** reason. Typical requirements:

- GPU is gfx950 (CDNA4) with **exactly 256 CUs** (not partitioned).
- Specific **TP size** (e.g. 4 for MiniMax-M3, 8 for Kimi-K3 / DS-V4-Pro).
- No PP / DP / EP / context-parallel / ubatching / EPLB / LoRA (model-dependent).
- Correct **KV cache** dtype/layout (e.g. FP8, block size 128) and quant layout
  (e.g. `AITER_MXFP4_MXFP4`).
- **Model Runner V2**, the right MoE backend, no sleep mode / weight transfer /
  KV connector, etc.

All ranks must **agree** on the decision — `tp_agree` is a `MIN` all-reduce over
the gloo CPU group, so if any rank refuses, all refuse. (One rank silently
turning the kernel off would leave the others spinning on it forever.)

### Per-step gate (fall back quietly)

Even on a valid deployment, each individual step is checked, and anything that
doesn't fit runs the **normal layers** instead:

- **Pure decode** (or spec-verify) — no prefill, no mixed step.
- **Row count** in the built widths (e.g. 1–16).
- **Graph mode** is NONE/FULL, never PIECEWISE.
- Per-step routing reads **only rank-identical metadata**, so every rank makes
  the same choice (`routing.py` `decide_step`).

Padded rows in a FULL graph (marked by a negative KV slot) pass through the
kernel harmlessly — their MoE output comes out zero.

### The motivating numbers

**vLLM — GLM-5.2 on 8× MI350X** (RFC #60904):

| Metric | Stock | MonoKernel | Speedup |
| --- | --- | --- | --- |
| Kernels per decode step | ~2,459 | ~210 | — |
| Time/token, batch 1 (BF16 attn) | 13.70 ms | 5.92 ms | 2.31× |
| Time/token, batch 8 (BF16 attn) | 17.33 ms | 11.44 ms | 1.51× |
| Time/token, batch 1 (FP8 attn opt-in) | 13.70 ms | 4.80 ms | 2.85× |

**SGLang — DeepSeek-V4.1-Flash, TP4 on MI350X** (RFC #43949): time-per-token cut
**43% at 1 user**, **18% at 4 and 8 users**.

The gain is biggest at **batch 1** (overhead dominates most there) and shrinks as
concurrency rises (more real work per launch).

---

## Part 6 — How it's wired into vLLM

vLLM carries the kernels **in-tree**, per model, under `vllm/models/<model>/amd/`.
There is no shared package in this checkout: `vllm/models/common/` holds only
`ops/`. Models that want shared runtime helpers **import DeepSeek-V4.1's**
`mono/common/` (MiniMax-M3 does), or keep their own adapted copy (DS-V4-Pro,
Kimi-K3, Muse Spark). That shared toolkit is itself adapted from **ROCm/ATOM**'s
`atom/mono`.

### The switch

`--monokernel` sets `KernelConfig.monokernel`
(`vllm/config/kernel.py:302`, wired in `vllm/engine/arg_utils.py`):

```python
monokernel: bool = False
"""If True, run the eligible layers of decode steps on the model's
MonoKernel: a persistent kernel that runs most or all of one decoder layer
in a single launch (...). A deployment that cannot run it is refused at startup."""
```

Caveat: not every model reads the same switch. **Kimi-K3** reads
`--additional-config '{"decode_backend": "latency_specialized"}'` instead, and
**MiniMax-M3's whole-layer path** (in this checkout) still reads the env var
`VLLM_ROCM_MINIMAX_M3_MONO`. Passing the "wrong" switch for a model is silently
ignored today (you just get stock decode, no error). Open PRs tidy this up:
[#66184](https://github.com/vllm-project/vllm/pull/66184) moves MiniMax onto
`--monokernel`, and [#67315](https://github.com/vllm-project/vllm/pull/67315)
aliases `--monokernel` and `decode_backend=latency_specialized` into one switch.

### The four integration styles

vLLM models attach the kernel in one of four ways:

1. **Layer mode (the RFC #60904 style)** — the layer loop asks for a *plan* once
   per step, then each layer uses its kernel or falls back to the stock layer.
   Used by **MiniMax-M3's FFN path** and **Kimi-K3**. This is the reference; code
   below.
2. **Per-layer `self.mono`** — the decoder layer holds a `MonoDecodeLayer` and
   tries it first in `forward`. Used by **DeepSeek-V4 / V4.1**
   (`deepseek_v41/amd/model.py:303`, `self.mono = MonoDecodeLayer.create(...)`).
3. **`maybe_attach_mono` hook** — the kernel is attached after weight loading and
   `forward` calls `self.mono.try_forward(...)`. Used by **Muse Spark**
   (`muse_spark/amd/model.py:1217` and `:974`). Health is polled in
   `compute_logits`.
4. **Whole-model forward replacement** — the gate swaps in a different forward.
   Used by **MiniMax-M3's whole-layer path** (`mono/dispatch.py`,
   `VLLM_ROCM_MINIMAX_M3_MONO`).

### The smallest complete example: MiniMax-M3 FFN launch

`vllm/models/minimax_m3/amd/specialized_decode/` (~2,850 lines) is the cleanest
end-to-end example. Its layout:

- `runtime/spec.py` — `DecodeConfig` (`tp_size=4, blocks=256, max_rows=16,
  widths=1..16, first_layer=3, last_layer=59`), `refusal_reasons()` (20+ checks),
  and `specialized_decode_config()` which returns `None` unless `monokernel` is
  set, else raises listing every reason.
- `runtime/guards.py` — `model_refusals`: ROCm, FlyDSL, gfx950, CUs == 256.
- `runtime/weights.py` — `MonoLayerWeights.from_layer`: zero-copy, checked views
  of the loaded weights (requires the `AITER_MXFP4_MXFP4` layout).
- `runtime/routing.py` — `decide_step`: the per-step gate.
- `runtime/decode.py` — `SpecializedDecode.plan()`, `Step.layer()`,
  `after_step()` (the fail-stop assert).
- `runtime/executor.py` — `FfnExecutor`: builds the `PeerBuffer`, warms every
  width, runs `tp_agree`, handles epoch resets.
- `mono/ffn.py` — the kernel (`run_ffn`, `build_ffn`).
- `mono/runner.py` — `FfnRunner`: one compiled kernel + scratch per width; its
  `warm()` refuses a build when `per_cu * compute_units < BLOCKS`.

**The layer-mode hook** (`model.py:1806`): plan once, then per layer use the
kernel or the stock layer, and run the fail-stop check after the loop:

```python
step = decode.plan(positions, intermediate_tensors)
failure: Exception | None = None
try:
    for idx, layer in enumerate(layers, start=self.start_layer):
        kernel = step.layer(layer.layer_id) if step else None
        hidden_states, residual = (kernel or layer)(positions, hidden_states, residual)
        ...
except Exception as err:
    failure = err
decode.after_step(step, failure)   # device-side assert the error word is clear
```

**The per-layer kernel** (`executor.py`): note vLLM does the norm and attention,
then one launch (`runner(...)`) does everything after — this is the K2 pattern:

```python
def run(positions, hidden_states, residual):
    hidden_states, residual = layer.input_layernorm(hidden_states, residual)
    part = layer.self_attn(positions=positions, hidden_states=hidden_states)
    launches[layer_id] += 1
    return runner(part, residual, weights, self._slots, final)
```

**Model Runner V2 lifecycle** (`model_state.py`): `MiniMaxM3ModelState`
(returned from `model.py:2002` `get_model_state_cls`) subclasses
`DefaultModelState` with three hooks — build the decode after load
(`build_specialized_decode`), `bind_caches()` at KV init, and
`executor.output_step()` after each step.

---

## Part 7 — How it's wired into SGLang

SGLang's design is different: **only the contract lives in-tree**; the kernels
live in an **external plugin** (`sgl-amd-mono`) loaded via a Python entry point.
(The local SGLang checkout predates this feature — the facts below are from
upstream `main` + the plugin repo.)

### The contract (`python/sglang/kernels/plugin_kernels/amd/spec.py`)

This file defines `MONOKERNEL_API_VERSION`, the backend name `"amd"`, the flag
`--enable-amd-monokernel`, and the data types a plugin fills in:

- `Feature` — an enum of SGLang features that are **refused unless a spec opts
  in** (speculative decoding, EP, LoRA, PD disaggregation, partitioned GPUs, …).
- `DeviceLimits(widths, tp_sizes, resident_blocks)` — what a spec supports on one
  ISA.
- `Requirements(dtypes, attention_backends, server_args, hf_config, env)` —
  config the spec needs; empty fields accept anything.
- `Numerics(kind, bounds)` — `BITWISE` or `TOLERANCE`, with bounds named
  `<launch>.<output>.<metric>`.
- `ShapeKey(forward_mode, batch_size, tokens_per_request)` — the **graph key** a
  step is routed by.
- `MonoSpec` — the whole thing a plugin provides: `name`, `devices`, `features`,
  `requires`, `numerics`, `model_disable_reasons`, `build`, `api_version`, plus
  optional `config_disable_reasons`, `load_tensor`, `tokens_per_request`.
- `register(backend=, architecture=, path=, model_type=)` / `find_spec(...)` —
  the registry.

### The glue (`python/sglang/srt/kernel_plugins/amd/binding.py`)

- `handle_monokernel(server_args)` runs at **argument resolution**: it validates
  the flag, refuses with every reason if the deployment doesn't fit, and aligns
  the CUDA-graph capture batch sizes to the spec's widths
  (`covered_capture_sizes`).
- `monokernel_weights_iterator(...)` lets a spec observe checkpoint tensors
  **before** SGLang converts them (for the weights it must copy), called in
  `model_loader/loader.py:1100`.
- `install_monokernel_forward(...)` runs **after load, before KV-cache sizing**
  (`model_runner.py:1410`): it `all_gather`s the spec choice so all ranks agree,
  builds the spec, and **wraps `model.forward`**.

The forward wrapper picks a `ShapeKey` from the batch; if the key is served it
calls the built kernel, otherwise it calls the original forward:

```python
original = model.forward

@functools.wraps(original)
def forward(*args, **kwargs):
    batch = kwargs["forward_batch"] if "forward_batch" in kwargs else args[2]
    key = shape_key(batch)
    if key not in served:
        return original(*args, **kwargs)   # stock path
    return built.forward(Plan(key=key), *args, **kwargs)

model.forward = forward
```

So unlike vLLM (which swaps *layers* inside the model's own loop), SGLang
replaces the **whole forward** for a served step, and routing is keyed on the
CUDA-graph shape (so a replayed graph keeps the routing it had at capture).

### The plugin (`sgl-project/sgl-amd-mono`)

Packaged with an entry point:

```toml
[project.entry-points."sglang.srt.plugins"]
sgl_amd_mono = "sgl_amd_mono:register_specs"
```

`register_specs()` registers a `SPECS` dict keyed by `(architecture, model_type)`
→ `"module:SPEC"`. `common/` holds the device runtime (epoch, guard, peer, plan,
sync, ops, hip_ext). Each model's `spec.py` provides a `MonoSpec` and a `_Built`
object whose `forward` runs SGLang's own embedding/attention plus the mono
launch(es). Supported models today: DS-V4.1-Flash, DS-V4-Pro, GLM-5.3(-Flash),
Kimi-K3/K3.1, MiniMax-M3/M3.1-Flash, and two Qwen3.8 variants.

---

## Part 8 — How to add a mono kernel

Common first steps (framework-independent):

1. **Pick one exact target.** One model, one GPU (gfx950), one TP size, one quant
   format, one row cap (e.g. 1–16). Mono kernels are specialized, not general.
2. **Profile the stock decode at low concurrency** and count kernels per step.
   That's your budget to beat (recall ~2,459 → ~210 for GLM-5.2).
3. **Write down the exact compute contract** — which ops you fuse and *exactly*
   how the stock path computes each one (dtypes, reduction order, quant layout).
   Your target is bitwise (or tightly bounded) equality against stock.
4. **Write the kernel in FlyDSL** against the stock ops, reusing `common/`
   (mailbox sync, peer memory, `first_task`, build keys). Start with **K2 (FFN)**
   — it's self-contained — then add **K1 (attention)** if the profile justifies
   it.
5. **Write the host runner**: one compiled kernel + scratch per width, an epoch
   word, warm every width at startup, and **check residency** (refuse a build
   whose grid can't stay resident).
6. **Make all three refusal points report every reason**, with all ranks
   agreeing (`tp_agree`), and make per-step routing read only rank-identical
   metadata.
7. **Make it fail-stop**: bounded waits + sticky error word + a post-loop assert.
8. **Validate**: numerics vs. stock first, then accuracy evals, then long-context
   and latency benchmarks, then fault injection.

### vLLM-specific wiring

- Put the code under `vllm/models/<model>/amd/specialized_decode/` with
  `runtime/` (refusals, routing, decode, executor, weights, guards) and `mono/`
  (the kernel + runner). Reuse `deepseek_v41/amd/mono/common/`.
- Read `config.kernel_config.monokernel` in the model's `__init__` and return a
  config only when set (else raise listing refusals).
- Add a `DefaultModelState` subclass (returned from `get_model_state_cls`) that
  builds the decode after load, binds caches at KV init, and does per-step
  bookkeeping.
- Hook the layer loop (layer mode) or the layer/model forward (the other styles).

### SGLang-specific wiring

Put the kernels in the `sgl-amd-mono` plugin under
`sgl_amd_mono/<model>/gfx950/`, waiting through `common/`, then write the spec.
A minimal spec skeleton:

```python
from sglang.kernels.plugin_kernels.amd.spec import (
    DeviceLimits, MonoSpec, Numerics, NumericsKind, Requirements,
)

def model_disable_reasons(model, hf_config, copies) -> list[str]:
    reasons = []
    # ... check weights/layouts the kernel assumes ...
    return reasons

def build(model, hf_config, ctx) -> "BuiltMonokernels":
    # allocate PeerBuffer, build FfnRunner, warm ctx.widths, return _Built(...)
    ...

SPEC = MonoSpec(
    name="mymodel_hip",
    devices={"gfx950": DeviceLimits(widths=(1, 2, 4, 8),
                                    tp_sizes=(4,),
                                    resident_blocks=256)},
    features=frozenset(),                      # refuse everything not opted into
    requires=Requirements(dtypes=frozenset({"bfloat16"}),
                          env={"SGLANG_USE_AITER": True}),
    numerics=Numerics(NumericsKind.BITWISE),
    model_disable_reasons=model_disable_reasons,
    build=build,
    api_version=1,
)
```

Then register it (in the plugin's `SPECS` dict):

```python
("MyModelForCausalLM", None): "sgl_amd_mono.mymodel.spec:SPEC",
```

The spec's `_Built.forward` must return exactly what the model's forward returns
and write the KV cache for **every** row (padded rows included), staying
CUDA-graph capturable (no host sync, shapes fixed by `plan.key`).

---

## Part 9 — Pitfalls (learned from the real PRs)

- **JIT cache key must include build params *and* source.** Build parameters go
  in a **frozen dataclass of plain scalars**, and the key includes a **hash of
  the kernel source files** (`key_tuple`, `source_digest`). Miss this and an edit
  to the kernel silently reuses a stale compiled binary.
- **Reduction order is part of the numerics.** The all-reduce sums partials in
  **rank order in fp32**. Change the order and you lose bit-exactness (and rank
  agreement).
- **Flags must not overtake their data.** Drain plain stores (`s_waitcnt`) and
  barrier *before* raising a mailbox flag (`publish`).
- **Grid must equal the CU count.** 256 workgroups, all resident. Partitioned
  GPUs, or too few CUs, → refuse (the runner checks `per_cu * CUs >= BLOCKS`).
- **Wrong CUDA-graph mode.** FULL or eager only; PIECEWISE stays stock.
- **Keep register use low** enough that a whole workgroup fits on each CU — the
  runner refuses a build that can't stay resident.
- **All ranks must branch together.** Any rank that turns the kernel off alone
  leaves its peers spinning. Route on rank-identical metadata and gate with
  `tp_agree`.

---

## Reading list

Start here, in order:

1. `vllm/models/deepseek_v41/amd/mono/README.md` — the reference two-launch
   design, and `mono/common/` (the shared toolkit).
2. `vllm/models/minimax_m3/amd/specialized_decode/` — the smallest complete
   end-to-end example (and its `README.md`).
3. `vllm/config/kernel.py` — the `--monokernel` definition.
4. The `sgl-amd-mono` README + `minimax_m31/spec.py` — the SGLang contract in
   practice.

### References

- vLLM RFC #60904 — [A standard for Specialized (MonoKernels) Decodes](https://github.com/vllm-project/vllm/issues/60904)
- vLLM PR #60397 — [DS-V4.1 mono decode](https://github.com/vllm-project/vllm/pull/60397) (merged 2026-10-10; the first one in)
- vLLM PR #62256 — [ROCm decode megakernel authoring guide](https://github.com/vllm-project/vllm/pull/62256) (open)
- vLLM PR #66184 — [MiniMax-M3 onto --monokernel](https://github.com/vllm-project/vllm/pull/66184) (open)
- vLLM PR #67315 — [unify --monokernel and decode_backend](https://github.com/vllm-project/vllm/pull/67315) (open)
- SGLang RFC #43949 — [Monokernel framework for SGLang](https://github.com/sgl-project/sglang/issues/43949)
- [ROCm/ATOM](https://github.com/ROCm/ATOM) — `atom/mono`, the design the shared runtime is adapted from.
- FlyDSL ships inside **AITER** — no separate checkout needed to read or run the kernels.
