# 00 — About Kimi K3, in plain English

*A non-specialist introduction. No prior knowledge of transformers assumed. Everything here is
described more precisely in docs 01–05.*

---

## What it is

**Kimi K3** is a large language model from Moonshot AI — the same category of thing as GPT, Claude, or
DeepSeek. You give it text (or images), it produces text. **SGLang** is the *serving engine*: the
software that actually runs the model on GPUs and answers requests fast enough to be useful. This
documentation set is about how SGLang runs K3, not about how Moonshot trained it.

Two framings that usually land:

- The model is the *recipe*; SGLang is the *kitchen*. Same recipe, a good kitchen serves a hundred
  covers an hour instead of five.
- Or: the model file is a few terabytes of numbers sitting on disk. It does nothing. SGLang is what
  turns it into something you can type at.

K3 is **multimodal** (it can look at images) and it is a **reasoning model** (it can think privately
before answering — the "thinking" text is marked off in the output so the serving layer can hide it).

---

## The one thing to understand: it's a stack of 93 layers

Text comes in at the bottom, passes through 93 near-identical processing stages, and a prediction
comes out the top. Each stage does two jobs:

1. **Attention** — "which earlier words matter for the word I'm writing now?" This is how the model
   knows that "it" in a sentence refers back to some noun three paragraphs ago.
2. **A feed-forward step** — the actual "thinking" about the word, once attention has gathered the
   relevant context.

Nearly every modern LLM is that pattern. What makes K3 distinctive is *how* it does each of the two,
plus one genuinely unusual thing about how information flows between the layers.

---

## What the architecture contains

### 1. Two different kinds of attention, mixed together (69 + 24)

Standard attention has an expensive habit: to answer about word 100,000 it re-examines a stored
record of all 99,999 earlier words. That stored record is called the **KV cache**, and it grows
without limit as the conversation gets longer. Long conversations get slow and memory-hungry — this
is *the* central cost problem in serving LLMs.

K3 splits the difference:

- **69 layers use KDA (Kimi Delta Attention)** — a *recurrent* design. Instead of keeping a record of
  every past word, each of these layers keeps a **fixed-size summary** that it updates as it reads.
  Think of a running set of notes rather than a full transcript. The notes are the same size whether
  the conversation is 1,000 words or a million. Cheap, constant memory — but lossy, since a summary
  is a summary.

  The "delta" in the name is the clever bit: when a new word arrives, the layer doesn't blindly append
  it to the notes. It first checks *what the notes already predict about this word*, and writes down
  only the part it got wrong. Correcting your notes rather than re-copying the page.

- **24 layers use MLA (Multi-head Latent Attention)** — real, exact attention that does look back at
  everything, but stores each past word in a compressed form rather than at full size. This is the
  design DeepSeek popularised; K3 borrows it essentially unchanged.

**Why mix them?** The recurrent layers are cheap but forgetful; the exact layers are precise but
expensive. Roughly three-quarters cheap and one-quarter exact keeps quality up while making long
contexts affordable. Only 24 of the 93 layers pay the growing-memory cost.

> The practical consequence: K3's memory footprint at long context is dominated by a *fixed* per-user
> allocation rather than a per-word one. Operators tune the balance between the two pools with a single
> knob, `--mamba-full-memory-ratio`.

### 2. A mixture of experts — 896 specialists, 17 consulted per word

The feed-forward half of each layer is not one big block. It's **896 smaller ones ("experts")**, plus
one generalist that always participates. For each individual word, a small router picks the
**16 experts** most relevant to it, and only those 16 run. The other 880 sit idle.

This is why the model can be enormous in total size yet fast in practice: it's a large panel of
consultants where you only ever call in the handful you need. Different words wake up different
experts.

K3 adds a wrinkle here that is unusual: before consulting the experts, it **shrinks the data to half
size** (7168 numbers per word → 3584), runs the experts in that compressed space, then expands back.
Since the experts typically live on *different GPUs*, the data has to be shipped across the network to
reach them — and shrinking it first **halves the network traffic**. In a big deployment that network
hop is often the slowest step in the whole layer, so halving it is a large win.

### 3. The attention-residual bank — K3's most unusual feature

In an ordinary model, information passes from layer 1 to 2 to 3 in a single stream, each layer adding
its contribution. Later layers see only the accumulated total; the intermediate states are gone.

K3 keeps a **scrapbook**. Every few layers, it saves a snapshot of the current state (up to 8
snapshots). Then at each layer, instead of just reading the latest state, it looks at *all* the saved
snapshots plus the current one, and **learns how much weight to give each**. A layer can decide "for
this word, the state as of layer 20 is more useful than the state right now" and lean on it.

Version history with the ability to blend versions, rather than a single overwritten document. This is
the feature with no equivalent in DeepSeek-V3 or most other open models, and it's what the largest
share of the custom kernel work in the SGLang branch exists to make fast.

### 4. No position numbers in the exact-attention layers

Most models explicitly tag each word with its position ("this is word 47"). K3's 24 MLA layers **don't
do this at all**. Position information reaches them implicitly, via the 69 recurrent layers, which
inherently read in order and so know where they are.

A quirk with real consequences for engineers: a lot of highly-optimised attention code assumes those
position tags exist and fuses their computation into the main kernel. For K3 there is nothing to fuse.

### 5. Vision

Images go through a separate component — a **vision tower** — that converts a picture into the same
kind of numbers the text layers consume, after which the model treats image content and text content
uniformly. K3 handles transparent images by compositing them onto a configurable background rather
than mangling the transparency, which is a small but real improvement over the previous generation.

---

## What SGLang does to make it fast

If you only remember one idea from the engineering side, make it this one: **at the scale these models
run, the bottleneck is usually moving data, not computing on it.** GPUs finish the arithmetic and then
wait — for memory, or for the network between GPUs.

So most of the optimisation work in the SGLang K3 branch is a variation on one theme: **stop moving
things twice.**

| Technique | The plain version |
|---|---|
| **Kernel fusion** | Two operations that ran back-to-back get merged into one, so the intermediate result never leaves the chip. K3 does this a dozen different ways — even fusing a *network operation* with the math that follows it. |
| **Merged weights** | Three separate small multiplications that all read the same input become one bigger multiplication. Read the input once instead of three times. |
| **Overlapping** | While the GPU waits on the network, give it unrelated math to do. K3 runs the "generalist expert" during the network wait for the specialists — worth about 4–5% throughput for free. |
| **Splitting work across GPUs** | Several different axes: split the layers, split the experts, split the attention heads, split the words. Each has different traffic costs, and the right combination depends on the hardware. |

Most of this only runs on **NVIDIA Blackwell** (B200/B300/GB200/GB300) hardware, which has capabilities
the older generations lack. On NVIDIA Hopper (H100/H200) and on AMD, the model still runs correctly —
it just falls back to generic, slower paths for these specific optimisations.

---

## Numbers worth remembering

| | |
|---|---|
| Layers | 93 — **69 recurrent (KDA) + 24 exact (MLA)** |
| Experts | 896 specialists + 1 generalist; **16 consulted per word** |
| Working width | 7168 numbers per word, **halved to 3584** inside the expert stage |
| Vocabulary | 163,840 distinct tokens |
| Snapshots in the residual bank | up to 8 |
| Typical deployment | 8 to 32 GPUs across 1–4 machines |

---

## A 30-second version

> Kimi K3 is a large multimodal reasoning model. Its defining trick is a hybrid memory design: about
> three-quarters of its layers keep a small fixed-size running summary of the conversation instead of a
> full transcript, so long contexts stay cheap, while the remaining quarter does exact lookback to keep
> quality high. Its feed-forward compute is a mixture of 896 experts of which only 16 run per word, and
> it compresses the data in half before shipping it to those experts to halve the network cost. It also
> keeps a small "scrapbook" of earlier internal states that every layer can blend from — an unusual
> feature most models don't have. SGLang is the engine that serves it, and most of the engineering
> there is about fusing operations together so data gets moved once instead of several times.

---

## Where to go next

| Doc | For |
|---|---|
| [01 — Model architecture](01-model-architecture.md) | the block diagrams, layer dataflow, attention-residual mechanism |
| [02 — Attention: KDA + MLA](02-attention-kda-mla.md) | the two attention types in detail, state cache, speculative decoding |
| [03 — Latent MoE and expert parallelism](03-moe-latent-and-ep.md) | routing, experts, all-to-all communication |
| [04 — Kernels and collectives](04-kernels-and-collectives.md) | the fused CUDA kernels, env vars |
| [05 — Serving and deployment](05-serving-and-deployment.md) | per-hardware launch recipes, Docker, parsers |
| `kimi-k3-architecture.html` | all of the above as one browsable page |
