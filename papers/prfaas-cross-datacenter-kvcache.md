# Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter

**Paper ID**: arXiv:2604.15039v1
**Authors**: Ruoyu Qin, Weiran He, Yaoyu Wang, et al. (Moonshot AI & Tsinghua University)
**Date**: April 2026
**Conference**: Submitted
**Tags**: `LLM-serving` `distributed-systems` `inference-optimization` `hybrid-attention` `prefill-decode-disaggregation`

---

## TL;DR

PrfaaS enables cross-datacenter LLM serving by leveraging hybrid attention models that reduce KVCache by ~10×. By selectively offloading long-context prefill to compute-dense clusters and transferring KVCache over commodity Ethernet, it achieves **54% higher throughput** than homogeneous deployments while using only **13 Gbps** of cross-cluster bandwidth.

---

## Problem Statement

### Current Limitations
Traditional prefill-decode (PD) disaggregation is confined to single RDMA-connected datacenters because:
- **Massive KVCache transfer**: Dense attention models generate 60+ Gbps for 32K tokens
- **Tight coupling**: Prefill and decode must share the same high-bandwidth fabric
- **Limited heterogeneity**: Can't deploy compute-optimized (prefill) and bandwidth-optimized (decode) accelerators in different locations
- **Poor elasticity**: Fixed prefill-to-decode hardware ratio can't adapt to traffic changes

### Key Bottleneck
```
KV Throughput Φ_kv = KVCache_size / Prefill_time

Dense models (MiniMax-M2.5, 32K tokens): 59.93 Gbps
→ Requires RDMA-class interconnect
→ Forces deployment into single datacenter
```

---

## Core Contribution

### 1. Hybrid Attention Changes the Game

**Model Architecture Evolution:**
- **Dense models**: All layers use full attention (O(n²) complexity, O(n) KVCache)
- **Hybrid models**: Mix of linear/SWA layers + sparse full-attention layers

**Representative Models:**

| Model | Architecture | Ratio | Params | KV Throughput (32K) |
|-------|-------------|-------|--------|---------------------|
| Ring-2.5-1T | Lightning + MLA | 7:1 | 1T | 2.59 Gbps |
| MiMo-V2-Flash | SWA + GQA | 5:1 | 309B | 4.66 Gbps |
| Qwen3.5-397B | GDN + GQA | 3:1 | 397B | 8.25 Gbps |
| Kimi Linear | KDA + MLA | 3:1 | 48B | 3.87 Gbps |

**Impact**: 10-13× reduction in KVCache bandwidth vs dense models

### 2. PrfaaS Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Requests                           │
└────────────────┬────────────────────────────────────────────┘
                 │
         ┌───────▼────────┐
         │ Request Router │  (Length-based threshold)
         └───┬────────┬───┘
             │        │
    L > t   │        │  L ≤ t
             │        │
┌────────────▼─────┐  │  ┌──────────────────────┐
│  PrfaaS Cluster  │  │  │  Local PD Cluster    │
│  (Compute-Dense) │  │  │ (Bandwidth-Optimized)│
│                  │  │  │                      │
│  ┌────────────┐  │  │  │ ┌────────┐┌────────┐│
│  │ Prefill    │  │  └─►│ │Prefill ││ Decode ││
│  │ Nodes      │  │     │ │ Nodes  ││ Nodes  ││
│  │ (MI300X)   │  │     │ │(MI355) ││(MI355) ││
│  └─────┬──────┘  │     │ └───┬────┘└────▲───┘│
│        │         │     │     │          │    │
│   ┌────▼─────┐   │     │ ┌───▼──────────┴─┐  │
│   │Local KV  │   │     │ │  Local KV      │  │
│   │  Store   │   │     │ │  Store         │  │
│   └────┬─────┘   │     │ └────────────────┘  │
└────────┼─────────┘     └─────────────────────┘
         │                         ▲
         │  100G Ethernet          │
         └─────────────────────────┘
              Cross-Cluster KVCache
```

**Three Subsystems:**

1. **Compute**:
   - PrfaaS clusters (e.g., MI300X, H200) - prefill only
   - Local PD clusters (e.g., MI355, H20) - prefill + decode

2. **Network**:
   - Intra-cluster: 400G+ RDMA (InfiniBand/RoCE)
   - Cross-cluster: 100G Ethernet (VPC/dedicated lines)

3. **Storage**: Hybrid prefix cache pool
   - Linear attention states (request-level, fixed size)
   - Full attention KVCache (block-level, length-dependent)
   - Prefix-cache blocks (reusable, aligned)
   - Transfer-cache blocks (cross-cluster, discarded after transfer)

### 3. Selective Offloading Strategy

**Not all requests benefit equally from PrfaaS:**
- Short requests: Memory/communication-bound, can't fully utilize compute-dense accelerators
- Long requests: Compute-bound, benefit significantly from faster prefill hardware

**Length-Based Routing:**
```python
if incremental_length > threshold_t:
    route_to_prfaas()  # Remote compute-dense cluster
else:
    route_to_local_pd()  # Local prefill nodes
```

**Optimal Threshold:**
- Balances PrfaaS and local PD throughput
- Accounts for bandwidth constraints
- Example: t = 19.4K tokens for 1T hybrid model

### 4. Dual-Timescale Scheduling

**Short-term (seconds to minutes):**
- Monitor cross-cluster bandwidth utilization
- Adjust routing threshold dynamically
- If bandwidth > 85%: temporarily raise threshold
- Consider prefix cache affinity

**Long-term (hours to days):**
- Monitor pipeline stage utilization
- Rebalance prefill/decode instance ratio
- Convert nodes between roles within PD cluster
- Re-optimize routing threshold

### 5. Throughput Model

**System throughput limited by slowest stage:**

```
PrfaaS throughput:
Θ_prfaas = min(N_prfaas / T_prefill(l_long), B_out / S_kv(l_long))
           ↑ compute bound            ↑ bandwidth bound

PD-Prefill throughput:
Θ_pd-p = N_p / T_prefill(l_short)

PD-Decode throughput:
Θ_pd-d = (N_d × BS_max) / (T_decode × L_out)

System throughput:
Λ_max = min(Θ_prfaas/p, Θ_pd-p/(1-p), Θ_pd-d)
        where p = P(L > t) = fraction routed to PrfaaS
```

**Optimization variables:**
1. Routing threshold `t` (determines `p`, `l_long`, `l_short`)
2. PD cluster ratio `N_p / N_d`

**Optimal configuration satisfies:**
```
Θ_prfaas/p = Θ_pd-p/(1-p)  (balance producers)
Θ_prfaas + Θ_pd-p = Θ_pd-d  (balance producer-consumer)
```

---

## Experimental Results

### Setup
- **Model**: Internal 1T parameter hybrid (Kimi Linear architecture, 3:1 KDA:MLA ratio)
- **PrfaaS cluster**: 32 H200 GPUs (4 instances @ 8 GPUs each)
- **Local PD cluster**: 64 H20 GPUs (3 prefill + 5 decode instances)
- **Baseline**: 96 H20 GPUs homogeneous PD cluster
- **Network**: 100 Gbps cross-cluster Ethernet, 800 Gbps intra-cluster RDMA
- **Workload**: Log-normal distribution (μ=9.90, σ=1.00, range [128, 128K]), mean ~27K tokens

### Performance

| Metric | PrfaaS-PD | Homogeneous PD | Naive Heterogeneous | Improvement |
|--------|-----------|----------------|---------------------|-------------|
| **Threshold t** | 19.4K | — | — | — |
| **Instance Config** | 4 / 3 / 5 | — / 9 / 3 | 4 / — / 8 | — |
| **Mean TTFT** | 2.22s | 4.44s | 1.74s | **50% faster** |
| **P90 TTFT** | 3.51s | 9.73s | 3.51s | **64% faster** |
| **Throughput** | 3.24 req/s | 2.11 req/s | 2.45 req/s | **54% gain** |
| **vs Baseline** | 1.54× | 1.00× | 1.16× | — |

### Bandwidth Analysis
- **Cross-cluster traffic**: ~13 Gbps average
- **Link utilization**: Only 13% of 100 Gbps Ethernet
- **Offloaded requests**: 49.6% (those with L > 19.4K)
- **Mean offloaded length**: 44K tokens

**Key insight**: Selective offloading + hybrid models keep bandwidth demand well within commodity Ethernet capacity

### Comparison Breakdown

**vs Homogeneous PD (54% throughput gain):**
- Superior compute on PrfaaS allows fewer local prefill instances
- Freed capacity allocated to decode
- Long requests complete faster despite cross-cluster transfer

**vs Naive Heterogeneous (32% throughput gain):**
- Naive approach: all prefill on H200, all decode on H20
- No load balancing → severe imbalance
- PrfaaS: selective offloading keeps both stages balanced

---

## Technical Deep-Dives

### Hybrid Prefix Cache Pool

**Challenge**: Different KVCache semantics in hybrid models
- Linear attention: Request-level recurrent states (fixed size, exact-match reuse)
- Full attention: Block-level KVCache (grows with length, prefix-match reuse)

**Solution**: Separate groups backed by unified block pool

```
┌──────────────────────────────────────────────────┐
│         Unified Hybrid Cache Pool                │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐   │
│  │ P   │ P   │ T   │ T   │ F   │ F   │ F   │...│
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┘   │
│   P = Prefix-cache  T = Transfer-cache  F = Free│
└──────────────────────────────────────────────────┘
         ▲                           ▲
         │                           │
┌────────┴─────────┐       ┌────────┴────────────┐
│ Linear Attention │       │  Full Attention     │
│      Group       │       │      Group          │
│                  │       │                     │
│ Request-level    │       │ Block-level KVCache │
│ Recurrent States │       │ (grows with length) │
│ (fixed size)     │       │                     │
└──────────────────┘       └─────────────────────┘
```

**Block Categories:**
1. **Prefix-cache blocks**: Reusable across requests, block-aligned
2. **Transfer-cache blocks**: For cross-cluster transfer, discarded after delivery
3. **Free blocks**: Available for allocation

### Cross-Cluster KVCache Transfer

**Requirements for stable Ethernet transport:**
1. **Layer-wise prefill pipelining**: Overlap computation and transmission
2. **Multi-connection TCP**: Fully utilize available bandwidth
3. **Congestion monitoring**: Detect packet loss/retransmission early
4. **Smooth transfer traffic**: Avoid bursty arrivals

**Transfer process:**
```
PrfaaS Node                        PD Decode Node
────────────                       ──────────────
1. Prefill layer 0  ────┐
2. Send KV layer 0  ────┼─────────► Receive & buffer
3. Prefill layer 1  ────┤
4. Send KV layer 1  ────┼─────────► Receive & buffer
   ...                  │
N. Complete prefill ────┤
   Free transfer blocks │
                        └─────────► Begin decode
```

### Bandwidth-Aware Routing with Cache Affinity

**Decision factors:**
1. Incremental length vs threshold
2. Current bandwidth utilization
3. Cache location (PrfaaS vs PD cluster)

**Algorithm:**
```python
def route_request(total_len, cached_prefix_prfaas, cached_prefix_pd):
    bw_util = get_bandwidth_utilization()

    # Adjust threshold based on bandwidth pressure
    if bw_util > 0.85:
        effective_threshold = threshold * 1.5
    else:
        effective_threshold = threshold

    # Compute incremental lengths
    incr_prfaas = total_len - cached_prefix_prfaas
    incr_pd = total_len - cached_prefix_pd

    if bw_util < 0.5:  # Abundant bandwidth
        # Can afford cross-cluster cache transfer
        best_cache = max(cached_prefix_prfaas, cached_prefix_pd)
        incremental = total_len - best_cache
    else:  # Scarce bandwidth
        # Evaluate clusters independently
        incremental = total_len - cached_prefix_pd

    if incremental > effective_threshold:
        return 'prfaas'
    else:
        return 'local_pd'
```

---

## AMD MI300/MI355 Deployment Guide

### Hardware Mapping

| Component | AMD Accelerator | Rationale |
|-----------|----------------|-----------|
| **PrfaaS Cluster** | MI300X | 192GB HBM3, 1.3 PFLOPs FP16 - ideal for long-context prefill |
| **PD Prefill** | MI355 | Balanced compute/bandwidth for short prefills |
| **PD Decode** | MI355 | 8TB/s memory bandwidth - excellent for decode |

### Example Configuration

```yaml
PrfaaS Cluster:
  location: datacenter-east
  gpus: 64x MI300X (8 nodes × 8 GPUs)
  instances: 8 @ tensor_parallel=8
  max_seq_len: 131072
  network: 100G Ethernet to datacenter-west

Local PD Cluster:
  location: datacenter-west
  gpus: 64x MI355 (8 nodes × 8 GPUs)
  prefill_instances: 3 @ tensor_parallel=8
  decode_instances: 5 @ tensor_parallel=8
  network: 400G RDMA intra-cluster

Routing:
  threshold: 19400 tokens
  expected_offload_rate: 50%
  cross_cluster_bw: 13 Gbps (13% of 100G link)
```

### Expected Performance (1T Hybrid Model)

**Throughput calculation:**
```python
# PrfaaS (MI300X) - 15% faster prefill than H200
Θ_prfaas = 8 / (1.5 * 0.85) ≈ 6.3 req/s  (compute-bound)

# PD-P (MI355) - similar to H20
Θ_pd_p = 3 / 0.5 = 6.0 req/s

# PD-D (MI355) - better bandwidth than H20
Θ_pd_d = (5 × 256) / (0.025 × 1024) ≈ 50 req/s

# System throughput (p ≈ 0.5)
Λ_max = min(6.3/0.5, 6.0/0.5, 50) = min(12.6, 12.0, 50) ≈ 12.0 req/s
```

**Expected improvement over homogeneous MI355 deployment:**
- Throughput: ~50% gain
- TTFT: ~60% reduction for long requests
- Bandwidth: <15 Gbps cross-datacenter

### ROCm Implementation Considerations

1. **Backend**: Use `device='hip'` instead of `device='cuda'`
2. **Frameworks**: vLLM/SGLang fully support ROCm via PyTorch
3. **Memory**: MI300X's 192GB HBM3 enables larger cache pools
4. **Networking**: ROCm supports RDMA via UCX for intra-cluster
5. **Cross-cluster**: Standard TCP/Ethernet (no special ROCm requirements)

---

## Key Insights

### What Makes This Work

1. **Model architecture is the enabler**: 10× KVCache reduction makes commodity Ethernet viable
2. **Selective offloading is the optimizer**: Don't send everything cross-cluster, only long requests
3. **Scheduling bridges the gap**: Bandwidth-aware routing prevents congestion
4. **Cache locality matters**: Consider prefix matches when routing

### When to Use PrfaaS

**Good fit:**
- ✅ Long-context workloads (>16K average)
- ✅ Hybrid attention models (KDA, SWA, Lightning, etc.)
- ✅ Heterogeneous hardware across datacenters
- ✅ Need independent prefill/decode scaling
- ✅ High traffic variance (bursty arrivals)

**Poor fit:**
- ❌ Short-context workloads (<4K average)
- ❌ Dense attention models (traditional Transformers)
- ❌ Homogeneous hardware in same datacenter
- ❌ Low-latency requirements (<100ms TTFT)

### Deployment Checklist

- [ ] Verify model uses hybrid attention (check architecture)
- [ ] Profile KV throughput (target <5 Gbps @ 32K tokens)
- [ ] Measure cross-datacenter bandwidth (need 100G+ dedicated)
- [ ] Analyze request length distribution (fit log-normal to find optimal threshold)
- [ ] Set up bandwidth monitoring (track utilization, adjust threshold)
- [ ] Configure hybrid cache pool (separate linear/full groups)
- [ ] Implement dual-timescale scheduler
- [ ] Test cross-cluster transfer latency (should add <200ms to TTFT)

---

## Limitations and Future Work

### Current Limitations

1. **Model dependency**: Only works well with hybrid attention architectures
2. **Transfer latency**: Cross-datacenter adds 100-500ms overhead
3. **Bandwidth sharing**: Must provision dedicated lines for production
4. **Cache cold start**: Initial requests have no prefix cache benefit
5. **Scheduling complexity**: Requires sophisticated monitoring and adaptation

### Future Directions

**From the paper:**
- KVCache compression (H2O, KIVI) for further bandwidth reduction
- CacheGen-style streaming for lower latency
- Cross-cluster prefix cache replication
- Integration with KVCache quantization (FP8/FP4)
- Phase-specialized hardware co-design (CPX, LPU, etc.)

**For AMD ecosystem:**
- MI400 series optimization (expected 2027)
- ROCm-native RDMA improvements
- Chiplet-based disaggregation (future AMD architectures)
- Integration with AMD Infinity Fabric for multi-die KVCache

---

## Related Work

### Disaggregated Serving
- **Splitwise** (ISCA 2024): Cost-optimal PD disaggregation framework
- **DistServe** (OSDI 2024): Goodput-optimized PD with scheduling
- **Mooncake** (ACM TOS 2024): KVCache-centric architecture (same authors)

### Heterogeneous LLM Serving
- **Helix** (ASPLOS 2025): Max-flow heterogeneous GPU serving
- **Hetis** (SC 2025): Fine-grained parallelism for heterogeneous clusters
- **LLM-PQ** (PPoPP 2024): Phase-aware partition + adaptive quantization

### KVCache Optimization
- **CacheGen** (SIGCOMM 2024): KVCache compression and streaming
- **CacheBlend** (EuroSys 2025): Fast RAG with cached knowledge fusion
- **KIVI** (ICML 2024): 2-bit asymmetric quantization for KVCache
- **H2O** (NeurIPS 2023): Heavy-hitter oracle for efficient inference

### Hybrid Attention Models
- **Kimi Linear** (2025): KDA for long-context efficiency
- **Qwen3.5** (2026): GDN + GQA hybrid
- **Ring-2.5** (2026): Lightning attention at 7:1 ratio
- **MiMo-V2-Flash** (2026): SWA-based hybrid architecture

---

## References

- **Paper**: arXiv:2604.15039v1 [cs.DC] 16 Apr 2026
- **Code**: Not yet released (check Moonshot AI GitHub)
- **Related**: Mooncake (https://github.com/kvcache-ai/Mooncake)
- **Frameworks**: vLLM, SGLang (both support hybrid models as of 2025)

---

## Personal Notes

### Why This Matters

This paper represents a systems turning point where model architecture directly enables new deployment paradigms. The key insight isn't just that hybrid models reduce KVCache (that's well-known), but that this reduction crosses a **qualitative threshold** where cross-datacenter disaggregation becomes practical.

The paper is also notable for:
1. **Production-driven design**: From Moonshot AI (real production deployment experience)
2. **End-to-end thinking**: Model + system + hardware co-design
3. **Realistic evaluation**: Uses bursty workloads, bandwidth constraints, cache hotspots
4. **AMD relevance**: MI300X/MI355 are perfect fit for this architecture

### Questions to Explore

1. How does this interact with speculative decoding? (Draft model in PrfaaS vs PD?)
2. What's the minimum viable cross-cluster bandwidth? (Can we go below 100G?)
3. How to handle multi-tenant scenarios with different models?
4. Can we apply similar ideas to multi-modal models? (vision encoder as "prefill"?)
5. What about edge deployments? (PrfaaS in cloud, decode on edge?)

### Action Items

- [ ] Test with Qwen3.5-397B on MI300X cluster
- [ ] Profile KV throughput for different hybrid ratios
- [ ] Benchmark ROCm RDMA vs TCP for intra-cluster
- [ ] Implement bandwidth monitor with auto-threshold adjustment
- [ ] Measure impact of cache replication on bandwidth
- [ ] Compare with dense model baseline (e.g., Llama 3.1 405B)

---

**Last Updated**: 2026-04-21
**Reviewed By**: Knowledge Vault Curator
**Status**: ⭐ High Impact - Recommended Reading
