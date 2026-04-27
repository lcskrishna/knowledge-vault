# DeepSeek-V4 Technical Summary

## Executive Overview

DeepSeek-V4 is a breakthrough series of Mixture-of-Experts (MoE) language models designed for efficient million-token context processing. Released in April 2026, it includes:

- **DeepSeek-V4-Pro**: 1.6T parameters (49B activated)
- **DeepSeek-V4-Flash**: 284B parameters (13B activated)

Both models natively support 1M token context length with dramatically reduced computational requirements compared to predecessors.

### Key Efficiency Gains vs DeepSeek-V3.2 (at 1M context)

**DeepSeek-V4-Pro:**
- **27%** of single-token inference FLOPs
- **10%** of KV cache size

**DeepSeek-V4-Flash:**
- **10%** of single-token inference FLOPs
- **7%** of KV cache size

---

## Model Architecture

### 1. Hybrid Attention Mechanism (CSA + HCA)

The core innovation enabling ultra-long context efficiency is the hybrid attention architecture combining two novel attention mechanisms:

#### Compressed Sparse Attention (CSA)

**Purpose:** Balance efficiency and accuracy through compression + sparsity

**Key Components:**

1. **KV Cache Compression**
   - Compresses every β tokens into 1 entry
   - Uses learnable compression weights and positional biases
   - Overlapped compression: each compressed entry uses 2β KV entries
   - Final compression ratio: 1/β

2. **Lightning Indexer for Sparse Selection**
   - Low-rank indexer queries: `q = (h × W_down) × W_up`
   - Computes index scores between queries and compressed blocks
   - Top-k selector chooses most relevant compressed KV entries
   - Uses FP4 precision for indexer computation

3. **Shared Key-Value Multi-Query Attention (MQA)**
   - Each compressed KV entry serves as both key and value
   - Multiple query heads attend to shared KV
   - Grouped output projection reduces computational overhead

4. **Sliding Window Addition**
   - Maintains w_win recent uncompressed KV entries
   - Preserves local fine-grained dependencies
   - Ensures causality within compressed blocks

**Technical Details:**
- Mixed precision: BF16 for RoPE dimensions, FP8 for other dimensions
- Partial RoPE: Applied to last 64 dimensions only
- Attention sink: Learnable sink logits to adjust attention scores

#### Heavily Compressed Attention (HCA)

**Purpose:** Extreme compression for non-critical layers

**Key Differences from CSA:**
- Higher compression rate (γ >> β)
- No overlapped compression
- No sparse attention (dense attention on all compressed entries)
- Same shared KV MQA and grouped output projection

**Compression Formula:**
```
Compression ratio: 1/γ where γ >> β
```

#### Hybrid Configuration

- CSA and HCA layers are interleaved throughout the model
- CSA provides detailed attention where needed
- HCA provides extreme efficiency in other layers
- Combined KV cache reduction: ~2% of baseline BF16 GQA8 at 1M context

### 2. Manifold-Constrained Hyper-Connections (mHC)

**Enhancement over standard residual connections:**

**Standard Hyper-Connections (HC):**
- Expands residual stream width by factor hc
- Input mapping Ω_in, residual transformation Φ, output mapping Ω_out
- Update: `x_{l+1} = Ω_out × Φ × x_l + Ω_in × F_l(Ω_in^T × x_l)`

**mHC Constraints:**
1. **Doubly Stochastic Matrix Constraint** on Φ:
   - `Φ ∈ M = {Φ | Φ·1 = 1, 1^T·Φ = 1, Φ ≥ 0}`
   - Ensures spectral norm ||Φ||_2 ≤ 1 (non-expansive)
   - Achieved via Sinkhorn-Knopp algorithm (20 iterations)

2. **Non-negative Bounded Input/Output:**
   - `Ω_in = σ(Ω̃_in)` where σ is Sigmoid
   - `Ω_out = 2·σ(Ω̃_out)`

**Dynamic Parameterization:**
- Parameters are input-dependent
- Decomposed into dynamic and static components
- Uses learned gating factors initialized to small values

**Benefits:**
- Enhanced training stability
- Preserved model expressivity
- Stable signal propagation in deep stacks

### 3. DeepSeekMoE Architecture

**Retained from DeepSeek-V3 with modifications:**

1. **Fine-grained routed experts + shared experts**
2. **Activation function change:** Sqrt(Softplus(·)) instead of Sigmoid(·)
3. **Load balancing:** Auxiliary-loss-free with slight sequence-wise balance loss
4. **Routing enhancements:**
   - Removed constraint on number of routing target nodes
   - Hash routing for initial Transformer blocks
   - Hash function based on input token ID

5. **FP4 Quantization:**
   - MoE expert weights use FP4 precision
   - Currently same FLOPs as FP8 on existing hardware
   - Theoretically 1/3 more efficient on future hardware

### 4. Multi-Token Prediction (MTP)

**Identical to DeepSeek-V3:**
- MTP modules and objectives unchanged
- Auxiliary prediction heads for multiple future tokens
- Improves speculative decoding capabilities

### 5. Muon Optimizer

**New optimization approach for DeepSeek-V4:**

**Algorithm:**
```
1. Compute gradients: g_t = ∇L(θ_{t-1})
2. Accumulate momentum: m_t = βm_{t-1} + g_t
3. Apply Nesterov + Newton-Schulz: ĝ_t = HybridNewtonSchulz(βm_t + g_t)
4. Update weights with rescaling factor α
```

**Benefits:**
- Faster convergence
- Greater training stability
- Effective for large-scale MoE models

---

## Infrastructure Optimizations

### Training Framework

1. **Fine-Grained Communication-Computation Overlap**
   - Single fused kernel for MoE modules
   - Overlaps computation, communication, and memory access

2. **TileLang DSL Integration**
   - Domain-Specific Language for kernel development
   - Balances development productivity and runtime efficiency

3. **Batch-Invariant Deterministic Kernels**
   - Ensures bitwise reproducibility
   - Consistent across training and inference

4. **FP4 Quantization-Aware Training**
   - Applied to MoE expert weights
   - Applied to indexer QK path

5. **Extended Automatic Differentiation**
   - Tensor-level checkpointing
   - Fine-grained recomputation control

6. **Contextual Parallelism**
   - Two-stage approach for compressed attention
   - Manages CSA/HCA across devices

7. **Hybrid ZeRO Strategy**
   - Optimized for Muon optimizer
   - Memory-efficient mHC implementation via recomputation

### Inference Framework

1. **Heterogeneous KV Cache Structure**
   - Mixed precision storage: BF16 (RoPE) + FP8 (other dims)
   - Compressed format for CSA/HCA

2. **On-Disk KV Cache Storage**
   - Enables shared-prefix reuse
   - Critical for million-token contexts

3. **Efficient Cache Management**
   - Hierarchical storage (GPU memory → disk)
   - Smart eviction policies

---

## Pre-Training

**Data:**
- DeepSeek-V4-Flash: 32T tokens
- DeepSeek-V4-Pro: 33T tokens
- Diverse, high-quality data sources

**Native 1M Context Support:**
- Both models support 1M context out-of-the-box after pre-training
- No additional fine-tuning required for long-context capability

**Performance:**
- DeepSeek-V4-Flash-Base: Surpasses DeepSeek-V3.2-Base on most benchmarks
- DeepSeek-V4-Pro-Base: New performance standard across reasoning, coding, long-context, and knowledge tasks

---

## Post-Training Pipeline

### Two-Stage Paradigm

**Stage 1: Domain-Specific Experts**

For each domain (math, coding, agent, instruction following):
1. **Supervised Fine-Tuning (SFT)** on high-quality domain data
2. **Reinforcement Learning (RL)** using GRPO (Group Relative Policy Optimization)
   - Domain-aligned behaviors
   - Reward models for specific success criteria

**Stage 2: Unified Model Consolidation**

**On-Policy Distillation (OPD):**
- Single unified model learns from specialist teachers
- Optimizes reverse KL loss
- Combines expertise from all domain specialists

### Infrastructure for Post-Training

1. **FP4 Quantization Integration**
   - Applied during RL training

2. **Efficient Teacher Scheduling**
   - Full-vocabulary OPD support

3. **Preemptible Fault-Tolerant Rollout**
   - Robust distributed RL training

4. **Million-Token Context RL Framework**
   - Scaled RL infrastructure for ultra-long contexts

5. **Sandbox Infrastructure**
   - Agentic AI training and evaluation

---

## Performance Benchmarks

### Knowledge & Reasoning (DeepSeek-V4-Pro-Max)

- **SimpleQA:** 75.6% (leading open-source)
- **HLE:** 80.6% (Pass@1)
- **Verified SWEBENCH:** 68.5% (Resolved)
- **Terminal 2.0:** 78.1% (Acc)

### Agent Capabilities

- **Toolathlon:** 80.8% (Pass@1)
- **Codeforces:** 2063 Rating
- Comparable to Claude Sonnet 4.5, approaching Opus 4.5 level

### Long-Context

- Supports 1M tokens natively
- Outperforms Gemini-3.1-Pro on academic benchmarks
- Strong performance on synthetic and real use cases

### Efficiency Comparison

At 1M context vs DeepSeek-V3.2:
- **V4-Pro:** 3.7× fewer FLOPs, 9.5× smaller KV cache
- **V4-Flash:** 9.8× fewer FLOPs, 13.7× smaller KV cache

---

## vLLM Support

### Official Support Status

✅ **Day-0 support announced April 23, 2026**

**Supported Models:**
- deepseek-ai/DeepSeek-V4-Pro
- deepseek-ai/DeepSeek-V4-Flash

**Key Features:**
- ✅ MoE expert parallelism
- ✅ Hybrid CSA+HCA attention architecture
- ✅ Efficient KV cache management for 1M-token contexts
- ✅ OpenAI-compatible API
- ✅ Kernel fusion optimizations
- ✅ Disaggregated serving support

**Hardware Support:**
- NVIDIA Hopper (H100, H200)
- NVIDIA Blackwell (B100, B200, GB200)
- AMD Instinct MI300X (via vLLM-ROCm)
- AMD Instinct MI325X, MI350X, MI355X

**Installation:**
```bash
# Requires vLLM >= 0.9.0
pip install vllm>=0.9.0

# Basic usage
vllm serve deepseek-ai/DeepSeek-V4-Flash
```

**Resources:**
- [DeepSeek V4 in vLLM Blog](https://vllm.ai/blog/deepseek-v4)
- [vLLM Recipes - DeepSeek-V4-Pro](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro)
- [vLLM Recipes - DeepSeek-V4-Flash](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)

---

## SGLang Support

### Official Support Status

✅ **Day-0 support announced April 24, 2026**

**SGLang + Miles:** First open-source stack for DeepSeek-V4 serving and training

**Key Features:**
- ✅ Inference for both V4-Pro (1.6T) and V4-Flash (284B)
- ✅ RL training support (first framework with this capability)
- ✅ Hybrid sparse-attention architecture (CSA/HCA)
- ✅ mHC support
- ✅ FP4 expert weights
- ✅ FP8 checkpoints available

**Serving Recipes:**

Three primary configurations on NVIDIA Blackwell/Hopper:
1. **Low-latency** - Optimized for minimal response time
2. **Balanced** - Throughput/latency tradeoff
3. **Max-throughput** - Maximum requests per second

Plus specialized recipes for:
- Long-context workloads (>100K tokens)
- Prefill/decode disaggregation

**Hardware Support:**
- NVIDIA Blackwell and Hopper GPUs
- AMD MI300X, MI325X, MI355X (via ROCm)

**Installation:**
```bash
# Clone from v0.5.9 branch
git clone -b v0.5.9 https://github.com/sgl-project/sglang.git
cd sglang
python setup_rocm.py install  # For AMD ROCm
```

**Pre-built Docker Images:**
- `lmsysorg/sglang:deepseek-v4-hopper` (NVIDIA H200)
- `lmsysorg/sglang` (ROCm builds available)

**Resources:**
- [LMSYS Blog - DeepSeek-V4 on Day 0](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [SGLang Docs - DeepSeek-V4](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4)
- [FP8 Checkpoints](https://huggingface.co/sgl-project/DeepSeek-V4-Pro-FP8)

---

## ROCm & AMD MI300 Deployment

### Current Support Status (2026)

✅ **ROCm is now a FIRST-CLASS platform for vLLM and SGLang**

Major milestones:
- **January 6, 2026:** First pre-built ROCm vLLM Docker image
- **December 29, 2025:** Dedicated ROCm CI pipeline live
- **Early 2026:** 93% of AMD CI tests passing (up from 37% in Nov 2025)

### Supported Hardware

**AMD Instinct Series:**
- MI355X (latest)
- MI350X
- MI325X
- MI300X (primary deployment target)

### vLLM on ROCm/MI300

**Installation:**
```bash
# Pull official ROCm-enabled Docker image
docker pull vllm/vllm-openai:latest-rocm

# Or build from source for MI300X
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

**Key Optimizations:**
- **ROCM_AITER_FA** for Multi-Head Attention
- **AITER MLA backends:** 1.2-4.4× higher throughput
- Tested with ROCm 7.0, PyTorch 2.9.0a0, vLLM 0.14.0rc2

**Resources:**
- [AMD ROCm vLLM Docs](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/vllm.html)
- [vLLM ROCm Installation](https://docs.vllm.ai/en/v0.6.5/getting_started/amd-installation.html)
- [ROCm Blog - vLLM First-Class Platform](https://rocm.blogs.amd.com/software-tools-optimization/vllm-omni/README.html)

### SGLang on ROCm/MI300

**Installation:**
```bash
# Clone and build for ROCm
git clone -b v0.5.9 https://github.com/sgl-project/sglang.git
cd sglang
python setup_rocm.py install

# Or use Docker
docker pull lmsysorg/sglang:latest-rocm
```

**Quantization Support on AMD:**
- ✅ FP8 (works out-of-the-box)
- ✅ MXFP4 (hardware-level support on gfx94x/MI300X)
- ✅ AWQ, W8A8, GPTQ
- ✅ compressed-tensors, Quark, petit_nvfp4

**Pre-quantized Models:**
- DeepSeek-V3/R1 FP8 models work immediately
- DeepSeek-V4 FP8 checkpoints available

**Resources:**
- [SGLang AMD GPU Docs](https://docs.sglang.io/platforms/amd_gpu.html)
- [ROCm SGLang Benchmark](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/sglang.html)

### DeepSeek-V4 Specific on MI300

**Current Status:**
- ✅ DeepSeek-R1 and V3 fully validated on MI300X
- 🔄 DeepSeek-V4 support in progress (CSA/HCA kernels being optimized)
- Expected full support Q2 2026

**Single-Node Deployment:**
- Full DeepSeek R1 671B fits on single MI300X (192GB VRAM)
- Quantized V4-Flash expected to fit on single MI300X

**Performance (DeepSeek-R1 on MI300X):**
- Up to **4× faster** inference with SGLang optimizations
- **5× higher throughput** vs NVIDIA H200 under high concurrency
- **$1.0 per 1M tokens** cost (55% savings vs DeepSeek API)

**AITER Optimizations:**
- Block-scale GEMM: up to **2× boost**
- Block-scale fused MoE: up to **3× boost**
- MLA for decode: up to **17× boost**
- MHA for prefill: up to **14× boost**

**Distributed Deployment:**
- 8× MI300X/MI325X per node recommended
- 32 GPU setup: 1.3× higher throughput vs EP8
- Requires InfiniBand or RoCE for multi-node

**Resources:**
- [AMD Blog - DeepSeek-R1 on MI300X](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html)
- [Microsoft - DeepSeek on AMD MI300](https://techcommunity.microsoft.com/blog/machinelearningblog/accelerating-deepseek-inference-with-amd-mi300-a-collaborative-breakthrough/4407673)
- [AMD ROCm DeepSeek Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/deepseekr1_sglang.html)

---

## Implementation Changes Needed for ROCm/MI300

### 1. Kernel Adaptations for CSA/HCA

**Priority: HIGH**

**Attention Kernels:**
```python
# Current: CUDA-optimized kernels
# Required: ROCm/HIP equivalents

# Key areas:
- Lightning Indexer FP4 operations
- Compressed KV cache operations
- Hybrid attention switching logic
```

**Specific Changes:**

**a) FP4 Indexer Kernel:**
```cpp
// NVIDIA CUDA version uses specific tensor core configs
// ROCm needs WMMA (Wave Matrix Multiply-Accumulate) for MI300

#ifdef USE_ROCM
  #include <hip/hip_fp16.h>
  #include <rocwmma/rocwmma.hpp>
  // Adapt FP4 gemm to use ROCWMMA on gfx94x
  // MI300X has native MXFP4 support - use rocwmma::mxfp4
#endif
```

**b) Compressed Attention Kernels:**
```python
# TileLang DSL needs ROCm backend
# Current implementation targets CUDA

# Required modifications:
1. Add ROCm backend to TileLang compiler
2. Map CUDA shared memory to LDS (Local Data Share)
3. Adapt warp primitives to wavefront (64 threads on AMD vs 32 on NVIDIA)
```

**c) KV Cache Compression:**
```cpp
// Softmax + weighted sum for compression
// Need to optimize for MI300's matrix cores

// CUDA version uses cutlass templates
// ROCm version should use composable_kernel or rocwmma
#ifdef USE_ROCM
  #include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
  // Use Composable Kernel for fused compression ops
#endif
```

### 2. MoE Routing and Expert Parallel

**Priority: HIGH**

**Current Challenges:**
- DeepSeek-V4 uses Hash routing + top-k routing
- Requires efficient all-to-all communication
- NVIDIA uses NCCL, ROCm uses RCCL

**Required Changes:**

```python
# Expert parallelism communication
# File: deepseek_v4/moe/expert_parallel.py

def expert_parallel_all_to_all(input_tensor, expert_map):
    if torch.cuda.is_available() and torch.version.hip:
        # ROCm path - use RCCL optimized path
        # MI300X has faster GPU-GPU communication via Infinity Fabric
        return rccl_all_to_all_optimized(input_tensor, expert_map)
    else:
        # NVIDIA path - use NCCL
        return nccl_all_to_all(input_tensor, expert_map)
```

**Hash Routing Kernel:**
```cpp
// hash_routing_kernel.cu -> hash_routing_kernel.hip

#ifdef USE_ROCM
  #include <hip/hip_runtime.h>

  __global__ void hash_routing_kernel_hip(
      const int* token_ids,
      int* expert_indices,
      int num_tokens,
      int num_experts) {
    // AMD-specific optimizations
    // Use LDS for hash table lookups
    __shared__ int hash_table[MAX_EXPERTS];
    // wavefront-level operations (64 threads)
    int lane_id = threadIdx.x % 64;  // waveSize = 64 on AMD
  }
#endif
```

### 3. FP4/FP8 Mixed Precision

**Priority: MEDIUM-HIGH**

**MI300X Capabilities:**
- Native MXFP4 support (gfx94x)
- FP8 support via WMMA
- Need to enable in PyTorch/ROCm stack

**Required Changes:**

```python
# deepseek_v4/quantization/fp4_qat.py

def get_fp4_dtype():
    if torch.version.hip:
        # ROCm path - use torch.mxfp4 when available
        # Currently requires ROCm 7.0+
        if hasattr(torch, 'mxfp4'):
            return torch.mxfp4
        else:
            # Fallback to custom kernel
            return CustomMXFP4Tensor
    else:
        # NVIDIA path
        return torch.float8_e4m3fn

# Weight quantization for MoE experts
def quantize_expert_weights_fp4_rocm(weights):
    """
    Quantize to MXFP4 format for MI300X
    Uses block-wise scaling (32 elements per block)
    """
    # Implementation using rocBLAS or composable_kernel
    pass
```

### 4. Memory Management

**Priority: HIGH**

**MI300X Memory Characteristics:**
- 192GB HBM3 per GPU
- Unified memory architecture
- Different memory hierarchy than NVIDIA

**Required Adaptations:**

```python
# deepseek_v4/kv_cache/cache_manager.py

class KVCacheManager:
    def __init__(self, device):
        if torch.version.hip:
            # MI300X has 192GB - can cache more aggressively
            self.max_cache_size = 180 * 1024 * 1024 * 1024  # 180GB
            # Use ROCm memory pools
            self.use_hip_memory_pool = True
            # Different eviction strategy for larger memory
            self.eviction_policy = "lru_large_memory"
        else:
            # NVIDIA H100 has 80GB
            self.max_cache_size = 70 * 1024 * 1024 * 1024
            self.eviction_policy = "lru_standard"

    def allocate_kv_cache(self, shape, dtype):
        if self.use_hip_memory_pool:
            # Use hipMallocManaged for unified memory
            return hip_managed_alloc(shape, dtype)
        else:
            return torch.empty(shape, dtype=dtype, device=self.device)
```

### 5. Distributed Communication

**Priority: MEDIUM**

**RCCL Optimizations:**

```python
# deepseek_v4/distributed/comm.py

def init_distributed_backend():
    if torch.version.hip:
        # ROCm - use RCCL with MI300X optimizations
        torch.distributed.init_process_group(
            backend="rccl",  # Not "nccl"
            init_method="env://",
            timeout=datetime.timedelta(minutes=30)
        )

        # Enable MI300X Infinity Fabric optimization
        os.environ["RCCL_NET_GDR_LEVEL"] = "5"  # Enable GPU Direct RDMA
        os.environ["RCCL_CROSS_NIC"] = "2"      # Optimize multi-NIC
        os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
    else:
        # NVIDIA - use NCCL
        torch.distributed.init_process_group(backend="nccl")
```

### 6. Build System Changes

**Priority: HIGH**

**CMakeLists.txt modifications:**

```cmake
# CMakeLists.txt

if(USE_ROCM)
    # Set HIP compiler
    set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")

    # Find ROCm packages
    find_package(hip REQUIRED)
    find_package(rocblas REQUIRED)
    find_package(rocwmma REQUIRED)
    find_package(composable_kernel REQUIRED)

    # Set architecture for MI300X
    set(CMAKE_HIP_ARCHITECTURES "gfx942")  # MI300X

    # Add ROCm include directories
    include_directories(
        ${ROCM_PATH}/include
        ${ROCM_PATH}/include/rocwmma
        ${ROCM_PATH}/include/composable_kernel
    )

    # Link ROCm libraries
    target_link_libraries(deepseek_v4
        hip::device
        roc::rocblas
        composable_kernel::device_operations
    )

    # Compile .cu files as HIP
    set_source_files_properties(
        kernels/attention_kernels.cu
        kernels/moe_kernels.cu
        PROPERTIES LANGUAGE HIP
    )
endif()
```

**setup.py modifications:**

```python
# setup.py

def get_rocm_extensions():
    """Build extensions for ROCm/MI300X"""
    extensions = []

    if torch.version.hip:
        # ROCm-specific extensions
        rocm_extension = CUDAExtension(
            name="deepseek_v4_rocm",
            sources=[
                "csrc/attention/csa_kernel.hip",
                "csrc/attention/hca_kernel.hip",
                "csrc/moe/expert_routing.hip",
                "csrc/quantization/fp4_ops.hip",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--gpu-architecture=gfx942",  # MI300X
                    "-DUSE_ROCM",
                    "-DUSE_MXFP4",
                ]
            },
            libraries=["rocblas", "rocwmma", "composable_kernel"],
        )
        extensions.append(rocm_extension)

    return extensions
```

### 7. Testing and Validation

**Priority: MEDIUM**

**Create ROCm-specific tests:**

```python
# tests/test_rocm_kernels.py

import pytest
import torch

@pytest.mark.skipif(not torch.version.hip, reason="ROCm only")
def test_csa_kernel_rocm():
    """Test CSA kernel on MI300X"""
    device = torch.device("cuda")  # Still uses "cuda" in PyTorch for ROCm

    # Test with 1M context
    seq_len = 1_000_000
    hidden_size = 2048

    hidden_states = torch.randn(seq_len, hidden_size, device=device)

    # Run CSA
    output = compressed_sparse_attention(
        hidden_states,
        compression_ratio=16,
        top_k=128
    )

    assert output.shape == (seq_len, hidden_size)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

@pytest.mark.skipif(not torch.version.hip, reason="ROCm only")
def test_moe_routing_mi300x():
    """Test MoE routing with Hash + TopK on MI300X"""
    # Test expert parallel communication
    # Test FP4 expert weights
    # Test load balancing
    pass
```

### 8. Documentation Updates

**Priority: LOW**

**Create ROCm deployment guide:**

```markdown
# docs/rocm_deployment.md

## DeepSeek-V4 on AMD MI300X

### System Requirements
- AMD Instinct MI300X GPU (192GB VRAM)
- ROCm 7.0 or later
- PyTorch 2.9+ with ROCm support
- 8× MI300X recommended for full DeepSeek-V4-Pro

### Installation
...

### Performance Tuning
- Enable Infinity Fabric GPU-GPU communication
- Use RCCL optimizations for distributed training
- Tune MXFP4 quantization for MI300X
...
```

---

## Summary of ROCm/MI300 Implementation Checklist

### High Priority (Required for Basic Functionality)
- [ ] Port CSA/HCA attention kernels to HIP
- [ ] Implement FP4 indexer using ROCWMMA/MXFP4
- [ ] Adapt MoE routing and expert parallelism for RCCL
- [ ] Update memory management for 192GB HBM3
- [ ] Modify build system (CMake + setup.py)
- [ ] Create KV cache manager optimized for MI300X

### Medium Priority (Performance Optimizations)
- [ ] Optimize FP4/FP8 mixed precision for MI300X
- [ ] Tune distributed communication with RCCL
- [ ] Implement composable_kernel fused operations
- [ ] Add MI300X-specific memory pool management
- [ ] Create performance benchmarks for MI300X

### Low Priority (Nice to Have)
- [ ] Write comprehensive ROCm documentation
- [ ] Add MI300X-specific profiling tools
- [ ] Create Docker images for MI300X deployment
- [ ] Implement auto-tuning for kernel parameters

### Expected Timeline
- **Basic functionality:** 4-6 weeks
- **Performance optimization:** 8-12 weeks
- **Production-ready:** 3-4 months

### Key Challenges
1. **FP4 Lightning Indexer:** Most complex kernel to port
2. **TileLang ROCm Backend:** May need significant DSL compiler work
3. **Memory Hierarchy Differences:** Requires careful tuning
4. **Testing at Scale:** Need access to MI300X clusters

---

## Additional Resources

### Official Documentation
- [DeepSeek-V4 Paper](https://arxiv.org/abs/2604.xxxxx)
- [DeepSeek-V4 HuggingFace](https://huggingface.co/collections/deepseek-ai/deepseek-v4)
- [DeepSeek API Docs](https://api-docs.deepseek.com/news/news260424)

### Implementation Guides
- [vLLM Blog - DeepSeek V4](https://vllm.ai/blog/deepseek-v4)
- [LMSYS Blog - DeepSeek-V4 Day 0](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [NVIDIA Blog - DeepSeek V4 on Blackwell](https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/)

### ROCm Resources
- [AMD ROCm vLLM First-Class Platform](https://rocm.blogs.amd.com/software-tools-optimization/vllm-omni/README.html)
- [SGLang AMD GPU Docs](https://docs.sglang.io/platforms/amd_gpu.html)
- [Microsoft Azure - DeepSeek on MI300](https://techcommunity.microsoft.com/blog/machinelearningblog/accelerating-deepseek-inference-with-amd-mi300-a-collaborative-breakthrough/4407673)

---

## Conclusion

DeepSeek-V4 represents a significant breakthrough in efficient long-context language modeling through its hybrid CSA/HCA attention architecture and advanced training techniques. The model is well-supported by both vLLM and SGLang with day-0 releases, making deployment straightforward on NVIDIA hardware.

For AMD MI300X deployment, while the core frameworks (vLLM and SGLang) have solid ROCm support, DeepSeek-V4 specific optimizations are still in development. The primary implementation work required focuses on:

1. **Porting attention kernels** (CSA/HCA) to HIP/ROCm
2. **Optimizing FP4/FP8 operations** for MI300X's MXFP4 support
3. **Adapting MoE routing** for RCCL distributed communication
4. **Tuning memory management** for 192GB HBM3

Given the strong foundation of ROCm support in vLLM and SGLang, and AMD's demonstrated success with DeepSeek-R1 on MI300X, full DeepSeek-V4 support on MI300X is expected in Q2 2026.

The MI300X's 192GB memory and superior multi-GPU bandwidth make it an excellent platform for DeepSeek-V4, potentially offering better price/performance than NVIDIA alternatives for long-context workloads.
