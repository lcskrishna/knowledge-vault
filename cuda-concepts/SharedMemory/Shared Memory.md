# Shared Memory
### Shared Memory

1.  Sometimes misalign non-coalesced memory anyway.
    1.  In such cases we can improve perofrmance of our kernels using shared memory.
2.  GPU memory can be categorized into two types.
    1.  Global Memory & L2 cache.
    2.  On Chip: Shared Memory, L1 cache, Readonly memory, constant memory
    3.  Shared memory only contains 64K.

<figure class="image image-style-block-align-left image_resized" style="width:70.35%;"><img style="aspect-ratio:1536/801;" src="3_Shared Memory_image.png" width="1536" height="801"></figure>

1.  **Shared memory usages**
    1.  Intra block thread communication channel.
    2.  Program managed cache for global memory data.
    3.  Scratch pad memory for transforming data to improve global memory access patterns.
2.  Shared memory Access
    1.  Shared memory is allocated to each threadblock when it starts executing.
    2.  This shared memory is shared by all threads in thread block.
    3.  Its contents have same lifetime as the threadblocks.

**Optimize the memory access**

1.  Counting on L1 cache to store repeatedly access memory. (Not programmed)
2.  Store repeatedly access memory explicilty in shraed memory (Programmed).

<figure class="image"><img style="aspect-ratio:1683/901;" src="2_Shared Memory_image.png" width="1683" height="901"></figure>

```src
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define SHARED_MEMORY_SIZE 128

__global__ void smem_static_test(int * in, int * output, int size)
{
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + tid;
	
	__shared__ int smem[SHARED_ARRAY_SIZE];
	if (gid < size) {
		smem[tid] = in[gid];
		out[gid] = smem[tid];
	}
}

__global__ void smem_dynamic_test(int * in, int * output, int size)
{
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + tid;
	
	extern __shared__ int smem[]; // Will notify kernel will be enabled somewhere.
	if (gid < size) {
		smem[tid] = in[gid];
		out[gid] = smem[tid];
	}
}

### HOST SIDE

if (not dynamic)
	smem_static_test <<< grid, block >>> (d_in, d_out);
else
	smem_dynamic_test <<< grid, block, sizeof(int *) * SHARED_ARRAY_SIZE >>>(d_in, d_out);
	// Here the sizeof(int *) is nothing but the shared meomry size in bytes. 
```

**Compilation Options**

```plain
nvcc --ptxas-options=-v -o shmem_test intro_shmem.cu common.cu -I common.h

ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z17smem_dynamic_testPiS_i' for 'sm_75'
ptxas info    : Function properties for _Z17smem_dynamic_testPiS_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 3.067 ms
ptxas info    : Compiling entry function '_Z16smem_static_testPiS_i' for 'sm_75'
ptxas info    : Function properties for _Z16smem_static_testPiS_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 372 bytes cmem[0]
ptxas info    : Compile time = 0.678 ms

ptxas info    : 0 bytes gmem
```

## Shared Memory Banks & Access Modes

Let's understand how shared memory is arranged and access patterns.

*   Shared memory to achieve higher memory bandwidth - we divide into banks.
    *   Its divided into 32 banks - because we have 32 threads in a block.
    *   <figure class="image"><img style="aspect-ratio:1027/476;" src="1_Shared Memory_image.png" width="1027" height="476"></figure>
    *   The operation can be serviced by one memory transaction - if warp doens't access more memory locations in a bank.
    *   Otherwise multiple address in a shared memory request fall in the same memory bank. This is called Bank conflict.
*   **Parallel Access of Memory banks:**
    *   Let's say multiple addresses accessed by a warp fall into mulitple banks, for example all 32 banks - then parallel access to shared memory occurs. This implies some of the addresses can be serviced using one memory transaction.
    *   **Ideal case - conflict free access.**
    *   **Here one thread accesses one bank.**
    *   <figure class="image"><img style="aspect-ratio:1049/252;" src="Shared Memory_image.png" width="1049" height="252"></figure>
    *   If multiple threads want to access same memory bank - then multiple transactions.
    *   If all threads access same bank. Its called broadcast access.
    *   This results in serialized access and diminished memory bandwidth.
*   Access mode depends on architectures either 32 bit or 64 Bit