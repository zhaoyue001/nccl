/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Simplified: ncclDevFuncTable dispatch removed — host always picks
 * the right specialized kernel.
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "common.h"
#include <cuda.h>

__shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ < 700
  __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif

struct RunWorkNop {
  __device__ void run(ncclWork *w) {}
};

// Fallback generic kernel (never used in production, kept for safety)
__global__ void ncclDevKernel_Generic(struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead) {
  ncclKernelMain<-1, RunWorkNop>(comm, channelMask, workHead);
}

// Host helper to set shared memory attribute — must be in nvcc-compiled context
// cudaFuncSetAttribute fails with "named symbol not found" when called from
// g++-compiled code, even if the kernel is in the same .so.
extern "C" __attribute__((visibility("default")))
void ncclOpSetShmemAttr(void* fn, size_t smem) {
  if (smem > 0) {
    cuFuncSetAttribute((CUfunction)fn,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem);
  }
}
