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
