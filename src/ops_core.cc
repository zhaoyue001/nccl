/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Dynamic kernel registration for per-operation .so architecture.
 * Each collective operation .so calls ncclRegisterKernel/ncclRegisterFuncKernel
 * to register its device kernels at load time.
 ************************************************************************/

#include "device.h"
#include <cstring>

// Dynamic kernel tables — populated by per-op .so registration
int ncclDevKernelCount = 0;
void* ncclDevKernelList[NCCL_MAX_DEV_KERNELS];
int ncclDevFuncRowToId[NCCL_MAX_DEV_FUNCS];
void* ncclDevKernelForFunc[NCCL_MAX_DEV_FUNCS];
bool ncclDevKernelForFuncIsSpecialized[NCCL_MAX_DEV_FUNCS];
ncclOpLaunchFunc_t ncclDevKernelForFuncLaunch[NCCL_MAX_DEV_FUNCS];  // per-func launch function

// Called once during library init
void ncclDevTablesInit() {
  memset(ncclDevKernelList, 0, sizeof(ncclDevKernelList));
  memset(ncclDevFuncRowToId, 0xff, sizeof(ncclDevFuncRowToId));  // -1
  memset(ncclDevKernelForFunc, 0, sizeof(ncclDevKernelForFunc));
  memset(ncclDevKernelForFuncIsSpecialized, 0, sizeof(ncclDevKernelForFuncIsSpecialized));
  memset(ncclDevKernelForFuncLaunch, 0, sizeof(ncclDevKernelForFuncLaunch));
  ncclDevKernelCount = 0;
}

void ncclRegisterKernel(void* kernelFn) {
  if (ncclDevKernelCount < NCCL_MAX_DEV_KERNELS) {
    ncclDevKernelList[ncclDevKernelCount++] = kernelFn;
  }
}

void ncclRegisterFuncKernel(int funcId, void* kernelFn, bool specialized, void* launchFn) {
  if (funcId >= 0 && funcId < NCCL_MAX_DEV_FUNCS) {
    ncclDevKernelForFunc[funcId] = kernelFn;
    ncclDevKernelForFuncIsSpecialized[funcId] = specialized;
    ncclDevKernelForFuncLaunch[funcId] = (ncclOpLaunchFunc_t)launchFn;
  }
}
