
#ifndef CUDA_INTRO_ALLOC_H
#define CUDA_INTRO_ALLOC_H

#include <cuda_runtime_api.h>
#include <stdlib.h>

#include "check_cuda.h"

static float* hostAllocFloat(size_t size) {
#if 1
  // allocate page-locked memory: faster to copy from/to GPU
  float* addr = NULL;
  CHECK_CUDA(cudaMallocHost(&addr, size * sizeof(float)));
  return addr;
#else
  void* addr = malloc(size * sizeof(float));
  if (addr == NULL) {
    throw std::runtime_error("out of memory");
  }
  return (float*)addr;
#endif
}

static void hostFree(void* addr) {
  if (addr != NULL) {
#if 1
    CHECK_CUDA(cudaFreeHost(addr));
#else
    free(addr);
#endif
  }
}

static float* gpuAllocFloat(size_t size) {
  float* addr = NULL;
#if CUDA_ARCH >= 60
  CHECK_CUDA(cudaMallocAsync(&addr, size * sizeof(float), cudaStreamPerThread));
#else
  CHECK_CUDA(cudaMalloc(&addr, size * sizeof(float)));
#endif
  return addr;
}

static void gpuFree(void* addr) {
  if (addr != NULL) {
#if CUDA_ARCH >= 60
    cudaFreeAsync(addr, cudaStreamPerThread);
#else
    cudaFree(addr);
#endif
  }
}

#endif  // CUDA_INTRO_ALLOC_H
