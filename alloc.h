
#ifndef CUDA_INTRO_ALLOC_H
#define CUDA_INTRO_ALLOC_H

#include <cuda_runtime_api.h>
#include <stdlib.h>

#include "check.h"

static float* hostAllocFloat(size_t size) {
  void* addr = malloc(size * sizeof(float));
  if (addr == NULL) {
    throw std::runtime_error("out of memory");
  }
  return addr;
}

static void hostFree(void* addr) {
  if (addr != NULL) {
    free(addr);
  }
}

static float* gpuAllocFloat(size_t size) {
  float* addr = NULL;
  CHECK(cudaMallocAsync(&addr, size * sizeof(float), cudaStreamPerThread));
  return addr;
}

static void gpuFree(void* addr) {
  if (addr != NULL) {
    cudaFreeAsync(addr, cudaStreamPerThread);
  }
}

#endif  // CUDA_INTRO_ALLOC_H
