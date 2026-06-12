
#ifndef CUDA_INTRO_ALLOC_H
#define CUDA_INTRO_ALLOC_H

#include <cuda_runtime_api.h>
#include <stdlib.h>

#include "check.h"

static float* hostAllocFloat(size_t size) {
  return (float*)malloc(size * sizeof(float));
}

static void hostFree(void* ptr) { return free(ptr); }

static float* gpuAllocFloat(size_t size) {
  float* ptr = NULL;
  CHECK(cudaMallocAsync(&ptr, size * sizeof(float), cudaStreamPerThread));
  return ptr;
}

static void gpuFree(void* ptr) {  //
  cudaFreeAsync(ptr, cudaStreamPerThread);
}

#endif  // CUDA_INTRO_ALLOC_H
