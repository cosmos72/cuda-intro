#include <math.h>

#include <iostream>

#include "alloc.h"
#include "check_cuda.h"
#include "timer.h"

#define BLOCK_SIZE 512  // number of threads per block

// Kernel function to kernel_add the elements of two arrays
__global__ void kernel_add(size_t size, const float* __restrict__ x,
                           float* __restrict__ y) {
  size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = index; i < size; i += stride) {
    if (i < size) {
      y[i] += x[i];
    }
  }
}

double cuda_add_benchmark(size_t size) {
  float* h = hostAllocFloat(size);
  float* x = gpuAllocFloat(size);
  float* y = gpuAllocFloat(size);

  // initialize x and y arrays on the host
  for (size_t i = 0; i < size; i++) {
    h[i] = (float)drand48();
  }
  CHECK_CUDA(cudaMemcpyAsync(x, h, size * sizeof(float), cudaMemcpyHostToDevice,
                             cudaStreamPerThread));
  CHECK_CUDA(cudaMemcpyAsync(y, h, size * sizeof(float), cudaMemcpyHostToDevice,
                             cudaStreamPerThread));

  dim3 threads(BLOCK_SIZE, 1, 1);
  dim3 blocks((size + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

  // warm-up
  kernel_add<<<blocks, threads, 0, cudaStreamPerThread>>>(size, x, y);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaStreamSynchronize(cudaStreamPerThread));

  const struct timespec start = now();

  size_t run_n = 1000;
  for (size_t i = 0; i < run_n; ++i) {
    kernel_add<<<blocks, threads, 0, cudaStreamPerThread>>>(size, x, y);
    CHECK_CUDA(cudaGetLastError());
  }

  // Wait for GPU to finish before accessing on host
  CHECK_CUDA(cudaStreamSynchronize(cudaStreamPerThread));

  const double elapsed = now() - start;

  // Free memory
  gpuFree(x);
  gpuFree(y);
  hostFree(h);

  return size * run_n / elapsed;
}
