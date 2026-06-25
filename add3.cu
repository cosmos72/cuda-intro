#include <math.h>
#include <time.h>

#include <iostream>

#include "alloc.h"
#include "check_cuda.h"

#define BLOCK_SIZE 512  // number of threads per block

// Kernel function to kernel_add the elements of two arrays
__global__ void kernel_add(size_t size, const float* __restrict__ x,
                           float* __restrict__ y) {
  size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = index; i < size; i += stride) {
    if (i < size) {
      y[i] = x[i] + y[i];
    }
  }
}

static int run(void) {
  size_t size = 1 << 20;
  float* hx = hostAllocFloat(size);
  float* hy = hostAllocFloat(size);
  float* hsum = hostAllocFloat(size);
  float* x = gpuAllocFloat(size);
  float* y = gpuAllocFloat(size);
  float maxerror = 0.0f;

  // initialize x and y arrays on the host
  for (size_t i = 0; i < size; i++) {
    hx[i] = (float)drand48();
    hy[i] = (float)drand48();
  }
  CHECK_CUDA(cudaMemcpyAsync(x, hx, size * sizeof(float),
                             cudaMemcpyHostToDevice, cudaStreamPerThread));
  CHECK_CUDA(cudaMemcpyAsync(y, hy, size * sizeof(float),
                             cudaMemcpyHostToDevice, cudaStreamPerThread));

  dim3 threads(BLOCK_SIZE, 1, 1);
  dim3 blocks((size + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

  kernel_add<<<blocks, threads, 0, cudaStreamPerThread>>>(size, x, y);

  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpyAsync(hsum, y, size * sizeof(float),
                             cudaMemcpyDeviceToHost, cudaStreamPerThread));

  // Wait for GPU to finish before accessing on host
  CHECK_CUDA(cudaStreamSynchronize(cudaStreamPerThread));

  // Check for errors
  for (size_t i = 0; i < size; i++) {
    maxerror = fmaxf(maxerror, fabsf(hx[i] + hy[i] - hsum[i]));
  }
  std::cout << "Max difference: " << maxerror << '\n';

  // Free memory
  gpuFree(x);
  gpuFree(y);
  hostFree(hx);
  hostFree(hy);

  return 0;
}

int main(void) {
  try {
    srand48(time(NULL));
    CHECK_CUDA(cudaSetDevice(0));
    run();
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
  return 0;
}
