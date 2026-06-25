#include <math.h>
#include <time.h>

#include <iostream>

#include "alloc.h"
#include "check_cuda.h"

// Kernel function to add the elements of two arrays
__global__ void kernel_add(size_t size, const float* __restrict__ x,
                           float* __restrict__ y) {
  for (size_t i = 0; i < size; i++) {
    y[i] += x[i];
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

  // Run kernel on 1M elements on the GPU
  kernel_add<<<1, 1, 0, cudaStreamPerThread>>>(size, x, y);

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
