#include <math.h>

#include <iostream>

#include "alloc.h"
#include "check.h"

// Kernel function to add the elements of two arrays
__global__ void add(size_t n, float* x, float* y) {
  size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    if (i < n) {
      y[i] = x[i] + y[i];
    }
  }
}

static int run(void) {
  size_t N = 1 << 20;
  float* hx = hostAllocFloat(N);
  float* hy = hostAllocFloat(N);
  float* x = gpuAllocFloat(N);
  float* y = gpuAllocFloat(N);
  float maxError = 0.0f;

  // initialize x and y arrays on the host
  for (size_t i = 0; i < N; i++) {
    hx[i] = 1.0f;
    hy[i] = 2.0f;
  }
  CHECK(cudaMemcpyAsync(x, hx, N * sizeof(float), cudaMemcpyHostToDevice,
                        cudaStreamPerThread));
  CHECK(cudaMemcpyAsync(y, hy, N * sizeof(float), cudaMemcpyHostToDevice,
                        cudaStreamPerThread));

  unsigned blocksize = 512;  // number of threads per block
  dim3 threads(blocksize, 1, 1);
  dim3 blocks((N + blocksize - 1) / blocksize, 1, 1);

  add<<<blocks, threads, 0, cudaStreamPerThread>>>(N, x, y);

  CHECK(cudaMemcpyAsync(hy, y, N * sizeof(float), cudaMemcpyDeviceToHost,
                        cudaStreamPerThread));

  // Wait for GPU to finish before accessing on host
  CHECK(cudaStreamSynchronize(cudaStreamPerThread));

  // Check for errors (all values should be 3.0f)
  for (size_t i = 0; i < N; i++) {
    maxError = fmaxf(maxError, fabs(hy[i] - 3.0f));
  }
  std::cout << "Max difference: " << maxError << '\n';

  // Free memory
  gpuFree(x);
  gpuFree(y);
  hostFree(hx);
  hostFree(hy);

  return 0;
}

int main(void) {
  try {
    CHECK(cudaSetDevice(0));
    run();
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
  return 0;
}
