#include <math.h>
#include <time.h>

#include <iostream>

#include "alloc.h"
#include "check.h"

#define BLOCK_SIZE 512  // number of threads per block

// Kernel function to kernel_add the elements of two arrays
__global__ void kernel_add(size_t n, float* x, float* y) {
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
  float* hsum = hostAllocFloat(N);
  float* x = gpuAllocFloat(N);
  float* y = gpuAllocFloat(N);
  float maxerror = 0.0f;

  // initialize x and y arrays on the host
  for (size_t i = 0; i < N; i++) {
    hx[i] = (float)drand48();
    hy[i] = (float)drand48();
  }
  CHECK(cudaMemcpyAsync(x, hx, N * sizeof(float), cudaMemcpyHostToDevice,
                        cudaStreamPerThread));
  CHECK(cudaMemcpyAsync(y, hy, N * sizeof(float), cudaMemcpyHostToDevice,
                        cudaStreamPerThread));

  dim3 threads(BLOCK_SIZE, 1, 1);
  dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

  kernel_add<<<blocks, threads, 0, cudaStreamPerThread>>>(N, x, y);

  CHECK(cudaGetLastError());

  CHECK(cudaMemcpyAsync(hsum, y, N * sizeof(float), cudaMemcpyDeviceToHost,
                        cudaStreamPerThread));

  // Wait for GPU to finish before accessing on host
  CHECK(cudaStreamSynchronize(cudaStreamPerThread));

  // Check for errors
  for (size_t i = 0; i < N; i++) {
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
    CHECK(cudaSetDevice(0));
    run();
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
  return 0;
}
