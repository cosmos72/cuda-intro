
#include <math.h>

#include <iostream>

#include "check_cuda.h"

// Kernel function to add the elements of two arrays.
//
// double arithmetic is SLOW, especially on consumer GPUs.
// prefer integers, float or even bfloat16.
//
// NOTE: bfloat16 requires CUDA_ARCH >= sm_80
__global__ void kernel_add(int size, float* x, float* y) {
  for (int i = 0; i < size; i++) {
    y[i] += x[i];
  }
}

static int run(void) {
  int size = 1 << 20;
  float *x = NULL, *y = NULL, *sum = NULL;
  float maxerror = 0.0f;

  // Allocate Unified Memory – accessible from CPU or GPU
  CHECK_CUDA(cudaMallocManaged(&x, size * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged(&y, size * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged(&sum, size * sizeof(float)));

  // initialize x and y arrays on the host
  for (int i = 0; i < size; i++) {
    x[i] = (float)drand48();
    y[i] = (float)drand48();
    sum[i] = x[i] + y[i];
  }

  // Run kernel on 1M elements on the GPU
  kernel_add<<<1, 1>>>(size, x, y);

  CHECK_CUDA(cudaGetLastError());

  // Wait for GPU to finish before accessing on host
  CHECK_CUDA(cudaStreamSynchronize(cudaStreamPerThread));

  // Check for errors
  for (int i = 0; i < size; i++) {
    maxerror = fmaxf(maxerror, fabs(sum[i] - y[i]));
  }
  std::cout << "Max difference: " << maxerror << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  return 0;
}

int main(void) {
  try {
    CHECK_CUDA(cudaSetDevice(0));
    run();
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
  return 0;
}
