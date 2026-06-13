
#include <math.h>

#include <iostream>

#include "check.h"

// Kernel function to add the elements of two arrays
__global__ void kernel_add(int n, float* x, float* y) {
  for (int i = 0; i < n; i++) {
    y[i] = x[i] + y[i];
  }
}

static int run(void) {
  int N = 1 << 20;
  float *x = NULL, *y = NULL, *sum = NULL;
  float maxerror = 0.0f;

  // Allocate Unified Memory – accessible from CPU or GPU
  CHECK(cudaMallocManaged(&x, N * sizeof(float)));
  CHECK(cudaMallocManaged(&y, N * sizeof(float)));
  CHECK(cudaMallocManaged(&sum, N * sizeof(float)));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = (float)drand48();
    y[i] = (float)drand48();
    sum[i] = x[i] + y[i];
  }

  // Run kernel on 1M elements on the GPU
  kernel_add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  CHECK(cudaStreamSynchronize(cudaStreamPerThread));

  // Check for errors
  for (int i = 0; i < N; i++) {
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
    CHECK(cudaSetDevice(0));
    run();
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
  return 0;
}
