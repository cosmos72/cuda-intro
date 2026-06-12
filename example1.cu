
#include <math.h>

#include <iostream>

#include "check.h"

// Kernel function to add the elements of two arrays
__global__ void add(int n, float* x, float* y) {
  for (int i = 0; i < n; i++) {
    y[i] = x[i] + y[i];
  }
}

static int run(void) {
  int N = 1 << 20;
  float *x = NULL, *y = NULL;
  float maxError = 0.0f;

  // Allocate Unified Memory – accessible from CPU or GPU
  CHECK(cudaMallocManaged(&x, N * sizeof(float)));
  CHECK(cudaMallocManaged(&y, N * sizeof(float)));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  CHECK(cudaStreamSynchronize(cudaStreamPerThread));

  // Check for errors (all values should be 3.0f)
  for (int i = 0; i < N; i++) {
    maxError = fmaxf(maxError, fabs(y[i] - 3.0f));
  }
  std::cout << "Max error: " << maxError << std::endl;

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
