#include <math.h>

#include <iostream>

// Kernel function to add the elements of two arrays
__global__ void add(size_t n, float* x, float* y) {
  for (size_t i = 0; i < n; i++) {
    y[i] += x[i];
  }
}

static bool failed(const char file[], int line, cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error " << err << " at " << file << ':' << line << ' '
              << cudaGetErrorString(err) << '\n';
    return true;
  }
  return false;
}

#define CHECK(expr)                              \
  do {                                           \
    if (failed(__FILE__, __LINE__, expr) != 0) { \
      goto out;                                  \
    }                                            \
  } while (0)

static float* hostAllocFloat(size_t size) {
  return (float*)malloc(size * sizeof(float));
}

static void hostFree(void* ptr) { return free(ptr); }

static float* gpuAllocFloat(size_t size) {
  float* ptr = NULL;
  CHECK(cudaMallocAsync(&ptr, size * sizeof(float), cudaStreamPerThread));
out:
  return ptr;
}

static void gpuFree(void* ptr) { cudaFreeAsync(ptr, cudaStreamPerThread); }

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
  CHECK(cudaMemcpyAsync(x, hx, N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyAsync(y, hy, N * sizeof(float), cudaMemcpyHostToDevice));

  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);

  CHECK(cudaMemcpyAsync(hy, y, N * sizeof(float), cudaMemcpyDeviceToHost));

  // Wait for GPU to finish before accessing on host
  CHECK(cudaStreamSynchronize(cudaStreamPerThread));

  // Check for errors (all values should be 3.0f)
  for (size_t i = 0; i < N; i++) {
    maxError = fmaxf(maxError, fabs(hy[i] - 3.0f));
  }
  std::cout << "Max error: " << maxError << '\n';

  // Free memory
out:
  gpuFree(x);
  gpuFree(y);
  hostFree(hx);
  hostFree(hy);

  return 0;
}

int main(void) {
  CHECK(cudaSetDevice(0));
  run();
out:
  return 0;
}
