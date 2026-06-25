#include <math.h>

#include <iostream>

#include "alloc.h"
#include "check.h"
#include "timer.h"

#define BLOCK_SIZE 512  // number of threads per block

// Kernel function to kernel_add the elements of two arrays
__global__ void kernel_add(size_t n, const float* __restrict__ x,
                           float* __restrict__ y) {
  size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) {
    if (i < n) {
      y[i] += x[i];
    }
  }
}

static void host_add(size_t n, const float* __restrict__ x,
                     float* __restrict__ y) {
  for (size_t i = 0; i < n; i++) {
    y[i] += x[i];
  }
}

static void kernel_benchmark(size_t N) {
  float* h = hostAllocFloat(N);
  float* x = gpuAllocFloat(N);
  float* y = gpuAllocFloat(N);

  // initialize x and y arrays on the host
  for (size_t i = 0; i < N; i++) {
    h[i] = (float)drand48();
  }
  CHECK(cudaMemcpyAsync(x, h, N * sizeof(float), cudaMemcpyHostToDevice,
                        cudaStreamPerThread));
  CHECK(cudaMemcpyAsync(y, h, N * sizeof(float), cudaMemcpyHostToDevice,
                        cudaStreamPerThread));

  dim3 threads(BLOCK_SIZE, 1, 1);
  dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

  // warm-up
  kernel_add<<<blocks, threads, 0, cudaStreamPerThread>>>(N, x, y);
  CHECK(cudaGetLastError());

  CHECK(cudaStreamSynchronize(cudaStreamPerThread));

  const struct timespec start = now();

  size_t run_n = 1000;
  for (size_t i = 0; i < run_n; ++i) {
    kernel_add<<<blocks, threads, 0, cudaStreamPerThread>>>(N, x, y);
    CHECK(cudaGetLastError());
  }

  // Wait for GPU to finish before accessing on host
  CHECK(cudaStreamSynchronize(cudaStreamPerThread));

  const double elapsed = now() - start;

  std::cout << N << '\t' << (N * run_n / elapsed) << '\n';

  // Free memory
  gpuFree(x);
  gpuFree(y);
  hostFree(h);
}

static void host_benchmark(size_t N) {
  float* hx = hostAllocFloat(N);
  float* hy = hostAllocFloat(N);

  // initialize x and y arrays on the host
  for (size_t i = 0; i < N; i++) {
    hx[i] = (float)drand48();
    hy[i] = (float)drand48();
  }

  // warm-up
  host_add(N, hx, hy);

  const struct timespec start = now();

  size_t run_n = 1000;
  for (size_t i = 0; i < run_n; ++i) {
    host_add(N, hx, hy);

    __asm__ __volatile__(""
                         : /* no output */
                         : /* no input */
                         : "memory");
  }

  const double elapsed = now() - start;

  std::cout << N << '\t' << (N * run_n / elapsed) << '\n';

  // Free memory
  hostFree(hx);
  hostFree(hy);
}

int main(void) {
  try {
    srand48(time(NULL));
    CHECK(cudaSetDevice(0));
    std::cout << "# host\n";
    for (size_t N = 16; N <= 1048576 * 64; N <<= 1) {
      host_benchmark(N);
    }
    std::cout << "\n# kernel\n";
    for (size_t N = 16; N <= 1048576 * 64; N <<= 1) {
      kernel_benchmark(N);
    }
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
  return 0;
}
