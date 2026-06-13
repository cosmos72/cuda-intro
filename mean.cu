#include <stdlib.h>
#include <time.h>

#include <iostream>

#include "alloc.h"
#include "check.h"

#define BLOCK_SIZE 512

static float host_sum(unsigned n, const float* hin) {
  float ret = 0.0f;
  for (unsigned i = 0; i < n; i++) {
    ret += hin[i];
  }
  return ret;
}

// Kernel function to compute the sum of all elements in an array
template <unsigned blocksize>
__global__ void kernel_sum(const float* in, float* buf, unsigned n) {
  extern __shared__ float sdata[];

  unsigned tid = threadIdx.x;
  unsigned i = blockIdx.x * blocksize * 2 + threadIdx.x;
  unsigned gridsize = blocksize * 2 * gridDim.x;
  sdata[tid] = 0;

  while (i < n) {
    sdata[tid] += in[i];

    if (i + blocksize < n) {
      sdata[tid] += in[i + blocksize];
    }

    i += gridsize;
  }

  __syncthreads();

  if (blocksize >= 1024) {
    if (tid < 512) {
      sdata[tid] += sdata[tid + 512];
    }
    __syncthreads();
  }

  if (blocksize >= 512) {
    if (tid < 256) {
      sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
  }

  if (blocksize >= 256) {
    if (tid < 128) {
      sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
  }

  if (blocksize >= 128) {
    if (tid < 64) {
      sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();
  }

  if (tid < 32) {
    if (blocksize >= 64) {
      sdata[tid] += sdata[tid + 32];
    }
    for (unsigned offset = warpSize / 2; offset > 0; offset /= 2) {
      sdata[tid] += __shfl_down_sync(0xffffffff, sdata[tid], offset);
    }
  }

  if (tid == 0) {
    buf[blockIdx.x] = sdata[0];
  }
}

static float cuda_sum(unsigned n, const float* in, unsigned block_n,
                      float* hbuf, float* buf) {
  const unsigned thread_n = BLOCK_SIZE / 2;
  dim3 threads(thread_n, 1, 1);
  dim3 blocks(block_n, 1, 1);

  const unsigned smemsize = thread_n * sizeof(float);

  kernel_sum<thread_n>
      <<<blocks, threads, smemsize, cudaStreamPerThread>>>(in, buf, n);

  CHECK(cudaMemcpyAsync(hbuf, buf, sizeof(float) * block_n,
                        cudaMemcpyDeviceToHost, cudaStreamPerThread));

  // Wait for GPU to finish before accessing on host
  CHECK(cudaStreamSynchronize(cudaStreamPerThread));

  return host_sum(block_n, hbuf);
}

static int run(void) {
  unsigned block_n = 64u;
  unsigned n = 1 << 20;
  float* hx = hostAllocFloat(n);
  float hbuf[block_n];

  float* x = gpuAllocFloat(n);
  float* buf = gpuAllocFloat(block_n);

  // initialize x array on the host
  for (unsigned i = 0; i < n; i++) {
    hx[i] = (float)drand48();
  }
  CHECK(cudaMemcpyAsync(x, hx, n * sizeof(float), cudaMemcpyHostToDevice,
                        cudaStreamPerThread));

  float mean = cuda_sum(n, x, block_n, hbuf, buf) / float(n);
  float hmean = host_sum(n, hx) / float(n);

  std::cout << "cuda mean = " << mean << ", host mean = " << hmean
            << ", difference: " << fabsf(mean - hmean) << '\n';

  // Free memory
  gpuFree(x);
  gpuFree(buf);
  hostFree(hx);

  return 0;
}

int main(void) {
  try {
    srand48(time(NULL));  // initialize random generator

    CHECK(cudaSetDevice(0));
    run();
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
  return 0;
}
