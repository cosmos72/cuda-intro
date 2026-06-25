#include <stdlib.h>
#include <time.h>

#include <iomanip>
#include <iostream>

#include "alloc.h"
#include "check_cublas.h"

/**
 *     / --- n1 --- \       / --- n3 --- \       / --- n3 --- \
 *     |            |       |            |       |            |
 * A = n2           |   B = n1           |   C = n2           |
 *     |            |       |            |       |            |
 *     \            /       \            /       \            /
 */

/* compute c = alpha * a * b */
static void host_matmul(int n1, int n2, int n3, float alpha,
                        const float* a /*n1*n2*/, const float* b /*n3*n1*/,
                        float* c /*n3*n2*/) {
  for (int k = 0; k < n3; k++) {
    for (int j = 0; j < n2; j++) {
      float sum = 0.0f;
      for (int i = 0; i < n1; i++) {
        sum += a[i + j * n1] * b[k + i * n3];
      }
      c[k + j * n3] = sum * alpha;
    }
  }
}

static void compare_matrix(int n2, int n3, const float* c /*n3*n2*/,
                           const float* tc /*n2*n3*/) {
  float delta = 0.0f;
  for (int k = 0; k < n3; k++) {
    for (int j = 0; j < n2; j++) {
      delta = fmaxf(delta, fabsf(c[k + j * n3] - tc[k * n2 + j]));
    }
  }
  std::cout << std::setprecision(7) /**/
            << "max difference: " << delta << '\n';
}

static cublasHandle_t init_cublas(void) {
  // warning: using the same handle from multiple threads
  // may cause race conditions - see docs
  cublasHandle_t handle = NULL;
  CHECK_CUBLAS(cublasCreate(&handle));
  return handle;
}

static void quitBlas(cublasHandle_t handle) {  //
  CHECK_CUBLAS(cublasDestroy(handle));
}

static int run(void) {
  cublasHandle_t handle = init_cublas();

  int n1 = 100, n2 = 30, n3 = 50;
  float* ha = hostAllocFloat(n1 * n2);
  float* hb = hostAllocFloat(n3 * n1);
  float* hc = hostAllocFloat(n3 * n2);
  float* hmul = hostAllocFloat(n3 * n2);

  float* a = gpuAllocFloat(n1 * n2);
  float* b = gpuAllocFloat(n3 * n1);
  float* c = gpuAllocFloat(n3 * n2);

  // initialize ha matrix on the host
  for (int i = 0; i < n1 * n2; i++) {
    ha[i] = (float)drand48();
  }
  // initialize hb matrix on the host
  for (int i = 0; i < n3 * n1; i++) {
    hb[i] = (float)drand48();
  }
  CHECK_CUDA(cudaMemcpyAsync(a, ha, n1 * n2 * sizeof(float),
                             cudaMemcpyHostToDevice, cudaStreamPerThread));
  CHECK_CUDA(cudaMemcpyAsync(b, hb, n3 * n1 * sizeof(float),
                             cudaMemcpyHostToDevice, cudaStreamPerThread));
  CHECK_CUDA(cudaMemsetAsync(c, 0, n3 * n2, cudaStreamPerThread));

  CHECK_CUBLAS(cublasSetStream(handle, cudaStreamPerThread));

  float alpha = 1.0f;
  float beta = 0.0f;

  // compute C = alpha * A * B + beta * C
  CHECK_CUBLAS(cublasSgemm(
      handle,
      CUBLAS_OP_T,  // transpose A, because BLAS uses column-major layout
      CUBLAS_OP_T,  // transpose B, because BLAS uses column-major layout
      n2, n3, n1,
      &alpha,   // cuda memory is supported too
      a, n1,    // A column stride, >= n1
      b, n3,    // B column stride, >= n3
      &beta,    // cuda memory is supported too
      c, n2));  // C column stride, >= n2 (cannot transpose C)

  CHECK_CUDA(cudaMemcpyAsync(hc, c, n3 * n2 * sizeof(float),
                             cudaMemcpyDeviceToHost, cudaStreamPerThread));

  // Wait for GPU to finish before accessing on host
  CHECK_CUDA(cudaStreamSynchronize(cudaStreamPerThread));

  host_matmul(n1, n2, n3, alpha, ha, hb, hmul);

  compare_matrix(n2, n3, hmul, hc);

  // Free memory
  gpuFree(c);
  gpuFree(b);
  gpuFree(a);
  hostFree(hmul);
  hostFree(hc);
  hostFree(hb);
  hostFree(ha);

  quitBlas(handle);

  return 0;
}

int main(void) {
  try {
    srand48(time(NULL));  // initialize random generator

    CHECK_CUDA(cudaSetDevice(0));
    run();
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
  return 0;
}
