#include <stdlib.h>
#include <time.h>

#define USE_CBLAS
#ifdef USE_CBLAS
#include <cblas-netlib.h>
#endif

#include <iomanip>
#include <iostream>

#include "alloc.h"
#include "check_cublas.h"
#include "timer.h"

/**
 *     / --- n --- \       / --- n --- \       / --- n --- \
 *     |           |       |           |       |           |
 * A = n           |   B = n           |   C = n           |
 *     |           |       |           |       |           |
 *     \           /       \           /       \           /
 */

/* compute C = A * B manually */
static double host_matmul(size_t n, const float* ha /*n*n*/,
                          const float* hb /*n*n*/, float* hc /*n*n*/) {
  const struct timespec start = now();
  for (size_t k = 0; k < n; k++) {
    for (size_t j = 0; j < n; j++) {
      float sum = 0.0f;
      for (size_t i = 0; i < n; i++) {
        sum += ha[i * n + j] * hb[i + k * n];
      }
      hc[j + k * n] = sum;
    }
  }
  const double elapsed = now() - start;
  return n * n * n / elapsed;
}

#ifdef USE_CBLAS
/* compute C = A * B using BLAS library */
static double blas_matmul(size_t n, const float* ha /*n*n*/,
                          const float* hb /*n*n*/, float* hc /*n*n*/) {
  const struct timespec start = now();
  float alpha = 1.0f;
  float beta = 0.0f;
  cblas_sgemm(CblasColMajor,   // column-major as CUBLAS does
              CblasNoTrans,    // don't transpose A
              CblasNoTrans,    // don't transpose B
              n, n, n, alpha,  //
              ha, n,           // A column stride, >= n
              hb, n,           // B column stride, >= n
              beta,            //
              hc,              //
              n);              // C column stride, >= n (cannot transpose C)
  const double elapsed = now() - start;
  return n * n * n / elapsed;
}
#endif

/* compute C = A * B using CUBLAS */
static double cuda_matmul(cublasHandle_t handle, size_t n,
                          const float* a /*n*n*/, const float* b /*n*n*/,
                          float* c /*n*n*/) {
  const struct timespec start = now();
  float alpha = 1.0f;
  float beta = 0.0f;
  CHECK_CUBLAS(cublasSgemm(handle,
                           CUBLAS_OP_N,  // don't transpose A
                           CUBLAS_OP_N,  // don't transpose B
                           n, n, n,
                           &alpha,  // cuda memory is supported too
                           a, n,    // A column stride, >= n
                           b, n,    // B column stride, >= n
                           &beta,   // cuda memory is supported too
                           c,       //
                           n));  // C column stride, >= n (cannot transpose C)

  // wait for GPU to finish before stopping timer
  CHECK_CUDA(cudaStreamSynchronize(cudaStreamPerThread));
  const double elapsed = now() - start;
  return n * n * n / elapsed;
}

static void compare_matrix(size_t n, const float* c1 /*n*n*/,
                           const float* c2 /*n*n*/, const char* label) {
  float delta = 0.0f;
  for (size_t i = 0; i < n * n; i++) {
    delta = fmaxf(delta, fabsf(c1[i] - c2[i]));
  }
  std::cout << std::setprecision(7) /**/
            << label << " max difference: " << delta << '\n';
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

  size_t maxn = 8192;

  float* ha = hostAllocFloat(maxn * maxn);
  float* hb = hostAllocFloat(maxn * maxn);
  float* hc = hostAllocFloat(maxn * maxn);
  float* hc_blas = hostAllocFloat(maxn * maxn);
  float* hc_cuda = hostAllocFloat(maxn * maxn);

  float* a = gpuAllocFloat(maxn * maxn);
  float* b = gpuAllocFloat(maxn * maxn);
  float* c = gpuAllocFloat(maxn * maxn);

  // initialize ha, hb matrix on the host
  for (size_t i = 0; i < maxn * maxn; i++) {
    ha[i] = (float)drand48();
    hb[i] = (float)drand48();
  }
  size_t n = 400;
  CHECK_CUDA(cudaMemcpyAsync(a, ha, n * n * sizeof(float),
                             cudaMemcpyHostToDevice, cudaStreamPerThread));
  CHECK_CUDA(cudaMemcpyAsync(b, hb, n * n * sizeof(float),
                             cudaMemcpyHostToDevice, cudaStreamPerThread));
  CHECK_CUDA(cudaMemsetAsync(c, 0, n * n, cudaStreamPerThread));

  CHECK_CUBLAS(cublasSetStream(handle, cudaStreamPerThread));

  cuda_matmul(handle, n, a, b, c);

  CHECK_CUDA(cudaMemcpyAsync(hc_cuda, c, n * n * sizeof(float),
                             cudaMemcpyDeviceToHost, cudaStreamPerThread));

  // Wait for GPU to finish before accessing on host
  CHECK_CUDA(cudaStreamSynchronize(cudaStreamPerThread));

  host_matmul(n, ha, hb, hc);

  compare_matrix(n, hc, hc_cuda, "cuda");

#ifdef USE_CBLAS
  blas_matmul(n, ha, hb, hc_blas);

  compare_matrix(n, hc, hc_blas, "blas");
#endif

  /* warm-up */
  cuda_matmul(handle, maxn, a, b, c);

  std::cout << "\n# cuda\n";
  for (n = 16; n <= maxn; n <<= 1) {
    double speed = cuda_matmul(handle, n, a, b, c);
    std::cout << (n * n * n) << '\t' << speed << '\n';
  }

#ifdef USE_CBLAS
  std::cout << "\n# blas\n";
  for (n = 16; n <= 2048; n <<= 1) {
    double speed = blas_matmul(n, ha, hb, hc_blas);
    std::cout << (n * n * n) << '\t' << speed << '\n';
  }
#endif

  std::cout << "\n# host\n";
  for (n = 16; n <= 1024; n <<= 1) {
    double speed = host_matmul(n, ha, hb, hc);
    std::cout << (n * n * n) << '\t' << speed << '\n';
  }
  // Free memory
  gpuFree(c);
  gpuFree(b);
  gpuFree(a);
  hostFree(hc_cuda);
  hostFree(hc_blas);
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
    return run();
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
