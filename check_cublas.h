
#ifndef CUDA_INTRO_CHECK_CUBLAS_H
#define CUDA_INTRO_CHECK_CUBLAS_H

#include <cublas_v2.h>

#include <sstream>
#include <stdexcept>

static void throw_if_cublas_failed(const char file[], int line,
                                   cublasStatus_t err) {
  if (err != CUBLAS_STATUS_SUCCESS) {
    std::stringstream buf;
    buf << "CUBLAS error " << err << " at " << file << ':' << line << ' '
        << cublasGetStatusString(err);
    throw std::runtime_error(buf.str());
  }
}

#define CHECK_CUBLAS(expr) throw_if_cublas_failed(__FILE__, __LINE__, expr)

#endif  // CUDA_INTRO_CHECK_CUBLAS_H
