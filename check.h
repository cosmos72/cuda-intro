
#ifndef CUDA_INTRO_CHECK_H
#define CUDA_INTRO_CHECK_H

#include <sstream>
#include <stdexcept>

static void throw_if_failed(const char file[], int line, cudaError_t err) {
  if (err != cudaSuccess) {
    std::stringstream buf;
    buf << "CUDA error " << err << " at " << file << ':' << line << ' '
        << cudaGetErrorString(err);
    throw std::runtime_error(buf.str());
  }
}

#define CHECK(expr) throw_if_failed(__FILE__, __LINE__, expr)

#endif  // CUDA_INTRO_CHECK_H
