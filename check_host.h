
#ifndef CUDA_INTRO_CHECK_HOST_H
#define CUDA_INTRO_CHECK_HOST_H

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

static void throw_if_host_failed(const char file[], int line, int err) {
  if (err != 0) {
    err = errno;
    std::stringstream buf;
    buf << "host error " << err << " at " << file << ':' << line << ' '
        << strerror(err);
    throw std::runtime_error(buf.str());
  }
}

#define CHECK_HOST(expr) throw_if_host_failed(__FILE__, __LINE__, expr)

#endif  // CUDA_INTRO_CHECK_HOST_H
