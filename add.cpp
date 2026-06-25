#include <math.h>

#include <fstream>
#include <iostream>
#include <string>
#include <thread>  // std::thread::hardware_concurrency();

#include "check_cuda.h"

double host_add_benchmark(size_t size);
double host_add_benchmark_multithread(size_t size, unsigned thread_n);
double cuda_add_benchmark(size_t size);

static void run_host_add_benchmark(void) {
  std::cout << "\n# host1\n";
  std::ofstream file("bench_host1.txt", std::ios::out | std::ios::trunc);

  for (size_t size = 1024; size <= 1048576 * 64; size <<= 1) {
    double speed = host_add_benchmark(size);
    std::cout << size << '\t' << speed << '\n';
    file << size << '\t' << speed << '\n';
  }
}

static void run_host_add_benchmark_multithread(void) {
  unsigned thread_max = std::thread::hardware_concurrency();

  for (unsigned thread_n = 2; thread_n <= thread_max; thread_n <<= 1) {
    std::cout << "\n# host" << thread_n << '\n';
    std::ofstream file("bench_host" + std::to_string(thread_n) + ".txt",
                       std::ios::out | std::ios::trunc);

    for (size_t size = 1024; size <= 1048576 * 64; size <<= 1) {
      double speed = host_add_benchmark_multithread(size, thread_n);
      std::cout << size << '\t' << speed << '\n';
      file << size << '\t' << speed << '\n';
    }
  }
}

static void run_cuda_add_benchmark(void) {
  std::cout << "\n# cuda\n";
  std::ofstream file("bench_cuda.txt", std::ios::out | std::ios::trunc);
  for (size_t size = 1024; size <= 1048576 * 64; size <<= 1) {
    double speed = cuda_add_benchmark(size);
    std::cout << size << '\t' << speed << '\n';
    file << size << '\t' << speed << '\n';
  }
}

int main(void) {
  try {
    srand48(time(NULL));
    CHECK_CUDA(cudaSetDevice(0));

    run_host_add_benchmark();
    run_host_add_benchmark_multithread();
    run_cuda_add_benchmark();

  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
  return 0;
}
