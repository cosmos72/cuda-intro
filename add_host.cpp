
#include <barrier>
#include <future>
#include <vector>

#include "alloc.h"
#include "timer.h"

static void host_add(size_t size, const float* __restrict__ hx,
                     float* __restrict__ hy) {
  for (size_t i = 0; i < size; i++) {
    hy[i] += hx[i];
  }
}

double host_add_benchmark(size_t size) {
  float* hx = hostAllocFloat(size);
  float* hy = hostAllocFloat(size);

  // initialize hx and hy arrays on the host
  for (size_t i = 0; i < size; i++) {
    hx[i] = (float)drand48();
    hy[i] = (float)drand48();
  }

  // warm-up
  host_add(size, hx, hy);

  const struct timespec start = now();

  size_t run_n = 1000;
  for (size_t i = 0; i < run_n; ++i) {
    host_add(size, hx, hy);

    __asm__ __volatile__(""
                         : /* no output */
                         : /* no input */
                         : "memory");
  }

  const double elapsed = now() - start;

  // Free memory
  hostFree(hx);
  hostFree(hy);

  return size * run_n / elapsed;
}

using CompletionFunction = void (*)();

static void host_add_benchmark_thread(
    size_t n, const float* __restrict__ hx, float* __restrict__ hy,
    std::barrier<CompletionFunction>* barrier) {
  barrier->arrive_and_wait();
  barrier->arrive_and_wait();

  size_t run_n = 1000;
  for (size_t i = 0; i < run_n; ++i) {
    host_add(n, hx, hy);

    __asm__ __volatile__(""
                         : /* no output */
                         : /* no input */
                         : "memory");
  }

  barrier->arrive_and_wait();
}

double host_add_benchmark_multithread(size_t size, unsigned thread_n) {
  float* hx = hostAllocFloat(size);
  float* hy = hostAllocFloat(size);

  // initialize hx and hy arrays on the host
  for (size_t i = 0; i < size; i++) {
    hx[i] = (float)drand48();
    hy[i] = (float)drand48();
  }

  // warm-up
  host_add(size, hx, hy);

  std::vector<std::future<void>> threads(thread_n);
  size_t size_per_thread = size / thread_n;

  std::barrier<CompletionFunction> barrier(thread_n + 1, []() {});

  for (size_t i = 0; i < thread_n; i++) {
    size_t offset = i * size_per_thread;
    threads[i] =
        std::async(std::launch::async, host_add_benchmark_thread,
                   size_per_thread, hx + offset, hy + offset, &barrier);
  }

  barrier.arrive_and_wait();
  const struct timespec start = now();
  barrier.arrive_and_wait();
  barrier.arrive_and_wait();
  const double elapsed = now() - start;

  for (size_t i = 0; i < thread_n; i++) {
    if (threads[i].valid()) {
      threads[i].wait();
    }
  }

  // Free memory
  hostFree(hx);
  hostFree(hy);

  size_t run_n = 1000;
  return (size * run_n / elapsed);
}
