#ifndef CUDA_INTRO_TIMER_H
#define CUDA_INTRO_TIMER_H

#include <time.h>

#include "check_host.h"

static struct timespec now(void) {
  struct timespec ts;
  CHECK_HOST(clock_gettime(CLOCK_MONOTONIC, &ts));
  return ts;
}

static double operator-(struct timespec left, struct timespec right) {
  return (left.tv_sec - right.tv_sec) +
         (int(left.tv_nsec) - int(right.tv_nsec)) / 1e9;
}

#endif  // CUDA_INTRO_TIMER_H
