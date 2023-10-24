//===- SODARunnerExtensions.cpp - Additional runtime functions that support MLIR runner libraries -===//
//===----------------------------------------------------------------------===//
//
// This file implements several additional runtime functions that extends the 
// runtime used by mlir-runner, mlir-cpu-runner, etc.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string.h>

#include "soda/ExecutionEngine/SODARunnerExtensions.h"

#ifdef SODA_SODARUNNEREXTENSIONS_DEFINE_FUNCTIONS

extern "C" void printTensorComponent(int64_t component, int64_t level, int64_t tid) {
  std::string tid_str = "";
  if (tid < 0) tid_str = "unknown";
  else tid_str = std::to_string(tid);

  switch (component) {
  case 0:
    printf("pointers [lvl = %ld] [tensor id = %s]\n", level, tid_str.c_str());
    break;
  case 1:
    printf("indices [lvl = %ld] [tensor id = %s]\n", level, tid_str.c_str());
    break;
  case 2:
    printf("values [lvl = %ld] [tensor id = %s]\n", level, tid_str.c_str());
    break;
  case 3: // DEPRECATED
    printf("dense [lvl = %ld] [tensor id = %s]\n", level, tid_str.c_str());
    break;
  }
  return;
}

extern "C" void rtclock_interval(double start, double end) {
  fprintf(stderr, "\n=============\nrtclock_interval\n%lf secs\n=============\n\n", end - start);
}

#endif // SODA_SODARUNNEREXTENSIONS_DEFINE_FUNCTIONS
