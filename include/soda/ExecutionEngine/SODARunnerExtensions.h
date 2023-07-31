//===- SODARunnerExtensions.h - Additional runtime functions that support MLIR runner libraries -===//
//===----------------------------------------------------------------------===//
//
// This file declares several additional runtime functions that extends the 
// runtime used by mlir-runner, mlir-cpu-runner, etc.
//
//===----------------------------------------------------------------------===//

#include <cstdint>

#ifndef SODA_EXECUTIONENGINE_SODARUNNEREXTENSIONS_H
#define SODA_EXECUTIONENGINE_SODARUNNEREXTENSIONS_H

#define SODA_SODARUNNEREXTENSIONS_EXPORT __attribute__((visibility("default")))
#define SODA_SODARUNNEREXTENSIONS_DEFINE_FUNCTIONS

//===----------------------------------------------------------------------===//
// Extra runtime functions to handling debugging sparse tensors in SODA.
//===----------------------------------------------------------------------===//
extern "C" SODA_SODARUNNEREXTENSIONS_EXPORT void printTensorComponent(int64_t component, int64_t level, int64_t tid);
extern "C" SODA_SODARUNNEREXTENSIONS_EXPORT void rtclock_interval(double start, double end);

#endif // SODA_EXECUTIONENGINE_SODARUNNEREXTENSIONS_H
