#ifndef MLIR_SCHEDULING_UTILS_H
#define MLIR_SCHEDULING_UTILS_H

#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "soda/Analysis/DataFlowGraph.h"
#include "mlir/IR/BuiltinOps.h"
#include "soda/Transforms/LoopSchedule.h"

namespace mlir {

AffineForOp extractInnermostLoop(func::FuncOp &Function, int loop);

void conditionLoopExecution(AffineForOp *forOp, LoopSchedule &loopSchedule);

} // namespace mlir

#endif // MLIR_SCHEDULING_UTILS_H