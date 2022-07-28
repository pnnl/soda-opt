#ifndef MLIR_SCHEDULE_H
#define MLIR_SCHEDULE_H

#include <fstream>
#include <map>
#include <vector>

#include "soda/Analysis/DataFlowGraph.h"

namespace mlir {
/// A base class representing a schedule of a loop body.
/// It consists of the prologue, loop iteration and epilogue.
/// Both prologue and epilogue are represented by arrays of incomplete
/// iterations.
class LoopSchedule {
public:
  /// A structure containg all the information about a scheduled operation.
  struct ScheduledOperation {
    ScheduledOperation(Operation *_operation, int _cycle, int _resourceID,
                       int _originalIteration)
        : operation(_operation), resourceID(_resourceID),
          originalIteration(_originalIteration) {}

    /// Original loop operation.
    Operation *operation;
    /// Resource used by this operation.
    int resourceID;
    /// Number of the original iteration this operation belonged to.
    int originalIteration;
  };

  /// This class represents one clock cycle of the schedule it is a part of.
  /// It contains the operations scheduled inside of it.
  class Cycle {
  public:
    Cycle() : number(0) {}
    Cycle(int _number) : number(_number) {}

    std::vector<ScheduledOperation> &getOperations() { return operations; }

    void addOperation(Operation *operation, int resourceID,
                      int originalIteration = 0) {
      operations.push_back(
          ScheduledOperation(operation, number, resourceID, originalIteration));
    }

    void addOperation(ScheduledOperation &scheduledOperation) {
      operations.push_back(scheduledOperation);
    }

  private:
    std::vector<ScheduledOperation> operations;
    int number;
  };

  /// An iteration represents a container class for the cycles within one loop
  /// iteration. While it can refer to an iteration inside a loop, it can also
  /// refer to an incomplete iteration within prologue or epilogue.
  class Iteration {
  public:
    Iteration() = default;

    Cycle &addCycle(int number) {
      cycles.push_back(Cycle(number));
      return cycles[cycles.size() - 1];
    }

    std::vector<Cycle> &getCycles() { return cycles; }

  private:
    std::vector<Cycle> cycles;
  };

  LoopSchedule(DataFlowGraph &_DFG) : DFG(_DFG) {}

  std::vector<Iteration> &getPrologue() { return prologue; }

  Iteration &getIteration() { return loopIteration; }

  std::vector<Iteration> &getEpilogue() { return epilogue; }

  unsigned int getInitiationInterval() { return II; }

protected:
  /// Iteration interval.
  int II;
  int latency;

  std::vector<Iteration> prologue;
  Iteration loopIteration;
  std::vector<Iteration> epilogue;

  DataFlowGraph &DFG;
};

/// This class represents a schedule of the loop body received from an external scheduler.
class ExternalLoopSchedule : public LoopSchedule {
public:
  ExternalLoopSchedule(std::ifstream &_scheduleFile, DataFlowGraph &_DFG,
                   AffineForOp *_forOp);

private:
  /// Extracts iteration interval from the schedule file.
  void extractInitiationInterval();

  /// Helper function for extracting data from the scheduled file.
  void extractRawSchedule();

  /// Populate schedule iterations using the extracted raw schedule.
  void populateSchedule();

  std::map<int, Cycle> rawSchedule;
  std::ifstream &scheduleFile;
  AffineForOp *originalForOp;
};

} // namespace mlir

#endif // MLIR_SCHEDULE_H