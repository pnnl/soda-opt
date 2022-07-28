#include "soda/Transforms/LoopSchedule.h"

using namespace mlir;

mlir::ExternalLoopSchedule::ExternalLoopSchedule(std::ifstream &_scheduleFile,
                                         DataFlowGraph &_DFG,
                                         AffineForOp *_forOp)
    : LoopSchedule(_DFG), scheduleFile(_scheduleFile), originalForOp(_forOp) {
  extractInitiationInterval();
  extractRawSchedule();
  populateSchedule();
}

void mlir::ExternalLoopSchedule::extractInitiationInterval() {
  std::string line;
  if (!scheduleFile.eof()) {
    std::getline(scheduleFile, line);
    if (scheduleFile.bad() || scheduleFile.fail()) {
      return;
    }

    II = std::stoi(line.substr(5));
  }
}

void mlir::ExternalLoopSchedule::extractRawSchedule() {
  std::string line;

  while (!scheduleFile.eof()) {
    std::getline(scheduleFile, line);
    if (scheduleFile.bad() || scheduleFile.fail()) {
      break;
    }

    if (line.empty() || line[0] == '#') {
      continue;
    }

    // Extract node ID
    int firstDelimiterPosition = line.find(";");
    int secondDelimiterPosition = line.find(";", firstDelimiterPosition + 1);
    std::string nodeName = line.substr(0, firstDelimiterPosition);
    int nodeID = std::stoi(nodeName.substr(nodeName.find("_") + 1));
    DFGNode *graphNode = DFG.getNode(nodeID);

    // Extract cycle number
    int cycleNumber = std::stoi(
        line.substr(firstDelimiterPosition + 1,
                    secondDelimiterPosition - firstDelimiterPosition - 1));

    // Extract resource ID
    int resourceID = std::stoi(line.substr(secondDelimiterPosition + 1));

    // Add operation to the schedule
    if (rawSchedule.find(cycleNumber) == rawSchedule.end()) {
      Cycle operations(cycleNumber);
      operations.addOperation(graphNode->getOperation(), resourceID);
      rawSchedule.insert({cycleNumber, operations});
    } else {
      rawSchedule[cycleNumber].addOperation(graphNode->getOperation(),
                                            resourceID);
    }
  }

  // Latency is the number of last cycle incremented by one as cycles are
  // numbered starting from 0
  latency = rawSchedule.rbegin()->first + 1;
}

void mlir::ExternalLoopSchedule::populateSchedule() {
  int operationCount = 0;
  for (auto it = rawSchedule.begin(); it != rawSchedule.end(); ++it) {
    operationCount += it->second.getOperations().size();
  }

  int prologueIterationNumber = 0;
  int currentOperationCount = 0;

  // Populate prologue and loop iteration
  while (currentOperationCount != operationCount) {
    Iteration iteration;
    currentOperationCount = 0;

    for (int i = 0; i < II; i++) {
      Cycle &cycle = iteration.addCycle(i);

      for (int j = 0; i + j < latency && prologueIterationNumber - j / II >= 0;
           j += II) {
        for (auto scheduledOperation : rawSchedule[i + j].getOperations()) {
          scheduledOperation.originalIteration =
              prologueIterationNumber - j / II;
          cycle.addOperation(scheduledOperation);
          currentOperationCount++;
        }
      }
    }

    if (currentOperationCount != operationCount) {
      prologue.push_back(iteration);
      prologueIterationNumber++;
    } else {
      loopIteration = iteration;
    }
  }

  int epilogueIterationNumber = 0;

  // Populate epilogue
  while (currentOperationCount != 0) {
    Iteration epilogueIteration;
    currentOperationCount = 0;

    for (int i = 0; i < II; i++) {
      Cycle &cycle = epilogueIteration.addCycle(i);

      for (int j = II * (epilogueIterationNumber + 1); i + j < latency;
           j += II) {
        for (auto scheduledOperation : rawSchedule[i + j].getOperations()) {
          scheduledOperation.originalIteration =
              prologueIterationNumber + 1 + epilogueIterationNumber - j / II;
          cycle.addOperation(scheduledOperation);
          currentOperationCount++;
        }
      }
    }

    if (currentOperationCount != 0) {
      epilogue.push_back(epilogueIteration);
      epilogueIterationNumber++;
    }
  }
}