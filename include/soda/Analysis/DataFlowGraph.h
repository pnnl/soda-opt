#ifndef MLIR_ANALYSIS_DATAFLOWGRAPH_H
#define MLIR_ANALYSIS_DATAFLOWGRAPH_H

#include <fstream>
#include <string>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DirectedGraph.h"

namespace mlir {

class DFGNode;
class DFGEdge;

/// Node in the data flow graph containing a pointer to the operation.
class DFGNode : public llvm::DGNode<DFGNode, DFGEdge> {
public:
  enum class NodeKind { Computational, Load, Store, Subfunction };

  DFGNode(Operation *_operation, NodeKind _kind)
      : operation(_operation), kind(_kind) {
    nodeID = ID++;
  }

  int getID() { return nodeID; }

  NodeKind getKind() { return kind; }

  Operation *getOperation() { return operation; }

  std::string getOperationName() {
    std::string name = operation->getName().getStringRef().data();
    return name;
  }

  void printToGraphml(std::ofstream &file) {
    std::string operationName = getOperationName();
    std::string resource;
    switch (kind) {
    case NodeKind::Computational:
      resource = operationName;
      break;
    case NodeKind::Subfunction:
      resource = "subfunction";
      break;
    case NodeKind::Load:
    case NodeKind::Store:
      resource = "mem";
    }

    file << "    <node id=\"" + std::to_string(nodeID) + "\">" << std::endl;
    file << "        <data key=\"name\">" + operationName + "_" +
                std::to_string(nodeID) + "</data>"
         << std::endl;
    file << "        <data key=\"uses_resource\">" + resource + "</data>"
         << std::endl;
    file << "    </node>" << std::endl;
  }

private:
  Operation *operation;
  NodeKind kind;
  int nodeID;

  static int ID;
};

/// Edge in the data flow graph between DFGNodes. It can represent a memory
/// dependence or a simple result usage.
class DFGEdge : public llvm::DGEdge<DFGNode, DFGEdge> {
public:
  enum class EdgeKind {
    MemoryDependence,
    DefUse,
  };

  DFGEdge(DFGNode &_node, EdgeKind _kind, int _distance)
      : DGEdge(_node), kind(_kind), distance(_distance) {}

  EdgeKind getKind() { return kind; }

  void printToGraphml(DFGNode *sourceNode, std::ofstream &file) {
    std::string edgeKind;
    switch (kind) {
    case EdgeKind::MemoryDependence:
      edgeKind = "Data";
      break;
    case EdgeKind::DefUse:
      edgeKind = "Precedence";
      break;
    }

    std::string sourceNodeID = std::to_string(sourceNode->getID());
    std::string targetNodeID = std::to_string(TargetNode.getID());

    file << "    <edge id=\"" + sourceNodeID + "_" + targetNodeID +
                "\" source=\"" + sourceNodeID + "\" target=\"" + targetNodeID +
                "\">"
         << std::endl;
    file << "        <data key=\"delay\">0</data>" << std::endl;
    file << "        <data key=\"distance\">" + std::to_string(distance) +
                "</data>"
         << std::endl;
    file << "        <data key=\"deptype\">" + edgeKind + "</data>"
         << std::endl;
    file << "    </edge>" << std::endl;
  }

private:
  EdgeKind kind;
  int distance;
};

/// Data flow graph with support of printing to a .graphml file.
class DataFlowGraph : public llvm::DirectedGraph<DFGNode, DFGEdge> {
public:
  DFGNode *getNode(Operation *operation) {
    for (auto *node : *this) {
      if (node->getOperation() == operation) {
        return node;
      }
    }

    return nullptr;
  }

  DFGNode *getNode(int nodeID) {
    for (auto *node : *this) {
      if (node->getID() == nodeID) {
        return node;
      }
    }

    return nullptr;
  }

  void printToGraphmlFile(std::string filePath);
};

/// This class is responsible for building a data flow graph from the operations
/// in the passed loop.
class LoopDataFlowGraphBuilder {
public:
  LoopDataFlowGraphBuilder(DataFlowGraph &dfg) : graph(dfg) {}

  void buildGraphFromLoop(AffineForOp *forOp);

protected:
  void addNodes(AffineForOp *forOp);

  void addDefUseEdges(AffineForOp *forOp);

  void addMemoryEdges(AffineForOp *forOp);

  DataFlowGraph &graph;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_DATAFLOWGRAPH_H