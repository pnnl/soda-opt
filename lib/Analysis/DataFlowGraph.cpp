#include "soda/Analysis/DataFlowGraph.h"

#include <fstream>
#include <iostream>

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"

using namespace mlir;

int DFGNode::ID = 1;

void DataFlowGraph::printToGraphmlFile(std::string filePath) {
  std::ofstream graphmlFile(filePath);

  graphmlFile << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;
  graphmlFile << "<graphml>" << std::endl;
  graphmlFile << "    <graph>" << std::endl;

  for (DFGNode *node : *this) {
    node->printToGraphml(graphmlFile);
  }

  for (DFGNode *node : *this) {
    for (DFGEdge *edge : node->getEdges()) {
      edge->printToGraphml(node, graphmlFile);
    }
  }

  graphmlFile << "    </graph>" << std::endl;
  graphmlFile << "</graphml>" << std::endl;

  graphmlFile.close();
}

void LoopDataFlowGraphBuilder::buildGraphFromLoop(AffineForOp *forOp) {
  addNodes(forOp);
  addDefUseEdges(forOp);
  addMemoryEdges(forOp);
}

void LoopDataFlowGraphBuilder::addNodes(AffineForOp *forOp) {
  forOp->walk([&](Operation *operation) {
    if (operation->getBlock() != forOp->getBody()) {
      return;
    }

    if (isa<AffineYieldOp>(operation)) {
      return;
    }

    if (isa<AffineLoadOp>(operation)) {
      graph.addNode(*(new DFGNode(operation, DFGNode::NodeKind::Load)));
      return;
    }

    if (isa<AffineStoreOp>(operation)) {
      graph.addNode(*(new DFGNode(operation, DFGNode::NodeKind::Store)));
      return;
    }

    if (isa<arith::AddIOp, arith::AndIOp, arith::MulIOp, arith::OrIOp, arith::ShLIOp, arith::DivSIOp,
            arith::FloorDivSIOp, arith::CeilDivSIOp, arith::RemSIOp,
            arith::ShRSIOp, arith::SubIOp, arith::DivUIOp, arith::RemUIOp,
            arith::ShRUIOp, arith::XOrIOp, arith::AddFOp, math::CopySignOp, arith::DivFOp, arith::MulFOp,
            arith::RemFOp, arith::SubFOp, arith::SelectOp, arith::CmpFOp>(operation)) {
      graph.addNode(
          *(new DFGNode(operation, DFGNode::NodeKind::Computational)));
      return;
    }

    if (isa<AffineMaxOp, AffineMinOp, AffineApplyOp, AffineIfOp, func::CallOp>(
            operation)) {
      graph.addNode(*(new DFGNode(operation, DFGNode::NodeKind::Subfunction)));
      return;
    }

    // If the operation doesn't fall into the defined categories, it is
    // classified as a subfunction
    graph.addNode(*(new DFGNode(operation, DFGNode::NodeKind::Subfunction)));
  });
}

void LoopDataFlowGraphBuilder::addDefUseEdges(AffineForOp *forOp) {
  for (DFGNode *node : graph) {
    Operation *operation = node->getOperation();
    for (auto user : operation->getUsers()) {
      auto destinationNode = graph.getNode(user);

      auto parentOperation = user->getParentOp();
      while (destinationNode == nullptr && parentOperation != nullptr) {
        destinationNode = graph.getNode(parentOperation);
        parentOperation = parentOperation->getParentOp();
      }

      if (destinationNode != nullptr) {
        auto edge = new DFGEdge(*destinationNode, DFGEdge::EdgeKind::DefUse, 0);
        graph.connect(*node, *destinationNode, *edge);
      }
    }
  }

  if (forOp->getNumRegionIterArgs() > 0) {
    int numArguments = forOp->getBody()->getNumArguments();
    Operation *sourceOperation =
        forOp->getBody()->getTerminator()->getOperand(0).getDefiningOp();
    for (auto user :
         forOp->getBody()->getArgument(numArguments - 1).getUsers()) {
      auto destinationNode = graph.getNode(user);
      auto edge = new DFGEdge(*destinationNode, DFGEdge::EdgeKind::DefUse, 1);
      graph.connect(*graph.getNode(sourceOperation), *destinationNode, *edge);
    }
  }
}

void LoopDataFlowGraphBuilder::addMemoryEdges(AffineForOp *forOp) {
  for (DFGNode *sourceNode : graph) {
    Operation *sourceOperation = sourceNode->getOperation();
    // We are only interested in memory operations.
    if (!isa<AffineLoadOp, AffineStoreOp>(sourceOperation))
      continue;

    MemRefAccess sourceAccess(sourceOperation);
    for (DFGNode *destinationNode : graph) {
      Operation *destinationOperation = destinationNode->getOperation();
      // We are only interested in memory operations.
      if (!isa<AffineLoadOp, AffineStoreOp>(destinationOperation))
        continue;

      MemRefAccess destinationAccess(destinationOperation);
      unsigned numCommonLoops =
          getNumCommonSurroundingLoops(*sourceOperation, *destinationOperation);
      FlatAffineValueConstraints dependenceConstraints;
      SmallVector<DependenceComponent, 2> dependenceComponents;

      DependenceResult result = checkMemrefAccessDependence(
          sourceAccess, destinationAccess, /*loopDepth*/ numCommonLoops + 1,
          &dependenceConstraints, &dependenceComponents);

      if (hasDependence(result)) {
        auto edge = new DFGEdge(*destinationNode,
                                DFGEdge::EdgeKind::MemoryDependence, 0);
        graph.connect(*sourceNode, *destinationNode, *edge);
      } else {
        DependenceResult result = checkMemrefAccessDependence(
            sourceAccess, destinationAccess, /*loopDepth*/ numCommonLoops,
            &dependenceConstraints, &dependenceComponents);

        if (hasDependence(result)) {
          int distance = dependenceComponents[dependenceComponents.size() - 1]
                             .lb.getValue();

          int scaledDistance = distance / forOp->getStep();
          if (scaledDistance * forOp->getStep() != distance) {
            scaledDistance++;
          }

          auto edge =
              new DFGEdge(*destinationNode, DFGEdge::EdgeKind::MemoryDependence,
                          scaledDistance);
          graph.connect(*sourceNode, *destinationNode, *edge);
        }
      }
    }
  }
}
