//===- GenerateLinalgSummary.cpp - GenerateLinalgSummaryPass --------------===//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that generates a summary of linalg generic
// operations.
//
// See test/Analysis/linalg/operation-info.mlir for summary example.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "soda/Misc/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/Support/FileUtilities.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "misc-passes"

using namespace mlir;
using namespace soda;

namespace {

// Struct to hold linalg operation info.
// This info includes:
// - operation name
// - number of inputs
// - number of outputs
// - size of each input/output
// - data type of each input/output
// - bitwidth of each input/output
// - direction of each input/output
// - number of arithmetic instructions inside the kernel
// - number of memory instructions inside the kernel
// - given known sizes:
//   - estimative on total number of arithmetic operations
//   - estimative on total number of memory operations
struct LinalgOpInfo {
  std::string opName;
  int numInputs;
  int numOutputs;
  std::vector<int> inputSizes;
  std::vector<int> outputSizes;
  std::vector<std::string> inputTypes;
  std::vector<std::string> outputTypes;
  std::vector<int> inputTypesBitwidth;
  std::vector<int> outputTypesBitwidth;
  std::vector<std::string> inputDirections;
  std::vector<std::string> outputDirections;
  int numArithmeticOpsInKernel;
  int numMemoryOpsInKernel;
  int numArithmeticOpsEstimative;
  int numMemoryOpsEstimative;
};

// Make getting the size of a memref shape reusable
static int getSizeOfMemRefShape(MemRefType type) {
  auto shape = type.getShape();
  int size = 1;
  for (auto dim : shape) {
    size *= dim;
  }
  return size;
}

static int getSizeOfMemRefShape(TensorType type) {
  auto shape = type.getShape();
  int size = 1;
  for (auto dim : shape) {
    size *= dim;
  }
  return size;
}

static void getInputSizes(mlir::linalg::GenericOp op, std::vector<int> &sizes) {
  for (auto x : op.getInputs()) {
    Type type = x.getType();
    if (MemRefType mr = type.dyn_cast<MemRefType>())
      sizes.push_back(getSizeOfMemRefShape(
          mr)); // can likely use MemRefType.getNumElements()
    else if (TensorType t = type.dyn_cast<TensorType>())
      sizes.push_back(getSizeOfMemRefShape(t));
    else
      sizes.push_back(0);
  }
}

static void getOutputSizes(mlir::linalg::GenericOp op,
                           std::vector<int> &sizes) {
  for (auto x : op.getOutputs()) {
    Type type = x.getType();
    if (MemRefType mr = type.dyn_cast<MemRefType>())
      sizes.push_back(getSizeOfMemRefShape(mr));
    else if (TensorType t = type.dyn_cast<TensorType>())
      sizes.push_back(getSizeOfMemRefShape(t));
    else
      sizes.push_back(0);
  }
}

static void pushTypeToVector(Type type, std::vector<std::string> &types) {
  // Type does not have a str() method.
  // Thus we print the type to a stream,
  // then convert the stream to a string.
  std::string strType;
  llvm::raw_string_ostream rso(strType);
  type.print(rso);
  types.push_back(rso.str());
}

static void getInputElementType(mlir::linalg::GenericOp op,
                                std::vector<std::string> &types) {
  for (auto x : op.getInputs()) {
    pushTypeToVector(x.getType(), types);
  }
}

static void getOuputElementType(mlir::linalg::GenericOp op,
                                std::vector<std::string> &types) {
  for (auto x : op.getOutputs()) {
    pushTypeToVector(x.getType(), types);
  }
}

static void getInputElementTypeBitwidth(mlir::linalg::GenericOp op,
                                        std::vector<int> &bitwidths) {
  for (auto x : op.getInputs()) {
    Type type = x.getType();
    if (MemRefType mr = type.dyn_cast<MemRefType>())
      bitwidths.push_back(mr.getElementTypeBitWidth());
    else if (TensorType t = type.dyn_cast<TensorType>())
      bitwidths.push_back(t.getElementTypeBitWidth());
    else
      bitwidths.push_back(0);
  }
}

static void getOuputElementTypeBitwidth(mlir::linalg::GenericOp op,
                                        std::vector<int> &bitwidths) {
  for (auto x : op.getOutputs()) {
    Type type = x.getType();
    if (MemRefType mr = type.dyn_cast<MemRefType>())
      bitwidths.push_back(mr.getElementTypeBitWidth());
    else if (TensorType t = type.dyn_cast<TensorType>())
      bitwidths.push_back(t.getElementTypeBitWidth());
    else
      bitwidths.push_back(0);
  }
}

static void getNumArithmeticOpsInKernel(mlir::linalg::GenericOp op,
                                        int &numArithmeticOps) {
  // Implement a walk over the inner kernel of a linalg.generic op
  // and count the number of arithmetic operations
  numArithmeticOps = 0;
  for (auto &x : /*bb*/ op.getBody()->getOperations()) {
    // Get the operation name and if contains `arith` then increment the counter
    if (x.getName().getStringRef().contains("arith")) {
      numArithmeticOps++;
    }
  }
}

static void getNumMemoryOpsInKernel(mlir::linalg::GenericOp op,
                                    int &numMemoryOps) {
  // To account for the memory operations we must check which basic block
  // arguments are used inside the kernel If a bba is used, it means that a load
  // operation is being performed If there is yielded values in linalg.yield, it
  // means that a store operation is being performed
  int numLoadOps = 0;
  int numStoreOps = 0;
  for (auto &bba : op.getBody()->getArguments()) {
    if (bba.use_empty())
      continue;
    numLoadOps++;
  }

  auto yieldOp = op.getBody()->getTerminator();
  assert(isa<linalg::YieldOp>(yieldOp));
  numStoreOps = yieldOp->getNumOperands();

  numMemoryOps = numLoadOps + numStoreOps;
}

static int getNumberOfIterations(mlir::linalg::GenericOp op) {
  // To provide an estimative on the number of arithmetic operations,
  // we can use the number of loops and the loop bounds for this GenericOp.

  // The number of loops derives from the number of iterator types (e.g.
  // parallel, reduction, etc.)
  // int numLoops = op.getIteratorTypes().size();

  // The bounds of the loops can be obtained from the `indexing_maps` attribute
  // and the dimensions of the input and output tensors Each genericOp has a
  // list of indexing maps (affine_map), one for each input and output tensor
  // ex: #map = affine_map<(d0, d1, d2) -> (d0, d2)>
  // In the example the input/output that uses this map has 2 dimensions and the
  // loop bounds 0 and 2 are d0 and d2
  // We will reduce with the product of the loop bounds
  std::vector<int> loopBounds;

  // All maps have the same number of dimensions, so we can use the first one to
  // get the number of dimensions
  auto mapAttr = op.getIndexingMaps()[0];
  auto map = mapAttr.cast<AffineMapAttr>().getValue();
  int numDims = map.getNumDims();

  // Now we create a dictionary to store if we have already processed a
  // dimension so we don't count dimensions twice
  std::map<int, bool> processedDims;
  for (int i = 0; i < numDims; i++) {
    processedDims[i] = false;
  }

  // Iterate over all indexing_map, input/output pairs.
  int count = 0;
  int numInputs = op.getInputs().size();
  int numOutputs = op.getOutputs().size();
  for (auto &mapAttr : op.getIndexingMaps()) {
    // Assert count is in range of inputs and outputs.
    assert(count < numInputs + numOutputs &&
           "Number of indexing_maps does not match number of inputs and "
           "outputs.");

    // Inputs and outputs are split in two different ranges.
    Value v;
    if (count < numInputs) {
      v = op.getInputs()[count];
    } else {
      v = op.getOutputs()[count - numInputs];
    }
    count++;

    auto map = mapAttr.cast<AffineMapAttr>().getValue();

    for (unsigned i = 0; i < map.getNumResults(); i++) {
      if (processedDims[map.getDimPosition(i)] == false) {
        processedDims[map.getDimPosition(i)] = true;
        loopBounds.push_back(v.getType().cast<ShapedType>().getDimSize(i));
      } else {
        // If we have already processed this dimension, we can skip it
        continue;
      }
    }

    // ---------------------------
    // Print debug information
    // llvm::outs() << "\n";
    // map.print(llvm::outs());
    // llvm::outs() << "\n";
    // llvm::outs() << "NumDims: " << map.getNumDims() << "\n";
    // llvm::outs() << "NumSymbols: " << map.getNumSymbols() << "\n";
    // llvm::outs() << "NumResults: " << map.getNumResults() << "\n";
    // llvm::outs() << "NumInputs: " << map.getNumInputs() << "\n";
    // llvm::outs() << "\n";
    // // Print results
    // for (int i = 0; i < map.getNumResults(); i++) {
    //   llvm::outs() << "Result " << i << ": " << map.getResult(i) << "\n";
    // }
    // // Print DimPosition
    // for (int i = 0; i < map.getNumResults(); i++) {
    //   llvm::outs() << "DimPos " << i << ": " << map.getDimPosition(i) <<
    //   "\n";
    // }
    // // Print input.getType()
    // llvm::outs() << "Argument Type: " << v.getType() << "\n";
    // llvm::outs() << "\n";
    // ---------------------------

    // Check if we processed all dimensions so we can early exit the loop
    bool allProcessed = true;
    for (int i = 0; i < numDims; i++) {
      if (processedDims[i] == false) {
        allProcessed = false;
        break;
      }
    }
    if (allProcessed) {
      break;
    }
  }

  // ---------------------------
  // Print debug information
  // llvm::outs() << "\n";
  // llvm::outs() << "Loop Bounds: ";
  // for (auto &bound : loopBounds) {
  //   llvm::outs() << bound << " ";
  // }
  // llvm::outs() << "\n";
  // ---------------------------

  // Now we can calculate the number of iterations
  int numberOfIterations = 1;
  for (auto &bound : loopBounds) {
    numberOfIterations *= bound;
  }

  return numberOfIterations;
}

static void getNumArithmeticOpsEstimative(mlir::linalg::GenericOp op,
                                          LinalgOpInfo &opInfo) {
  int numOfIterations = getNumberOfIterations(op);
  opInfo.numArithmeticOpsEstimative =
      numOfIterations * opInfo.numArithmeticOpsInKernel;
}

static void getNumMemoryOpsEstimative(mlir::linalg::GenericOp op,
                                      LinalgOpInfo &opInfo) {
  int numOfIterations = getNumberOfIterations(op);
  opInfo.numMemoryOpsEstimative = numOfIterations * opInfo.numMemoryOpsInKernel;
}

static void collectLinaglOperationInfo(LinalgOpInfo &opInfo,
                                       mlir::linalg::GenericOp op) {
  std::string kernelFnName =
      // opInfo.opName =
      // op->getParentOfType<func::FuncOp>().getName()+"/"+op->getName();
      opInfo.opName = Twine(op->getParentOfType<func::FuncOp>().getName(),
                            "/linalg.generic")
                          .str();
  opInfo.numInputs = op.getInputs().size();
  opInfo.numOutputs = op.getOutputs().size();
  getInputSizes(op, opInfo.inputSizes);
  getOutputSizes(op, opInfo.outputSizes);
  getInputElementType(op, opInfo.inputTypes);
  getOuputElementType(op, opInfo.outputTypes);
  getInputElementTypeBitwidth(op, opInfo.inputTypesBitwidth);
  getOuputElementTypeBitwidth(op, opInfo.outputTypesBitwidth);
  // getInputDirections();
  // getOutputDirections();
  getNumArithmeticOpsInKernel(op, opInfo.numArithmeticOpsInKernel);
  getNumMemoryOpsInKernel(op, opInfo.numMemoryOpsInKernel);
  getNumArithmeticOpsEstimative(op, opInfo);
  getNumMemoryOpsEstimative(op, opInfo);
}

class GenerateLinalgSummaryPass
    : public mlir::soda::GenerateLinalgSummaryBase<GenerateLinalgSummaryPass> {

  void runOnOperation() override {

    getOperation().walk([this](mlir::linalg::GenericOp op) {
      // Prepare the output streams
      std::string errorMessage;
      // std::string filename = op.getKernelName().getValue().str() +
      // "_linalg_summary.txt";
      std::string filename = "linalg_summary.txt";
      auto output = openOutputFile(filename, &errorMessage);
      outputStream = &output->os();

      if (writeToTerminal) {
        outputStream = &llvm::outs();
      }

      // Populate the stream with the xml vector
      resetIndent();

      if (!writeToTerminal) {
        output->keep();
      }

      LinalgOpInfo opInfo;
      collectLinaglOperationInfo(opInfo, op);
      printLinalgOpInfo(opInfo);
    });
  }

  // To keep track of what name to use for the XML arguments
  int pointerId = 0;

  // Resets pointer ID
  // Should be called at each new testbench
  void resetPointerId() { pointerId = 0; }
  int incPointerId() { return pointerId++; }

  void printAestPreamble() {
    printIndent() << "<?xml version=\"1.0\"?>\n"
                  << "<function>\n";
  }

  void initTestbench() {
    printIndent() << "<testbench\n";
    resetPointerId();
  }

  void closeTestbench() { printIndent() << "/>\n"; }

  void printLinalgOpInfo(const LinalgOpInfo &opInfo) {
    printA() << "=========================\n";
    printA() << "REPORT BEGIN\n";
    printA() << "LinalgOpInfo: " << opInfo.opName << "\n";
    printA() << "  numInputs: " << opInfo.numInputs << "\n";
    printA() << "  numOutputs: " << opInfo.numOutputs << "\n";
    printA() << "  inputSizes: ";
    for (auto s : opInfo.inputSizes) {
      printA() << s << " ";
    }
    printA() << "\n";
    printA() << "  outputSizes: ";
    for (auto s : opInfo.outputSizes) {
      printA() << s << " ";
    }
    printA() << "\n";
    printA() << "  inputTypes: ";
    for (auto t : opInfo.inputTypes) {
      printA() << t << " ";
    }
    printA() << "\n";
    printA() << "  outputTypes: ";
    for (auto t : opInfo.outputTypes) {
      printA() << t << " ";
    }
    printA() << "\n";
    printA() << "  inputTypesBitwidth: ";
    for (auto t : opInfo.inputTypesBitwidth) {
      printA() << t << " ";
    }
    printA() << "\n";
    printA() << "  outputTypesBitwidth: ";
    for (auto t : opInfo.outputTypesBitwidth) {
      printA() << t << " ";
    }
    printA() << "\n";
    printA() << "  inputDirections: ";
    for (auto d : opInfo.inputDirections) {
      printA() << d << " ";
    }
    printA() << "\n";
    printA() << "  outputDirections: ";
    for (auto d : opInfo.outputDirections) {
      printA() << d << " ";
    }
    printA() << "\n";
    printA() << "  numArithmeticOpsInKernel: "
             << opInfo.numArithmeticOpsInKernel << "\n";
    printA() << "  numMemoryOpsInKernel: " << opInfo.numMemoryOpsInKernel
             << "\n";
    printA() << "  numArithmeticOpsEstimative: "
             << opInfo.numArithmeticOpsEstimative << "\n";
    printA() << "  numMemoryOpsEstimative: " << opInfo.numMemoryOpsEstimative
             << "\n";
    printA() << "REPORT END\n";
    printA() << "=========================\n";
  }

  /// Manages the indentation as we traverse the IR nesting.
  int indent;
  struct IndentRAII {
    int &indent;
    IndentRAII(int &indent) : indent(indent) {}
    ~IndentRAII() { --indent; }
  };
  void resetIndent() { indent = 0; }
  IndentRAII pushIndent() { return IndentRAII(++indent); }

  /// Output streams to the generated XML files or terminal output
  raw_ostream *outputStream;
  raw_ostream &analysisOut() { return *outputStream; }

  llvm::raw_ostream &printIndent() {
    for (int i = 0; i < indent; ++i)
      analysisOut() << " ";
    return analysisOut();
  }

  llvm::raw_ostream &printA() { return analysisOut(); }
};

} // end anonymous namespace

// Generate linalg summary pass
std::unique_ptr<mlir::Pass> mlir::soda::createGenerateLinalgSummaryPass() {
  return std::make_unique<GenerateLinalgSummaryPass>();
}