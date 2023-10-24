//===- Passes.td - SparseTensor Reports for SODA -*- tablegen -*-----------------===//
//===----------------------------------------------------------------------===//
//
// Defines language-specific pattern match optimizations for SODA using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef SODA_SPARSETENSOR_TRANSFORMS_PASSES
#define SODA_SPARSETENSOR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Basic analysis passes
//===----------------------------------------------------------------------===//

def DenseBufferization : Pass<"soda-sparse-compiler-dense-bufferization", "ModuleOp"> {
  let summary = "Lowers dense tensor ops through BufferizableOpInterface implementations. Split from SparsificationAndBufferizationPass.";
  let description = [{
      Lowers dense tensor ops through BufferizableOpInterface implementations. Split from SparsificationAndBufferizationPass.
  }];
  let constructor = "mlir::soda::createDenseBufferizationPass()";
  let dependentDialects = [
    "bufferization::BufferizationDialect",
  ];
  let options = [
    Option<"testBufferizationAnalysisOnly", "test-bufferization-analysis-only", "bool", /*default=*/"false",
           "Run only the inplacability analysis (default: false)">
  ];
}

def InsertTensorCopies : Pass<"soda-sparse-compiler-insert-tensor-copies", "ModuleOp"> {
  let summary = "Lowers dense tensor ops through BufferizableOpInterface implementations. Split from SparsificationAndBufferizationPass.";
  let description = [{
      Lowers dense tensor ops through BufferizableOpInterface implementations. Split from SparsificationAndBufferizationPass.
  }];
  let constructor = "mlir::soda::createInsertTensorCopiesPass()";
  let dependentDialects = [
    "bufferization::BufferizationDialect",
  ];
  let options = [
    Option<"testBufferizationAnalysisOnly", "test-bufferization-analysis-only", "bool", /*default=*/"false",
           "Run only the inplacability analysis (default: false)">
  ];
}

def SparseTensorMemrefDebug : Pass<"soda-sparse-compiler-memref-debug", "ModuleOp"> {
  let summary = "Instruments .pointers, .indices, and .values operations with runtime debugging calls.";
  let description = [{
      Instruments .pointers, .indices, and .values operations with runtime debugging calls.
  }];
  let constructor = "mlir::soda::createSparseTensorMemrefDebugPass()";
  let dependentDialects = [
    "sparse_tensor::SparseTensorDialect",
  ];
}

#endif // SODA_SPARSETENSOR_TRANSFORMS_PASSES