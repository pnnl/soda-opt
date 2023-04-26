#!/bin/bash

<<<<<<< HEAD
set -e
set -o pipefail

docker run -u $(id -u):$(id -g) -v $(pwd):/working_dir --rm agostini01/soda \
mlir-opt \
  -pass-pipeline="builtin.module(func.func(tosa-to-tensor, tosa-to-linalg-named, tosa-to-linalg, tosa-to-arith))" \
  $1 \
  -o ${2}-inter

docker run -u $(id -u):$(id -g) -v $(pwd):/working_dir --rm agostini01/soda \
mlir-opt \
  --canonicalize \
  -convert-tensor-to-linalg \
  -empty-tensor-to-alloc-tensor \
  -eliminate-empty-tensors \
=======
docker run -u $(id -u):$(id -g) -v $(pwd):/working_dir --rm agostini01/soda \
mlir-opt \
  -pass-pipeline="any( tosa-to-arith, tosa-to-linalg-named, tosa-to-linalg)" \
  --canonicalize \
  -convert-tensor-to-linalg \
  -linalg-init-tensor-to-alloc-tensor \
  -eliminate-alloc-tensors \
>>>>>>> origin/main
  -linalg-bufferize -arith-bufferize \
  -tensor-bufferize -func-bufferize \
  -finalizing-bufferize -buffer-deallocation \
  --buffer-results-to-out-params \
  --canonicalize -cse \
<<<<<<< HEAD
${2}-inter \
=======
$1 \
>>>>>>> origin/main
-o $2
