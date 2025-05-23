#!/bin/bash

set -e
set -o pipefail

# Check if docker is available or if the needed binaries are available
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPT_DIR/check_docker.sh

$DOCKER_RUN \
mlir-opt \
  -pass-pipeline="builtin.module(func.func(tosa-to-arith, tosa-to-tensor, tosa-to-linalg-named, tosa-to-linalg))" \
  $1 \
  -o ${2}-inter

$DOCKER_RUN \
mlir-opt \
  --tosa-to-arith="include-apply-rescale=true" \
  --canonicalize \
  -convert-tensor-to-linalg \
  -empty-tensor-to-alloc-tensor \
  -eliminate-empty-tensors \
  -one-shot-bufferize="function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries allow-return-allocs-from-loops unknown-type-conversion=identity-layout-map" \
  -func-bufferize \
  -buffer-deallocation-simplification \
  -bufferization-lower-deallocations \
  --buffer-results-to-out-params \
  --canonicalize -cse \
${2}-inter \
-o $2
