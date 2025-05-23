#!/bin/bash

set -e
set -o pipefail

# Check if docker is available or if the needed binaries are available
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPT_DIR/check_docker.sh

$DOCKER_RUN \
tf-mlir-translate \
  --graphdef-to-mlir \
  --tf-input-arrays=x1 \
  --tf-input-data-types=DT_FLOAT \
  --tf-input-shapes=4,32,32,1 \
  --tf-output-arrays=Identity \
  $1 \
  -o output/tf.mlir

$DOCKER_RUN \
tf-opt \
  --tf-executor-to-functional-conversion \
  --tf-region-control-flow-to-functional \
  --tf-shape-inference \
  --tf-to-tosa-pipeline \
  output/tf.mlir \
  -o $2