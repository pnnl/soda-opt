#!/bin/bash

# list of needed binaries
NEEDED_BINARIES=(
  soda-opt
  soda-translate
  mlir-opt
  mlir-translate
  flatbuffer_translate
  tf-mlir-translate
  tf-opt
  torch-mlir-opt
  bambu
  openroad
  yosys
)

DOCKER_RUN="docker run -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd) --rm agostini01/soda"
if ! command -v docker &> /dev/null; then
  DOCKER_RUN=""
  
  # Loop over all needed binaries and check if they are available
  for binary in "${NEEDED_BINARIES[@]}"; do
    if ! command -v $binary &> /dev/null; then
      echo "WARNING: The docker binary could not be found. Verifying if all needed binaries are available locally..."
      echo "ERROR: $binary could not be found. Exiting."
      exit
    fi
  done
  # echo "SUCCESS: All needed binaries are available locally."
fi
