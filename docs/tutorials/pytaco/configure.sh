#!/usr/bin/env -S bash -x

# Set up arguments
MLIR_ROOT=$1
SODA_OPT_ROOT=$2
OPENMP_LIB=$3

echo "#!/usr/bin/env bash

# Set up PYTHONPATH
export PYTHONPATH=$SODA_OPT_ROOT/build/python_packages/soda:$MLIR_ROOT/build/tools/mlir/python_packages/mlir_core:$MLIR_ROOT/mlir/test/Integration/Dialect/SparseTensor/taco:\$PYTHONPATH

# Set up libraries
export RUNNER_UTILS=$MLIR_ROOT/build/lib/libmlir_runner_utils.so
export C_RUNNER_UTILS=$MLIR_ROOT/build/lib/libmlir_c_runner_utils.so
export SODA_RUNNER_EXT=$SODA_OPT_ROOT/build/lib/libsoda_runner_ext.so
export OPENMP_LIB=$OPENMP_LIB

# Set up PATH
export PATH=$MLIR_ROOT/build/bin:$SODA_OPT_ROOT/build/bin:\$PATH

# Set up other environment variables
export PTT=$PWD" > ENV
