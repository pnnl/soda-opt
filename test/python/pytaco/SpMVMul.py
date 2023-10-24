# RUN: PYTHONPATH=%pyextra:$PYTHONPATH \
# RUN:  RUNNER_UTILS=%mlir_lib_dir/libmlir_runner_utils%shlibext \
# RUN:  C_RUNNER_UTILS=%mlir_lib_dir/libmlir_c_runner_utils%shlibext \
# RUN:  SODA_RUNNER_EXT=%sodashlibdir/libsoda_runner_ext%shlibext \
# RUN:  SODA_TEST_DIR=%soda_test_dir \
# RUN:  %PYTHON %s | FileCheck %s

from typing import Tuple, List
import os

import mlir_soda.pytaco.mlir_pytaco_ext as ext
from mlir_soda.pytaco.soda_pytaco_check import run_and_check
from tools import mlir_pytaco_api as pt

def SpMVMul(rows: int, cols: int) -> List[Tuple['Tensor', str]]:
    # Set up matrix and vectors
    M = pt.tensor([rows, cols], ext.csr)
    v = pt.tensor([M.shape[1]], ext.dv)
    z = pt.tensor([M.shape[0]], ext.dv)

    # Declare index variables
    i, j = pt.get_index_vars(2)
    
    # Define computation and compile
    z[i] = M[i, j] * v[j]

    # Return the tensor assignments to compile
    return [(z, 'z')]


# This will run sequentially
run_and_check(
    kernel=SpMVMul,
    kernel_inputs=(16, 16),
    tensor_inputs=(
        f"{os.environ['SODA_TEST_DIR']}/Runner/SparseTensor/data/T16x16_0.tns", 
        f"{os.environ['SODA_TEST_DIR']}/Runner/SparseTensor/data/T16_1.tns"
    ),
    enable_openmp=False,
    omp_num_threads=1
)

# CHECK: 1 16
# CHECK: 16
# CHECK: 1 0.479979
# CHECK: 2 0.184463
# CHECK: 3 0
# CHECK: 4 0.412375
# CHECK: 5 0
# CHECK: 6 0
# CHECK: 7 0
# CHECK: 8 0.381202
# CHECK: 9 0
# CHECK: 10 0
# CHECK: 11 0
# CHECK: 12 0
# CHECK: 13 0
# CHECK: 14 0
# CHECK: 15 0
# CHECK: 16 0