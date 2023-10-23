# RUN: PYTHONPATH=%pyextra:$PYTHONPATH \
# RUN:  RUNNER_UTILS=%mlir_lib_dir/libmlir_runner_utils%shlibext \
# RUN:  C_RUNNER_UTILS=%mlir_lib_dir/libmlir_c_runner_utils%shlibext \
# RUN:  OPENMP_LIB=%openmp_lib \
# RUN:  SODA_RUNNER_EXT=%sodashlibdir/libsoda_runner_ext%shlibext \
# RUN:  SODA_TEST_DIR=%soda_test_dir \
# RUN:  %PYTHON %s | FileCheck %s

from typing import Tuple, List
import os

import mlir_soda.pytaco.mlir_pytaco_ext as ext
from mlir_soda.pytaco.soda_pytaco_check import run_and_check
from tools import mlir_pytaco_api as pt

def SpMSpMMul(ARows: int, ACols: int, BCols: int) -> List[Tuple['Tensor', str]]:
    # Set up matrices
    A = pt.tensor([ARows, ACols], ext.csr)
    B = pt.tensor([A.shape[1], BCols], ext.csc)
    Z = pt.tensor([A.shape[0], B.shape[1]], ext.dm)

    # Declare index variables
    i, j, k = pt.get_index_vars(3)

    # Define computation and compile
    Z[i, j] = A[i, k] * B[k, j]

    # Return the tensor assignments to compile
    return [(Z, 'Z')]

# This will run w/ any-storage-any-loop par, 4 omp threads by default
run_and_check(
    kernel=SpMSpMMul,
    kernel_inputs=(16, 16, 16),
    tensor_inputs=(
        f"{os.environ['SODA_TEST_DIR']}/Runner/SparseTensor/data/T16x16_0.tns", 
        f"{os.environ['SODA_TEST_DIR']}/Runner/SparseTensor/data/T16x16_1.tns"
    )
)

# CHECK: 2 256
# CHECK: 16 16
# CHECK: 1 1 0.611014
# CHECK: 1 2 0
# CHECK: 1 3 0
# CHECK: 1 4 0
# CHECK: 1 5 0
# CHECK: 1 6 0.0419861
# CHECK: 1 7 0.313736
# CHECK: 1 8 0
# CHECK: 1 9 0
# CHECK: 1 10 0.682485
# CHECK: 1 11 0.165686
# CHECK: 1 12 0
# CHECK: 1 13 0
# CHECK: 1 14 0
# CHECK: 1 15 0
# CHECK: 1 16 0.436722
# CHECK: 2 1 0.229735
# CHECK: 2 2 0
# CHECK: 2 3 0.168233
# CHECK: 2 4 0
# CHECK: 2 5 0
# CHECK: 2 6 0.394361
# CHECK: 2 7 0
# CHECK: 2 8 0.448312
# CHECK: 2 9 0
# CHECK: 2 10 0.444184
# CHECK: 2 11 0
# CHECK: 2 12 0
# CHECK: 2 13 0
# CHECK: 2 14 0
# CHECK: 2 15 0
# CHECK: 2 16 0.155885
# CHECK: 3 1 0
# CHECK: 3 2 0.57617
# CHECK: 3 3 0.112128
# CHECK: 3 4 0.245211
# CHECK: 3 5 0.708332
# CHECK: 3 6 0.790416
# CHECK: 3 7 0
# CHECK: 3 8 0
# CHECK: 3 9 0
# CHECK: 3 10 0
# CHECK: 3 11 0
# CHECK: 3 12 0.604531
# CHECK: 3 13 0
# CHECK: 3 14 0.817553
# CHECK: 3 15 0.593079
# CHECK: 3 16 0.583207
# CHECK: 4 1 0.365266
# CHECK: 4 2 0.0974632
# CHECK: 4 3 0
# CHECK: 4 4 0
# CHECK: 4 5 0
# CHECK: 4 6 0.0360726
# CHECK: 4 7 0
# CHECK: 4 8 0
# CHECK: 4 9 0
# CHECK: 4 10 0
# CHECK: 4 11 0
# CHECK: 4 12 0
# CHECK: 4 13 0
# CHECK: 4 14 0.0805368
# CHECK: 4 15 0.0255885
# CHECK: 4 16 0
# CHECK: 5 1 0
# CHECK: 5 2 0
# CHECK: 5 3 0
# CHECK: 5 4 0
# CHECK: 5 5 0
# CHECK: 5 6 0
# CHECK: 5 7 0.354123
# CHECK: 5 8 0
# CHECK: 5 9 0
# CHECK: 5 10 0
# CHECK: 5 11 0.187015
# CHECK: 5 12 0
# CHECK: 5 13 0
# CHECK: 5 14 0
# CHECK: 5 15 0
# CHECK: 5 16 0
# CHECK: 6 1 0
# CHECK: 6 2 0.27444
# CHECK: 6 3 0
# CHECK: 6 4 0
# CHECK: 6 5 0
# CHECK: 6 6 0
# CHECK: 6 7 0
# CHECK: 6 8 0
# CHECK: 6 9 0
# CHECK: 6 10 0
# CHECK: 6 11 0
# CHECK: 6 12 0
# CHECK: 6 13 0
# CHECK: 6 14 0.226778
# CHECK: 6 15 0.0720528
# CHECK: 6 16 0
# CHECK: 7 1 0.507128
# CHECK: 7 2 0.0203137
# CHECK: 7 3 0.0203171
# CHECK: 7 4 0.0444314
# CHECK: 7 5 0
# CHECK: 7 6 0
# CHECK: 7 7 0
# CHECK: 7 8 0.413347
# CHECK: 7 9 0
# CHECK: 7 10 0
# CHECK: 7 11 0
# CHECK: 7 12 0.265868
# CHECK: 7 13 0
# CHECK: 7 14 0.148138
# CHECK: 7 15 0
# CHECK: 7 16 0.105675
# CHECK: 8 1 0.337654
# CHECK: 8 2 0
# CHECK: 8 3 0
# CHECK: 8 4 0
# CHECK: 8 5 0
# CHECK: 8 6 0.0333457
# CHECK: 8 7 0
# CHECK: 8 8 0
# CHECK: 8 9 0
# CHECK: 8 10 0
# CHECK: 8 11 0
# CHECK: 8 12 0
# CHECK: 8 13 0
# CHECK: 8 14 0
# CHECK: 8 15 0
# CHECK: 8 16 0
# CHECK: 9 1 0
# CHECK: 9 2 0
# CHECK: 9 3 0
# CHECK: 9 4 0
# CHECK: 9 5 0
# CHECK: 9 6 0
# CHECK: 9 7 0
# CHECK: 9 8 0
# CHECK: 9 9 0
# CHECK: 9 10 0
# CHECK: 9 11 0
# CHECK: 9 12 0
# CHECK: 9 13 0
# CHECK: 9 14 0
# CHECK: 9 15 0
# CHECK: 9 16 0
# CHECK: 10 1 0.188245
# CHECK: 10 2 0.10137
# CHECK: 10 3 0.101387
# CHECK: 10 4 0.221723
# CHECK: 10 5 0
# CHECK: 10 6 0
# CHECK: 10 7 0
# CHECK: 10 8 0
# CHECK: 10 9 0
# CHECK: 10 10 0.0918926
# CHECK: 10 11 0
# CHECK: 10 12 0.546623
# CHECK: 10 13 0
# CHECK: 10 14 0.905862
# CHECK: 10 15 0.0899846
# CHECK: 10 16 0.527342
# CHECK: 11 1 0
# CHECK: 11 2 0
# CHECK: 11 3 0
# CHECK: 11 4 0
# CHECK: 11 5 0
# CHECK: 11 6 0
# CHECK: 11 7 0
# CHECK: 11 8 0
# CHECK: 11 9 0
# CHECK: 11 10 0
# CHECK: 11 11 0
# CHECK: 11 12 0
# CHECK: 11 13 0
# CHECK: 11 14 0
# CHECK: 11 15 0
# CHECK: 11 16 0
# CHECK: 12 1 0
# CHECK: 12 2 0
# CHECK: 12 3 0.214067
# CHECK: 12 4 0
# CHECK: 12 5 0
# CHECK: 12 6 0.481272
# CHECK: 12 7 0
# CHECK: 12 8 0.570453
# CHECK: 12 9 0
# CHECK: 12 10 0.255221
# CHECK: 12 11 0
# CHECK: 12 12 0
# CHECK: 12 13 0
# CHECK: 12 14 0
# CHECK: 12 15 0
# CHECK: 12 16 0
# CHECK: 13 1 0.525107
# CHECK: 13 2 0.072399
# CHECK: 13 3 0.248027
# CHECK: 13 4 0.158356
# CHECK: 13 5 0
# CHECK: 13 6 0.429368
# CHECK: 13 7 0
# CHECK: 13 8 0.500558
# CHECK: 13 9 0
# CHECK: 13 10 0.187184
# CHECK: 13 11 0.0323494
# CHECK: 13 12 1.04229
# CHECK: 13 13 0
# CHECK: 13 14 0.52797
# CHECK: 13 15 0
# CHECK: 13 16 0.778438
# CHECK: 14 1 0
# CHECK: 14 2 0
# CHECK: 14 3 0.00084597
# CHECK: 14 4 0
# CHECK: 14 5 0
# CHECK: 14 6 0
# CHECK: 14 7 0
# CHECK: 14 8 0
# CHECK: 14 9 0
# CHECK: 14 10 0
# CHECK: 14 11 0
# CHECK: 14 12 0
# CHECK: 14 13 0
# CHECK: 14 14 0
# CHECK: 14 15 0.0447912
# CHECK: 14 16 0
# CHECK: 15 1 0
# CHECK: 15 2 0.060997
# CHECK: 15 3 0.0610072
# CHECK: 15 4 0.133416
# CHECK: 15 5 0
# CHECK: 15 6 0
# CHECK: 15 7 0
# CHECK: 15 8 0
# CHECK: 15 9 0
# CHECK: 15 10 0
# CHECK: 15 11 0
# CHECK: 15 12 0.328918
# CHECK: 15 13 0
# CHECK: 15 14 0.444821
# CHECK: 15 15 0.0821331
# CHECK: 15 16 0.317316
# CHECK: 16 1 0
# CHECK: 16 2 0
# CHECK: 16 3 0
# CHECK: 16 4 0
# CHECK: 16 5 0
# CHECK: 16 6 0
# CHECK: 16 7 0
# CHECK: 16 8 0
# CHECK: 16 9 0
# CHECK: 16 10 0
# CHECK: 16 11 0
# CHECK: 16 12 0
# CHECK: 16 13 0
# CHECK: 16 14 0
# CHECK: 16 15 0
# CHECK: 16 16 0