# RUN: python3 python_codegen.py --kernel SpMVMul --inps 16 32
# [16 x 32] * [32]
# Substitute '16' and '32' for any matrix dimensions

from typing import Tuple, List

from tools import mlir_pytaco_api as pt
import mlir_soda.taco.mlir_pytaco_ext as ext

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