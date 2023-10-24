from typing import Tuple, List

import mlir_soda.pytaco.mlir_pytaco_ext as ext
from tools import mlir_pytaco_api as pt

def SpMVMul(rows: int, cols: int) -> List[Tuple['Tensor', str]]:
    # Set up matrix and vectors. Each tensor is 
    # instantiated with a storage format (e.g. 
    # pt.csr) ; see the API in ${PYTACO}.
    #
    # Sparse storage formats are exposed in 'ext'.
    M = pt.tensor([rows, cols], ext.csr)
    v = pt.tensor([M.shape[1]], ext.dv)
    z = pt.tensor([M.shape[0]], ext.dv)

    # Declare index variables
    i, j = pt.get_index_vars(2)
    
    # Define computation and compile
    z[i] = M[i, j] * v[j]

    # Return the tensor assignments to compile
    return [(z, 'z')]
