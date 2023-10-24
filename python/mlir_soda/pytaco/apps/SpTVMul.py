# Run w/ soda-pytaco

from typing import Tuple, List

from tools import mlir_pytaco_api as pt
import mlir_soda.pytaco.mlir_pytaco_ext as ext

def SpTVMul(M0: int, M1: int, M2: int) -> List[Tuple['Tensor', str]]:
    # Set up tensor and vectors
    M = pt.tensor([M0, M1, M2], ext.cdc)
    v = pt.tensor([M.shape[2]], ext.dv)
    Z = pt.tensor([M.shape[0], M.shape[1]], ext.dm)

    # Declare index variables
    i, j, k = pt.get_index_vars(3)
    
    # Define computation and compile
    Z[i, j] = M[i, j, k] * v[k]

    # Return the tensor assignments to compile
    return [(Z, 'Z')]