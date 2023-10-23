# Run w/ soda-pytaco

from typing import Tuple, List

from tools import mlir_pytaco_api as pt
import mlir_soda.pytaco.mlir_pytaco_ext as ext

def MTTKRP(A1: int, B0: int, B1: int, B2: int) -> List[Tuple['Tensor', str]]:
    # Set up matrices (@A0, @B0, @B1, @B2 are all dimensions)
    B = pt.tensor([B0, B1, B2], ext.csf)
    C = pt.tensor([B.shape[1], A1], ext.dm)
    D = pt.tensor([B.shape[2], A1], ext.dm)
    A = pt.tensor([B.shape[0], A1], ext.dm)

    # Declare index variables
    i, j, k, l = pt.get_index_vars(4)

    # Define computation and compile
    A[i, j] = B[i, k, l] * D[l, j] * C[k, j]

    # Return the tensor assignments to compile
    return [(A, 'A')]