# Run w/ soda-pytaco

from typing import Tuple, List

from tools import mlir_pytaco_api as pt
import mlir_soda.pytaco.mlir_pytaco_ext as ext

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