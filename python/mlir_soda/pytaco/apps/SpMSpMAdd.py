# Run w/ soda-pytaco

from typing import Tuple, List

from tools import mlir_pytaco_api as pt
import mlir_soda.pytaco.mlir_pytaco_ext as ext

def SpMSpMAdd(rows: int, cols: int) -> List[Tuple['Tensor', str]]:
    # Set up matrices
    A = pt.tensor([rows, cols], ext.csr)
    B = pt.tensor([rows, cols], ext.csr)
    Z = pt.tensor([rows, cols], ext.dm)

    # Declare index variables
    i, j = pt.get_index_vars(2)

    # Define computation and compile
    Z[i, j] = A[i, j] + B[i, j]

    # Return the tensor assignments to compile
    return [(Z, 'Z')]