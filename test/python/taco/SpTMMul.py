# RUN: python3 codegen.py --kernel SpTMMul --inps 16 32 64 64
# Can substitute dimensions

from typing import Tuple, List

from tools import mlir_pytaco_api as pt
import mlir_soda.taco.mlir_pytaco_ext as ext

def SpTMMul(A0: int, A1: int, A2: int, B0: int) -> List[Tuple['Tensor', str]]:
    # Set up tensor and matrices
    A = pt.tensor([A0, A1, A2], ext.csf)
    B = pt.tensor([B0, A.shape[2]], ext.dm)
    Z = pt.tensor([A.shape[0], A.shape[1], B.shape[0]], ext.dt)

    # Declare index variables
    i, j, k, l = pt.get_index_vars(4)
    
    # Define computation and compile
    Z[i, j, k] = A[i, j, l] * B[k, l]

    # Return the tensor assignments to compile
    return [(Z, 'Z')]