# Run w/ soda-pytaco
#
# **WARNING** : --parallelization-strategy='none' is required on this app.
#
# Nested parallel reductions do not seem to be supported by the 
# sparse-tensor compiler at this moment, and it is unclear whether
# using scf.parallel + scf.reduce is the right strategy for multi-
# level reductions rather than something like scf.forall.
#

from typing import Tuple, List

from tools import mlir_pytaco_api as pt
import mlir_soda.pytaco.mlir_pytaco_ext as ext

def InnerProd(dim0: int, dim1: int, dim2) -> List[Tuple['Tensor', str]]:
    # Set up matrices
    A = pt.tensor([dim0, dim1, dim2], ext.dcc)
    B = pt.tensor([dim0, dim1, dim2], ext.dcc)
    z = pt.tensor(0) # A scalar

    # Declare index variables
    i, j, k = pt.get_index_vars(3)

    # Define computation and compile
    z[0] = A[i, j, k] * B[i, j, k] # Reduce to a scalar must be explicit via '[0]'

    # Return the tensor assignments to compile
    return [(z, 'z')]