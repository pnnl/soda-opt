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

def Reduce2D(rows: int, cols: int) -> List[Tuple['Tensor', str]]:
    # Set up matrix
    M = pt.tensor([rows, cols], ext.csr)
    z = pt.tensor(0) # A scalar

    # Declare index variable
    i, j = pt.get_index_vars(2)

    # Define computation and compile
    z[0] = M[i, j] # Scalar reference via '[0]' explicitly

    # Return the tensor assignments to compile
    return [(z, 'z')]