# Run w/ soda-pytaco

from typing import Tuple, List

from tools import mlir_pytaco_api as pt
import mlir_soda.pytaco.mlir_pytaco_ext as ext

def Reduce1D(elms: int) -> List[Tuple['Tensor', str]]:
    # Set up vector
    v = pt.tensor([elms], ext.cv)
    z = pt.tensor(0) # A scalar

    # Declare index variable
    i = pt.get_index_vars(1)[0] # Returns a list -> index into it.

    # Define computation and compile
    z[0] = v[i] # Scalar reference via '[0]' explicitly

    # Return the tensor assignments to compile
    return [(z, 'z')]