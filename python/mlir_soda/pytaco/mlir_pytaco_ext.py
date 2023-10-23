"""Extensions to MLIR-PyTACO with sparse tensor support.

See http://tensor-compiler.org/ for TACO tensor compiler.

This module implements extensions to MLIR-PyTACO that allows
it to separately compile PyTACO code without invoking the 
MLIR execution engine.

The methods in this file are defined externally to the IndexExpr
and Tensor classes to avoid LLVM mainline integration and upkeep.
They are stricly for internal use by the SODA toolchain (for now).
"""
import sys
import os

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)

from typing import List, Tuple, Optional

from mlir import ir
import tools.mlir_pytaco as mlir_pytaco
from tools.mlir_pytaco import Tensor, IndexVar, IndexExpr, Format
from tools.mlir_pytaco_api import dense, compressed


# =================================================================

def emit_mlir(
    self,
    dst: "Tensor",
    dst_indices: Tuple["IndexVar", ...],
    prefix: str = ""
) -> Tuple[str, List[ir.Type]]:
    """Emits the tensor assignment dst[dst_indices] = expression as MLIR.

    No compilation via the "sparse-compiler" pipeline is performed in this 
    method. The high-level MLIR (in the linalg dialect w/ sparse_tensor
    attributes) is emitted.

    Args:
        dst: The destination tensor.
        dst_indices: The tuple of IndexVar used to access the destination tensor.
        prefix: A string prepended to the name for the emitted MLIR function.

    Returns:
        MLIR function generated as a string ; the list of input tensor types (ShapedTypes).
    """
    expr_to_info = self._validate_and_collect_expr_info(dst, dst_indices)
    input_accesses = self.get_input_accesses()

    # Parse and generate high-level MLIR for the expression.
    mlir_pytaco._ENTRY_NAME = f"{prefix}.main"
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        self._emit_assignment(
            module, dst, dst_indices, expr_to_info, input_accesses,
        )

        input_types = [a.tensor.mlir_tensor_type() for a in input_accesses]
        mlir_code = str(module)

        return mlir_code, input_types

# Set method
IndexExpr.emit_mlir = emit_mlir

# =================================================================

def separately_compile(
    self, 
    force_recompile: bool = False, 
    prefix: str = ""
) -> Optional[Tuple[str, List[ir.Type]]]:
    """Emits the tensor assignment as high-level MLIR for separate compilation.

    The code generation takes place in the underlying assignment
    (IndexExpr) and does not invoke the "sparse-compiler" pipeline.

    Args:
        force_recompile: Flag to force recompilation of the MLIR.
        prefix: A string prepended to the name for the emitted MLIR function.

    Returns:
        MLIR function generated as a string ; list of input tensor types (ShapedTypes)
    """
    if self._assignment is None or (
        self._engine is not None and not force_recompile
    ):
        return None
    
    return self._assignment.expression.emit_mlir(
        self, self._assignment.indices, prefix
    )

# Set method
Tensor.separately_compile = separately_compile

# =================================================================

# Formats
csr = Format([dense, compressed]) # Compressed sparse row
csc = Format([dense, compressed], [1, 0]) # Compressed sparse column
csf = Format([compressed, compressed, compressed]) # Compressed sparse fiber
dcc = Format([dense, compressed, compressed]) # Dense-sparse-sparse (unclear name)
dv = Format([dense]) # Dense vector
cv = Format([compressed]) # Compressed vector
dm = Format([dense, dense]) # Dense 2D matrix
dt = Format([dense, dense, dense]) # Dense 3D tensor