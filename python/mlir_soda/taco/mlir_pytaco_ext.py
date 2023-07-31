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

from typing import List, Tuple, Union

from mlir import execution_engine
from mlir import ir
from mlir import runtime
import tools.mlir_pytaco as mlir_pytaco
from tools.mlir_pytaco import Tensor, IndexVar, IndexExpr, Format
from tools.mlir_pytaco_api import dense, compressed

# =================================================================

def _emit_entry_point_and_kernel(
    self, 
    module: ir.Module, 
    add_timing: bool = False,
    print_frostt: bool = False) -> Tuple[str, List[List[int]]]:
    """Inserts @main as an entry point to the kernel function with
    tensor inputs and optional print/timing instrumentation.

    TODO: Use MLIR-python bindings

    Args:
        module: The MLIR module to insert the kernel function.
        add_timing: Flag to control timing instrumentation to the kernel.
        print_frostt: Flag to insert sparse_tensor.out to dump tensors in FROSTT format.

    Returns:
        The transformed MLIR module as a string ; the shapes of the input tensors.
    """

    # Assumptions:
    # 1. MLIR generated using the modified PyTACO frontend
    #    and codegen.py have a specific format -> a single 
    #    function within the module -> i.e. the "kernel."
    # 2. Kernels have only tensor inputs and outputs
    # 3. Each tensor is marked with a sparse encoding, even
    #    for dense tensors.
    # 4. All tensor dimensions are known at compile time.

    # Fetch the kernel function from @module
    kernel_func = module.body.operations[0]

    # Set up stem for @main:
    main_func = "func.func private @getTensorFilename(index) -> (!llvm.ptr<i8>)\nfunc.func @main() {\n"

    # Set up the null pointer for sparse_tensor.out
    if print_frostt:
        main_func += "%none = llvm.mlir.null : !llvm.ptr<i8>\n"

    # Set up timing function if necessary
    if add_timing:
        main_func = "\
            func.func private @rtclock() -> (f64)\n\
            func.func private @rtclock_interval(f64, f64) -> ()\n" + main_func

    # Parse the kernel function arguments and return value
    arg_shapes = []
    arg_types = []
    for i, arg in enumerate(kernel_func.arguments):
        # Get the tensor type, shape, encoding, and internal type
        tensor_type = arg.type
        assert isinstance(tensor_type, ir.RankedTensorType)
        shape = tensor_type.shape
        arg_shapes.append(shape)
        arg_types.append(tensor_type)

        # Emit the tensor as a constant, convert it to the corresponding
        # sparse format, and dump it for debugging -> as MLIR operations
        input_tensor_ops = f"\
            %c{i} = arith.constant {i} : index\
            %fileName{i} = call @getTensorFilename(%c{i}) : (index) -> (!llvm.ptr<i8>)\
            %t{i} = sparse_tensor.new %fileName{i}: !llvm.ptr<i8> to {tensor_type}\n"

        if print_frostt: input_tensor_ops += f"sparse_tensor.out %t{i}, %none : {tensor_type}, !llvm.ptr<i8>\n"
        main_func += input_tensor_ops

    # Add the call and result sparse_tensor.out to the function. 
    # If @add_timing is set, we want to record the execution time
    # via calls to the runtime function rtclock and rtclock_interval.
    call_ops = []
    if add_timing: call_ops.append(f"%start = call @rtclock() : () -> (f64)") # Start of interval
    call_ops.append(f"%res = call @{str(kernel_func.name)[1:-1]}({', '.join([f'%t{i}' for i in range(len(arg_types))])}) : ({', '.join([str(at) for at in arg_types])}) -> {kernel_func.type.results[0]}\n") # Kernel call
    if add_timing: call_ops.append(f"%end = call @rtclock() : () -> (f64)") # End of interval
    if print_frostt: call_ops.append(f"sparse_tensor.out %res, %none : {kernel_func.type.results[0]}, !llvm.ptr<i8>\n") # Result output
    if add_timing: call_ops.append(f"call @rtclock_interval(%start, %end) : (f64, f64) -> ()") # Print interval
    main_func += ''.join(call_ops)
    
    # Add return
    main_func += "return\n}\n"

    # Add it to the module as string
    curr_module = str(module)
    mlir_code = main_func.join(curr_module.rsplit('}', 1)) + "\n}\n"
    return mlir_code, arg_shapes

# Set method
IndexExpr._emit_entry_point_and_kernel = _emit_entry_point_and_kernel

# =================================================================

def emit_mlir(
    self,
    dst: "Tensor",
    dst_indices: Tuple["IndexVar", ...],
    prefix: str = "",
    emit_entry_point: bool = False,
    add_timing: bool = False,
    print_frostt: bool = False
) -> Tuple[str, List[List[int]]]:
    """Emits the tensor assignment dst[dst_indices] = expression as MLIR.

    No compilation via the "sparse-compiler" pipeline is performed in this 
    method. The high-level MLIR (in the linalg dialect w/ sparse_tensor
    attributes) is emitted.

    Args:
        dst: The destination tensor.
        dst_indices: The tuple of IndexVar used to access the destination tensor.
        prefix: A string prepended to the name for the emitted MLIR function.
        emit_entry_point: Flag that controls entry point function emission (@main)
        add_timing: Flag to control timing instrumentation to the kernel.
        print_frostt: Flag to insert sparse_tensor.out to dump tensors in FROSTT format.

    Returns:
        MLIR function generated as a string ; the shapes of the input tensors.

    Raises:
        ValueError: If the expression is not proper or not supported.
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
        input_shapes = []
        if emit_entry_point: mlir_code, input_shapes = self._emit_entry_point_and_kernel(module, add_timing, print_frostt)
        else: mlir_code = str(module)
        return mlir_code, input_shapes

# Set method
IndexExpr.emit_mlir = emit_mlir

# =================================================================

def separately_compile(
    self, 
    force_recompile: bool = False, 
    prefix: str = "", 
    emit_entry_point: bool = False,
    add_timing: bool = False,
    print_frostt: bool = False
) -> Union[None, Tuple[str, List[List[int]]]]:
    """Emits the tensor assignment as high-level MLIR for separate compilation.

    The code generation takes place in the underlying assignment
    (IndexExpr) and does not invoke the "sparse-compiler" pipeline.

    Args:
        force_recompile: A boolean value to enable recompilation, such as for the
        purpose of timing.
        prefix: A string prepended to the name for the emitted MLIR function.
        emit_entry_point: Flag that controls entry point function emission (@main)
        add_timing: Flag to control timing instrumentation to the kernel.
        print_frostt: Flag to insert sparse_tensor.out to dump tensors in FROSTT format.

    Returns:
        MLIR function generated as a string ; list of input tensor shapes

    Raises:
        ValueError: If the assignment is not proper or not supported.
    """
    if self._assignment is None or (
        self._engine is not None and not force_recompile
    ):
        return None

    return self._assignment.expression.emit_mlir(
        self, self._assignment.indices, prefix, emit_entry_point, add_timing, print_frostt
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
