import sys
import os
import inspect

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)

from typing import Tuple, Callable

# TODO: Consider using unittest framework for python testing

from mlir_soda.pytaco.soda_sparse_compiler import SODASparseCompiler, MLIRGenOptions, ParOption, LLVMIRGenOptions, RunOptions, Capture
from tools import mlir_pytaco_api as pt

# ===========================================================

def run_and_check(
    kernel: Callable,
    kernel_inputs: Tuple[int],
    tensor_inputs: Tuple[str],
    enable_openmp: bool = True,
    omp_num_threads: int = 4,
    parallelization_strategy: ParOption = ParOption.Any_Storage_Any_Loop
) -> None:
    """
    Small wrapper for running and checking a PyTACO defined kernel w/ soda-sparse-compiler.inspect
    
    This is intended to be used w/ FileCheck in the MLIR testing system.

    Args:
        kernel: The PyTACO defined kernel to run
        kernel_inputs: The inputs to the kernel function in Python
        tensor_inputs: The tensor inputs to the kernel
        enable_openmp: Whether to enable OpenMP
        omp_num_threads: The number of OpenMP threads to use
        parallelization_strategy: The parallelization strategy to use
    """

    # Set up options
    mlir_options = MLIRGenOptions()
    llvm_options = LLVMIRGenOptions(print_after_each=False, omp=enable_openmp, par=parallelization_strategy)
    run_options = RunOptions(tensor_inputs, omp_num_threads)

    # Inspect target kernel
    kernel_name: str = kernel.__name__

    # Build a compiler
    compiler = SODASparseCompiler(kernel_name, kernel_func = kernel)

    # Run the compiler
    cap: Capture = compiler.compile_and_run(mlir_options, llvm_options, run_options, kernel_inputs)

    # Print the tensor output -> FileCheck will compare the output
    print(cap.tensor_output.getvalue())
    return


