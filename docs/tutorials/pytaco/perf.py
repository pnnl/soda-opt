import sys
import os
import matplotlib.pyplot as plt

from typing import Tuple, List, Callable

import mlir_soda.pytaco.mlir_pytaco_ext as ext
from tools import mlir_pytaco_api as pt
from mlir_soda.pytaco.soda_sparse_compiler import SODASparseCompiler, MLIRGenOptions, ParOption, LLVMIRGenOptions, RunOptions, Capture

# =============================================================================
# Kernel definitions
# =============================================================================

def SpMSpM(ARows: int, ACols: int, BCols: int) -> List[Tuple['Tensor', str]]:
    # A and B are sparse
    A = pt.tensor([ARows, ACols], ext.csc)
    B = pt.tensor([A.shape[1], BCols], ext.csr)
    Z = pt.tensor([A.shape[0], B.shape[1]], ext.dm)

    # Declare index variables
    i, j, k = pt.get_index_vars(3)

    # Define computation and compile
    Z[i, j] = A[i, k] * B[k, j]

    # Return the tensor assignments to compile
    return [(Z, 'Z')]


def DMM(ARows: int, ACols: int, BCols: int) -> List[Tuple['Tensor', str]]:
    # A and B are dense
    A = pt.tensor([ARows, ACols], ext.dm)
    B = pt.tensor([A.shape[1], BCols], ext.dm)
    Z = pt.tensor([A.shape[0], B.shape[1]], ext.dm)

    # Declare index variables
    i, j, k = pt.get_index_vars(3)

    # Define computation and compile
    Z[i, j] = A[i, k] * B[k, j]

    # Return the tensor assignments to compile
    return [(Z, 'Z')]


# =============================================================================
# Compiler wrapper
# =============================================================================

def get_time(ts: str) -> float:
    ts = " ".join(ts.split()).split(" ")
    return float(ts[2]) # "=== ...", "rtclock_interval", "TIME" <- 3rd element

def run(
    kernel: Callable,
    kernel_inputs: Tuple[int],
    tensor_inputs: Tuple[str],
    num_threads: List[int]
) -> List[float]:
    """
    Small wrapper on SODASparseCompiler that sweeps @num_threads and returns
    a list of running times for each thread count.

    Args:
        kernel: The PyTACO defined kernel to run
        kernel_inputs: The inputs to the kernel function in Python
        tensor_inputs: The tensor inputs to the kernel
        num_threads: List of number of threads to use

    Returns:
        A list of running times for each thread count
    """

    # Set up options ; 
    mlir_options = MLIRGenOptions(timing=True)
    llvm_options = LLVMIRGenOptions(print_after_each=False)
    run_options = RunOptions(tensor_inputs)

    # Inspect target kernel
    kernel_name: str = kernel.__name__

    # Build a compiler
    compiler = SODASparseCompiler(kernel_name, kernel_func = kernel)

    # Sweep @num_threads
    times: List[float] = []
    for num_thread in num_threads:
        run_options.num_threads = num_thread

        # Run the compiler
        cap: Capture = compiler.compile_and_run(mlir_options, llvm_options, run_options, kernel_inputs)
        times.append(get_time(cap.timing.getvalue()))

    return times


if __name__ == "__main__":

    # NOTE: Modify this script to take arguments if you'd like to run it from the command line.

    # =============================================================================
    # Setup
    # =============================================================================
    num_threads: List[int] = [1, 2, 4, 8, 16, 32]
    kernel_inputs: Tuple[int] = (2048, 2048, 2048)
    tensor_inputs: List[str] = [f'{os.environ["PTT"]}/tensors/M2048_0.tns', f'{os.environ["PTT"]}/tensors/M2048_1.tns']
    
    print("===========================================================")    
    print("Running performance comparison between SpMSpM and DMM")
    print("-----------------------------------------------------------")
    print("Configurations:")
    print("  kernel_inputs: ", kernel_inputs)
    print("  tensor_inputs: ", tensor_inputs)
    print("  num_threads: ", num_threads, "\n")

    # =============================================================================
    # Run SpMSpM
    # =============================================================================

    print("-----------------------------------------------------------")
    print("Running SpMSpM ...")
    sparse_times: List[float] = run(SpMSpM, kernel_inputs, tensor_inputs, num_threads)
    print("DONE. Times: ", sparse_times)
    print("-----------------------------------------------------------\n")

    # =============================================================================
    # Run DMM
    # =============================================================================

    print("-----------------------------------------------------------")
    print("Running DMM ...")
    dense_times: List[float] = run(DMM, kernel_inputs, tensor_inputs, num_threads)
    print("DONE. Times: ", dense_times)
    print("-----------------------------------------------------------\n")

    # =============================================================================
    # Plot results as different colored lines in the same figure and show dots for each data point and ticks
    # =============================================================================
    print("Plotting results...")
    plt.plot(num_threads, sparse_times, 'r', label='SpMSpM')
    plt.plot(num_threads, dense_times, 'b', label='DMM')
    plt.plot(num_threads, sparse_times, 'ro')
    plt.plot(num_threads, dense_times, 'bo')
    plt.xticks(num_threads)
    plt.xlabel('Number of Threads')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.savefig('perf.png', dpi=300)
    print("DONE. Saved to perf.png")


