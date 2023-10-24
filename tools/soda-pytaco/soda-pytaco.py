#!/usr/bin/env python3

import sys
import os

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)

from typing import Optional, List, Tuple, Any

import sparse
import ast
import argparse
from enum import Enum

from mlir_soda.pytaco.soda_sparse_compiler import SODASparseCompiler, MLIRGenOptions, ParOption, LLVMIRGenOptions, RunOptions, Capture

# ------------------------------------------------------------------

DOUBLE = "============================================================================"
SINGLE = "----------------------------------------------------------------------------"
DONE = "DONE."
DEBUG="[DEBUG]:"
ERROR="[ERROR]:"
COMMAND="[CMD]:"

def directory_str(pstr) -> str:
    if os.path.isdir(pstr): return pstr
    else: raise NotADirectoryError(pstr)

def file_str(pstr) -> str:
    if os.path.isfile(pstr): return pstr
    else: raise FileNotFoundError(pstr)

def tensor_specs(pstr) -> List[Tuple[Tuple[int], float]]:
    specs = []
    pstr = pstr.strip().lower()
    for spec in pstr.split(','):
        dims, density = spec.split(':')
        assert density.isdigit() or density.replace(".", "", 1).isdigit(), f"Invalid density specification: {density} in {spec}"
        dim_split = [d.strip() for d in dims.split('x')]
        assert all([d.isdigit() for d in dim_split]), f"Invalid dimension specification: {dims} in {spec}"
        specs.append((tuple([int(d) for d in dim_split]), float(density)))
    return specs

def tobool(pstr) -> bool:
    val = pstr.strip().lower().capitalize()
    val = ast.literal_eval(pstr)
    assert type(val) == bool or type(val) == int, f"Invalid boolean specification: {pstr}"
    return bool(val)

# ------------------------------------------------------------------

def create_file_path(
    output_dir: str,
    parts: List[Any],
    extension: str # Include . in extension
) -> str:
    """Create a file path from the specified parts."""
    parts = [str(p) for p in parts]
    return f'{output_dir}/{"_".join(parts)}{extension}'


def generate_tensors(
    output_dir: str,
    tensor_specs: List[Tuple[Tuple[int], float]]
) -> List[str]:
    """Generate random tensor inputs in extended FROSTT format based on @tensor_specs."""

    # Generate random tensor in COO format
    input_tensor_files = []
    for shape, density in tensor_specs:
        if args.v: print(DEBUG, f'Generating random tensor input : {"x".join(map(str, shape))} at {density} density ...')

        # Generate random tensor
        tensor = sparse.random(shape, density=density)

        # Output to file as FROSTT format ; don't overwrite files
        tensor_file = f'{output_dir}/T{"x".join(map(str, shape))}_{density}_0.tns'
        duplicate = 1
        while os.path.isfile(tensor_file):
            tensor_file = f'{output_dir}/T{"x".join(map(str, shape))}_{density}_{duplicate}.tns'
            duplicate += 1

        input_tensor_files.append(tensor_file)
        if args.v: print(DEBUG, f'Outputting {tensor_file} .')

        with open(tensor_file, 'w') as file:
            file.write(f'{tensor.ndim} {len(tensor.data)}\n') # First line = "rank nnz"
            for r in tensor.shape: file.write(f'{r} ') # Next line = dim sizes
            file.write("\n")
            for i in range(len(tensor.data)):
                for j in range(len(tensor.shape)): 
                    file.write(f'{tensor.coords[j][i] + 1} ')
                file.write('{0:.5f}'.format(tensor.data[i]) + "\n")

    return input_tensor_files


# ------------------------------------------------------------------

def main(args):
    # Handle tensor generation if specified
    if args.new_tensor_specs is not None:
        if args.v: print(DOUBLE, f"Generating tensors: {args.new_tensor_specs} ...", SINGLE, sep="\n")
        generate_tensors(args.output_dir, args.new_tensor_specs)
        if args.v:
            print(DEBUG, DONE)
            print(DOUBLE, "\n")

    if 'mlir' not in args.actions: return

    # Handle kernel compilation if specified
    compiler = SODASparseCompiler(args.kernel, args.kernel_dir)
    cap = Capture()
    stem = []

    if args.v: print(DOUBLE, compiler.kernel_name, DOUBLE, sep="\n")   

    # ------------------------------------------------------------------

    if 'mlir' in args.actions:
        mlir_options = MLIRGenOptions(
            entry=args.emit_entry_point,
            print_input=args.print_input_tensors,
            print_output=args.print_output_tensors,
            timing=args.add_timing
        )
        
        # Generate the MLIR
        if args.v: 
            print(DOUBLE, f"Generating MLIR for {compiler.kernel_name} ...", SINGLE, sep="\n")
            print(DEBUG, f"Options:")
            print(DEBUG, f"  emit-entry-point: {mlir_options.entry}")
            print(DEBUG, f"  print-input-tensors: {mlir_options.print_input}")
            print(DEBUG, f"  print-output-tensors: {mlir_options.print_output}")
            print(DEBUG, f"  add-timing: {mlir_options.timing}")
        
        compiler.compile_pytaco_to_mlir(mlir_options, cap, args.inps)

        # Write the MLIR to file
        stem = [compiler.kernel_name, compiler.tensor_name, *args.inps]
        mlir_file_name = create_file_path(args.output_dir, stem, '.mlir')
        with open(mlir_file_name, 'w') as f: f.write(cap.mlir)

        if args.print_pytaco_mlir: print(DEBUG, "MLIR generated:\n", cap.mlir)
        if args.v: 
            print(DEBUG, f"MLIR written to {mlir_file_name}")
            print(DEBUG, DONE)
            print(DOUBLE, "\n")

    # ------------------------------------------------------------------

    if 'llvm' in args.actions:
        llvm_options = LLVMIRGenOptions(
            par=args.parallelization_strategy, runtime_library=args.enable_runtime, 
            omp=args.enable_omp, print_after_each=args.print_after_all, extra=args.sc_extra
        )

        # Generate the lowered MLIR, lowering logs, and LLVM-IR
        lowered_mlir_file_name = create_file_path(args.output_dir, stem, '.lowered.mlir')
        log_file_name = create_file_path(args.output_dir, stem, '.lowering.out')
        llvm_file_name = create_file_path(args.output_dir, stem, '.ll')

        if args.v: 
            print(DOUBLE, f"Lowering {compiler.kernel_name} [{mlir_file_name}] ...", SINGLE, sep="\n")
            print(COMMAND, f"{llvm_options.get_lowered_mlir_cmd()} {mlir_file_name} -o {lowered_mlir_file_name} 2>&1 {log_file_name}")
            print(COMMAND, f"{llvm_options.get_lowered_llvm_cmd()} {lowered_mlir_file_name} -o {llvm_file_name}")

        compiler.compile_mlir_to_llvm_with_shell(llvm_options, cap, cap.mlir)
        with open(lowered_mlir_file_name, 'w') as f: f.write(cap.lowered_mlir)
        with open(log_file_name, 'w') as f: f.write(cap.lowering_log.getvalue())
        with open(llvm_file_name, 'w') as f: f.write(cap.llvm)

        if args.v: 
            print(DEBUG, f"Lowered MLIR written to {lowered_mlir_file_name}")
            print(DEBUG, f"Lowering log written to {log_file_name}")
            print(DEBUG, f"LLVM-IR written to {llvm_file_name}")
            print(DEBUG, DONE)
            print(DOUBLE, "\n")

    # ------------------------------------------------------------------

    if 'run' in args.actions:
        run_options = RunOptions(
            num_threads=args.omp_num_threads,
            tensor_files=args.input_tensor_files
        )

        # Run the kernel, write any outputs to file
        tensor_output_file_name = create_file_path(args.output_dir, stem, '.run.out')
        timing_output_file_name = create_file_path(args.output_dir, stem, '.time')

        if args.v: 
            print(DOUBLE, f"Running {compiler.kernel_name} [{lowered_mlir_file_name}]...", SINGLE, sep="\n")
        
            # Rebuild the mlir-cpu-runner command
            env_vars = f"OMP_NUM_THREADS={run_options.num_threads} {' '.join([f'TENSOR{i}={f}' for i, f in enumerate(run_options.tensor_files)])}"
            cmd = f'{env_vars} {str(run_options)} {lowered_mlir_file_name}'
            if args.print_output_tensors or args.print_input_tensors: cmd += f' 1> {tensor_output_file_name}'
            if args.add_timing: cmd += f' 2> {timing_output_file_name}'
            print(COMMAND, cmd)

        compiler.run_compiled_kernel_with_shell(run_options, cap, cap.lowered_mlir)

        if args.print_output_tensors or args.print_input_tensors:
            with open(tensor_output_file_name, 'w') as f: f.write(cap.tensor_output.getvalue())

        if args.add_timing:
            with open(timing_output_file_name, 'w') as f: f.write(cap.timing.getvalue())
        
        if args.v: 
            if args.print_output_tensors or args.print_input_tensors:
                print(DEBUG, f"Tensors printed to {tensor_output_file_name}")
            if args.add_timing:
                print(DEBUG, f"Timing recorded in {timing_output_file_name}")
            print(DEBUG, DONE)
            print(DOUBLE, "\n")

    return
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    # Actions
    # ---
    # mlir: compile kernel to MLIR (.mlir)
    # llvm: compile kernel to LLVM (.lowered.mlir, .lowering.out, .ll)
    # run: run kernel via mlir-cpu-runner (.run.out)
    # gen-tensors: generate new tensors in FROSTT format (.tns)
    argparser.add_argument(
        '--actions',
        type=str,
        nargs='+',
        choices=['mlir', 'llvm', 'run', 'gen-tensors'],
        help='Supported functionality for soda-pytaco.py ; see documentation in script for more details',
        required=True)

    # Path flags
    argparser.add_argument(
        '--output-dir',
        type=directory_str,
        help='Path to output directory for all generated files (scripts, compilation results, new tensors).',
        required=True)

    # Kernel specification
    kernel_spec_group = argparser.add_argument_group('Kernel specification', 'Specify the kernel to compile and/or run')
    kernel_spec_group.add_argument(
        '--kernel',
        type=str,
        help='Name of the kernel ; should be in a file of the same name')
    kernel_spec_group.add_argument(
        '--kernel-dir',
        type=directory_str,
        help='Directory to kernel file ; @kernel must be in this directory in a file of the same name')
    kernel_spec_group.add_argument(
        '--inps',
        type=int,
        nargs='*',
        help='Input arguments to the kernel (python function)')
    
    # MLIR kernel compilation flags
    mlir_compilation_group = argparser.add_argument_group('MLIR kernel compilation flags')
    mlir_compilation_group.add_argument(
        '--emit-entry-point',
        type=tobool,
        default=True,
        help='Emit entry point function (@main) with kernel MLIR (to invoke w/ mlir-cpu-runner)')
    mlir_compilation_group.add_argument(
        '--print-input-tensors',
        type=tobool,
        default=False,
        help='Add input tensor printing in FROSTT format via sparse_tensor.out')
    mlir_compilation_group.add_argument(
        '--print-output-tensors',
        type=tobool,
        default=True,
        help='Add output tensor printing in FROSTT format via sparse_tensor.out')
    mlir_compilation_group.add_argument(
        '--add-timing',
        type=tobool,
        default=False,
        help='Add timing instrumentation to the kernel via rtclock() calls')
    mlir_compilation_group.add_argument(
        '--print-pytaco-mlir',
        type=tobool,
        default=False,
        help='Print the PyTACO-generated MLIR to stdout')

    # soda-opt compilation flags
    soda_opt_group = argparser.add_argument_group('soda-opt compilation flags')
    soda_opt_group.add_argument(
        '--parallelization-strategy',
        type=str,
        choices=['none', 'dense-outer-loop', 'any-storage-outer-loop', 'dense-any-loop', 'any-storage-any-loop'],
        default='any-storage-any-loop',
        help='Option for "soda-opt -soda-sparse-compiler"')
    soda_opt_group.add_argument(
        '--enable-runtime',
        type=tobool,
        default=False,
        help='Option for "soda-opt -soda-sparse-compiler"')
    soda_opt_group.add_argument(
        '--enable-omp',
        type=tobool,
        default=True,
        help='Option for "soda-opt -soda-sparse-compiler"')
    soda_opt_group.add_argument(
        '--print-after-all',
        type=tobool,
        default=True,
        help='Option "-mlir-print-ir-after-all" for "soda-opt -soda-sparse-compiler"')
    soda_opt_group.add_argument(
        '--sc-extra',
        type=str,
        default='',
        help='Extra arguments to pass to "soda-opt -soda-sparse-compiler"')

    # Tensor inputs and runtime flags
    runtime_group = argparser.add_argument_group('Tensor inputs and runtime flags')
    runtime_group.add_argument(
        '--omp-num-threads',
        type=int,
        default=4,
        help='Number of threads to use for OpenMP parallelization')
    runtime_group.add_argument(
        '--input-tensor-files',
        type=file_str,
        nargs='*',
        help='List of input tensor files in COO/MTX/FROSTT format ; represents the input tensors to the kernel')

    # Extra functionality for tensor generation
    tensor_gen_group = argparser.add_argument_group('Extra functionality for tensor generation')
    tensor_gen_group.add_argument(
        '--new-tensor-specs',
        type=tensor_specs,
        help='List of sparse tensor specifications to generate ; each tensor spec is of the form: "[dim]x[dim]...:[density]>, ..." (e.g. --new-tensor-specs="16x16x16:0.1, 32x32x32:0.2")')

    argparser.add_argument(
        '--v',
        action='store_true',
        default=False,
        help='Print verbose output')

    args = argparser.parse_args()

    # Handle dependencies
    if 'mlir' in args.actions:
        assert args.kernel is not None, "Must specify --kernel when invoking 'mlir'"
        assert args.kernel_dir is not None, "Must specify --kernel-dir when invoking 'mlir'"
        assert args.inps is not None, "Must specify --inps when invoking 'mlir' action"

    if 'llvm' in args.actions:
        assert 'mlir' in args.actions, "Must specify 'mlir' when invoking 'llvm'"

    if 'run' in args.actions:
        assert 'llvm' in args.actions, "Must specify 'llvm' when invoking 'run'"
        assert (args.input_tensor_files is not None or args.input_tensor_dir is not None), \
                "Must specify --input-tensor-files or --input-tensor-dir when invoking 'run'"

    if 'gen-tensors' in args.actions:
        assert args.new_tensor_specs is not None, "Must specify --new-tensor-specs when invoking 'gen-tensors'"

    main(args)