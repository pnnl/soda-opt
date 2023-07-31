import sys
import os

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)

from typing import List, Tuple

OUTPUT_DIR = './pytaco-output'
KERNELS_DIR = './pytaco-kernels'
TENSORS_DIR = './pytaco-tensors'

DEBUG="[DEBUG]:"
ERROR="[ERROR]:"
COMMAND="[CMD]:"

OPT = 'soda-opt'
OPT_FLAGS = '-soda-sparse-compiler="parallelization-strategy='
DEFAULT_PAR = 'any-storage-any-loop'
DEFAULT_RUNTIME = 'enable-runtime-library=false'
DEFAULT_OMP = 'enable-openmp'
DEFAULT_DUMP = '-mlir-print-ir-after-all'
DEFAULT_EXTRA = ''

TRANSLATE = 'mlir-translate'
TRANSLATE_FLAGS = '-mlir-to-llvmir'

RUNNER = 'mlir-runner'
RUNNER_FLAGS = f"-e=main -entry-point-result=void -shared-libs={os.environ['RUNNERLIB']},{os.environ['SUPPORTLIB']}"

import argparse
import importlib
import subprocess
import sparse

# from mlir_soda.taco import mlir_pytaco_ext
# from tools import mlir_pytaco, mlir_pytaco_api

def generate_tensor_inputs(args, reuse: bool, input_shapes: List[List[int]], density: List[float]) -> List[str]:
    # Generate random tensor inputs based on @input_shapes and @density.
    # If @reuse is True, this function first checks if there already existss
    # a tensor input file for the given shape in @TENSORS_DIR.

    # Sanity check
    if not density: density = [0.1] * len(input_shapes)
    else: assert len(input_shapes) == len(density), f"{ERROR} Need to specify a density per kernel input tensor!"

    # Generate random tensor in COO format
    input_tensor_files = []
    for i, shape in enumerate(input_shapes):
        if args.v: print(DEBUG, f'Generating random tensor input for arg{i} : {"x".join(map(str, shape))} ...')
        
        # Check if tensor input file already exists, if necessary
        if reuse:
            tensor_file = f'{TENSORS_DIR}/{"x".join(map(str, shape))}.{i}.tns'
            if os.path.isfile(tensor_file):
                if args.v: print(DEBUG, f'Found {tensor_file}!')
                continue

        # Generate random tensor
        tensor = sparse.random(shape, density=density[i])

        # Output to file as FROSTT format
        tensor_file = f'{TENSORS_DIR}/{"x".join(map(str, shape))}.{i}.tns'
        input_tensor_files.append(tensor_file)
        if args.v: print(DEBUG, f'Outputting {tensor_file} ...')

        with open(tensor_file, 'w') as file:
            file.write(f'{tensor.ndim} {len(tensor.data)}\n') # First line = "rank nnz"
            for r in tensor.shape: file.write(f'{r} ') # Next line = dim sizes
            file.write("\n")
            for i in range(len(tensor.data)):
                for j in range(len(tensor.shape)): 
                    file.write(f'{tensor.coords[j][i] + 1} ')
                file.write('{0:.5f}'.format(tensor.data[i]) + "\n")

        if args.v: print(DEBUG, "DONE.")
    
    return input_tensor_files


def pytaco_to_mlir(args) -> Tuple[List[str], List[List[str]]]:
    # Returns a list of function prefixes generated per tensor.
    # E.g. ['SpMV.z.0']

    # Fetch the python function based on the kernel
    try:
        kernel_module = importlib.import_module(args.kernel)
        kernel = getattr(kernel_module, args.kernel)
        assert kernel , f"{ERROR} Can't find {args.kernel}!"
    except:
        assert False, f"{ERROR} Can't find {args.kernel}!!"

    if args.v: print(DEBUG, f'Found kernel in [{KERNELS_DIR}/{args.kernel}.py].')

    # Generate output and tensors directory, if necessary
    print(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_DIR):
        if args.v: print(DEBUG, f'mkdir {OUTPUT_DIR}') 
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(TENSORS_DIR):
        if args.v: print(DEBUG, f'mkdir {TENSORS_DIR}') 
        os.makedirs(TENSORS_DIR)

    # Process the kernel and fetch the tensor assignments to compile
    if args.v: print(DEBUG, 'Generating tensor assignments ...')
    tensor_assignments = kernel(*args.inps)
    if args.v: print(DEBUG, 'DONE.')

    # For each tensor assignment, generate a MLIR function and 
    # write it to a .mlir file.
    #
    # Function names will be formatted as: '@KERNEL.TENSOR.ID.main'
    # For example: 'SpMV.z.0.main'
    #
    # Output MLIR files will be similarly named: 'OUTPUT_DIR/KERNEL.TENSOR.ID.mlir'
    # For example: ./output/SpMV.z.0.mlir
    assignment_num = 0
    prefixes = []
    input_tensor_files = []
    for ta, tensor_name in tensor_assignments:
        # Set up prefix
        function_prefix = f'{args.kernel}.{tensor_name}.{assignment_num}'
        prefixes.append(function_prefix)
        if args.v: print(DEBUG, f'Generating MLIR for [{tensor_name}] to @{function_prefix}.main ...')

        # MLIR codegen
        mlir_function, input_shapes = ta.separately_compile(
            prefix=function_prefix, 
            emit_entry_point=args.emit_entry_point,
            add_timing=args.add_timing, 
            print_frostt=args.print_frostt
        )

        # If necessary, generate random inputs for the kernel
        if args.tensor_inputs == 'generate':
            if args.v: print(DEBUG, f'Generating random inputs for {tensor_name} ...')
            itf = generate_tensor_inputs(args, args.tensor_inputs == 'reuse', input_shapes, density=args.density)
            input_tensor_files.append(itf)

        assignment_num += 1

        # Write MLIR to a file
        with open(f'{OUTPUT_DIR}/{function_prefix}.mlir', 'w') as file:
            if args.v: print(DEBUG, f'Outputting @{function_prefix}.main to {OUTPUT_DIR}/{function_prefix}.mlir ...')
            file.write(f'{mlir_function}\n')

        # Reformat MLIR code
        if args.v: print(DEBUG, f'Reformatting {OUTPUT_DIR}/{function_prefix}.mlir ...')
        cmd = f'{OPT} {OUTPUT_DIR}/{function_prefix}.mlir -o {OUTPUT_DIR}/{function_prefix}.reformatted.mlir ; mv {OUTPUT_DIR}/{function_prefix}.reformatted.mlir {OUTPUT_DIR}/{function_prefix}.mlir'
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        
        if args.v: print(DEBUG, "DONE.")

    return prefixes, input_tensor_files


def lower_mlir(args, prefixes: List[str]) -> None:
    # Walk through @prefixes and invoke mlir-opt for each
    # .mlir file corresponding to each prefix
    for prefix in prefixes:
        # Sanity check
        mlir_file = f'{OUTPUT_DIR}/{prefix}.mlir'
        assert os.path.isfile(mlir_file) , f"{ERROR} Can't find {mlir_file}!!"

        # Set up file names
        lowered_mlir = f'{OUTPUT_DIR}/{prefix}.lowered.mlir'
        lowered_ll = f'{OUTPUT_DIR}/{prefix}.lowered.ll'

        # Fetch configurations from @args
        par = DEFAULT_PAR if args.sc_par_strategy is None else args.sc_par_strategy
        runtime = DEFAULT_RUNTIME if not args.sc_target_runtime else ''
        omp = DEFAULT_OMP if not args.sc_disable_omp else ''
        dump = DEFAULT_DUMP if not args.disable_ir_printing else ''
        extra = DEFAULT_EXTRA if args.sc_extra is None else args.sc_extra

        # Build mlir-opt command
        dump_fname = f'{OUTPUT_DIR}/{prefix}.lowering.out'
        cmd = f'{OPT} {OPT_FLAGS}{par} {runtime} {omp} {extra}" {dump} {mlir_file} -o {lowered_mlir}'
        print(COMMAND, f"Lowering {mlir_file}:\n{cmd}")

        # Run the command
        dump_file = open(dump_fname, 'w')
        process = subprocess.Popen(cmd, shell=True, stdout=dump_file, stderr=dump_file)
        process.wait()
        dump_file.close()
        if args.v: print(DEBUG, "DONE.")

        # Build mlir-translate command
        cmd = f'{TRANSLATE} {TRANSLATE_FLAGS} {lowered_mlir} -o {lowered_ll}'
        print(COMMAND, f"Lowering {lowered_ll} to LLVM:\n{cmd}")
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        if args.v: print(DEBUG, "DONE.")

    return


def run_mlir(args, prefixes: List[str], all_input_tensor_files: List[List[str]]) -> None:
    # Walk through @prefixes and invoke mlir-cpu-runner
    # for each .lowered.mlir file corresponding to each prefix
    for i, prefix in enumerate(prefixes):
        # Build mlir-cpu-runner command
        lowered_mlir = f'{OUTPUT_DIR}/{prefix}.lowered.mlir'
        dump_fname = f'{OUTPUT_DIR}/{prefix}.run.out'

        # Enable OpenMP, if necessary
        rflags = RUNNER_FLAGS
        if not args.sc_disable_omp: rflags += f",{os.environ['OMPLIB']}"
        cmd = f'{RUNNER} {rflags} {lowered_mlir} &> {dump_fname}'

        # Add inputs to command
        cmd = f'{" ".join([f"TENSOR{i}={tf}" for i, tf in enumerate(all_input_tensor_files[i])])} {cmd}'
        print(COMMAND, f"Running mlir-cpu-runner on {lowered_mlir}:\n{cmd}")

        # Run the command
        dump_file = open(dump_fname, 'w')
        process = subprocess.Popen(cmd, shell=True, stdout=dump_file, stderr=dump_file)
        process.wait()
        dump_file.close()
        if args.v: print(DEBUG, "DONE.")

    return


def main(args):
    print(f"================================\n{args.kernel}\n================================")

    # Generate high-level MLIR from PyTaco kernel
    tensor_prefixes, all_input_tensor_files = pytaco_to_mlir(args)

    # # Lower MLIR to the LLVM dialect
    lower_mlir(args, tensor_prefixes)

    # Run the generated code, if necessary
    if args.run: run_mlir(args, tensor_prefixes, all_input_tensor_files)

    print("================================\n")
    return

if __name__ == "__main__":
    # Invoke the script as:
    # python3 mlir_codegen.py --kernel NAME --inps INT ...
    #
    # Example: python3 mlir_codegen.py --kernel SpMV --inps 16 32
    # This invokes the SpMV PyTaco kernel in SpMV.py and passes
    # inputs 16 and 32 (i.e. matrix dimensions) to the kernel.
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--kernel',
        type=str,
        help='[Required] Name of the kernel ; should be in a file of the same name',
        required=True)
    argparser.add_argument(
        '--inps',
        type=int,
        nargs='*',
        help='[Required] Input arguments to the kernel (python function)',
        required=True)
    
    # Path flags
    argparser.add_argument(
        '--output-dir',
        type=str,
        help='Path to output directory')
    argparser.add_argument(
        '--kernels-dir',
        type=str,
        help='Path to kernels directory')
    argparser.add_argument(
        '--tensors-dir',
        type=str,
        help='Path to input tensors directory (in COO format/FROSTT Format)')
    
    # Compilation flags
    argparser.add_argument(
        '--sc-par-strategy',
        type=str,
        help='Option "parallelization-strategy" for "sparse-compiler" ; see mlir-opt --help for more')
    argparser.add_argument(
        '--sc-target-runtime',
        action='store_true',
        help='Lower to sparse-tensor runtime calls')
    argparser.add_argument(
        '--sc-disable-omp',
        action='store_true',
        help='Disable OpenMP parallelization and lowering')
    argparser.add_argument(
        '--sc-extra',
        type=str,
        help='Extra options to pass to "sparse-compiler" ; see mlir-opt --help')
    argparser.add_argument(
        '--emit-entry-point',
        action='store_true',
        help='Emit entry point function (@main) with kernel MLIR (to invoke w/ mlir-cpu-runner)')
    argparser.add_argument(
        '--print-frostt',
        action='store_true',
        help='Add input/output tensor printing in FROSTT format via sparse_tensor.out')
    argparser.add_argument(
        '--add-timing',
        action='store_true',
        help='Add timing instrumentation to the kernel via rtclock() calls')
    argparser.add_argument(
        '--disable-ir-printing',
        action='store_true',
        help='Disable printing of IR after each pass')

    # Tensor inputs and runtime flags
    argparser.add_argument(
        '--run',
        action='store_true',
        help='Run the generated code using mlir-cpu-runner | co-dependent on --tensor-inputs')
    argparser.add_argument(
        '--tensor-inputs',
        type=str,
        choices=['reuse', 'generate'],
        help='Generate random inputs for target kernel execution | co-dependent on --run')
    argparser.add_argument(
        '--density',
        type=float,
        nargs='*',
        help='Density within [0, 1] for input tensors | dependent on --tensor-inputs')

    # Other flags
    argparser.add_argument(
        '--v',
        action='store_true',
        help='Verbose')
    
    args = argparser.parse_args()

    # Handle dependences
    if args.sc_target_runtime:
        if args.sc_disable_omp: argparser.error("--sc-target-runtime requires --sc-disable-omp to be disabled.")

    if not args.emit_entry_point:
        if args.print_frostt: argparser.error("--print-frostt requires --emit-entry-point.")
        if args.add_timing: argparser.error("--add-timing requires --emit-entry-point.")

    if args.run and not args.tensor_inputs:
        argparser.error("--run requires --tensor-inputs.")

    if args.density:
        if not args.tensor_inputs: argparser.error("--density requires --tensor-inputs.")

    # Set paths
    if args.output_dir is not None: OUTPUT_DIR = args.output_dir
    if args.kernels_dir is not None: KERNELS_DIR = args.kernels_dir
    if args.tensors_dir is not None: TENSORS_DIR = args.tensors_dir
    sys.path.append(KERNELS_DIR)

    main(args)