import sys
import os

_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SCRIPT_PATH)

from typing import List, Tuple, Callable, Dict

import importlib
import subprocess
from dataclasses import dataclass, field
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from enum import Enum

from mlir import execution_engine
from mlir import ir
from mlir import passmanager
from tools import mlir_pytaco_api as pt

ERROR = "[ERROR]:"

# =================================================================

@dataclass
class MLIRGenOptions:
    """Options for MLIR generation via soda-sparse-compiler"""
    entry: bool = True
    print_output: bool = True
    print_input: bool = False
    timing: bool = False

    def __post_init__(self) -> None:
        if not self.entry:
            self.print_output = self.print_input = self.timing = False
        return

    def __hash__(self) -> int:
        return (self.entry, self.print_output, self.print_input, self.timing).__hash__()
    
    def __eq__(self, other: "MLIRGenOptions") -> bool:
        return (self.entry, self.print_output, self.print_input, self.timing) == (other.entry, other.print_output, other.print_input, other.timing)

# =================================================================

class ParOption(Enum):
    """Parallelization strategy for the soda-sparse-compiler"""
    No=0
    Dense_Outer_Loop=1
    Any_Storage_Outer_Loop=2
    Dense_Any_Loop=3
    Any_Storage_Any_Loop=4

    def __str__(self) -> str:
        return self.name.lower().replace('_', '-')

# =================================================================

@dataclass
class LLVMIRGenOptions:
    """Options for LLVM-IR generation via soda-sparse-compiler"""
    par: ParOption = ParOption.Any_Storage_Any_Loop
    runtime_library: bool = False
    omp: bool = True
    print_after_each: bool = True
    extra: str = ""

    def __post_init__(self) -> None:
        if self.par == ParOption.No:
            self.runtime_library = self.omp = False
        return

    def __hash__(self) -> int:
        return (self.par, self.runtime_library, self.omp, self.print_after_each, self.extra).__hash__()
    
    def __eq__(self, other: "LLVMIRGenOptions") -> bool:
        return (self.par, self.runtime_library, self.omp, self.print_after_each, self.extra) == (other.par, other.runtime_library, other.omp, other.print_after_each, other.extra)
    
    def __str__(self) -> str:
        return f"builtin.module(soda-sparse-compiler{{\
parallelization-strategy={self.par},\
enable-runtime-library={int(self.runtime_library)},\
enable-openmp={int(self.omp)}\
{f',self.extra' if self.extra else ''}}})"

    def get_lowered_mlir_cmd(self) -> str:
        # __str__ defaults to pass pipeline representation of the lowering step
        return f"soda-opt -soda-sparse-compiler=\
\"parallelization-strategy={self.par} \
enable-runtime-library={int(self.runtime_library)} \
enable-openmp={int(self.omp)} \
{self.extra}\" \
-mlir-print-ir-after-all={int(self.print_after_each)}"

    def get_lowered_llvm_cmd(self) -> str:
        return "mlir-translate -mlir-to-llvmir"
    
# =================================================================

@dataclass
class RunOptions:
    """Options for MLIR generation via soda-sparse-compiler"""
    tensor_files: List[str] # file paths
    num_threads: int = 4
    libs: List[str] = field(default_factory=lambda: 
        [os.environ['RUNNER_UTILS'], os.environ['C_RUNNER_UTILS'], os.environ['SODA_RUNNER_EXT']])
    
    def __post_init__(self) -> None:
        if 'OPENMP_LIB' in os.environ:
            self.libs.append(os.environ['OPENMP_LIB'])

    def __hash__(self) -> int:
        return (*self.tensor_files, self.num_threads, *self.libs).__hash__()
    
    def __eq__(self, other: "RunOptions") -> bool:
        return (*self.tensor_files, self.num_threads, *self.libs) == (*other.tensor_files, other.num_threads, *self.libs)

    def __str__(self) -> str:
        return f"mlir-cpu-runner -e=main -entry-point-result=void -shared-libs={','.join(self.libs)}"

# =================================================================

@dataclass
class Capture:
    """Capture all logs, intermediate IRs, and outputs"""
    function_prefix: str = ""
    input_shapes: List[List[int]] = field(default_factory=list)
    mlir: str = "" # via PyTACO
    lowered_mlir: str = "" # in LLVM dialect, via soda-opt
    lowering_log: StringIO = field(default_factory=StringIO)
    llvm: str = "" # via mlir-translate
    tensor_output: StringIO = field(default_factory=StringIO) # via entry point
    timing: StringIO = field(default_factory=StringIO) # via entry point

    def capture_lowering_logs(self):
        return redirect_stderr(self.lowering_log)
    
    def capture_result(self):
        return redirect_stdout(self.result)

# =================================================================

class SODASparseCompiler:
    """SODA Sparse Compiler: Compiles PyTACO kernels to LLVM-IR w/ OpenMP parallelization"""

    def __init__(
        self, 
        kernel_name: str,
        path_to_kernel: str = "",
        kernel_func: Callable = None
    ):
        self.kernel_name: str = kernel_name
        self.path_to_kernel: str = path_to_kernel
        self.kernel: Callable = kernel_func if kernel_func is not None else self._load_kernel()
        self.tensor_name: str = None


    def _load_kernel(self) -> Callable:
        """Fetch the kernel definition function"""

        # Use the path given, if necessary
        if self.path_to_kernel: sys.path.append(self.path_to_kernel)

        # Load
        try:
            kernel_module = importlib.import_module(self.kernel_name)
            kernel = getattr(kernel_module, self.kernel_name)
            assert kernel , f"{ERROR} Can't find {self.kernel_name}!"
        except:
            assert False, f"{ERROR} Can't find {self.kernel_name}!!"

        # debug.print(DEBUG, f'Found kernel in [{self.kernel_name}.py].')
        return kernel


    def _reformat_mlir(self, mlir_code: str) -> ir.Module:
        """
        Runs the execution engine w/ empty pass pipeline to reformat MLIR

        Args:
            mlir_code: The MLIR code to reformat.

        Returns:
            The ir.Module w/ reformatted code
        
        """
        with ir.Context(), ir.Location.unknown():
            module = ir.Module.parse(mlir_code)
            passmanager.PassManager.parse('builtin.module()').run(module)
            return module


    def _emit_entry_point_and_kernel(
        self, 
        mlir_str: str,
        input_types: List[ir.Type],
        options: MLIRGenOptions
    ) -> str:
        """
        Emits the entry point function and kernel function for the given MLIR code.

        Args:
            mlir_str: The MLIR code to emit the entry point and kernel for.
            input_types: The list of ShapedTypes of the input tensors.
            options: MLIRGenOptions object

        Returns:
            MLIR function generated as a string
        """

        # Assumptions:
        # 1. MLIR generated using the modified PyTACO frontend
        #    and codegen.py have a specific format -> a single 
        #    function within the module -> i.e. the "kernel."
        # 2. Kernels have only tensor inputs and outputs
        # 3. Each tensor is marked with a sparse encoding, even
        #    for dense tensors.
        # 4. All tensor dimensions are known at compile time.'
        # 5. @input_accesses must match kernel function args.

        with ir.Context(), ir.Location.unknown():
            module = ir.Module.parse(mlir_str)

            # Fetch the kernel function from @module
            kernel_func = module.body.operations[0]

            # Sanity check -- input access types should match kernel function args
            assert len(kernel_func.arguments) == len(input_types) , '_emit_entry_point_and_kernel: Inconsistent number of arguments and input accesses.'

            # We're emitting this code as string manipulations since
            # FuncOp.parse is unavailable. It's simpler to parse strings.
            module_code: List[str] = ["module {\n"]

            # All entry point additions will read tensors from files
            # via sparse_tensor.new; setup the symbols
            module_code.append("func.func private @getTensorFilename(index) -> (!llvm.ptr<i8>)\n")

            # If we need timing, set up the symbols
            if options.timing:
                module_code.append("func.func private @rtclock() -> (f64)\n")
                module_code.append("func.func private @rtclock_interval(f64, f64) -> ()\n")

            # If we need to use sparse_tensor.out, set up the symbols
            if options.print_input or options.print_output:
                module_code.append("""
llvm.mlir.global internal constant @none("\0A")\n\
func.func @getStdOut() -> (!llvm.ptr<i8>) {\n\
    %base = llvm.mlir.addressof @none : !llvm.ptr<array<2 x i8>>\n\
    %off = llvm.mlir.constant(0 : index) : i64\n\
    %stdout = llvm.getelementptr %base[%off, %off] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>\n\
    return %stdout : !llvm.ptr<i8>\n\
}\n""")

            # Add the kernel function
            module_code.append(str(kernel_func))

            # Set up stem for @main:
            module_code.append("func.func @main() {\n")

            # If we're printing inputs, we need a pointer to stdout
            if options.print_input:
                module_code.append("%stdout = call @getStdOut() : () -> (!llvm.ptr<i8>)\n")

            # Parse the kernel function arguments and return value; recreate
            # the input tensor types to use ir.RankedTensorType.
            arg_types = []
            for i, tensor_type in enumerate(input_types):
                # Get the internal type
                arg_types.append(tensor_type)

                # Emit the tensor as a constant, convert it to the corresponding
                # sparse format, and dump it for debugging -> as MLIR operations
                input_tensor_ops = f"\
                    %c{i} = arith.constant {i} : index\n\
                    %fileName{i} = call @getTensorFilename(%c{i}) : (index) -> (!llvm.ptr<i8>)\n\
                    %t{i} = sparse_tensor.new %fileName{i}: !llvm.ptr<i8> to {tensor_type}\n"

                if options.print_input: 
                    input_tensor_ops += f"\
                        sparse_tensor.out %t{i}, %stdout : {tensor_type}, !llvm.ptr<i8>\n"
                
                # Record
                module_code.append(input_tensor_ops)

            # The kernel will be in a wrapper function, which will be called
            # from main (we want to avoid returning complex types in main).
            #
            # We'll make a call to the wrapper. It has the same function
            # signature as the kernel, but is simply called "wrapper"
            module_code.append(f"call @wrapper(\
                               {', '.join([f'%t{i}' for i in range(len(arg_types))])}) :\
                                ({', '.join([str(at) for at in arg_types])}) -> ()\n")
            
            # End w/ return
            module_code.append("return\n}\n")

            # Finally, we'll add the wrapper function. First, the signature:
            module_code.append(f"func.func private @wrapper({', '.join([f'%t{i} : {str(at)}' for i, at in enumerate(arg_types)])}) -> () {{\n")

            # If we need timing, start the interval
            if options.timing:
                module_code.append("%start = call @rtclock() : () -> (f64)\n")

            # Call the kernel
            module_code.append(f"%res = call @{str(kernel_func.name)[1:-1]}(\
                               {', '.join([f'%t{i}' for i in range(len(arg_types))])})\
                                  : ({', '.join([str(at) for at in arg_types])}) -> {kernel_func.type.results[0]}\n")

            # If we need timing, end the interval and print it
            if options.timing:
                module_code.append("%end = call @rtclock() : () -> (f64)\n")
                module_code.append("call @rtclock_interval(%start, %end) : (f64, f64) -> ()\n")

            # If we need to print the output, we need a pointer to stdout
            if options.print_output:
                module_code.append("%stdout = call @getStdOut() : () -> (!llvm.ptr<i8>)\n")
                
                # Print the output
                module_code.append(f"sparse_tensor.out %res, %stdout : {kernel_func.type.results[0]}, !llvm.ptr<i8>\n")

            # Return and close the module
            module_code.append("return\n}\n}\n")
            return ''.join(module_code)


    def compile_pytaco_to_mlir(
        self,
        options: MLIRGenOptions,
        cap: Capture,
        kernel_inputs: Tuple[int] = None
    ) -> ir.Module:
        """
        Compiles the kernel to MLIR with the given options. Records
        the MLIR code and input shapes for the kernel in the current
        configuration.

        If no @kernel_inputs is provided, we assume that self.tensor
        will be reused w/ the same input sizes to the kernel.

        Args:
            options: MLIRGenOptions object
            cap: Capture object
            kernel_inputs: A tuple of input sizes to compile the kernel with.

        Returns:
            MLIR module. This is returned rather than a string of the 
            code to enable generality -> users of this method can invoke
            the MLIR pass manager on the module to perform additional
            transformations or use the MLIR code string for other purposes.

        TODO: Avoid reparsing and rebuilding modules ; fix corruption of module
              across multiple uses.

        TODO: Standardize returns for all compile methods
        """

        # Process the kernel and fetch the tensor assignments to compile
        #
        # TODO: Cache the generated Tensor object -> need to determine 
        # how the object can safely persist across multiple compiler uses.
        tensor_assignments: List[Tuple[pt.tensor, str]] = self.kernel(*kernel_inputs)

        # NOTE: We only handle a single tensor asasignment for now.
        assert len(tensor_assignments) == 1, f"{ERROR} Can only handle a single tensor assignment for now!"

        tensor, tensor_name = tensor_assignments[0]
        function_prefix: str = f"{self.kernel_name}_{tensor_name}"
        self.tensor_name = tensor_name

        # MLIR kernel codegen
        mlir_str, input_types = tensor.separately_compile(prefix=function_prefix)
        input_shapes: List[List[int]] = [t.shape for t in input_types]

        # Entry point generation
        if options.entry:
            mlir_str = self._emit_entry_point_and_kernel(mlir_str, input_types, options)

        # Reformat MLIR code
        module: ir.Module = self._reformat_mlir(mlir_str)

        # Record the metadata on the generated code as strings
        cap.mlir = str(module)
        cap.function_prefix = function_prefix
        cap.input_shapes = input_shapes

        return module
    

    def compile_mlir_to_llvm_with_python_bindings(
        self,
        options: LLVMIRGenOptions,
        cap: Capture,
        input_module: ir.Module
    ) -> ir.Module:
        """
        Compiles the MLIR code to LLVM-IR with the given options using
        the pass manager available in the Python bindings.

        Args:
            options: LLVMIRGenOptions object
            cap: Capture object
            input_module: A ir.Module object

        Returns:
            Lowered LLVM-IR as an ir.Module object.
        """

        # NOTE: This function is currently broken. Use compiler_mlir_to_llvm_with_shell instead.
        raise RuntimeWarning("compile_mlir_to_llvm_with_python_bindings: mlir_soda.passmanager fails to capture the open ir.Context and fails to recognize -soda-sparse-compiler!")

        with cap.capture_lowering_logs():
            with ir.Context() as ctx, ir.Location.unknown():
                passmanager.PassManager.parse(str(options)).run(input_module)
                return input_module
            
    
    def compile_mlir_to_llvm_with_shell(
        self,
        options: LLVMIRGenOptions,
        cap: Capture,
        input_module: str
    ) -> None:
        """
        Compiles the MLIR code to LLVM-IR by invoking soda-opt
        and mlir-translate as a subprocess.

        Args:
            options: LLVMIRGenOptions object
            cap: Capture object
            input_module: The MLIR module as a string.

        Returns:
            None. All output is stored in @cap
        """

        # Launch the first process -> This generates lowered MLIR 
        # in the LLVM dialect by invoking soda-opt
        result = subprocess.run(
            options.get_lowered_mlir_cmd(), 
            input=input_module, capture_output=True, 
            text=True, check=True, shell=True
        )

        # Capture output
        cap.lowered_mlir = result.stdout
        cap.lowering_log.write(result.stderr)

        # Launch the second process -> This generates LLVM-IR
        # by invoking mlir-translate
        result = subprocess.run(
            options.get_lowered_llvm_cmd(), 
            input=result.stdout, capture_output=True, 
            text=True, check=True, shell=True
        )

        # Capture output 
        cap.llvm = result.stdout
        return
                
    
    def run_compiled_kernel_with_python_bindings(
        self,
        options: RunOptions,
        cap: Capture,
        input_module: ir.Module
    ) -> None:
        """
        Runs the compiled kernel (to LLVM) with the given options using
        the execution engine.

        Args:
            options: RunOptions object
            cap: Capture object
            input_module: The LLVM-IR module to run.
            
        Returns: 
            None. All outputs are set inside @cap
        """

        # NOTE: This function is currently broken. Use run_compiled_kernel_with_shell instead.
        raise RuntimeWarning("run_compiled_kernel_with_python_bindings: mlir_soda.execution_engine fails to capture the open ir.Context!")

        # TODO: Set environment variables based on @options
        raise NotImplementedError("Not yet implemented @options.set_environment()")
        raise NotImplementedError("Not yet capturing LLVM-IR, only lowers the MLIR to LLVM dialect")

        # Run and capture output
        with cap.capture_result():
            with ir.Context(), ir.Location.unknown():
                engine = execution_engine.ExecutionEngine(input_module, shared_libs=options.libs)
                engine.invoke('main')
        
        return
    

    def run_compiled_kernel_with_shell(
        self,
        options: RunOptions,
        cap: Capture,
        input_module: str
    ) -> None:
        """
        Runs the compiled kernel (to LLVM dialect) with the given options 
        by invoking mlir-cpu-runner as a subprocess.

        Args:
            options: RunOptions object
            cap: Capture object
            input_module: The lowered MLIR module to run.
            
        Returns: 
            None. All outputs are set inside @cap
        """

        # Build the environment
        new_env = os.environ.copy()
        new_env["OMP_NUM_THREADS"] = str(options.num_threads)
        for i, f in enumerate(options.tensor_files):
            new_env[f"TENSOR{i}"] = f

        # Run mlir-cpu-runner
        result = subprocess.run(
            str(options), input=input_module, capture_output=True,
            text=True, check=True, shell=True, env=new_env
        )

        # Capture output
        cap.tensor_output.write(result.stdout)
        cap.timing.write(result.stderr)
        return

    
    def compile_and_run(
        self,
        mlir_options: MLIRGenOptions,
        llvm_options: LLVMIRGenOptions,
        run_options: RunOptions,
        kernel_inputs: Tuple[int]
    ) -> Capture:
        """
        End-to-end driver for compiling and running the kernel with the
        given options.

        Args:
            mlir_options: MLIRGenOptions object
            llvm_options: LLVMIRGenOptions object
            run_options: RunOptions object
            kernel_inputs: A tuple of input sizes to compile the kernel with.

        Returns:
            Capture w/ all intermediate IRs, logs, and runs.
        """

        # Set up capture
        cap: Capture = Capture()

        # Compile to MLIR
        self.compile_pytaco_to_mlir(mlir_options, cap, kernel_inputs)

        # Compile to LLVM
        self.compile_mlir_to_llvm_with_shell(llvm_options, cap, cap.mlir)

        # Run
        self.run_compiled_kernel_with_shell(run_options, cap, cap.lowered_mlir)

        return cap