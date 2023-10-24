# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'SODA_PROJ'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir','.ll']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.soda_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%sodashlibdir', config.soda_lib_root))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

# MLIR runner libraries are installed in the llvm project lib dir.
config.substitutions.append(("%mlir_lib_dir", config.llvm_lib_dir))

# export the SODA test directory
config.substitutions.append(("%soda_test_dir", config.test_source_root))

# export the OpenMP library
config.substitutions.append(("%openmp_lib", config.openmp_lib))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.soda_obj_root, 'test')
config.soda_tools_dir = os.path.join(config.soda_obj_root, 'bin')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [config.soda_tools_dir, config.mlir_tools_dir, config.llvm_tools_dir]
tools = [
    'soda-opt',
    'soda-translate',
    'soda-capi-test',
    'mlir-runner',
    # Tools from llvm-project
    'opt',
    'mlir-cpu-runner',
    ToolSubst('%PYTHON', config.python_executable, unresolved='ignore'),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

llvm_config.with_environment('PYTHONPATH', [
    os.path.join(config.mlir_obj_root, 'python_packages', 'soda'),
], append_path=True)

if config.enable_pytaco:
    config.substitutions.append(('%pyextra', ':'.join([
        os.path.join(config.mlir_obj_root, 'python_packages', 'soda'),
        os.path.join(config.llvm_obj_root, 'tools', 'mlir', 'python_packages', 'mlir_core'),
        os.path.join(config.llvm_obj_root, '..', 'mlir', 'test', 'Integration', 'Dialect', 'SparseTensor', 'taco')
    ])))
