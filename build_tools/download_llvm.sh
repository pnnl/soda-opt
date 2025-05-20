#!/bin/bash

set -e

if [[ $# -ne 2 ]] ; then
  echo "Usage: $0 <path/to/llvm> <llvm_branch_name>"
  exit 1
fi

# LLVM source
LLVM_SRC_DIR="$1"
LLVM_BRANCH="$2"

if [ -f "$LLVM_SRC_DIR/llvm/CMakeLists.txt" ]; then
  echo "LLVM already found at: $LLVM_SRC_DIR, trying to retrieve the correct branch"
  git -C "$LLVM_SRC_DIR" fetch --all
  git -C "$LLVM_SRC_DIR" checkout "$LLVM_BRANCH"
  exit 0
elif [ -d "$LLVM_SRC_DIR" ]; then
  echo "LLVM source directory already exists at $LLVM_SRC_DIR, but is not a git repository. Please remove it and try again."
  exit 1
fi

# Clone the LLVM project
mkdir -p "$LLVM_SRC_DIR"
git clone --depth=1 --branch "$LLVM_BRANCH" https://github.com/llvm/llvm-project.git "$LLVM_SRC_DIR"
