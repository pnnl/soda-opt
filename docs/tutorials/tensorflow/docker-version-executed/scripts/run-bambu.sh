#!/bin/bash

set -e
set -o pipefail

if [ -d output/$1 ] 
then
    rm -rf output/$1
    mkdir -p output/$1
else
    mkdir -p output/$1
fi

# PLATFORM=asap7
# DEVICE=asap7-BC
# DEVICE=asap7-TC
# DEVICE=asap7-WC

PLATFORM=nangate45
DEVICE=nangate45

# Check if docker is available or if the needed binaries are available
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPT_DIR/check_docker.sh

cp output/05$1.ll output/$1/input.ll
cp main_kernel_testbench.c output/$1/main_kernel_testbench.c

pushd output/$1;

$DOCKER_RUN \
opt -O2 -strip-debug input.ll \
  -S -o visualize.ll

$DOCKER_RUN \
bambu -v3 --print-dot \
  -lm --soft-float \
--compiler=I386_CLANG16  \
--device=${DEVICE} \
--clock-period=5 \
--experimental-setup=BAMBU-BALANCED-MP \
--channels-number=2 \
--memory-allocation-policy=ALL_BRAM \
--disable-function-proxy \
--generate-tb=main_kernel_testbench.c \
--simulate --simulator=VERILATOR --verilator-parallel \
--top-fname=main_kernel \
input.ll 2>&1 | tee bambu-log

popd

# Patch nangate
SCRIPTDIR=scripts
BAMBUDIR=output/$1
KERNELNAME=main_kernel
source ${SCRIPTDIR}/patch_openroad_synt.sh
