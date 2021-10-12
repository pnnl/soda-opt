#!/bin/bash

# USAGE
#  This file is supposed to be run in the same folder where it exists.

# Kernel configs
NAME=gemm
KSIZE=2

# Directories
KERNELDIR=$(pwd)

# Bambu configs
CLKPERIOD=10
CHANNELSNUMBER=2

# Load debug flags for bambu
source ${KERNELDIR}/../../scripts/bambu-debug-flags.sh

# Perform the synthesis
# source ${KERNELDIR}/../../scripts/outline-affine-for-optimize-full-bambu-flopoco.sh

# With SSDCS
source ${KERNELDIR}/../../scripts/outline-affine-for-optimize-full-bambu-soft-float-with-ssdcs.sh
source ${KERNELDIR}/../../scripts/outline-affine-for-optimize-none-bambu-soft-float-with-ssdcs.sh

# NO SSDCS
source ${KERNELDIR}/../../scripts/outline-affine-for-optimize-full-bambu-soft-float-no-ssdcs.sh
source ${KERNELDIR}/../../scripts/outline-affine-for-optimize-none-bambu-soft-float-no-ssdcs.sh