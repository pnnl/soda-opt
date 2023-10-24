#!/usr/bin/env -S bash -x

# Set up arguments
MLIR_ROOT=$1
SODA_OPT_ROOT=$2
OPENMP_LIB=$3

# Configure
./configure.sh $MLIR_ROOT $SODA_OPT_ROOT $OPENMP_LIB ;
source ./ENV ;

# Tutorial - Part 1
#
# End-to-end compilation/execution for SpMVMul (this is the last
# command in the example from README.md)
mkdir -p ${PTT}/output ;
soda-pytaco --v \
    --actions mlir llvm run \
    --output-dir ${PTT}/output \
    --kernel SpMVMul --kernel-dir ${PTT} --inps 16 16 \
    --emit-entry-point=True --add-timing=True \
    --input-tensor-files ${PTT}/tensors/T0.tns ${PTT}/tensors/T1.tns ;

# Sanity check w/ ${PTT}/tensors/expected_result.tns - check that the files are the same
DIFF=$(diff ${PTT}/output/SpMVMul_z_16_16.run.out ${PTT}/tensors/expected_result.tns) 
if [ "$DIFF" != "" ] 
then
    echo "ERROR: SpMVMul_z_16_16.run.out and expected_result.tns are different"
    exit 1
fi

# Tutorial - Part 2
#
# Generate random tensors
mkdir -p ${PTT}/new-tensors ;
soda-pytaco --v \
    --actions gen-tensors \
    --output-dir ${PTT}/new-tensors \
    --new-tensor-specs="16x16x16:0.1, 32x32x32:0.2" ;

# Tutorial - Part 3
python3 ${PTT}/perf.py ;


