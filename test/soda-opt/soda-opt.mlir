// RUN: soda-opt --show-dialects | FileCheck %s
// RUN: soda-opt --h | FileCheck %s -check-prefix=CHECKHELP
//
// WIP: accel
// CHECK: affine
// CHECK: arith
// CHECK: builtin
// CHECK: cf
// CHECK: func
// CHECK: linalg
// CHECK: llvm
// CHECK: ml_program
// CHECK: pdl
// CHECK: scf
// CHECK: snn
// CHECK: soda
// CHECK: transform
// CHECK: vector

// CHECKHELP: convert-linalg-matmul-to-soda
// CHECKHELP: soda-generate-bambu-accelcode
// CHECKHELP: soda-generate-bambu-hostcode
// CHECKHELP: soda-outline-bambu-code
// CHECKHELP: soda-opt-pipeline