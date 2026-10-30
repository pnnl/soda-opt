//===- mm_4x4_v4.cpp - SystemC backend for mm_4x4_v4 ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SyscBackend.h"

// The model must come last: it defines short macros such as M, N and K.
#include "soda/ExecutionEngine/axi/accelerators/mm_4x4_v4/accelerator.sc.h"

SODA_AXI_DEFINE_SYSC_BACKEND("sysc mm_4x4_v4")
