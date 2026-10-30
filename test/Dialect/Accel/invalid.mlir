// RUN: soda-opt %s --allow-unregistered-dialect -split-input-file -verify-diagnostics

// expected-error @below {{unknown opcode action 'sned', expected one of: send, send_literal, send_dim, send_idx, recv}}
"test.op"() {a = #accel.action<sned(0)>} : () -> ()

// -----

// expected-error @below {{unknown recv mode 'add', expected 'replace' or 'accumulate'}}
"test.op"() {a = #accel.action<recv(2, add)>} : () -> ()

// -----

// Only recv actions take a mode.
// expected-error @below {{expected ')'}}
"test.op"() {a = #accel.action<send(0, accumulate)>} : () -> ()

// -----

// expected-error @below {{send_literal value 4294967296 does not fit in 32 bits}}
"test.op"() {a = #accel.action<send_literal(0x100000000)>} : () -> ()

// -----

// expected-error @below {{operand id must be non-negative, got -1}}
"test.op"() {a = #accel.action<send(-1)>} : () -> ()

// -----

// expected-error @below {{opcode 'sA' is defined more than once}}
"test.op"() {m = #accel.opcode_map<sA = [send(0)], sA = [send(1)]>} : () -> ()

// -----

// expected-error @below {{expected '['}}
"test.op"() {m = #accel.opcode_map<sA = send(0)>} : () -> ()

// -----

// expected-error @below {{a flow group can contain at most one nested group}}
"test.op"() {f = #accel.opcode_flow<((a) (b))>} : () -> ()

// -----

// expected-error @below {{opcode flow must contain at least one opcode}}
"test.op"() {f = #accel.opcode_flow<(())>} : () -> ()

// -----

// expected-error @below {{has unknown accel attribute 'accel.opcodes'}}
"test.op"() {accel.opcodes = 1} : () -> ()

// -----

// expected-error @below {{'accel.opcode_map' must be an #accel.opcode_map, got #accel.opcode_flow<(s)>}}
"test.op"() {accel.opcode_map = #accel.opcode_flow<(s)>} : () -> ()

// -----

// expected-error @below {{has 'accel.opcode_flow' but no 'accel.opcode_map' that defines its opcodes}}
"test.op"() {accel.opcode_flow = #accel.opcode_flow<(s)>} : () -> ()

// -----

// expected-error @below {{'accel.opcode_flow' uses opcode 'r', which is not defined in 'accel.opcode_map'}}
"test.op"() {accel.opcode_map = #accel.opcode_map<s = [send(0)]>,
             accel.opcode_flow = #accel.opcode_flow<(s r)>} : () -> ()

// -----

// expected-error @below {{'accel.init_opcodes' must be a single group without nesting}}
"test.op"() {accel.opcode_map = #accel.opcode_map<s = [send(0)]>,
             accel.init_opcodes = #accel.opcode_flow<(s (s))>} : () -> ()

// -----

// expected-error @below {{'accel.permutation_map' must be a permutation}}
"test.op"() {accel.permutation_map = affine_map<(m, n) -> (m, m)>} : () -> ()

// -----

// expected-error @below {{'accel.accel_dim' results must be non-negative constants}}
"test.op"() {accel.accel_dim = affine_map<(m, n) -> (m, 4)>} : () -> ()

// -----

// Checks against the linalg op that carries the attributes.

func.func @operand_out_of_range(%A: memref<8x8xi32>, %B: memref<8x8xi32>,
                                %C: memref<8x8xi32>) {
  // expected-error @below {{opcode 's' action send(3) refers to operand 3, but the op has 3 operands}}
  linalg.matmul {accel.opcode_map = #accel.opcode_map<s = [send(3)]>}
      ins(%A, %B : memref<8x8xi32>, memref<8x8xi32>)
      outs(%C : memref<8x8xi32>)
  return
}

// -----

func.func @recv_into_input(%A: memref<8x8xi32>, %B: memref<8x8xi32>,
                           %C: memref<8x8xi32>) {
  // expected-error @below {{opcode 'r' action recv(0) must receive into an output operand}}
  linalg.matmul {accel.opcode_map = #accel.opcode_map<r = [recv(0)]>}
      ins(%A, %B : memref<8x8xi32>, memref<8x8xi32>)
      outs(%C : memref<8x8xi32>)
  return
}

// -----

func.func @send_dim_out_of_range(%A: memref<8x8xi32>, %B: memref<8x8xi32>,
                                 %C: memref<8x8xi32>) {
  // expected-error @below {{opcode 'd' action send_dim(1) uses dimension 2, but the operand has rank 2}}
  linalg.matmul {accel.opcode_map = #accel.opcode_map<d = [send_dim(1, 2)]>}
      ins(%A, %B : memref<8x8xi32>, memref<8x8xi32>)
      outs(%C : memref<8x8xi32>)
  return
}

// -----

func.func @accumulate_without_add(%A: memref<8xi32>, %C: memref<i32>) {
  // expected-error @below {{opcode 'r' action recv(1) uses accumulate mode, but the op does not compute this output as an add reduction}}
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                   affine_map<(d0) -> ()>],
                  iterator_types = ["reduction"],
                  accel.opcode_map = #accel.opcode_map<r = [recv(1, accumulate)]>}
      ins(%A : memref<8xi32>) outs(%C : memref<i32>) {
  ^bb0(%a: i32, %c: i32):
    %0 = arith.maxsi %a, %c : i32
    linalg.yield %0 : i32
  }
  return
}

// -----

func.func @accel_dim_arity(%A: memref<8x8xi32>, %B: memref<8x8xi32>,
                           %C: memref<8x8xi32>) {
  // expected-error @below {{'accel.accel_dim' must map the op's 3 loops to 3 tile sizes}}
  linalg.matmul {accel.accel_dim = affine_map<(m, n) -> (4, 4)>}
      ins(%A, %B : memref<8x8xi32>, memref<8x8xi32>)
      outs(%C : memref<8x8xi32>)
  return
}

// -----

func.func @permutation_arity(%A: memref<8x8xi32>, %B: memref<8x8xi32>,
                             %C: memref<8x8xi32>) {
  // expected-error @below {{'accel.permutation_map' must permute the op's 3 loops}}
  linalg.matmul {accel.permutation_map = affine_map<(m, n) -> (n, m)>}
      ins(%A, %B : memref<8x8xi32>, memref<8x8xi32>)
      outs(%C : memref<8x8xi32>)
  return
}

// -----

func.func @flow_too_deep(%A: memref<8x8xi32>, %B: memref<8x8xi32>,
                         %C: memref<8x8xi32>) {
  // expected-error @below {{'accel.opcode_flow' has 3 nested groups, but only 1 loops are tiled for the accelerator (at most 2 groups)}}
  linalg.matmul {accel.opcode_map = #accel.opcode_map<s = [send(0)]>,
                 accel.opcode_flow = #accel.opcode_flow<(s ((s)))>,
                 accel.accel_dim = affine_map<(m, n, k) -> (4, 0, 0)>}
      ins(%A, %B : memref<8x8xi32>, memref<8x8xi32>)
      outs(%C : memref<8x8xi32>)
  return
}

// -----

// Op verifiers.

func.func @send_dim_op(%T: memref<4x4xi32>) {
  // expected-error @below {{'accel.send_dim' op dimension 2 is out of range for a memref of rank 2}}
  %0 = accel.send_dim %T[2] : (memref<4x4xi32>) -> i32
  return
}

// -----

func.func @send_idx_op(%T: memref<4xi32>) {
  // expected-error @below {{'accel.send_idx' op dimension 1 is out of range for a memref of rank 1}}
  %0 = accel.send_idx %T[1] : (memref<4xi32>) -> i32
  return
}

// -----

func.func @recv_accumulate_index(%T: memref<4xindex>) {
  // expected-error @below {{'accel.recv' op accumulate mode needs an integer or float element type, got 'index'}}
  %0 = accel.recv %T mode = accumulate : (memref<4xindex>) -> i32
  return
}
