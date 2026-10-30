// RUN: soda-opt %s --convert-accel-to-axi-runtime -split-input-file -verify-diagnostics

func.func @unsupported_type(%A: memref<4xi8>) {
  // expected-error @below {{'accel.send' op has unsupported element type 'i8' (supported: i32, f32)}}
  %0 = accel.send %A : (memref<4xi8>) -> i32
  return
}

// -----

func.func @unsupported_recv_type(%A: memref<4xf64>) {
  // expected-error @below {{'accel.recv' op has unsupported element type 'f64' (supported: i32, f32)}}
  %0 = accel.recv %A : (memref<4xf64>) -> i32
  return
}

// -----

func.func private @dma_wait_send(i32)

func.func @signature_mismatch(%v: i32) {
  // expected-error @below {{runtime function 'dma_wait_send' is used with type '() -> ()', but it is declared with type '(i32) -> ()'}}
  %0 = accel.send_literal %v : (i32) -> i32
  return
}
