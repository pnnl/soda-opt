// RUN: soda-opt %s | FileCheck %s
// RUN: soda-opt %s | soda-opt | FileCheck %s

// CHECK-LABEL: func.func @init_dma
// CHECK:         accel.init_dma %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (i32, i32, i32, i32, i32)
func.func @init_dma(%addr: i32, %in: i32, %in_size: i32, %out: i32,
                    %out_size: i32) {
  accel.init_dma %addr, %in, %in_size, %out, %out_size
      : (i32, i32, i32, i32, i32)
  return
}

// CHECK-LABEL: func.func @send
// CHECK:         %[[O0:.*]] = accel.send %{{.*}} : (memref<60x80xf32>) -> i32
// CHECK:         accel.send %{{.*}}, %[[O0]] : (memref<60x80xf32>, i32) -> i32
func.func @send(%A: memref<60x80xf32>) -> i32 {
  %0 = accel.send %A : (memref<60x80xf32>) -> i32
  %1 = accel.send %A, %0 : (memref<60x80xf32>, i32) -> i32
  return %1 : i32
}

// CHECK-LABEL: func.func @send_literal
// CHECK:         %[[O0:.*]] = accel.send_literal %{{.*}} : (i32) -> i32
// CHECK:         accel.send_literal %{{.*}}, %[[O0]] : (i32, i32) -> i32
func.func @send_literal(%opcode: i32) -> i32 {
  %0 = accel.send_literal %opcode : (i32) -> i32
  %1 = accel.send_literal %opcode, %0 : (i32, i32) -> i32
  return %1 : i32
}

// CHECK-LABEL: func.func @send_dim_idx
// CHECK:         %[[O0:.*]] = accel.send_dim %{{.*}}[1] : (memref<10x20xi32>) -> i32
// CHECK:         accel.send_idx %{{.*}}[0], %[[O0]] : (memref<10x20xi32>, i32) -> i32
func.func @send_dim_idx(%T: memref<10x20xi32>) -> i32 {
  %0 = accel.send_dim %T[1] : (memref<10x20xi32>) -> i32
  %1 = accel.send_idx %T[0], %0 : (memref<10x20xi32>, i32) -> i32
  return %1 : i32
}

// CHECK-LABEL: func.func @recv
// CHECK:         %[[O0:.*]] = accel.recv %{{.*}} : (memref<4x4xi32>) -> i32
// CHECK:         accel.recv %{{.*}}, %[[O0]] mode = accumulate : (memref<4x4xi32>, i32) -> i32
// CHECK:         accel.recv %{{.*}} mode = replace : (memref<4x4xi32>) -> i32
func.func @recv(%C: memref<4x4xi32>) -> i32 {
  %0 = accel.recv %C : (memref<4x4xi32>) -> i32
  %1 = accel.recv %C, %0 mode = accumulate : (memref<4x4xi32>, i32) -> i32
  %2 = accel.recv %C mode = replace : (memref<4x4xi32>) -> i32
  return %1 : i32
}
