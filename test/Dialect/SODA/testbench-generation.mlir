// RUN: soda-opt -allow-unregistered-dialect --soda-extract-arguments-to-xml="write-to-terminal" %s | FileCheck %s --check-prefixes=CHECK_TERMINAL_XML
// RUN: soda-opt -allow-unregistered-dialect --soda-extract-arguments-to-xml="write-to-terminal using-bare-ptr" %s | FileCheck %s --check-prefixes=CHECK_BARE_XML

// RUN: soda-opt -allow-unregistered-dialect --soda-extract-arguments-to-c-testbench="write-to-terminal" %s | FileCheck %s --check-prefixes=CHECK_TERMINAL_C
// TODO: soda-opt -allow-unregistered-dialect --soda-extract-arguments-to-c-testbench="write-to-terminal using-bare-ptr" %s | FileCheck %s --check-prefixes=CHECK_BARE_C

// TODO: Sometimes, filecheck will try to verify the file before the file is written. 
// This causes the test to fail. Commenting this test for now
// TODO: soda-opt -allow-unregistered-dialect --soda-extract-arguments-to-xml %s
// TODO:   FileCheck %s -input-file=driver_kernel_test.xml --check-prefixes=CHECK_FILE

module attributes {soda.container_module}  {
  func.func @driver() {
    %0 = "loadA"() : () -> memref<4x7xf32>
    %1 = "loadB"() : () -> memref<7x3xf32>
    %2 = "allocateC"() : () -> memref<4x3xf32>
    soda.launch_func  @driver_kernel::@driver_kernel args(%0 : memref<4x7xf32>, %1 : memref<7x3xf32>, %2 : memref<4x3xf32>)
    return
  }
  soda.module @driver_kernel {
    soda.func @driver_kernel(%arg0: memref<4x7xf32>, %arg1: memref<7x3xf32>, %arg2: memref<4x3xf32>) kernel {
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      linalg.matmul ins(%arg0, %arg1 : memref<4x7xf32>, memref<7x3xf32>) outs(%arg2 : memref<4x3xf32>)
      soda.return
    }
  }
}

// TODO: This test is not executing due to the issue mentioned above on RUN line.
// CHECK_FILE: <?xml version="1.0"?>
// CHECK_FILE: <function>
// CHECK_FILE:  <testbench
// CHECK_FILE:   P0="{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}"
// CHECK_FILE:   P1="{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}"
// CHECK_FILE:   P2="0" P3="4" P4="7" P5="7" P6="1" 
// CHECK_FILE:   P7="{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}"
// CHECK_FILE:   P8="{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}"
// CHECK_FILE:   P9="0" P10="7" P11="3" P12="3" P13="1" 
// CHECK_FILE:   P14="{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}"
// CHECK_FILE:   P15="{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}"
// CHECK_FILE:   P16="0" P17="4" P18="3" P19="3" P20="1" 
// CHECK_FILE:  />
// CHECK_FILE:  </function>

// CHECK_TERMINAL_XML: <?xml version="1.0"?>
// CHECK_TERMINAL_XML: <function>
// CHECK_TERMINAL_XML:  <testbench
// CHECK_TERMINAL_XML:   P0="{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}"
// CHECK_TERMINAL_XML:   P1="{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}"
// CHECK_TERMINAL_XML:   P2="0" P3="4" P4="7" P5="7" P6="1" 
// CHECK_TERMINAL_XML:   P7="{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}"
// CHECK_TERMINAL_XML:   P8="{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}"
// CHECK_TERMINAL_XML:   P9="0" P10="7" P11="3" P12="3" P13="1" 
// CHECK_TERMINAL_XML:   P14="{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}"
// CHECK_TERMINAL_XML:   P15="{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}"
// CHECK_TERMINAL_XML:   P16="0" P17="4" P18="3" P19="3" P20="1" 
// CHECK_TERMINAL_XML:  />
// CHECK_TERMINAL_XML:  </function>

// CHECK_BARE_XML: <?xml version="1.0"?>
// CHECK_BARE_XML: <function>
// CHECK_BARE_XML:  <testbench
// CHECK_BARE_XML:   P0="{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}"
// CHECK_BARE_XML:   P1="{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}"
// CHECK_BARE_XML:   P2="{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}"
// CHECK_BARE_XML:  />
// CHECK_BARE_XML:  </function>



// CHECK_TERMINAL_C: #define _FILE_OFFSET_BITS 64
// CHECK_TERMINAL_C: #define __Inf (1.0 / 0.0)
// CHECK_TERMINAL_C: #define __Nan (0.0 / 0.0)
// CHECK_TERMINAL_C: #ifdef __cplusplus
// CHECK_TERMINAL_C: #undef printf

// CHECK_TERMINAL_C: #include <cstdio>
// CHECK_TERMINAL_C: #include <cstdlib>

// CHECK_TERMINAL_C: typedef bool _Bool;
// CHECK_TERMINAL_C: #else
// CHECK_TERMINAL_C: #include <stdio.h>
// CHECK_TERMINAL_C: #include <stdlib.h>

// CHECK_TERMINAL_C: extern void exit(int status);
// CHECK_TERMINAL_C: #endif

// CHECK_TERMINAL_C: #include <sys/types.h>

// CHECK_TERMINAL_C: #ifdef __AC_NAMESPACE
// CHECK_TERMINAL_C: using namespace __AC_NAMESPACE;
// CHECK_TERMINAL_C: #endif



// CHECK_TERMINAL_C: #ifndef CDECL
// CHECK_TERMINAL_C: #ifdef __cplusplus
// CHECK_TERMINAL_C: #define CDECL extern "C"
// CHECK_TERMINAL_C: #else
// CHECK_TERMINAL_C: #define CDECL
// CHECK_TERMINAL_C: #endif
// CHECK_TERMINAL_C: #endif

// CHECK_TERMINAL_C: #ifndef EXTERN_CDECL
// CHECK_TERMINAL_C: #ifdef __cplusplus
// CHECK_TERMINAL_C: #define EXTERN_CDECL extern "C"
// CHECK_TERMINAL_C: #else
// CHECK_TERMINAL_C: #define EXTERN_CDECL extern
// CHECK_TERMINAL_C: #endif
// CHECK_TERMINAL_C: #endif

// CHECK_TERMINAL_C: #include <mdpi/mdpi_user.h>

// CHECK_TERMINAL_C: CDECL void driver_kernel(void*, void*, void*);

// CHECK_TERMINAL_C: int main()
// CHECK_TERMINAL_C: {
// CHECK_TERMINAL_C:    void* P0;
// CHECK_TERMINAL_C:    void* P1;
// CHECK_TERMINAL_C:    void* P2;
// CHECK_TERMINAL_C:    {
// CHECK_TERMINAL_C:       float P0_temp[] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
// CHECK_TERMINAL_C:       P0 = (void*)P0_temp;
// CHECK_TERMINAL_C:       m_param_alloc(0, sizeof(P0_temp));
// CHECK_TERMINAL_C:       float P1_temp[] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
// CHECK_TERMINAL_C:       P1 = (void*)P1_temp;
// CHECK_TERMINAL_C:       m_param_alloc(1, sizeof(P1_temp));
// CHECK_TERMINAL_C:       float P2_temp[] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
// CHECK_TERMINAL_C:       P2 = (void*)P2_temp;
// CHECK_TERMINAL_C:       m_param_alloc(2, sizeof(P2_temp));
// CHECK_TERMINAL_C:       driver_kernel((void*) P0, (void*) P1, (void*) P2);
// CHECK_TERMINAL_C:    }
// CHECK_TERMINAL_C:    return 0;
// CHECK_TERMINAL_C: }