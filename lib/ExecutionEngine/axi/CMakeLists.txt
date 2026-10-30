# AXI DMA runtime libraries. They share AxiRuntime.cpp (the C interface in
# soda/ExecutionEngine/axi/AxiRuntime.h) and each link one DMA backend.

set(LLVM_OPTIONAL_SOURCES
  AxiRuntime.cpp
  HwBackend.cpp
  MockBackend.cpp
  )

# add_soda_axi_runtime(name backend sources...)
function(add_soda_axi_runtime name)
  add_mlir_library(${name}
    SHARED
    AxiRuntime.cpp
    ${ARGN}

    EXCLUDE_FROM_LIBMLIR
    )
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

# Loopback stream without an accelerator, for tests.
add_soda_axi_runtime(soda_axi_runtime_mock MockBackend.cpp)

# Xilinx AXI DMA mapped through /dev/mem.
if(UNIX)
  add_soda_axi_runtime(soda_axi_runtime_hw HwBackend.cpp)
endif()
