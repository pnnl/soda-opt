# AXI DMA runtime libraries. They share AxiRuntime.cpp (the C interface in
# soda/ExecutionEngine/axi/AxiRuntime.h) and each link one DMA backend.

set(LLVM_OPTIONAL_SOURCES
  AxiRuntime.cpp
  HwBackend.cpp
  MockBackend.cpp
  SyscBackend.cpp
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

# SystemC simulations of the DMA engine and an accelerator model from
# include/soda/ExecutionEngine/axi/accelerators, one library per model:
# soda_axi_runtime_sysc_<model>.
set(SODA_AXI_SYSC_ACCELERATORS
  mm_4x4_v1
  mm_4x4_v2
  mm_4x4_v3
  mm_4x4_v4
  conv_v3
  )
if(SODA_HAVE_SYSTEMC)
  add_subdirectory(sysc)
endif()
