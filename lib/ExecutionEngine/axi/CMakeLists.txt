# Exclude these from libMLIR.so because the JIT infrastructure
# is a big dependency which most don't need.

add_subdirectory(api)

set(LLVM_OPTIONAL_SOURCES
  AxiUtils.cpp
  AxiUtilsMock.cpp
  AxiUtilsSysc.cpp
)

add_mlir_library(mlir_mockaxi_runner_utils
  SHARED
  AxiUtilsMock.cpp

  EXCLUDE_FROM_LIBMLIR
)
target_compile_definitions(mlir_mockaxi_runner_utils PRIVATE mlir_mockaxi_runner_utils_EXPORTS)

add_mlir_library(mlir_axi_runner_utils
  SHARED
  AxiUtils.cpp
  EXCLUDE_FROM_LIBMLIR
)
target_compile_definitions(mlir_axi_runner_utils PRIVATE mlir_axi_runner_utils_EXPORTS)
add_dependencies(mlir_axi_runner_utils
  axi_api_v1
)
target_link_libraries(mlir_axi_runner_utils PUBLIC axi_api_v1) # Needed to call implemented functions

# Only generate systemc libraries for native builds
if(AXI_CROSSCOMPILING)
  message(STATUS "Cross-compiling, SystemC runner libraries are disabed")
else()
  message(STATUS "Not Cross-compiling, SystemC runner libraries are enabled")

  if(DEFINED ENV{SYSTEMC_HOME})
    message(STATUS "SYSTEMC_HOME is set: $ENV{SYSTEMC_HOME} -- building sysc runner libs")

    add_mlir_library(mlir_syscaxi_runner_utils
      SHARED
      AxiUtilsSysc.cpp
      EXCLUDE_FROM_LIBMLIR
    )
    target_compile_definitions(mlir_syscaxi_runner_utils PRIVATE mlir_syscaxi_runner_utils_EXPORTS)
    add_dependencies(mlir_syscaxi_runner_utils
      axi_api_v1_sysc
    )
    target_link_libraries(mlir_syscaxi_runner_utils PUBLIC axi_api_v1_sysc) # Needed to call implemented functions

    # ----------------------------
    # Matmul accelerator
    # Same for accelerator v1
    add_mlir_library(mlir_syscaxi_runner_utils_accv1
      SHARED
      AxiUtilsSysc.cpp
      EXCLUDE_FROM_LIBMLIR
    )
    target_compile_definitions(mlir_syscaxi_runner_utils_accv1 PRIVATE mlir_syscaxi_runner_utils_v1_EXPORTS)
    add_dependencies(mlir_syscaxi_runner_utils_accv1
      axi_api_v1_sysc_accv1
    )
    target_link_libraries(mlir_syscaxi_runner_utils_accv1 PUBLIC axi_api_v1_sysc_accv1) # Needed to call implemented functions

    # Same for accelerator v2
    add_mlir_library(mlir_syscaxi_runner_utils_accv2
      SHARED
      AxiUtilsSysc.cpp
      EXCLUDE_FROM_LIBMLIR
    )
    target_compile_definitions(mlir_syscaxi_runner_utils_accv2 PRIVATE mlir_syscaxi_runner_utils_v2_EXPORTS)
    add_dependencies(mlir_syscaxi_runner_utils_accv2
      axi_api_v1_sysc_accv2
    )
    target_link_libraries(mlir_syscaxi_runner_utils_accv2 PUBLIC axi_api_v1_sysc_accv2) # Needed to call implemented functions

    # Same for accelerator v3
    add_mlir_library(mlir_syscaxi_runner_utils_accv3
      SHARED
      AxiUtilsSysc.cpp
      EXCLUDE_FROM_LIBMLIR
    )
    target_compile_definitions(mlir_syscaxi_runner_utils_accv3 PRIVATE mlir_syscaxi_runner_utils_v3_EXPORTS)
    add_dependencies(mlir_syscaxi_runner_utils_accv3
      axi_api_v1_sysc_accv3
    )
    target_link_libraries(mlir_syscaxi_runner_utils_accv3 PUBLIC axi_api_v1_sysc_accv3) # Needed to call implemented functions

    # Same for accelerator v4
    add_mlir_library(mlir_syscaxi_runner_utils_accv4
      SHARED
      AxiUtilsSysc.cpp
      EXCLUDE_FROM_LIBMLIR
    )
    target_compile_definitions(mlir_syscaxi_runner_utils_accv4 PRIVATE mlir_syscaxi_runner_utils_v4_EXPORTS)
    add_dependencies(mlir_syscaxi_runner_utils_accv4
      axi_api_v1_sysc_accv4
    )
    target_link_libraries(mlir_syscaxi_runner_utils_accv4 PUBLIC axi_api_v1_sysc_accv4) # Needed to call implemented functions

    # ----------------------------
    # Conv accelerator

    # Same for accelerator v1
    add_mlir_library(mlir_syscaxi_runner_utils_conv_accv1
      SHARED
      AxiUtilsSysc.cpp
      EXCLUDE_FROM_LIBMLIR
    )
    target_compile_definitions(mlir_syscaxi_runner_utils_conv_accv1 PRIVATE mlir_syscaxi_runner_utils_conv_v1_EXPORTS)
    add_dependencies(mlir_syscaxi_runner_utils_conv_accv1
      axi_api_v1_sysc_conv_accv1
    )
    target_link_libraries(mlir_syscaxi_runner_utils_conv_accv1 PUBLIC axi_api_v1_sysc_conv_accv1) # Needed to call implemented functions
  endif()
endif()

set(CMAKE_CXX_FLAGS "${tmpcxxflags}") # Revert to normal CXX flags