# Exclude these from libMLIR.so because the JIT infrastructure
# is a big dependency which most don't need.

set(LLVM_OPTIONAL_SOURCES
  api_v0.cpp
  api_v1.cpp
  api_v1_sysc.cpp
  api_v2.cpp
  api_v2_sysc.cpp
)

add_mlir_library(axi_api_v0
  SHARED
  api_v0.cpp

  EXCLUDE_FROM_LIBMLIR
)

add_mlir_library(axi_api_v1
  SHARED
  api_v1.cpp

  EXCLUDE_FROM_LIBMLIR
)

set(tmpcxxflags ${CMAKE_CXX_FLAGS})
string(REPLACE "-Werror=global-constructors" "" FIXED ${CMAKE_CXX_FLAGS})
string(REPLACE "-Wcast-qual" "-Wno-vla-extension" FIXED ${FIXED})

if(AXI_CROSSCOMPILING)
  message(STATUS "Cross-compiling, SystemC api libraries are disabed")
  string(APPEND FIXED " -mfpu=neon")
else()
  message(STATUS "Not Cross-compiling, SystemC api libraries are enabled")

  if(DEFINED ENV{SYSTEMC_HOME})
    message(STATUS "SYSTEMC_HOME is set: $ENV{SYSTEMC_HOME} -- building sysc api libs")

    add_mlir_library(axi_api_v1_sysc
      SHARED
      api_v1_sysc.cpp

      EXCLUDE_FROM_LIBMLIR
    )
    target_include_directories(axi_api_v1_sysc PUBLIC $ENV{SYSTEMC_HOME}/include/)
    set_target_properties(axi_api_v1_sysc PROPERTIES COMPILE_FLAGS "")
    target_link_libraries(axi_api_v1_sysc PUBLIC $ENV{SYSTEMC_HOME}/lib-linux64/libsystemc.a)

    # ----------------------------
    # Matmul accelerator

    # Same for accelerator v1
    add_mlir_library(axi_api_v1_sysc_accv1
      SHARED
      api_v1_sysc.cpp

      EXCLUDE_FROM_LIBMLIR
    )
    target_include_directories(axi_api_v1_sysc_accv1 PUBLIC $ENV{SYSTEMC_HOME}/include/)
    set_target_properties(axi_api_v1_sysc_accv1 PROPERTIES COMPILE_FLAGS "-DACC_V1")
    target_link_libraries(axi_api_v1_sysc_accv1 PUBLIC $ENV{SYSTEMC_HOME}/lib-linux64/libsystemc.a)

    # Same for accelerator v2
    add_mlir_library(axi_api_v1_sysc_accv2
      SHARED
      api_v1_sysc.cpp

      EXCLUDE_FROM_LIBMLIR
    )
    target_include_directories(axi_api_v1_sysc_accv2 PUBLIC $ENV{SYSTEMC_HOME}/include/)
    set_target_properties(axi_api_v1_sysc_accv2 PROPERTIES COMPILE_FLAGS "-DACC_V2")
    target_link_libraries(axi_api_v1_sysc_accv2 PUBLIC $ENV{SYSTEMC_HOME}/lib-linux64/libsystemc.a)

    # Same for accelerator v3
    add_mlir_library(axi_api_v1_sysc_accv3
      SHARED
      api_v1_sysc.cpp

      EXCLUDE_FROM_LIBMLIR
    )
    target_include_directories(axi_api_v1_sysc_accv3 PUBLIC $ENV{SYSTEMC_HOME}/include/)
    set_target_properties(axi_api_v1_sysc_accv3 PROPERTIES COMPILE_FLAGS "-DACC_V3")
    target_link_libraries(axi_api_v1_sysc_accv3 PUBLIC $ENV{SYSTEMC_HOME}/lib-linux64/libsystemc.a)

    # Same for accelerator v4
    add_mlir_library(axi_api_v1_sysc_accv4
      SHARED
      api_v1_sysc.cpp

      EXCLUDE_FROM_LIBMLIR
    )
    target_include_directories(axi_api_v1_sysc_accv4 PUBLIC $ENV{SYSTEMC_HOME}/include/)
    set_target_properties(axi_api_v1_sysc_accv4 PROPERTIES COMPILE_FLAGS "-DACC_V4")
    target_link_libraries(axi_api_v1_sysc_accv4 PUBLIC $ENV{SYSTEMC_HOME}/lib-linux64/libsystemc.a)

    # ----------------------------
    # Conv accelerator

    # Same for accelerator v1
    add_mlir_library(axi_api_v1_sysc_conv_accv1
      SHARED
      api_v1_sysc.cpp

      EXCLUDE_FROM_LIBMLIR
    )
    target_include_directories(axi_api_v1_sysc_conv_accv1 PUBLIC $ENV{SYSTEMC_HOME}/include/)
    set_target_properties(axi_api_v1_sysc_conv_accv1 PROPERTIES COMPILE_FLAGS "-DCONV_V1")
    target_link_libraries(axi_api_v1_sysc_conv_accv1 PUBLIC $ENV{SYSTEMC_HOME}/lib-linux64/libsystemc.a)
  endif()
endif()

set(CMAKE_CXX_FLAGS "${FIXED}")

# No additional properties for now
# target_compile_definitions(axi_api_v1 PRIVATE axi_api_EXPORTS)