set(LLVM_OPTIONAL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../SyscBackend.cpp)
foreach(accelerator ${SODA_AXI_SYSC_ACCELERATORS})
  list(APPEND LLVM_OPTIONAL_SOURCES ${accelerator}.cpp)
endforeach()

# SystemC needs RTTI and exceptions, which LLVM builds without.
set(LLVM_REQUIRES_RTTI ON)
set(LLVM_REQUIRES_EH ON)

foreach(accelerator ${SODA_AXI_SYSC_ACCELERATORS})
  set(name soda_axi_runtime_sysc_${accelerator})
  add_mlir_library(${name}
    SHARED
    ../AxiRuntime.cpp
    ../SyscBackend.cpp
    ${accelerator}.cpp

    EXCLUDE_FROM_LIBMLIR

    LINK_LIBS PRIVATE
    SystemC::SystemC
    )
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  # The models use SC_HAS_PROCESS, which is deprecated in SystemC 3.
  target_compile_definitions(${name} PRIVATE SC_ALLOW_DEPRECATED_IEEE_API)
  # Keep the build output readable: the models are not warning clean.
  target_compile_options(${name} PRIVATE -Wno-unused-parameter
    -Wno-unused-variable -Wno-unused-but-set-variable -Wno-sign-compare
    -Wno-cast-qual -Wno-implicit-fallthrough -Wno-misleading-indentation)
  list(APPEND SODA_AXI_SYSC_RUNTIME_LIBS ${name})
endforeach()

set(SODA_AXI_SYSC_RUNTIME_LIBS ${SODA_AXI_SYSC_RUNTIME_LIBS}
  CACHE INTERNAL "SystemC AXI runtime libraries")
