add_mlir_library(soda_runner_ext
  SHARED
  SODARunnerExtensions.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/soda/ExecutionEngine
)
set_property(TARGET soda_runner_ext PROPERTY CXX_STANDARD 17)
target_compile_definitions(soda_runner_ext PUBLIC soda_runner_ext_EXPORTS)
