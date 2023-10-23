add_subdirectory(soda-opt)
add_subdirectory(soda-translate)
add_subdirectory(mlir-runner)

if (ENABLE_PYTACO)
    add_subdirectory(soda-pytaco)
endif()