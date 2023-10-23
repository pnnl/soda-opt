# Install Python dependencies
execute_process( 
    COMMAND ${Python3_EXECUTABLE} -m pip install -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
    OUTPUT_VARIABLE _
)

# Install soda-pytaco
#
# We add a separate target even though this is a script to 
# conform to the rest of the build system.
#
# TODO: Streamline, remove chmod +x
add_custom_target(soda-pytaco)
add_custom_command(
    TARGET soda-pytaco
    COMMAND chmod +x ${CMAKE_CURRENT_SOURCE_DIR}/soda-pytaco.py &&
            ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/soda-pytaco.py
            ${CMAKE_BINARY_DIR}/bin/soda-pytaco)