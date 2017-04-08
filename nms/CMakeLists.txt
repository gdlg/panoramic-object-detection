
set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/gpu_nms.pyx" "${CMAKE_CURRENT_SOURCE_DIR}/cpu_nms.pyx")
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build")

configure_file(${SETUP_PY_IN} ${SETUP_PY})

add_custom_command(OUTPUT ${OUTPUT}
    COMMAND ${PYTHON_EXECUTABLE}
                   ARGS ${SETUP_PY} build
                   DEPENDS ${DEPS}
                   COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/build/lib.linux-x86_64-2.7/gpu_nms.so ${CMAKE_CURRENT_SOURCE_DIR}/gpu_nms.so
                   COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/build/lib.linux-x86_64-2.7/cpu_nms.so ${CMAKE_CURRENT_SOURCE_DIR}/cpu_nms.so
                   )

add_custom_target(target ALL DEPENDS ${OUTPUT})

