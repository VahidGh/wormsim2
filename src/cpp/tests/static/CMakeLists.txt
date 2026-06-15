find_package(Python3 REQUIRED)

add_test(
    NAME    cpp_quality
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/src/cpp/tests/static/cpp_quality_checker.py
            ${CMAKE_SOURCE_DIR}
)
set_tests_properties(cpp_quality PROPERTIES
    LABELS "static"
    TIMEOUT 30
)
