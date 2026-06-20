add_executable(test_eigenworm test_eigenworm.cpp)
target_link_libraries(test_eigenworm PRIVATE wormsim2_body)
target_compile_definitions(test_eigenworm PRIVATE
    WORMSIM2_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_test(NAME test_eigenworm COMMAND test_eigenworm)
set_tests_properties(test_eigenworm PROPERTIES LABELS "body")
