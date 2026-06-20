add_executable(test_wcon test_wcon.cpp)
target_link_libraries(test_wcon PRIVATE wormsim2_body)

add_test(NAME test_wcon COMMAND test_wcon)
set_tests_properties(test_wcon PROPERTIES LABELS "output")
