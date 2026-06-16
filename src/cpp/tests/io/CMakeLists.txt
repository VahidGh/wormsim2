add_executable(test_nmodl_parser test_nmodl_parser.cpp)
target_link_libraries(test_nmodl_parser PRIVATE wormsim2_io)
add_test(NAME nmodl_parser COMMAND test_nmodl_parser)

add_executable(test_perturbation test_perturbation.cpp)
target_link_libraries(test_perturbation PRIVATE wormsim2_io)
add_test(NAME perturbation COMMAND test_perturbation)

add_executable(test_connectome_load test_connectome_load.cpp)
target_link_libraries(test_connectome_load PRIVATE wormsim2_neural)
add_test(NAME connectome_load COMMAND test_connectome_load)
set_tests_properties(connectome_load PROPERTIES
    LABELS "io"
    TIMEOUT 60
    ENVIRONMENT "WORMSIM2_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
)
