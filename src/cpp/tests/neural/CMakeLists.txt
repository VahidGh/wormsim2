add_executable(test_hh_single   test_hh_single.cpp)
add_executable(test_gap_junction test_gap_junction.cpp)
add_executable(test_nmj          test_nmj.cpp)

foreach(_t test_hh_single test_gap_junction test_nmj)
    target_link_libraries(${_t} PRIVATE wormsim2_neural)
    add_test(NAME ${_t} COMMAND ${_t})
    set_tests_properties(${_t} PROPERTIES LABELS "neural" TIMEOUT 30)
endforeach()
