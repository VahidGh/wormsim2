add_executable(neural_trace neural_trace.cpp)
target_link_libraries(neural_trace PRIVATE wormsim2_neural)

add_executable(connectome_trace connectome_trace.cpp)
target_link_libraries(connectome_trace PRIVATE wormsim2_neural)

add_executable(body_trace body_trace.cpp)
target_link_libraries(body_trace PRIVATE wormsim2_body)

if(WORMSIM2_HAS_DEALII)
    add_executable(fem_body_trace fem_body_trace.cpp)
    target_link_libraries(fem_body_trace wormsim2_body_fem)
    DEAL_II_SETUP_TARGET(fem_body_trace)
endif()
