# ---------------------------------------------------------------------------
# wormsim2 C++ core
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# io — network model loading (NEURON + NeuroML2) and perturbation config
# ---------------------------------------------------------------------------
add_library(wormsim2_io STATIC
    src/io/NetworkConfig.cpp
    src/io/NetworkInputParser.cpp
    src/io/NEURONLoader.cpp
    src/io/NeuroMLLoader.cpp
    src/io/PerturbationConfig.cpp
)
target_include_directories(wormsim2_io PUBLIC include)
target_compile_features(wormsim2_io PUBLIC cxx_std_20)

# ---------------------------------------------------------------------------
# neural — HH ODE integrator (Rush–Larsen gates, conductance method voltages)
# ---------------------------------------------------------------------------
add_library(wormsim2_neural STATIC
    src/neural/ChannelKinetics.cpp
    src/neural/NeuralState.cpp
    src/neural/NeuralIntegrator.cpp
)
target_include_directories(wormsim2_neural PUBLIC include)
target_link_libraries(wormsim2_neural PUBLIC wormsim2_io)
target_compile_features(wormsim2_neural PUBLIC cxx_std_20)

# ---------------------------------------------------------------------------
# body — eigenworm reduced-order body + WCON output
# ---------------------------------------------------------------------------
add_library(wormsim2_body STATIC
    src/body/EigenwormBody.cpp
    src/output/WCONExporter.cpp
)
target_include_directories(wormsim2_body PUBLIC include)
target_link_libraries(wormsim2_body PUBLIC wormsim2_neural)
target_compile_features(wormsim2_body PUBLIC cxx_std_20)

# ---------------------------------------------------------------------------
# Tools (notebook data runners — always built)
# ---------------------------------------------------------------------------
add_subdirectory(tools)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(WORMSIM2_BUILD_TESTS)
    add_subdirectory(tests)
endif()
