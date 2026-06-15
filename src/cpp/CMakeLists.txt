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
# Tests
# ---------------------------------------------------------------------------
if(WORMSIM2_BUILD_TESTS)
    add_subdirectory(tests)
endif()
