cmake_minimum_required(VERSION 3.22)

# ---------------------------------------------------------------------------
# deal.II must be found and initialised BEFORE project() — deal.II requirement.
# ---------------------------------------------------------------------------
find_package(deal.II 9.5 QUIET
    HINTS
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.5.1
        $ENV{DEAL_II_DIR}
)
if(deal.II_FOUND)
    DEAL_II_INITIALIZE_CACHED_VARIABLES()
    set(WORMSIM2_HAS_DEALII TRUE)
else()
    set(WORMSIM2_HAS_DEALII FALSE)
endif()

project(wormsim2 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(WORMSIM2_BUILD_TESTS   "Build CTest test suite"                  ON)
set(WORMSIM2_BACKEND "CPU" CACHE STRING "Compute backend: CPU | CUDA | OPENCL")

if(WORMSIM2_BUILD_TESTS)
    enable_testing()
endif()
set_property(CACHE WORMSIM2_BACKEND PROPERTY STRINGS CPU CUDA OPENCL)

# ---------------------------------------------------------------------------
# Compiler warnings  (NFR-QUAL-01)
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter   # suppressed: many pure-virtual stubs have unused args
    )
endif()
if(MSVC)
    add_compile_options(/W4 /permissive-)
endif()

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(src/cpp)
