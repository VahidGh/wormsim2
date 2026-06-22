add_executable(test_eigenworm test_eigenworm.cpp)
target_link_libraries(test_eigenworm PRIVATE wormsim2_body)
target_compile_definitions(test_eigenworm PRIVATE
    WORMSIM2_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_test(NAME test_eigenworm COMMAND test_eigenworm)
set_tests_properties(test_eigenworm PROPERTIES LABELS "body")

if(WORMSIM2_HAS_DEALII)
    add_executable(test_fem_body test_fem_body.cpp)
    # Plain (no-keyword) linking — DEAL_II_SETUP_TARGET uses plain style internally.
    target_link_libraries(test_fem_body wormsim2_body_fem)
    DEAL_II_SETUP_TARGET(test_fem_body)

    # MK Docker toolchain libraries are not on the default LD_LIBRARY_PATH.
    # Compute a colon-separated list of all pkg lib directories for the test env.
    set(_MK_PKG_ROOT "/u/sw/toolchains/gcc-glibc/11.2.0/pkgs")
    set(_MK_LIBS
        "${_MK_PKG_ROOT}/dealii/9.5.1/lib"
        "${_MK_PKG_ROOT}/trilinos/15.0.0/lib"
        "${_MK_PKG_ROOT}/boost/1.76.0/lib"
        "${_MK_PKG_ROOT}/arpack/3.8.0/lib"
        "${_MK_PKG_ROOT}/p4est/2.3.2/lib"
        "${_MK_PKG_ROOT}/hdf5/1.12.0/lib"
        "${_MK_PKG_ROOT}/petsc/3.15.1/lib"
        "${_MK_PKG_ROOT}/hypre/2.22.0/lib"
        "${_MK_PKG_ROOT}/mumps/5.4.0/lib"
        "${_MK_PKG_ROOT}/suitesparse/5.10.1/lib"
        "${_MK_PKG_ROOT}/openblas/0.3.15/lib"
        "${_MK_PKG_ROOT}/scalapack/2.1.0/lib"
        "${_MK_PKG_ROOT}/vtk/9.0.3/lib"
        "${_MK_PKG_ROOT}/tbb/2021.4.0/lib"
        "${_MK_PKG_ROOT}/metis/5.1.0/lib"
    )
    string(JOIN ":" _MK_LD_PATH ${_MK_LIBS})

    add_test(NAME test_fem_body COMMAND test_fem_body)
    set_tests_properties(test_fem_body PROPERTIES
        LABELS "body;fem"
        TIMEOUT 300
        ENVIRONMENT "LD_LIBRARY_PATH=${_MK_LD_PATH}")
endif()
