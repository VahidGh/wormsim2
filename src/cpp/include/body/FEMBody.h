#pragma once
// FEMBody — corotated linear-elastic FEM body for C. elegans locomotion.
// Matches MetaWorm's body model (BAAIWorm, Nat. Comput. Sci. 2024):
//   ~984 vertices / ~3341 tetrahedra, 95 BWM activation constraints,
//   corotated linear-elastic material (E=100 Pa, ν=0.30).
//
// Dynamics: overdamped (C. elegans Re ≈ 10⁻³).  Each step solves:
//   γ * (x_new - x_old) / dt = f_muscle(a) + f_elastic(x_old) + f_boundary
// where γ is the agar drag coefficient.  This is quasi-static implicit Euler
// (one linear solve per timestep, mass-less, viscosity-dominated).
//
// deal.II 9.5.1: FE_SimplexP<3>(1) P1 linear tets, UMFPACK direct solve.

#include "FEMBodyMesh.h"
#include "MuscleMap.h"
#include "CorotatedElastic.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_direct.h>

#include <array>
#include <memory>
#include <vector>
#include <cstdint>

namespace wormsim2 {

// Simulation parameters for the FEM body step.
struct FEMBodyParams {
    double dt         = 0.5e-3;  // timestep [s] — 0.5 ms
    // Volumetric drag coefficient [Pa·s/m²].
    // γ = 10.0 gives K_drag_ii ≈ 3e-11 N/m << K_elastic_ii ≈ 9.4e-5 N/m
    // (quasi-static, body equilibrates within ~1 step) while regularising the
    // K matrix condition number to ~3×10⁶ (safe for UMFPACK double precision).
    double gamma_drag = 10.0;
    double E          = kBodyE;  // Young's modulus [Pa]
    double nu         = kBodyNu; // Poisson ratio
};

// Per-vertex centerline output (head=0 → tail=1 in s).
struct CenterlinePoint {
    double s;          // arc-length parameter [0,1]
    double x, y, z;   // position [m]
    double kappa;      // local curvature κ(s) [m⁻¹] (filled by CurvatureSensor)
};

class FEMBody {
public:
    // Build mesh, assemble stiffness, factorise.  Call once.
    void init(const FEMBodyParams& p = FEMBodyParams{});

    // Advance one timestep.
    // activation[0..94]: per-muscle NMJ activation ∈ [0,1].
    void step(const std::array<double, kNMuscles>& activation);

    // Current vertex positions (length = n_vertices()).
    const std::vector<dealii::Tensor<1,3>>& positions() const { return pos_; }

    // Centerline extracted from body midline vertices at current step.
    // Returns kNSegments+1 points from head to tail.
    std::vector<CenterlinePoint> centerline() const;

    int n_vertices() const;

    const FEMBodyMesh& mesh() const { return mesh_; }

private:
    FEMBodyParams                        params_;
    FEMBodyMesh                          mesh_;
    MuscleMap                            muscle_map_;
    std::vector<CorotatedElasticElement> elements_;

    // deal.II FE infrastructure (displacement field, 3 DOFs per vertex).
    // FESystem is non-copyable/non-movable after construction — use unique_ptr.
    std::unique_ptr<dealii::FESystem<3>> fe_;
    dealii::DoFHandler<3>                dof_handler_;
    dealii::SparsityPattern              sparsity_;
    dealii::SparseMatrix<double>         K_;    // linear elastic stiffness (rest pose)
    dealii::SparseDirectUMFPACK          solver_;

    // Vertex position arrays (SI metres).
    std::vector<dealii::Tensor<1,3>>     pos_;      // current deformed positions
    std::vector<dealii::Tensor<1,3>>     pos_rest_; // rest-pose positions
    std::vector<dealii::Tensor<1,3>>     vel_;      // velocities (for damping)

    // Precomputed per-element vertex index lists for fast force assembly.
    std::vector<std::array<int,4>>       tet_verts_;

    // vertex_dof_[vi][comp] = global DOF index for vertex vi, component comp.
    std::vector<std::array<dealii::types::global_dof_index, 3>> vertex_dof_;
    // node_vol_[vi] = sum of (V_e/4) over cells sharing vertex vi [m³].
    std::vector<double>                  node_vol_;
    // Penalty-pinned DOFs (rigid-body suppression).  Set in init(), zeroed in step().
    std::vector<dealii::types::global_dof_index> pin_dofs_;

    void build_elements();
    void assemble_stiffness();
    void extract_positions_from_mesh();
    void build_dof_maps();

};

} // namespace wormsim2
