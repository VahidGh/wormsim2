#include "body/FEMBody.h"

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/grid_tools.h>

#include <cmath>
#include <stdexcept>
#include <numeric>

namespace wormsim2 {

void FEMBody::init(const FEMBodyParams& p)
{
    params_ = p;

    // Build mesh.
    mesh_.build();
    const auto& tria = mesh_.triangulation();

    // Set up DoF handler: FESystem of 3 × FE_SimplexP<3>(1) = P1 vector field.
    fe_ = std::make_unique<dealii::FESystem<3>>(dealii::FE_SimplexP<3>(1), 3);
    dof_handler_.reinit(tria);
    dof_handler_.distribute_dofs(*fe_);

    // Sparsity pattern.
    {
        dealii::DynamicSparsityPattern dsp(dof_handler_.n_dofs());
        dealii::DoFTools::make_sparsity_pattern(dof_handler_, dsp);
        sparsity_.copy_from(dsp);
    }
    K_.reinit(sparsity_);

    // Extract rest-pose vertex positions and build per-tet data.
    extract_positions_from_mesh();
    build_elements();

    // Build muscle map.
    muscle_map_.build(mesh_);

    // Assemble linear elastic stiffness at rest pose, build dof maps, apply BCs.
    assemble_stiffness();
    build_dof_maps();

    // Apply penalty BCs: clamp all vertices at x > 0.95·L (tail segment).
    // Pins all 6 rigid-body modes and mimics the tail adhering to agar.
    // Penalty 1e20 >> K_elastic (~0.01) forces u_pin → 0.
    {
        constexpr double kPenalty = 1.0e20;
        const double kLPin = 0.95 * FEMBodyMesh::kLength;
        const int nv = static_cast<int>(pos_rest_.size());
        pin_dofs_.clear();
        for (int i = 0; i < nv; ++i) {
            if (pos_rest_[i][0] < kLPin) continue;
            for (int c = 0; c < 3; ++c) {
                const auto dof = vertex_dof_[i][c];
                K_.add(dof, dof, kPenalty);
                pin_dofs_.push_back(dof);
            }
        }
    }

    solver_.initialize(K_);
}

void FEMBody::extract_positions_from_mesh()
{
    const auto& verts = mesh_.triangulation().get_vertices();
    const int nv = static_cast<int>(verts.size());
    pos_rest_.resize(nv);
    pos_.resize(nv);
    vel_.resize(nv, dealii::Tensor<1,3>{});
    for (int i = 0; i < nv; ++i) {
        for (int d = 0; d < 3; ++d)
            pos_rest_[i][d] = verts[i][d];
        pos_[i] = pos_rest_[i];
    }
}

void FEMBody::build_elements()
{
    const auto& tria = mesh_.triangulation();
    elements_.clear();
    tet_verts_.clear();
    const LameParams mat = LameParams::from_E_nu(params_.E, params_.nu);

    for (const auto& cell : tria.active_cell_iterators()) {
        if (cell->n_vertices() != 4)
            throw std::runtime_error("FEMBody: non-tet cell encountered");

        std::array<dealii::Tensor<1,3>, 4> rp;
        std::array<int, 4> vi;
        for (unsigned int v = 0; v < 4; ++v) {
            vi[v] = static_cast<int>(cell->vertex_index(v));
            rp[v] = pos_rest_[vi[v]];
        }
        CorotatedElasticElement elem;
        elem.init(rp, mat);
        elements_.push_back(std::move(elem));
        tet_verts_.push_back(vi);
    }
}

void FEMBody::build_dof_maps()
{
    const int nv = static_cast<int>(pos_.size());
    vertex_dof_.assign(nv, {dealii::numbers::invalid_dof_index,
                             dealii::numbers::invalid_dof_index,
                             dealii::numbers::invalid_dof_index});
    node_vol_.assign(nv, 0.0);

    std::vector<dealii::types::global_dof_index> local_dof_indices(fe_->dofs_per_cell);
    for (const auto& cell : dof_handler_.active_cell_iterators()) {
        cell->get_dof_indices(local_dof_indices);
        const double vol_e_4 = cell->measure() / 4.0;
        for (unsigned int i = 0; i < fe_->dofs_per_cell; ++i) {
            const auto [comp, shape_i] = fe_->system_to_component_index(i);
            const int vi = static_cast<int>(cell->vertex_index(shape_i));
            vertex_dof_[vi][comp] = local_dof_indices[i];
        }
        for (unsigned int v = 0; v < cell->n_vertices(); ++v)
            node_vol_[cell->vertex_index(v)] += vol_e_4;
    }

}

void FEMBody::assemble_stiffness()
{
    // Assemble the linearised stiffness at rest pose (F=I, R=I, S=I).
    // K_e[i][j] = V_e * B_e^T * C * B_e  where C is the elasticity tensor.
    // For P1 tet, B is constant per element.
    // We compute K_e analytically from Dm_inv and Lamé parameters.
    K_ = 0.0;

    const LameParams mat = LameParams::from_E_nu(params_.E, params_.nu);

    // MappingFE is required for simplex (tet) meshes.
    // Default MappingQ1 targets hexahedra and gives wrong Jacobians on tets.
    const dealii::MappingFE<3> mapping(dealii::FE_SimplexP<3>(1));

    dealii::QGaussSimplex<3>   quadrature(2);
    dealii::FEValues<3>        fe_values(mapping, *fe_,
                                         quadrature,
                                         dealii::update_gradients |
                                         dealii::update_JxW_values);
    const unsigned int dofs_per_cell = fe_->dofs_per_cell;
    dealii::FullMatrix<double> Ke(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto& cell : dof_handler_.active_cell_iterators()) {
        fe_values.reinit(cell);
        Ke = 0.0;

        for (unsigned int q = 0; q < quadrature.size(); ++q) {
            const double JxW = fe_values.JxW(q);
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                // Component and gradient of shape function i.
                const int ci = fe_->system_to_component_index(i).first;
                const auto& grad_i = fe_values.shape_grad(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    const int cj = fe_->system_to_component_index(j).first;
                    const auto& grad_j = fe_values.shape_grad(j, q);

                    // Isotropic linear elastic stiffness:
                    // K[i][j] = µ * (∇φ_i · ∇φ_j) * δ_{ci,cj}
                    //          + µ * ∂φ_i/∂x_{cj} * ∂φ_j/∂x_{ci}
                    //          + λ * ∂φ_i/∂x_{ci} * ∂φ_j/∂x_{cj}
                    double val = mat.mu * (grad_i * grad_j) * (ci == cj ? 1.0 : 0.0)
                               + mat.mu * grad_i[cj] * grad_j[ci]
                               + mat.lambda * grad_i[ci] * grad_j[cj];
                    Ke(i, j) += val * JxW;
                }
            }
        }
        // Add drag contribution: K_drag = (γ/dt) * M_e (lumped mass).
        // Lumped: each DOF gets V_e/4 * γ/dt.
        const double vol_e = cell->measure();
        const double diag_drag = vol_e / 4.0 * params_.gamma_drag / params_.dt;
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            Ke(i, i) += diag_drag;

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                K_.add(local_dof_indices[i], local_dof_indices[j], Ke(i,j));
    }

    // Boundary conditions are applied in init() after build_dof_maps() populates
    // vertex_dof_.  Nothing to do here.
}

void FEMBody::step(const std::array<double, kNMuscles>& activation)
{
    const int nv    = static_cast<int>(pos_.size());
    const int ndof  = static_cast<int>(dof_handler_.n_dofs());
    const double gamma_over_dt = params_.gamma_drag / params_.dt;

    // Muscle active forces (per vertex).
    std::vector<dealii::Tensor<1,3>> f_muscle(nv);
    muscle_map_.add_active_forces(activation, pos_, f_muscle);

    // Build RHS: backward-Euler overdamped dynamics.
    // K_total = K_elastic + K_drag; K_elastic handles elastic restoring implicitly.
    // RHS = K_drag * u_old + f_muscle  (no separate f_elastic — that would double-count).
    // At equilibrium: K_elastic * u* = f_muscle  ✓
    dealii::Vector<double> rhs(ndof);
    for (int vi = 0; vi < nv; ++vi) {
        if (vertex_dof_[vi][0] == dealii::numbers::invalid_dof_index) continue;
        const double drag = node_vol_[vi] * gamma_over_dt;
        for (int comp = 0; comp < 3; ++comp) {
            const double u_old = pos_[vi][comp] - pos_rest_[vi][comp];
            rhs[vertex_dof_[vi][comp]] = drag * u_old + f_muscle[vi][comp];
        }
    }

    // Enforce pinned DOFs: zero their rhs so u_pin = 0 exactly (penalty already large).
    for (auto dof : pin_dofs_)
        rhs[dof] = 0.0;

    // Solve: (K_elastic + K_drag) * u_new = rhs.
    dealii::Vector<double> u_new(ndof);
    solver_.vmult(u_new, rhs);

    // Update absolute positions: x_new = x_rest + u_new.
    // Skip orphan vertices (invalid DOF) — their pos_ stays at rest.
    for (int vi = 0; vi < nv; ++vi) {
        if (vertex_dof_[vi][0] == dealii::numbers::invalid_dof_index) continue;
        for (int comp = 0; comp < 3; ++comp)
            pos_[vi][comp] = pos_rest_[vi][comp] + u_new[vertex_dof_[vi][comp]];
    }
}

std::vector<CenterlinePoint> FEMBody::centerline() const
{
    // Extract midline by grouping vertices into the 9 hex axial planes
    // (n_axial=8 → planes at s = 0, 1/8, ..., 1) then linearly interpolating
    // to ns+1 = 25 output points.  Direct 1/24-bin assignment gives outlier
    // bins from intermediate cap vertices that cause spurious zigzags.
    const int ns     = FEMBodyMesh::kNSegments; // 24
    const int np     = 9;                        // hex axial planes: 0..8

    std::vector<CenterlinePoint> planes(np);
    std::vector<int>             cnt(np, 0);

    for (int i = 0; i < static_cast<int>(pos_.size()); ++i) {
        const double s = mesh_.vertex_data()[i].s;
        const int k = std::min(np - 1, (int)std::round(s * (np - 1)));
        planes[k].s += s;
        planes[k].x += pos_[i][0];
        planes[k].y += pos_[i][1];
        planes[k].z += pos_[i][2];
        ++cnt[k];
    }
    for (int k = 0; k < np; ++k) {
        if (cnt[k] > 0) {
            const double inv = 1.0 / cnt[k];
            planes[k].s *= inv; planes[k].x *= inv;
            planes[k].y *= inv; planes[k].z *= inv;
        } else {
            planes[k].s = static_cast<double>(k) / (np - 1);
            planes[k].x = planes[k].s * FEMBodyMesh::kLength;
        }
    }

    // Linearly interpolate to ns+1 output points.
    std::vector<CenterlinePoint> cl(ns + 1);
    for (int j = 0; j <= ns; ++j) {
        const double s_j  = static_cast<double>(j) / ns;
        const double frac = s_j * (np - 1);
        const int    lo   = std::min((int)frac, np - 2);
        const double a    = frac - lo;
        cl[j].s = s_j;
        cl[j].x = (1-a)*planes[lo].x + a*planes[lo+1].x;
        cl[j].y = (1-a)*planes[lo].y + a*planes[lo+1].y;
        cl[j].z = (1-a)*planes[lo].z + a*planes[lo+1].z;
    }
    return cl;
}

int FEMBody::n_vertices() const
{
    return static_cast<int>(pos_.size());
}

} // namespace wormsim2
