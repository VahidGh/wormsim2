#include "body/FEMBodyMesh.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <cmath>
#include <stdexcept>

namespace wormsim2 {

double FEMBodyMesh::radius_at(double s)
{
    // Linear taper at head (s<0.1) and tail (s>0.9), uniform mid-body.
    double r = kMaxRadius;
    if (s < 0.1)
        r *= (kTaperFrac + (1.0 - kTaperFrac) * (s / 0.1));
    else if (s > 0.9)
        r *= (kTaperFrac + (1.0 - kTaperFrac) * ((1.0 - s) / 0.1));
    return r;
}

void FEMBodyMesh::build()
{
    // Step 1: Generate hexahedral subdivided cylinder (deal.II native).
    // subdivided_cylinder(tria, n_axial, radius, half_length) places the
    // cylinder axis along x, centred at the origin.
    // Target after hex→tet conversion: ~984 vertices / ~3341 tets.
    // subdivided_cylinder with n_axial=8 → 8 axial × 8 angular cross-section
    // hex cells → after convert_hypercube_to_simplex_mesh: calibrated to MetaWorm scale.
    dealii::Triangulation<3> hex_tria;
    const double half_len = kLength / 2.0;
    dealii::GridGenerator::subdivided_cylinder(hex_tria,
                                               /*n_axial_cells=*/ 8,
                                               /*radius=*/        kMaxRadius,
                                               /*half_length=*/   half_len);

    // Step 2: Convert hex → simplex (tet) mesh (deal.II 9.3+).
    dealii::GridGenerator::convert_hypercube_to_simplex_mesh(hex_tria, tria_);

    // Step 3: Reattach cylindrical manifold (lost during simplex conversion).
    // cylinder() axis is x; CylindricalManifold needs explicit direction.
    const dealii::Tensor<1,3> axis({1.0, 0.0, 0.0});
    const dealii::Point<3>    axis_pt(0.0, 0.0, 0.0);
    tria_.set_manifold(0, dealii::CylindricalManifold<3>(axis, axis_pt));
    for (auto& cell : tria_.active_cell_iterators())
        cell->set_manifold_id(0);

    // Step 4: Apply body taper and shift x: [-L/2, L/2] → [0, L].
    // Use GridTools::transform for proper vertex manipulation.
    const double half_len_cap = half_len; // capture
    dealii::GridTools::transform(
        [half_len_cap](const dealii::Point<3>& pt) -> dealii::Point<3> {
            const double s = (pt[0] + half_len_cap) / (2.0 * half_len_cap);
            const double r = std::sqrt(pt[1]*pt[1] + pt[2]*pt[2]);
            const double r_max = kMaxRadius;
            const double scale = (r > 1.0e-12) ? (radius_at(s) / r_max) : 1.0;
            return dealii::Point<3>(pt[0] + half_len_cap,
                                    pt[1] * scale,
                                    pt[2] * scale);
        },
        tria_);

    // Step 6: Annotate vertices with arc-length and quadrant.
    assign_vertex_data();
}

void FEMBodyMesh::apply_taper()
{
    // Taper is now applied inline in build() via GridTools::transform.
}

void FEMBodyMesh::assign_vertex_data()
{
    const auto& verts = tria_.get_vertices();
    const std::size_t nv = verts.size();
    vdata_.resize(nv);

    for (std::size_t i = 0; i < nv; ++i) {
        const auto& pt = verts[i];
        const double s = pt[0] / kLength;
        vdata_[i].s = s;

        // Quadrant from angular position (y=dorsal, z=left in body frame).
        const double theta = std::atan2(pt[2], pt[1]); // ∈ (-π, π]
        if (theta >= 0.0 && theta < M_PI / 2.0)
            vdata_[i].quad = Quadrant::DL;
        else if (theta >= M_PI / 2.0)
            vdata_[i].quad = Quadrant::DR;
        else if (theta < 0.0 && theta >= -M_PI / 2.0)
            vdata_[i].quad = Quadrant::VL;
        else
            vdata_[i].quad = Quadrant::VR;

        // Surface detection: radius within 5% of expected taper radius.
        const double r = std::sqrt(pt[1]*pt[1] + pt[2]*pt[2]);
        vdata_[i].on_surface = (r > 0.90 * radius_at(std::max(0.0, std::min(1.0, s))));
    }
}

} // namespace wormsim2
