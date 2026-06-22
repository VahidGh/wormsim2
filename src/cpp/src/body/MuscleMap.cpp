#include "body/MuscleMap.h"

#include <deal.II/grid/tria.h>
#include <cmath>
#include <stdexcept>

// Anonymous namespace: avoids file-scope static (PS-02), same linkage.
// muscle_id = quad_offset + segment:
//   DL 0..23, DR 24..47, VL 48..71, VR 72..94
namespace {
constexpr int kQuadOffset[4] = {0, 24, 48, 72};
constexpr int kQuadCount[4]  = {24, 24, 24, 23};
} // namespace

namespace wormsim2 {

int MuscleMap::muscle_id(Quadrant q, int seg)
{
    const int qi = static_cast<int>(q);
    return kQuadOffset[qi] + seg;
}

Quadrant MuscleMap::quadrant_of(int mid)
{
    for (int q = 3; q >= 0; --q)
        if (mid >= kQuadOffset[q]) return static_cast<Quadrant>(q);
    return Quadrant::DL;
}

int MuscleMap::segment_of(int mid)
{
    return mid - kQuadOffset[static_cast<int>(quadrant_of(mid))];
}

Quadrant MuscleMap::quadrant_from_angle(double theta)
{
    // theta = atan2(z, y): y=dorsal(+), z=left(+)
    // DL: θ ∈ [0,   π/2)
    // DR: θ ∈ [π/2, π]   (or wrapping: |θ|>π/2 && y>0)
    // VL: θ ∈ (-π/2, 0)
    // VR: θ ∈ [-π, -π/2]
    if (theta >= 0.0 && theta <  M_PI / 2.0) return Quadrant::DL;
    if (theta >= M_PI / 2.0)                  return Quadrant::DR;
    if (theta < 0.0 && theta >= -M_PI / 2.0) return Quadrant::VL;
    return Quadrant::VR;
}

void MuscleMap::build(const FEMBodyMesh& mesh_obj)
{
    const auto& tria  = mesh_obj.triangulation();

    // Initialise muscle regions.
    regions_.resize(kNMuscles);
    for (int m = 0; m < kNMuscles; ++m) {
        regions_[m].muscle_id = m;
        regions_[m].quad      = quadrant_of(m);
        regions_[m].segment   = segment_of(m);
    }

    // Assign each surface cell (tet with ≥1 vertex on surface) to a muscle.
    cell_data_.clear();
    for (const auto& cell : tria.active_cell_iterators()) {
        // Check if this tet has a face on the outer surface (boundary_id != 0
        // is used by GridGenerator::cylinder for the curved wall).
        bool is_surface = false;
        for (unsigned int f = 0; f < cell->n_faces(); ++f)
            if (cell->face(f)->at_boundary() &&
                cell->face(f)->boundary_id() == 0)
            { is_surface = true; break; }
        if (!is_surface) continue;

        // Determine muscle from centroid arc-length and angle.
        dealii::Point<3> centroid;
        for (unsigned int v = 0; v < cell->n_vertices(); ++v)
            for (int d = 0; d < 3; ++d)
                centroid[d] += cell->vertex(v)[d] / cell->n_vertices();

        const double s     = centroid[0] / FEMBodyMesh::kLength;
        const double theta = std::atan2(centroid[2], centroid[1]);
        const Quadrant q   = quadrant_from_angle(theta);
        const int qi       = static_cast<int>(q);

        // Map s ∈ [0,1] → segment ∈ [0, kNSegments-1].
        int seg = static_cast<int>(s * FEMBodyMesh::kNSegments);
        seg = std::max(0, std::min(kQuadCount[qi] - 1, seg));

        const int mid = muscle_id(q, seg);
        regions_[mid].cell_ids.push_back(cell->active_cell_index());

        // Collect per-cell vertex indices and rest volume.
        CellData cd;
        cd.muscle_id = mid;
        for (unsigned int v = 0; v < 4; ++v)
            cd.vi[v] = static_cast<int>(cell->vertex_index(v));
        // Volume = det(Ds)/6 where Ds = edge matrix at rest pose.
        // Use vertex positions from the triangulation (rest pose).
        std::array<dealii::Tensor<1,3>, 4> rest;
        for (unsigned int v = 0; v < 4; ++v)
            for (int d = 0; d < 3; ++d)
                rest[v][d] = cell->vertex(v)[d];
        dealii::Tensor<2,3> Ds;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                Ds[i][j] = rest[j+1][i] - rest[0][i];
        cd.vol    = std::abs(dealii::determinant(Ds)) / 6.0;
        cd.Dm_inv = dealii::invert(Ds);  // for active-stress force
        cell_data_.push_back(cd);
    }
}

void MuscleMap::add_active_forces(const std::array<double, kNMuscles>& activation,
                                   const std::vector<Vec3>&             /*cur_pos*/,
                                   std::vector<Vec3>&                   f_out) const
{
    // Active stress: P_active = a * T_max * (e_x ⊗ e_x).
    // For a P1 tet, the contractile force on vertex k (k=1..3) is:
    //   F_k[x] = -V_e * a * T_max * Dm_inv[0][k-1]
    // and on vertex 0 (Newton's 3rd law):
    //   F_0[x] = +V_e * a * T_max * Σ_{k=0..2} Dm_inv[0][k]
    // Forces on y and z are zero (active stress along x only).
    // This is self-equilibrating per tet (zero net force) and creates
    // a contractile pair: the "tail-end" vertex is pulled toward the
    // "head-end" vertex, so the dorsal side shortens → C-bend.
    for (const auto& cd : cell_data_) {
        const double a = activation[cd.muscle_id];
        if (a < 1.0e-9) continue;

        const double fscale = cd.vol * a * kTmax;
        double sum_col = 0.0;
        for (int k = 0; k < 3; ++k) {
            const double fk = fscale * cd.Dm_inv[0][k];
            f_out[cd.vi[k + 1]][0] -= fk;
            sum_col += fk;
        }
        f_out[cd.vi[0]][0] += sum_col;
    }
}

} // namespace wormsim2
