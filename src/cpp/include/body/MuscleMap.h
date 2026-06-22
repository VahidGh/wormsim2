#pragma once
// MuscleMap — maps the 95 C. elegans body-wall muscles (BWMs) to surface
// tetrahedra in the FEMBodyMesh.
//
// Layout: 4 longitudinal quadrants × ~24 body segments = 95 muscles total:
//   DL (dorsal-left)   BWM ids  0..23   (24 muscles)
//   DR (dorsal-right)  BWM ids 24..47   (24 muscles)
//   VL (ventral-left)  BWM ids 48..71   (24 muscles)
//   VR (ventral-right) BWM ids 72..94   (23 muscles)
//
// Active stress model (MetaWorm-style):
//   τ_active = a_m * T_max * (f ⊗ f),  f = +x (longitudinal fibre)
//   T_max = 10 Pa (Petzold et al. 2011; Boyle & Cohen 2008).

#include "FEMBodyMesh.h"
#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>
#include <vector>
#include <array>

namespace wormsim2 {

static constexpr int    kNMuscles = 95;
static constexpr double kTmax     = 10.0; // Pa, peak isometric tension

// A single BWM region: set of surface-tet cell indices in the triangulation.
struct MuscleRegion {
    int                 muscle_id;  // 0..94
    Quadrant            quad;
    int                 segment;    // 0..23
    std::vector<int>    cell_ids;   // active_cell_index() values
};

class MuscleMap {
public:
    using Vec3 = dealii::Tensor<1, 3>;

    // Build from a fully constructed FEMBodyMesh.
    void build(const FEMBodyMesh& mesh);

    int n_muscles() const { return kNMuscles; }

    // Given per-muscle activation a[0..94] and current vertex positions,
    // add active contractile forces into f_out (same indexing as vertex array).
    // Uses the precomputed per-cell vertex lists and volumes from build().
    void add_active_forces(const std::array<double, kNMuscles>& activation,
                           const std::vector<Vec3>&             cur_pos,
                           std::vector<Vec3>&                   f_out) const;

    // muscle_id from quadrant + segment index.
    static int muscle_id(Quadrant q, int seg);
    static Quadrant quadrant_of(int muscle_id);
    static int      segment_of (int muscle_id);

private:
    // Per-cell precomputed data (built once from mesh).
    struct CellData {
        int                    muscle_id;
        std::array<int, 4>     vi;      // vertex indices of the tet
        double                 vol;     // rest volume
        dealii::Tensor<2, 3>   Dm_inv; // inverse rest-edge matrix — needed for active stress
    };

    std::vector<MuscleRegion> regions_;
    std::vector<CellData>     cell_data_; // only for surface muscle tets

    static Quadrant quadrant_from_angle(double theta); // theta ∈ (-π, π]
};

} // namespace wormsim2
