#pragma once
// FEMBodyMesh — tapered-cylinder tetrahedral mesh of the C. elegans body.
// Geometry: length 1.0 mm, max radius 0.040 mm (80 µm diameter), linear taper
// at head (x<0.1) and tail (x>0.9) to 50% radius — matching worm cross-section
// measurements (Fang-Yen et al. 2010).
// Mesh scale: ~984 vertices / ~3341 tetrahedra (MetaWorm BAAIWorm 2024 parity).
// Uses deal.II 9.5.1: GridGenerator::cylinder → convert_hypercube_to_simplex_mesh.

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/base/point.h>
#include <array>
#include <vector>

namespace wormsim2 {

// Muscle quadrant index (matching NMJDef muscle_id ordering).
enum class Quadrant : int { DL = 0, DR = 1, VL = 2, VR = 3 };

// Per-vertex body-space data attached after mesh build.
struct VertexData {
    double s;        // arc-length parameter ∈ [0,1] (head=0, tail=1)
    Quadrant quad;   // which muscle quadrant this surface vertex belongs to
    bool on_surface; // true if vertex is on the outer cylinder wall
};

class FEMBodyMesh {
public:
    // Body geometry (SI: metres).
    static constexpr double kLength    = 1.0e-3;  // 1 mm
    static constexpr double kMaxRadius = 0.040e-3; // 40 µm half-diameter
    // Taper: head (s<0.1) and tail (s>0.9) shrink to kTaperFrac * kMaxRadius.
    static constexpr double kTaperFrac = 0.50;
    // Longitudinal segments (matching 24-segment motor neuron column).
    static constexpr int kNSegments = 24;

    // Build the triangulation.  Call once at startup.
    // After build(), triangulation() and vertex_data() are valid.
    void build();

    const dealii::Triangulation<3>& triangulation() const { return tria_; }
    dealii::Triangulation<3>&       triangulation()       { return tria_; }
    const std::vector<VertexData>&  vertex_data()   const { return vdata_; }

    // Radius at normalised arc-length s ∈ [0,1].
    static double radius_at(double s);

private:
    dealii::Triangulation<3>  tria_;
    std::vector<VertexData>   vdata_;

    void apply_taper();
    void assign_vertex_data();
};

} // namespace wormsim2
