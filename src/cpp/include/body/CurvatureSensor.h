#pragma once
// CurvatureSensor — computes local body curvature κ(s) from the FEM centerline
// and converts it to proprioceptive stretch-receptor activation following
// Boyle, Berri & Cohen (2012) Front. Comput. Neurosci. 6:10.
//
// Curvature: κ(s_j) = |dT/ds| where T(s) = dx/ds is the unit tangent.
// Stretch-receptor activation:
//   α_D(j) = max(0,  κ(s_j) − κ_thresh)   (dorsal SR — excited by D-bend)
//   α_V(j) = max(0, -κ(s_j) − κ_thresh)   (ventral SR — excited by V-bend)
// Motor feedback: α_D(j) drives VB motoneurons at segment j+1 (forward coupling)
//                 α_V(j) drives DB motoneurons at segment j+1.
// This creates a traveling wave: curvature at s propagates excitation to s+Δs,
// generating the 90° phase lag between eigenworm modes a0 and a1.

#include "FEMBody.h"
#include <deal.II/base/tensor.h>
#include <vector>
#include <array>

namespace wormsim2 {

// SR model parameters (Boyle & Cohen 2012, Table 1 defaults).
struct SRParams {
    double kappa_thresh = 50.0;   // m⁻¹ — threshold curvature below which SR is silent
    double g_SR         = 0.5e-9; // S — max SR conductance injected per segment
    int    delta_seg    = 2;      // forward coupling offset in segments (≈head-to-tail)
};

// Output: per-motoneuron-segment stretch-receptor conductance to inject.
// Indexed as [segment 0..23][0=dorsal SR, 1=ventral SR].
using SRActivation = std::array<std::array<double, 2>, 24>;

class CurvatureSensor {
public:
    explicit CurvatureSensor(const SRParams& p = SRParams{}) : params_(p) {}

    // Compute κ(s) at each of the kNSegments centerline points and update
    // the centerline's kappa field in-place.
    static void compute_curvature(std::vector<CenterlinePoint>& cl);

    // Compute SR activation from a curvature-stamped centerline.
    // Returns conductance to inject into each [segment][D/V] motor input.
    SRActivation compute_sr(const std::vector<CenterlinePoint>& cl) const;

private:
    SRParams params_;
};

} // namespace wormsim2
