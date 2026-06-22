#include "body/CurvatureSensor.h"
#include <cmath>

namespace wormsim2 {

void CurvatureSensor::compute_curvature(std::vector<CenterlinePoint>& cl)
{
    const int n = static_cast<int>(cl.size());
    if (n < 3) return;

    // Compute arc-length spacing.
    // κ(j) ≈ |T(j+1) - T(j-1)| / (2 * Δs)  where T = unit tangent.
    for (int j = 0; j < n; ++j) {
        const int jp = std::min(j + 1, n - 1);
        const int jm = std::max(j - 1, 0);
        // Tangent vectors (unnormalised).
        double tx = cl[jp].x - cl[jm].x;
        double ty = cl[jp].y - cl[jm].y;
        double tz = cl[jp].z - cl[jm].z;
        const double norm_t = std::sqrt(tx*tx + ty*ty + tz*tz);
        if (norm_t < 1.0e-15) { cl[j].kappa = 0.0; continue; }
        // Normalise.
        tx /= norm_t; ty /= norm_t; tz /= norm_t;

        // Approximate curvature using second difference of position / |ds|².
        // κ ≈ |d²r/ds²| = |(r_{j+1} - 2*r_j + r_{j-1})| / Δs²
        const double ds = (cl[jp].s - cl[jm].s) * FEMBodyMesh::kLength / 2.0;
        if (ds < 1.0e-15) { cl[j].kappa = 0.0; continue; }
        const double ax = cl[jp].x - 2.0*cl[j].x + cl[jm].x;
        const double ay = cl[jp].y - 2.0*cl[j].y + cl[jm].y;
        const double az = cl[jp].z - 2.0*cl[j].z + cl[jm].z;
        cl[j].kappa = std::sqrt(ax*ax + ay*ay + az*az) / (ds * ds);
    }
}

SRActivation CurvatureSensor::compute_sr(const std::vector<CenterlinePoint>& cl) const
{
    SRActivation sr{};
    const int ns = FEMBodyMesh::kNSegments;
    const int n  = static_cast<int>(cl.size());

    for (int seg = 0; seg < ns; ++seg) {
        // Find the centerline point closest to this segment's midpoint.
        const double s_mid = (seg + 0.5) / ns;
        int best = 0;
        double best_d = 1.0e9;
        for (int j = 0; j < n; ++j) {
            double d = std::abs(cl[j].s - s_mid);
            if (d < best_d) { best_d = d; best = j; }
        }
        const double kappa = cl[best].kappa;
        // Dorsal/ventral bending: use signed curvature = y-component of κ vector.
        // Approximated by the y-displacement of the segment centroid from x-axis.
        const double y_dev = cl[best].y; // deviation from straight (y=0)

        // Boyle-Cohen 2012 SR activation:
        //   α_D = max(0,  y_dev/kappa_scale − κ_thresh)  [dorsal bend → VB excited]
        //   α_V = max(0, -y_dev/kappa_scale − κ_thresh)  [ventral bend → DB excited]
        const double kappa_signed = kappa * (y_dev >= 0 ? 1.0 : -1.0);
        sr[seg][0] = std::max(0.0, ( kappa_signed - params_.kappa_thresh) * params_.g_SR);
        sr[seg][1] = std::max(0.0, (-kappa_signed - params_.kappa_thresh) * params_.g_SR);
    }
    return sr;
}

} // namespace wormsim2
