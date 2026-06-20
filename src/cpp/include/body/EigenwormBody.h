#pragma once
// Quasi-static eigenworm body for C. elegans locomotion.
//
// Maps NMJ synaptic activations from NeuralState onto a reduced-order body
// representation using the 4-mode Stephens 2008 eigenworm basis.
//
// Physics: curvature at each body segment is proportional to the difference
// between dorsal (MDL/MDR) and ventral (MVL/MVR) NMJ-weighted activations.
// Body shape is represented as a tangent-angle profile θ(s), projected onto
// the 4-dimensional eigenworm subspace.

#include "body/EigenwormBasis.h"
#include "io/NetworkConfig.h"
#include "neural/NeuralState.h"

#include <string>
#include <vector>

namespace wormsim2 {

class EigenwormBody {
public:
    /// Muscle-to-curvature gain: net normalised activation → tangent-angle increment.
    static constexpr float kGain = 1.2f;

    /// Build body from loaded connectome. Parses NMJ post-cell names (MDL/MDR/MVL/MVR)
    /// and pre-computes the per-NMJ eigenworm segment assignment.
    explicit EigenwormBody(const NetworkConfig& cfg);

    /// Compute body shape from the current NMJ activation state.
    /// Call once per integrator step, after NeuralIntegrator::step().
    void step(const NeuralState& state);

    // ── Accessors ─────────────────────────────────────────────────────────────
    /// Mode amplitudes a[0..3] (Stephens 2008 eigenmodes, radians).
    [[nodiscard]] const float* mode_amplitudes() const noexcept { return a_; }

    /// Centerline x-positions (mm), kNPoints = 49 values (head…tail).
    [[nodiscard]] const float* centerline_x() const noexcept { return x_; }

    /// Centerline y-positions (mm), kNPoints = 49 values.
    [[nodiscard]] const float* centerline_y() const noexcept { return y_; }

    /// Tangent-angle profile θ[kNSeg] (radians), cumulative from head.
    [[nodiscard]] const float* tangent_angles() const noexcept { return theta_; }

    /// Per-segment curvature kappa[kNSeg] (radians/body-length, positive = dorsal bend).
    [[nodiscard]] const float* curvature() const noexcept { return kappa_; }

    /// Number of NMJ connections with a valid muscle-segment assignment.
    [[nodiscard]] int valid_nmj_count() const noexcept { return n_valid_nmj_; }

private:
    struct NMJInfo {
        int  j0;      ///< first eigenworm segment (even: 2 × c302_segment)
        int  j1;      ///< second eigenworm segment (j0 + 1)
        bool dorsal;  ///< true = MDL/MDR (dorsal), false = MVL/MVR (ventral)
        bool valid;   ///< false when muscle name couldn't be parsed
    };

    /// Parse "MDL05[0]" → NMJInfo{j0=8, j1=9, dorsal=true, valid=true}.
    static NMJInfo parse_nmj_info(const std::string& name) noexcept;

    // pre-computed per-NMJ lookup (indexed parallel to cfg.nmj_connections)
    std::vector<NMJInfo> nmj_info_;
    // per-NMJ connection weight (copy from NetworkConfig for cache locality)
    std::vector<float>   nmj_weight_;

    int n_valid_nmj_{0};

    float kappa_[EigenwormBasis::kNSeg]{};
    float theta_[EigenwormBasis::kNSeg]{};
    float a_[EigenwormBasis::kNModes]{};
    float x_[EigenwormBasis::kNPoints]{};
    float y_[EigenwormBasis::kNPoints]{};
};

} // namespace wormsim2
