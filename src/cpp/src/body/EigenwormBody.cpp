#include "body/EigenwormBody.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>

namespace wormsim2 {

// ---------------------------------------------------------------------------
// Static helper
// ---------------------------------------------------------------------------

EigenwormBody::NMJInfo
EigenwormBody::parse_nmj_info(const std::string& name) noexcept {
    // Expected formats from c302 NeuroML2: "MDL05[0]", "MVR24[0]"
    // Ignore head-region muscles that don't map to a body segment (segment > 24).
    if (name.size() < 5) return {-1, -1, false, false};
    if (name[0] != 'M') return {-1, -1, false, false};
    if (name[1] != 'D' && name[1] != 'V') return {-1, -1, false, false};

    const bool dorsal = (name[1] == 'D');

    // Find first digit after the 3-char prefix (MDL / MVR / etc.)
    std::size_t num_start = std::string::npos;
    for (std::size_t i = 3; i < name.size(); ++i) {
        if (std::isdigit(static_cast<unsigned char>(name[i]))) {
            num_start = i;
            break;
        }
    }
    if (num_start == std::string::npos) return {-1, -1, false, false};

    // Read digits
    std::size_t num_end = num_start;
    while (num_end < name.size() &&
           std::isdigit(static_cast<unsigned char>(name[num_end])))
        ++num_end;

    const int seg_1based = std::stoi(name.substr(num_start, num_end - num_start));
    if (seg_1based < 1 || seg_1based > 24) return {-1, -1, false, false};

    const int seg = seg_1based - 1; // 0-based c302 segment index (0..23)
    const int j0  = 2 * seg;        // eigenworm segment pair base (0, 2, 4, … 46)
    return {j0, j0 + 1, dorsal, true};
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

EigenwormBody::EigenwormBody(const NetworkConfig& cfg) {
    const std::size_t n = cfg.nmj_connections.size();
    nmj_info_.resize(n);
    nmj_weight_.resize(n);

    n_valid_nmj_ = 0;
    for (std::size_t ki = 0; ki < n; ++ki) {
        const auto& nmj = cfg.nmj_connections[ki];
        nmj_weight_[ki] = nmj.weight;

        // Resolve muscle name through the pre-synaptic neuron slot that holds
        // the muscle cell (post_neuron_id points to the muscle in cfg.neurons).
        if (nmj.post_neuron_id < 0 ||
            nmj.post_neuron_id >= static_cast<int>(cfg.neurons.size())) {
            nmj_info_[ki] = {-1, -1, false, false};
            continue;
        }
        nmj_info_[ki] = parse_nmj_info(cfg.neurons[nmj.post_neuron_id].name);
        if (nmj_info_[ki].valid) ++n_valid_nmj_;
    }

    std::fill(std::begin(kappa_), std::end(kappa_), 0.0f);
    std::fill(std::begin(theta_), std::end(theta_), 0.0f);
    std::fill(std::begin(a_),     std::end(a_),     0.0f);
    std::fill(std::begin(x_),     std::end(x_),     0.0f);
    std::fill(std::begin(y_),     std::end(y_),     0.0f);
}

// ---------------------------------------------------------------------------
// step()
// ---------------------------------------------------------------------------

void EigenwormBody::step(const NeuralState& state) {
    using B = EigenwormBasis;

    // Accumulators: weighted NMJ activation + total weight per eigenworm segment.
    float actD[B::kNSeg]{}, actV[B::kNSeg]{};
    float  wD [B::kNSeg]{},  wV [B::kNSeg]{};

    for (std::size_t ki = 0; ki < nmj_info_.size(); ++ki) {
        const auto& info = nmj_info_[ki];
        if (!info.valid) continue;

        const float s = state.s_nmj[ki] * nmj_weight_[ki];
        const float w = nmj_weight_[ki];

        if (info.dorsal) {
            actD[info.j0] += s; actD[info.j1] += s;
            wD  [info.j0] += w; wD  [info.j1] += w;
        } else {
            actV[info.j0] += s; actV[info.j1] += s;
            wV  [info.j0] += w; wV  [info.j1] += w;
        }
    }

    // Curvature: normalised dorsal − ventral activation.
    for (int j = 0; j < B::kNSeg; ++j) {
        const float aD = (wD[j] > 0.0f) ? actD[j] / wD[j] : 0.0f;
        const float aV = (wV[j] > 0.0f) ? actV[j] / wV[j] : 0.0f;
        kappa_[j] = kGain * (aD - aV);
    }

    // Integrate curvature → tangent angle (θ[j] = cumulative sum of κ * ds).
    const float ds = 1.0f / B::kNSeg;
    float cumth = 0.0f;
    for (int j = 0; j < B::kNSeg; ++j) {
        cumth    += kappa_[j] * ds;
        theta_[j] = cumth;
    }

    // Project onto eigenbasis.
    for (int m = 0; m < B::kNModes; ++m)
        a_[m] = B::project(m, theta_);

    // Integrate centerline positions.
    B::integrate_centerline(theta_, x_, y_);
}

} // namespace wormsim2
