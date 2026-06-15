#pragma once

#include <cmath>
#include <string_view>
#include <vector>

namespace wormsim2 {

// ---------------------------------------------------------------------------
// GateKinetics — Boltzmann steady-state + bell-shaped time constant
//
// Steady state : x_inf(V) = 1 / (1 + exp((v_half - V) / k))
//   k > 0 → activation (opens at depolarisation)
//   k < 0 → inactivation (closes at depolarisation)
//
// Time constant: tau(V) = tau_0 + tau_a * exp(-((V - tau_v) / tau_w)^2)
//   tau_a = 0 → constant tau_0
// ---------------------------------------------------------------------------
struct GateKinetics {
    float v_half{0.0f};    ///< half-activation voltage (mV)
    float k{10.0f};        ///< slope factor (mV); negative = inactivation
    float tau_0{5.0f};     ///< baseline time constant (ms)
    float tau_a{0.0f};     ///< Gaussian amplitude (ms); 0 = constant
    float tau_v{0.0f};     ///< Gaussian centre (mV)
    float tau_w{25.0f};    ///< Gaussian half-width (mV)
    float initial{0.0f};   ///< initial gate value
    int   power{1};        ///< exponent in conductance product (g ∝ x^power)
};

// ---------------------------------------------------------------------------
// ChannelSpec — full description of one channel type
//   gates empty → pure ohmic leak (gate product = 1)
// ---------------------------------------------------------------------------
struct ChannelSpec {
    float                     g_bar{0.0f};   ///< catalog default conductance (nS)
    float                     e_rev_mV{0.0f};
    std::vector<GateKinetics> gates;
};

// ---------------------------------------------------------------------------
// Pure kinetics helpers — [[nodiscard]], no side effects
// ---------------------------------------------------------------------------

[[nodiscard]] inline float x_inf(float v, const GateKinetics& g) noexcept {
    return 1.0f / (1.0f + std::exp((g.v_half - v) / g.k));
}

[[nodiscard]] inline float tau_x(float v, const GateKinetics& g) noexcept {
    if (g.tau_a == 0.0f) return g.tau_0;
    const float z = (v - g.tau_v) / g.tau_w;
    return g.tau_0 + g.tau_a * std::exp(-z * z);
}

// Rush–Larsen exponential update — exact integration for constant V over dt.
[[nodiscard]] inline float rush_larsen(float x, float v,
                                        const GateKinetics& g, float dt) noexcept {
    const float xi  = x_inf(v, g);
    const float tau = tau_x(v, g);
    return xi + (x - xi) * std::exp(-dt / tau);
}

// ---------------------------------------------------------------------------
// Built-in catalog for c302 channels.
// Returns a zero-conductance spec for unknown IDs (safe no-op).
// ---------------------------------------------------------------------------
[[nodiscard]] const ChannelSpec& channel_spec(std::string_view channel_id) noexcept;

} // namespace wormsim2
