#include "neural/NeuralIntegrator.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace wormsim2 {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

NeuralIntegrator::NeuralIntegrator(const NetworkConfig& cfg) : cfg_(cfg) {
    build_layout();

    std::size_t max_gates = 0;
    for (const auto& nl : layout_)
        max_gates = std::max(max_gates, nl.total_gates);

    state_ = NeuralState::allocate(cfg_, max_gates == 0 ? 1 : max_gates);
    reset();
}

void NeuralIntegrator::build_layout() {
    layout_.resize(cfg_.neurons.size());

    for (std::size_t i = 0; i < cfg_.neurons.size(); ++i) {
        std::size_t gate_idx = 0;
        for (const auto& ca : cfg_.neurons[i].channels) {
            const ChannelSpec& spec = channel_spec(ca.channel_id);
            if (spec.g_bar == 0.0f && spec.gates.empty()) continue;

            // Prefer NetworkConfig channel data when available
            float g = ca.conductance_density > 0.0f
                      ? ca.conductance_density : spec.g_bar;
            float e_rev = spec.e_rev_mV;
            if (auto it = cfg_.channels.find(ca.channel_id);
                    it != cfg_.channels.end() && it->second.e_rev_mV != 0.0f)
                e_rev = it->second.e_rev_mV;

            layout_[i].channels.push_back(ResolvedChannel{
                .g_nS      = g,
                .e_rev_mV  = e_rev,
                .gate_begin = gate_idx,
                .spec       = &spec
            });
            gate_idx += spec.gates.size();
        }
        layout_[i].total_gates = gate_idx;
    }
}

// ---------------------------------------------------------------------------
// reset — restore initial conditions from NetworkConfig
// ---------------------------------------------------------------------------

void NeuralIntegrator::reset() {
    const std::size_t gs = state_.gates_stride;
    for (std::size_t i = 0; i < cfg_.neurons.size(); ++i) {
        state_.v[i] = cfg_.neurons[i].v_initial_mV;
        for (const auto& rc : layout_[i].channels) {
            for (std::size_t gi = 0; gi < rc.spec->gates.size(); ++gi)
                state_.gate[i * gs + rc.gate_begin + gi] = rc.spec->gates[gi].initial;
        }
    }
    std::fill(state_.s_syn.begin(), state_.s_syn.end(), 0.0f);
    std::fill(state_.s_nmj.begin(), state_.s_nmj.end(), 0.0f);
}

// ---------------------------------------------------------------------------
// Public step
// ---------------------------------------------------------------------------

void NeuralIntegrator::step(float dt, std::span<const float> i_ext) {
    update_gates(dt);
    update_synapses(dt);
    update_voltages(dt, i_ext);
}

// ---------------------------------------------------------------------------
// Gate update — Rush–Larsen per gate per neuron
// ---------------------------------------------------------------------------

void NeuralIntegrator::update_gates(float dt) {
    const std::size_t gs = state_.gates_stride;
    for (std::size_t i = 0; i < cfg_.neurons.size(); ++i) {
        const float v = state_.v[i];
        for (const auto& rc : layout_[i].channels) {
            for (std::size_t gi = 0; gi < rc.spec->gates.size(); ++gi) {
                const std::size_t idx = i * gs + rc.gate_begin + gi;
                state_.gate[idx] = rush_larsen(state_.gate[idx], v,
                                               rc.spec->gates[gi], dt);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Synapse update — graded presynaptic release, Rush–Larsen for s variable
// ---------------------------------------------------------------------------

void NeuralIntegrator::update_synapses(float dt) {
    for (std::size_t si = 0; si < cfg_.chem_synapses.size(); ++si) {
        const auto& syn = cfg_.chem_synapses[si];
        const int   pre = syn.pre_id;
        if (pre < 0 || static_cast<std::size_t>(pre) >= state_.n_neurons()) continue;

        const float v_pre = state_.v[static_cast<std::size_t>(pre)];
        const float s_inf = 1.0f / (1.0f + std::exp((kVthSyn - v_pre) / kKSyn));

        state_.s_syn[si] = s_inf + (state_.s_syn[si] - s_inf)
                           * std::exp(-dt / syn.tau_decay_ms);
    }

    // NMJ update: same graded-release kinetics, slower decay
    for (std::size_t ki = 0; ki < cfg_.nmj_connections.size(); ++ki) {
        const auto& nmj = cfg_.nmj_connections[ki];
        const int   pre = nmj.motor_neuron_id;
        if (pre < 0 || static_cast<std::size_t>(pre) >= state_.n_neurons()) continue;

        const float v_pre = state_.v[static_cast<std::size_t>(pre)];
        const float s_inf = 1.0f / (1.0f + std::exp((kVthSyn - v_pre) / kKSyn));

        state_.s_nmj[ki] = s_inf + (state_.s_nmj[ki] - s_inf)
                           * std::exp(-dt / kNMJTauDecay);
    }
}

// ---------------------------------------------------------------------------
// Voltage update — conductance method (implicit diagonal, explicit gap coupling)
//
// For each neuron i:
//   (C_m + dt * g_total) * V_new = C_m * V + dt * (I_rhs + I_ext)
//
//   g_total  = Σ_ch g_nS * gate_prod  +  Σ_gj g_gap  +  Σ_syn g_max * s
//   I_rhs    = Σ_ch g_nS * gate_prod * E_rev_ch
//            + Σ_gj g_gap * V_j  (lagged — V_j from start of step)
//            + Σ_syn g_max * s * E_rev_syn
// ---------------------------------------------------------------------------

void NeuralIntegrator::update_voltages(float dt, std::span<const float> i_ext) {
    const std::vector<float> v_prev = state_.v;   // snapshot for gap coupling
    const int N = cfg_.neuron_count();

    for (int i = 0; i < N; ++i) {
        const float C_m = cfg_.neurons[static_cast<std::size_t>(i)].capacitance_nF;
        float g_total = 0.0f;
        float I_rhs   = 0.0f;

        // Ionic channels
        for (const auto& rc : layout_[static_cast<std::size_t>(i)].channels) {
            const float gp = gate_product(static_cast<std::size_t>(i), rc);
            g_total += rc.g_nS * gp;
            I_rhs   += rc.g_nS * gp * rc.e_rev_mV;
        }

        // Gap junctions (explicit cross-terms: use v_prev for V_j)
        for (const auto& gj : cfg_.gap_junctions) {
            int j = -1;
            if (gj.neuron_a == i) j = gj.neuron_b;
            else if (gj.neuron_b == i) j = gj.neuron_a;
            if (j < 0) continue;
            g_total += gj.conductance_nS;
            I_rhs   += gj.conductance_nS * v_prev[static_cast<std::size_t>(j)];
        }

        // Chemical synapses (post-synaptic current)
        for (std::size_t si = 0; si < cfg_.chem_synapses.size(); ++si) {
            const auto& syn = cfg_.chem_synapses[si];
            if (syn.post_id != i) continue;
            const float g_s = syn.g_max_nS * state_.s_syn[si];
            g_total += g_s;
            I_rhs   += g_s * syn.e_rev_mV;
        }

        // NMJ currents (motoneuron → body-wall muscle)
        for (std::size_t ki = 0; ki < cfg_.nmj_connections.size(); ++ki) {
            const auto& nmj = cfg_.nmj_connections[ki];
            if (nmj.post_neuron_id != i) continue;
            const float g_nmj = kNMJGMax * nmj.weight * state_.s_nmj[ki];
            g_total += g_nmj;
            I_rhs   += g_nmj * kNMJERev;
        }

        // Conductance-method solve (implicit in V_i)
        const float denom   = C_m + dt * g_total;
        const float I_ext_i = i_ext.empty() ? 0.0f
                              : i_ext[static_cast<std::size_t>(i)];
        const float V_new   = (C_m * v_prev[static_cast<std::size_t>(i)]
                               + dt * (I_rhs + I_ext_i)) / denom;
        state_.v[static_cast<std::size_t>(i)] =
            std::clamp(V_new, kVoltMin, kVoltMax);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

[[nodiscard]] float NeuralIntegrator::gate_product(std::size_t ni,
                                                    const ResolvedChannel& rc) const noexcept {
    if (rc.spec->gates.empty()) return 1.0f;
    const std::size_t gs = state_.gates_stride;
    float prod = 1.0f;
    for (std::size_t gi = 0; gi < rc.spec->gates.size(); ++gi) {
        const float x = state_.gate[ni * gs + rc.gate_begin + gi];
        const int   p = rc.spec->gates[gi].power;
        if (p == 1)      prod *= x;
        else if (p == 2) prod *= x * x;
        else { float r = 1.0f; for (int k = 0; k < p; ++k) r *= x; prod *= r; }
    }
    return prod;
}

[[nodiscard]] std::span<const float> NeuralIntegrator::voltages() const noexcept {
    return state_.v;
}

[[nodiscard]] int NeuralIntegrator::neuron_count() const noexcept {
    return cfg_.neuron_count();
}

} // namespace wormsim2
