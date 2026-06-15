#pragma once

#include "ChannelKinetics.h"
#include "NeuralState.h"
#include "../io/NetworkConfig.h"
#include <span>
#include <vector>

namespace wormsim2 {

// ---------------------------------------------------------------------------
// NeuralIntegrator — single-compartment HH network over the full connectome
//
// Numerical scheme (per step):
//   1. Gate update  : Rush–Larsen per gate per neuron  (exact exponential)
//   2. Synapse update: Rush–Larsen for s_syn driven by graded V_pre
//   3. Voltage update: conductance method — implicit diagonal, explicit
//      gap-junction cross-terms (stable for dt ≤ 0.05 ms, C. elegans params)
//
// Unit system: mV · ms · pA · pF  (consistent: pF·mV/ms = pA, nS·mV = pA)
//   NetworkConfig::capacitance_nF   treated as pF (typical CEL neuron 1–20 pF)
//   ChannelAssignment::conductance_density  treated as whole-cell nS
// ---------------------------------------------------------------------------
class NeuralIntegrator {
public:
    explicit NeuralIntegrator(const NetworkConfig& cfg);

    /// Advance state by dt (ms). i_ext: per-neuron external current (pA).
    /// Pass empty span for zero external current.
    void step(float dt, std::span<const float> i_ext = {});

    [[nodiscard]] std::span<const float> voltages()     const noexcept;
    [[nodiscard]] const NeuralState&     state()        const noexcept { return state_; }
    [[nodiscard]] int                    neuron_count() const noexcept;

    void reset();

private:
    const NetworkConfig& cfg_;
    NeuralState          state_;

    // ---------------------------------------------------------------------------
    // Per-neuron resolved channel entry (built once at construction)
    // ---------------------------------------------------------------------------
    struct ResolvedChannel {
        float              g_nS;       ///< effective whole-cell conductance (nS)
        float              e_rev_mV;
        std::size_t        gate_begin; ///< offset into neuron-local gate subarray
        const ChannelSpec* spec;       ///< non-owning ptr into static catalog
    };

    struct NeuronLayout {
        std::vector<ResolvedChannel> channels;
        std::size_t                  total_gates{0};
    };
    std::vector<NeuronLayout> layout_;

    // Graded synapse threshold / slope (shared across all chemical synapses)
    static constexpr float kVthSyn  = -55.0f; ///< presynaptic release threshold (mV)
    static constexpr float kKSyn    =   5.0f; ///< release slope (mV)
    static constexpr float kVoltMin = -150.0f;
    static constexpr float kVoltMax =   80.0f;

    void  build_layout();
    void  update_gates(float dt);
    void  update_synapses(float dt);
    void  update_voltages(float dt, std::span<const float> i_ext);

    [[nodiscard]] float gate_product(std::size_t neuron_idx,
                                     const ResolvedChannel& rc) const noexcept;
};

} // namespace wormsim2
