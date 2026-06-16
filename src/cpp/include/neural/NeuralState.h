#pragma once

#include "../io/NetworkConfig.h"
#include <cstddef>
#include <vector>

namespace wormsim2 {

// ---------------------------------------------------------------------------
// NeuralState — SoA simulation state owned by NeuralIntegrator
//
// Layout:
//   v[i]                          membrane voltage of neuron i (mV)
//   gate[i * gates_stride + k]    k-th gate variable of neuron i  [0, 1]
//   s_syn[si]                     activation variable of synapse si  [0, 1]
//   s_nmj[ki]                     activation variable of NMJ ki     [0, 1]
//
// The SoA layout is GPU-friendly: transposing the gate array to
// gate[k * n_neurons + i] (gate-major) gives fully coalesced GPU access.
// That transposition is deferred to the CUDA/OpenCL backend.
// ---------------------------------------------------------------------------
struct NeuralState {
    std::vector<float> v;             ///< [n_neurons] (mV)
    std::vector<float> gate;          ///< [n_neurons × gates_stride]
    std::size_t        gates_stride{0};
    std::vector<float> s_syn;         ///< [n_chem_synapses]
    std::vector<float> s_nmj;         ///< [n_nmj_connections]

    [[nodiscard]] std::size_t n_neurons()  const noexcept { return v.size(); }
    [[nodiscard]] std::size_t n_synapses() const noexcept { return s_syn.size(); }
    [[nodiscard]] std::size_t n_nmj()      const noexcept { return s_nmj.size(); }

    /// Allocate and zero-fill; initial values are filled by NeuralIntegrator::reset().
    [[nodiscard]] static NeuralState allocate(const NetworkConfig& cfg,
                                               std::size_t gates_stride);
};

} // namespace wormsim2
