#include "neural/NeuralState.h"

namespace wormsim2 {

NeuralState NeuralState::allocate(const NetworkConfig& cfg, std::size_t gs) {
    NeuralState s;
    s.gates_stride = gs;
    const std::size_t n = static_cast<std::size_t>(cfg.neuron_count());
    s.v.assign(n, 0.0f);
    s.gate.assign(n * gs, 0.0f);
    s.s_syn.assign(static_cast<std::size_t>(cfg.synapse_count()), 0.0f);
    s.s_nmj.assign(cfg.nmj_connections.size(), 0.0f);
    return s;
}

} // namespace wormsim2
