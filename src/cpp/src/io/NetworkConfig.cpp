#include "io/NetworkConfig.h"

namespace wormsim2 {

const NeuronDef* NetworkConfig::find_neuron(std::string_view name) const noexcept {
    for (const auto& n : neurons) {
        if (n.name == name) return &n;
    }
    return nullptr;
}

} // namespace wormsim2
