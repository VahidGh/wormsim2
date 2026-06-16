#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace wormsim2 {

// ---------------------------------------------------------------------------
// Ion channel kinetics (from NMODL .mod or NeuroML2 ChannelML)
// ---------------------------------------------------------------------------

/// One gating variable (m, h, n, …) with its initial value.
struct GateVar {
    std::string name;
    float       initial_value{0.0f};
};

/// Full channel definition loaded from a .mod or ChannelML source.
struct ChannelDef {
    std::string id;         ///< e.g. "NCA", "KD", "KA"
    std::string ion;        ///< "non_specific" | "k" | "na" | "ca"
    float e_rev_mV{0.0f};  ///< reversal potential (mV)
    float gbar{0.0f};       ///< default max conductance density (S/cm²)
    std::vector<GateVar> gates;
    std::string mod_source; ///< raw NMODL text (retained for future NMODL compiler)
};

// ---------------------------------------------------------------------------
// Per-neuron channel assignment
// ---------------------------------------------------------------------------

struct ChannelAssignment {
    std::string channel_id;
    float conductance_density{0.0f}; ///< S/cm² (overrides ChannelDef::gbar)
};

// ---------------------------------------------------------------------------
// Neuron
// ---------------------------------------------------------------------------

struct NeuronDef {
    int         id{-1};
    std::string name;
    std::string cell_type;             ///< NeuroML2 component id, e.g. "GenericNeuronCell"
    float       capacitance_nF{10.0f}; ///< membrane capacitance (treated as pF)
    float       v_initial_mV{-65.0f};  ///< initial membrane potential
    std::vector<ChannelAssignment> channels;
};

// ---------------------------------------------------------------------------
// Chemical synapse
// ---------------------------------------------------------------------------

struct ChemSynapseDef {
    int   pre_id{-1};
    int   post_id{-1};
    float g_max_nS{1.0f};    ///< max synaptic conductance
    float e_rev_mV{0.0f};
    float tau_rise_ms{0.5f};
    float tau_decay_ms{10.0f};
};

// ---------------------------------------------------------------------------
// Electrical synapse (gap junction)
// ---------------------------------------------------------------------------

struct GapJunctionDef {
    int   neuron_a{-1};
    int   neuron_b{-1};
    float conductance_nS{0.1f};
};

// ---------------------------------------------------------------------------
// Neuromuscular junction  (motoneuron → body-wall muscle)
// ---------------------------------------------------------------------------

/// Connects a motor neuron to one of the 95 body-wall muscles (BWM 0..94).
struct NMJDef {
    int   motor_neuron_id{-1};
    int   muscle_id{-1};        ///< 0..94 (population-relative index)
    int   post_neuron_id{-1};   ///< resolved index in cfg.neurons (302..396)
    float weight{1.0f};
};

// ---------------------------------------------------------------------------
// Aggregate network configuration
// ---------------------------------------------------------------------------

/// Central in-memory representation produced by either NEURONLoader or
/// NeuroMLLoader. All downstream simulation components consume this struct.
struct NetworkConfig {
    std::string source_format; ///< "NEURON" | "NeuroML2"
    std::string source_path;

    std::vector<NeuronDef>     neurons;
    std::unordered_map<std::string, ChannelDef> channels; ///< keyed by id
    std::vector<ChemSynapseDef> chem_synapses;
    std::vector<GapJunctionDef> gap_junctions;
    std::vector<NMJDef>         nmj_connections;

    [[nodiscard]] int neuron_count()       const noexcept { return static_cast<int>(neurons.size()); }
    [[nodiscard]] int synapse_count()      const noexcept { return static_cast<int>(chem_synapses.size()); }
    [[nodiscard]] int gap_junction_count() const noexcept { return static_cast<int>(gap_junctions.size()); }

    /// Returns nullptr when no neuron with the given name exists.
    [[nodiscard]] const NeuronDef* find_neuron(std::string_view name) const noexcept;
};

} // namespace wormsim2
