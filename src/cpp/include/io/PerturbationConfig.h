#pragma once

#include "NetworkConfig.h"

#include <filesystem>
#include <string>

namespace wormsim2 {

/// Reads a perturbation specification file and applies overrides to a
/// NetworkConfig before the simulation starts (FR-NET-03).
///
/// File format — simple key = value, one entry per line, comments with '#':
///
///   # Example perturbation spec
///   channel.KD.conductance_scale = 0.0      # knock out KD in all neurons
///   synapse.exc.g_max_scale       = 0.5      # halve all excitatory weights
///   gap_junction.conductance_scale = 0.8     # reduce all gap junctions
///   neuron.AVBL.v_initial_mV      = -60.0   # set initial voltage for AVBL
///
/// Supported keys:
///   channel.<CHANNEL_ID>.conductance_scale  — multiplies all assignments of that channel
///   synapse.<SYN_TYPE>.g_max_scale          — not yet implemented (stub, ISSUE-009)
///   gap_junction.conductance_scale          — multiplies all gap junction conductances
///   neuron.<NEURON_NAME>.v_initial_mV       — overrides initial voltage
///   neuron.<NEURON_NAME>.capacitance_nF     — overrides capacitance
class PerturbationConfig {
public:
    /// Parse @p spec_path and apply all perturbations to @p cfg in-place.
    /// Throws std::runtime_error if the file cannot be opened.
    /// Unknown keys emit a warning to stderr and are skipped.
    static void apply(const std::filesystem::path& spec_path,
                      NetworkConfig&               cfg);

private:
    static void applyLine(std::string_view key, std::string_view value,
                          NetworkConfig& cfg);
};

} // namespace wormsim2
