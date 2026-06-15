#pragma once

#include "NetworkConfig.h"

#include <filesystem>
#include <optional>
#include <string>

namespace wormsim2 {

/// Format tag for explicit override (otherwise auto-detected from extension).
enum class NetworkFormat { NEURON, NeuroML2 };

/// Top-level dispatcher: loads a network model from either NEURON or NeuroML2
/// format and returns a populated NetworkConfig (FR-NET-01).
///
/// Format detection:
///   - .nml  → NeuroML2
///   - .hoc  → NEURON (network description; requires mod_dir for channel kinetics)
///   - .mod  → NEURON channel-kinetics only (populates channels, no connectivity)
///   - Explicit @p format overrides extension detection.
///
/// For NEURON .hoc files, full connectivity parsing is not yet implemented
/// (ISSUE-009); the loader populates channel kinetics from mod_dir and logs a
/// recommendation to pre-convert via jNeuroML.
class NetworkInputParser {
public:
    /// Load from @p input_path.
    /// @param mod_dir  Directory of .mod files (used for both formats when channel
    ///                 kinetics are not embedded in the primary file).
    /// @param perturb_spec  Optional perturbation spec applied after loading.
    /// @param format   Optional explicit format override.
    [[nodiscard]] static NetworkConfig load(
        const std::filesystem::path&          input_path,
        const std::filesystem::path&          mod_dir       = {},
        const std::optional<std::filesystem::path>& perturb_spec = std::nullopt,
        std::optional<NetworkFormat>          format        = std::nullopt);

private:
    static NetworkFormat detectFormat(const std::filesystem::path& path);
};

} // namespace wormsim2
