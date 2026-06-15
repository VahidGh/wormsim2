#pragma once

#include "NetworkConfig.h"

#include <filesystem>
#include <string>

namespace wormsim2 {

/// Loads a neural network model from a NeuroML2 .nml file.
///
/// Parses the NeuroML2 XML file tree and populates a NetworkConfig with:
///   - Neuron list and biophysical properties (capacitance, channel densities)
///   - Chemical synapse parameters (max conductance, reversal potential, time constants)
///   - Gap junction conductances
///   - Neuromuscular junction connections
///
/// Ion-channel kinetics are resolved from NMODL .mod files referenced in the
/// channelDensity elements (same as NEURONLoader::parseModDir). If a mod_dir
/// is provided, channel kinetics are loaded automatically; otherwise only the
/// conductance density assignments are stored.
///
/// Implementation note (ISSUE-009): uses a minimal hand-written XML attribute
/// scanner sufficient for the c302 NeuroML2 output subset. A full XML library
/// (pugixml) is planned as a replacement when broader NeuroML2 support is needed.
class NeuroMLLoader {
public:
    /// Load network from @p nml_path.
    /// If @p mod_dir is non-empty, resolve channel kinetics from .mod files there.
    /// Throws std::runtime_error on file-not-found or fatal parse error.
    static void load(const std::filesystem::path& nml_path,
                     NetworkConfig&               cfg,
                     const std::filesystem::path& mod_dir = {});

private:
    // XML attribute scanner helpers
    static std::string getAttribute(const std::string& element_text,
                                    std::string_view   attr_name);
    static float getFloatAttr(const std::string& element_text,
                              std::string_view   attr_name,
                              float              fallback = 0.0f);

    static void parsePopulations(const std::string& xml, NetworkConfig& cfg);
    static void parseBiophysics(const std::string& xml,  NetworkConfig& cfg);
    static void parseElectricalProjections(const std::string& xml, NetworkConfig& cfg);
    static void parseContinuousProjections(const std::string& xml, NetworkConfig& cfg);
};

} // namespace wormsim2
