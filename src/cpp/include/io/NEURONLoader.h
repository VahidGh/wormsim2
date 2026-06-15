#pragma once

#include "NetworkConfig.h"

#include <filesystem>
#include <string>
#include <vector>

namespace wormsim2 {

/// Loads a neural network model from NEURON format files.
///
/// Two sub-tasks:
///  1. parseModDir()  — scans a directory for NMODL .mod files and populates
///     NetworkConfig::channels with channel kinetics.
///  2. loadHoc()      — parses a .hoc network description and populates
///     neurons, synapses, gap junctions, and NMJ connections.
///     NOTE: HOC parsing requires a HOC interpreter; this implementation
///     returns an error directing users to pre-convert via jNeuroML (ISSUE-009).
///
/// Typical usage:
///   NetworkConfig cfg;
///   cfg.source_format = "NEURON";
///   NEURONLoader::parseModDir("/path/to/c302/mod", cfg);
///   // For full connectivity, convert HOC→NeuroML2 first, then use NeuroMLLoader.
class NEURONLoader {
public:
    /// Scan @p mod_dir for *.mod files; parse each and insert into @p cfg.channels.
    /// Throws std::runtime_error on file-not-found or fatal parse error.
    static void parseModDir(const std::filesystem::path& mod_dir,
                            NetworkConfig&               cfg);

    /// Parse a single NMODL .mod file; inserts result into @p cfg.channels.
    static void parseMod(const std::filesystem::path& mod_path,
                         NetworkConfig&               cfg);

    /// Load full network (neurons, synapses, gap junctions) from a .hoc file.
    /// Currently returns false + populates error_msg directing users to ISSUE-009
    /// (convert HOC→NeuroML2 via jNeuroML, then use NeuroMLLoader).
    [[nodiscard]] static bool loadHoc(const std::filesystem::path& hoc_path,
                                      NetworkConfig&               cfg,
                                      std::string&                 error_msg);

private:
    // Internal NMODL block-level parser helpers
    static ChannelDef parseModSource(const std::string& source,
                                     const std::string& filename);

    static void parseNeuronBlock(const std::string& block, ChannelDef& ch);
    static void parseParameterBlock(const std::string& block, ChannelDef& ch);
    static void parseStateBlock(const std::string& block, ChannelDef& ch);
};

} // namespace wormsim2
