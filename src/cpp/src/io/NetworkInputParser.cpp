#include "io/NetworkInputParser.h"
#include "io/NEURONLoader.h"
#include "io/NeuroMLLoader.h"
#include "io/PerturbationConfig.h"

#include <stdexcept>

namespace wormsim2 {

// ---------------------------------------------------------------------------
// Format detection
// ---------------------------------------------------------------------------

NetworkFormat NetworkInputParser::detectFormat(const std::filesystem::path& path) {
    const std::string ext = path.extension().string();
    if (ext == ".nml")  return NetworkFormat::NeuroML2;
    if (ext == ".hoc")  return NetworkFormat::NEURON;
    if (ext == ".mod")  return NetworkFormat::NEURON;
    throw std::runtime_error(
        "NetworkInputParser: cannot detect format from extension '" + ext +
        "'. Provide format explicitly or use a .nml, .hoc, or .mod file.");
}

// ---------------------------------------------------------------------------
// Public load dispatcher
// ---------------------------------------------------------------------------

NetworkConfig NetworkInputParser::load(
    const std::filesystem::path&          input_path,
    const std::filesystem::path&          mod_dir,
    const std::optional<std::filesystem::path>& perturb_spec,
    std::optional<NetworkFormat>          format) {

    const NetworkFormat fmt = format.value_or(detectFormat(input_path));

    NetworkConfig cfg;

    switch (fmt) {
    case NetworkFormat::NeuroML2:
        NeuroMLLoader::load(input_path, cfg, mod_dir);
        break;

    case NetworkFormat::NEURON:
        cfg.source_format = "NEURON";
        cfg.source_path   = input_path.string();

        if (input_path.extension() == ".mod") {
            // Single .mod file: channel kinetics only, no connectivity
            NEURONLoader::parseMod(input_path, cfg);
        } else {
            // .hoc file: channel kinetics from mod_dir, connectivity stub
            if (!mod_dir.empty()) {
                NEURONLoader::parseModDir(mod_dir, cfg);
            }
            std::string error_msg;
            if (!NEURONLoader::loadHoc(input_path, cfg, error_msg)) {
                // Warn but don't throw: the caller may only need channel kinetics
                // or plans to supply a NeuroML2 file for connectivity.
                // Throw so the caller knows the full network was not loaded.
                throw std::runtime_error("NetworkInputParser: " + error_msg);
            }
        }
        break;
    }

    if (perturb_spec.has_value()) {
        PerturbationConfig::apply(*perturb_spec, cfg);
    }

    return cfg;
}

} // namespace wormsim2
