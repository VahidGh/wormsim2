#include "io/PerturbationConfig.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace wormsim2 {

// ---------------------------------------------------------------------------
// File-local helpers
// ---------------------------------------------------------------------------

namespace {

std::string_view trimSV(std::string_view sv) {
    const auto first = sv.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return {};
    const auto last = sv.find_last_not_of(" \t\r\n");
    return sv.substr(first, last - first + 1);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Per-line applicator
// ---------------------------------------------------------------------------

void PerturbationConfig::applyLine(std::string_view key,
                                   std::string_view value,
                                   NetworkConfig&   cfg) {
    const float fval = [&] {
        try { return std::stof(std::string(value)); }
        catch (...) { return 0.0f; }
    }();

    // -----------------------------------------------------------------------
    // channel.<CHANNEL_ID>.conductance_scale
    // -----------------------------------------------------------------------
    if (key.starts_with("channel.")) {
        const auto rest    = key.substr(8); // after "channel."
        const auto dot_pos = rest.find('.');
        if (dot_pos == std::string_view::npos) {
            std::cerr << "[PerturbationConfig] unrecognised key: " << key << '\n';
            return;
        }
        const std::string channel_id(rest.substr(0, dot_pos));
        const auto        subkey     = rest.substr(dot_pos + 1);

        if (subkey == "conductance_scale") {
            auto it = cfg.channels.find(channel_id);
            if (it != cfg.channels.end()) {
                it->second.gbar *= fval;
            }
            // Also scale per-neuron ChannelAssignment entries
            for (auto& neuron : cfg.neurons) {
                for (auto& ca : neuron.channels) {
                    if (ca.channel_id == channel_id) {
                        ca.conductance_density *= fval;
                    }
                }
            }
            return;
        }
        std::cerr << "[PerturbationConfig] unrecognised channel subkey: " << subkey << '\n';
        return;
    }

    // -----------------------------------------------------------------------
    // gap_junction.conductance_scale  — scales all gap junctions
    // -----------------------------------------------------------------------
    if (key == "gap_junction.conductance_scale") {
        for (auto& gj : cfg.gap_junctions) {
            gj.conductance_nS *= fval;
        }
        return;
    }

    // -----------------------------------------------------------------------
    // synapse.<TYPE>.g_max_scale  — stub (ISSUE-009)
    // -----------------------------------------------------------------------
    if (key.starts_with("synapse.")) {
        std::cerr << "[PerturbationConfig] synapse perturbations not yet "
                     "implemented (ISSUE-009); key ignored: " << key << '\n';
        return;
    }

    // -----------------------------------------------------------------------
    // neuron.<NAME>.v_initial_mV | capacitance_nF
    // -----------------------------------------------------------------------
    if (key.starts_with("neuron.")) {
        const auto rest    = key.substr(7); // after "neuron."
        const auto dot_pos = rest.rfind('.');
        if (dot_pos == std::string_view::npos) {
            std::cerr << "[PerturbationConfig] unrecognised key: " << key << '\n';
            return;
        }
        const std::string neuron_name(rest.substr(0, dot_pos));
        const auto        subkey      = rest.substr(dot_pos + 1);

        for (auto& n : cfg.neurons) {
            if (n.name == neuron_name) {
                if (subkey == "v_initial_mV") {
                    n.v_initial_mV = fval;
                } else if (subkey == "capacitance_nF") {
                    n.capacitance_nF = fval;
                } else {
                    std::cerr << "[PerturbationConfig] unrecognised neuron subkey: "
                              << subkey << '\n';
                }
                return;
            }
        }
        std::cerr << "[PerturbationConfig] neuron not found: " << neuron_name << '\n';
        return;
    }

    std::cerr << "[PerturbationConfig] unrecognised key: " << key << '\n';
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

void PerturbationConfig::apply(const std::filesystem::path& spec_path,
                                NetworkConfig&               cfg) {
    std::ifstream file(spec_path);
    if (!file) {
        throw std::runtime_error("PerturbationConfig: cannot open file: " +
                                 spec_path.string());
    }

    std::string line;
    int line_no = 0;
    while (std::getline(file, line)) {
        ++line_no;
        std::string_view sv = trimSV(line);

        // Skip blank lines and comments
        if (sv.empty() || sv.front() == '#') continue;

        // Strip inline comment
        const auto hash = sv.find('#');
        if (hash != std::string_view::npos) sv = trimSV(sv.substr(0, hash));

        const auto eq = sv.find('=');
        if (eq == std::string_view::npos) {
            std::cerr << "[PerturbationConfig] line " << line_no
                      << ": missing '=', skipping: " << sv << '\n';
            continue;
        }

        const auto key   = trimSV(sv.substr(0, eq));
        const auto value = trimSV(sv.substr(eq + 1));

        if (key.empty() || value.empty()) continue;

        applyLine(key, value, cfg);
    }
}

} // namespace wormsim2
