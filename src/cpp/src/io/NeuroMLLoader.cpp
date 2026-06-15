#include "io/NeuroMLLoader.h"
#include "io/NEURONLoader.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace wormsim2 {

// ---------------------------------------------------------------------------
// Minimal XML attribute scanner (sufficient for c302 NeuroML2 output subset)
// ---------------------------------------------------------------------------

namespace {

/// Parse a float from a string; returns fallback on failure.
float toFloat(const std::string& s, float fallback = 0.0f) {
    try { return std::stof(s); }
    catch (...) { return fallback; }
}

/// Convert string to int; returns -1 on failure.
int toInt(const std::string& s) {
    try { return std::stoi(s); }
    catch (...) { return -1; }
}

/// Extract the numerical index from a cell-instance id like "AVBL[0]" or "AVBL_0".
int extractInstanceIndex(const std::string& instance_id) {
    // Try "NAME[N]" form
    const std::regex bracket_re(R"(\[(\d+)\])");
    std::smatch m;
    if (std::regex_search(instance_id, m, bracket_re)) return toInt(m[1].str());
    // Try "NAME_N" form
    const auto pos = instance_id.rfind('_');
    if (pos != std::string::npos) return toInt(instance_id.substr(pos + 1));
    return 0;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// XML attribute scanner helpers
// ---------------------------------------------------------------------------

std::string NeuroMLLoader::getAttribute(const std::string& element_text,
                                         std::string_view   attr_name) {
    // Match attr_name="value" or attr_name='value'
    const std::string pat =
        std::string(attr_name) + R"(\s*=\s*["']([^"']*)["'])";
    const std::regex re(pat);
    std::smatch m;
    if (std::regex_search(element_text, m, re)) return m[1].str();
    return {};
}

float NeuroMLLoader::getFloatAttr(const std::string& element_text,
                                   std::string_view   attr_name,
                                   float              fallback) {
    const std::string val = getAttribute(element_text, attr_name);
    return val.empty() ? fallback : toFloat(val, fallback);
}

// ---------------------------------------------------------------------------
// Population parser — neuron list with ids
// ---------------------------------------------------------------------------

void NeuroMLLoader::parsePopulations(const std::string& xml, NetworkConfig& cfg) {
    // <population id="..." component="..." size="..." />  (NeuroML2 short form)
    // <population id="..." component="..."><instance id="..."/></population>
    const std::regex pop_re(R"(<population\s+([^>]*)>)",
                             std::regex::icase);
    const std::regex inst_re(R"(<instance\s+id\s*=\s*["']([^"']*)["'])",
                              std::regex::icase);

    std::sregex_iterator pop_it(xml.begin(), xml.end(), pop_re);
    const std::sregex_iterator end;
    for (; pop_it != end; ++pop_it) {
        const std::string pop_text = (*pop_it)[1].str();
        const std::string pop_id   = getAttribute(pop_text, "id");
        const std::string size_str = getAttribute(pop_text, "size");

        if (!size_str.empty()) {
            // Short form: size attribute gives neuron count
            const int size = toInt(size_str);
            for (int i = 0; i < size; ++i) {
                NeuronDef nd;
                nd.id   = static_cast<int>(cfg.neurons.size());
                nd.name = pop_id + "[" + std::to_string(i) + "]";
                cfg.neurons.push_back(std::move(nd));
            }
        }
        // Long form with <instance> elements is handled by scanning the full xml
    }

    // Long form: pick up any <instance id="..."> not already covered
    std::sregex_iterator inst_it(xml.begin(), xml.end(), inst_re);
    for (; inst_it != end; ++inst_it) {
        const std::string inst_id = (*inst_it)[1].str();
        // Avoid duplicate inserts — check by name
        bool found = false;
        for (const auto& n : cfg.neurons) {
            if (n.name == inst_id) { found = true; break; }
        }
        if (!found) {
            NeuronDef nd;
            nd.id   = static_cast<int>(cfg.neurons.size());
            nd.name = inst_id;
            cfg.neurons.push_back(std::move(nd));
        }
    }
}

// ---------------------------------------------------------------------------
// Biophysics parser — channel densities per cell group / per neuron
// ---------------------------------------------------------------------------

void NeuroMLLoader::parseBiophysics(const std::string& xml, NetworkConfig& cfg) {
    // <channelDensity id="..." ionChannel="..." condDensity="..." erev="..." />
    const std::regex cd_re(
        R"(<channelDensity\s+([^/]*)/>)",
        std::regex::icase);

    std::sregex_iterator it(xml.begin(), xml.end(), cd_re);
    const std::sregex_iterator end;
    for (; it != end; ++it) {
        const std::string elem = (*it)[1].str();
        const std::string channel_id = getAttribute(elem, "ionChannel");
        const float cond_density     = getFloatAttr(elem, "condDensity");
        const float erev             = getFloatAttr(elem, "erev");

        if (channel_id.empty()) continue;

        // Ensure channel exists in the catalogue (may be populated later from .mod)
        if (cfg.channels.find(channel_id) == cfg.channels.end()) {
            ChannelDef ch;
            ch.id        = channel_id;
            ch.gbar      = cond_density;
            ch.e_rev_mV  = erev;
            cfg.channels[channel_id] = std::move(ch);
        } else {
            // Update erev if we have a better value
            if (erev != 0.0f) cfg.channels[channel_id].e_rev_mV = erev;
        }

        // Apply this channel assignment to all neurons (global biophysics)
        // Cell-group–specific biophysics requires a more complex two-pass parse;
        // treat as a global density until that is implemented (ISSUE-009).
        for (auto& neuron : cfg.neurons) {
            // Check if assignment already present
            bool already = false;
            for (const auto& ca : neuron.channels) {
                if (ca.channel_id == channel_id) { already = true; break; }
            }
            if (!already) {
                ChannelAssignment ca;
                ca.channel_id          = channel_id;
                ca.conductance_density = cond_density;
                neuron.channels.push_back(std::move(ca));
            }
        }
    }

    // Also pick up membrane properties (capacitance)
    // <specificCapacitance value="..." />
    const std::regex cap_re(
        R"(<specificCapacitance\s+value\s*=\s*["']([^"']*)["'])",
        std::regex::icase);
    std::smatch cap_m;
    if (std::regex_search(xml, cap_m, cap_re)) {
        const float cap = toFloat(cap_m[1].str());
        if (cap > 0.0f) {
            for (auto& n : cfg.neurons) n.capacitance_nF = cap;
        }
    }
}

// ---------------------------------------------------------------------------
// Electrical projections (gap junctions)
// ---------------------------------------------------------------------------

void NeuroMLLoader::parseElectricalProjections(const std::string& xml,
                                                NetworkConfig&     cfg) {
    // <electricalProjection id="..." presynapticPopulation="..." postsynapticPopulation="...">
    //   <electricalConnectionInstanceW id="..." preCell="..." postCell="..." weight="..." />
    // </electricalProjection>

    const std::regex conn_re(
        R"(<electricalConnectionInstance(?:W)?\s+([^/]*)/>)",
        std::regex::icase);

    std::sregex_iterator it(xml.begin(), xml.end(), conn_re);
    const std::sregex_iterator end;
    for (; it != end; ++it) {
        const std::string elem     = (*it)[1].str();
        const std::string pre_cell = getAttribute(elem, "preCell");
        const std::string post_cell= getAttribute(elem, "postCell");
        const float weight         = getFloatAttr(elem, "weight", 1.0f);

        if (pre_cell.empty() || post_cell.empty()) continue;

        // Resolve cell references to neuron ids by instance index in the vector
        // c302 uses "POPULATION[index]" format
        auto find_by_name = [&](const std::string& cell_ref) -> int {
            const NeuronDef* nd = cfg.find_neuron(cell_ref);
            if (nd) return nd->id;
            // Try matching the tail of the cell reference path (../POPULATION[N])
            const auto slash = cell_ref.rfind('/');
            const std::string short_ref =
                (slash != std::string::npos) ? cell_ref.substr(slash + 1) : cell_ref;
            nd = cfg.find_neuron(short_ref);
            return nd ? nd->id : -1;
        };

        const int a = find_by_name(pre_cell);
        const int b = find_by_name(post_cell);
        if (a < 0 || b < 0) continue;

        GapJunctionDef gj;
        gj.neuron_a       = a;
        gj.neuron_b       = b;
        gj.conductance_nS = weight; // weight in NeuroML2 maps to relative conductance
        cfg.gap_junctions.push_back(gj);
    }
}

// ---------------------------------------------------------------------------
// Continuous projections (chemical synapses and NMJs)
// ---------------------------------------------------------------------------

void NeuroMLLoader::parseContinuousProjections(const std::string& xml,
                                                NetworkConfig&     cfg) {
    // <continuousProjection id="..." presynapticPopulation="..." postsynapticPopulation="...">
    //   <continuousConnectionInstanceW id="..." preCell="..." postCell="..."
    //     preComponent="..." postComponent="..." weight="..." />
    // </continuousProjection>

    const std::regex conn_re(
        R"(<continuousConnectionInstance(?:W)?\s+([^/]*)/>)",
        std::regex::icase);

    auto find_neuron_id = [&](const std::string& ref) -> int {
        const NeuronDef* nd = cfg.find_neuron(ref);
        if (nd) return nd->id;
        const auto slash = ref.rfind('/');
        const std::string short_ref =
            (slash != std::string::npos) ? ref.substr(slash + 1) : ref;
        nd = cfg.find_neuron(short_ref);
        return nd ? nd->id : -1;
    };

    std::sregex_iterator it(xml.begin(), xml.end(), conn_re);
    const std::sregex_iterator end;
    for (; it != end; ++it) {
        const std::string elem      = (*it)[1].str();
        const std::string pre_cell  = getAttribute(elem, "preCell");
        const std::string post_cell = getAttribute(elem, "postCell");
        const float weight          = getFloatAttr(elem, "weight", 1.0f);

        if (pre_cell.empty() || post_cell.empty()) continue;

        const int pre_id  = find_neuron_id(pre_cell);
        const int post_id = find_neuron_id(post_cell);
        if (pre_id < 0 || post_id < 0) continue;

        // Check if post-synaptic target is a muscle (muscle ids are in NMJ territory)
        // Heuristic: look for "BWM" or "muscle" in the cell reference
        const bool is_nmj =
            post_cell.find("BWM")    != std::string::npos ||
            post_cell.find("muscle") != std::string::npos ||
            post_cell.find("Muscle") != std::string::npos;

        if (is_nmj) {
            NMJDef nmj;
            nmj.motor_neuron_id = pre_id;
            nmj.muscle_id       = extractInstanceIndex(post_cell); // 0..94
            nmj.weight          = weight;
            cfg.nmj_connections.push_back(nmj);
        } else {
            ChemSynapseDef syn;
            syn.pre_id    = pre_id;
            syn.post_id   = post_id;
            syn.g_max_nS  = weight;
            cfg.chem_synapses.push_back(syn);
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

void NeuroMLLoader::load(const std::filesystem::path& nml_path,
                          NetworkConfig&               cfg,
                          const std::filesystem::path& mod_dir) {
    std::ifstream file(nml_path);
    if (!file) {
        throw std::runtime_error("NeuroMLLoader: cannot open file: " +
                                 nml_path.string());
    }
    const std::string xml(std::istreambuf_iterator<char>{file},
                           std::istreambuf_iterator<char>{});

    cfg.source_format = "NeuroML2";
    cfg.source_path   = nml_path.string();

    parsePopulations(xml, cfg);
    parseBiophysics(xml, cfg);
    parseElectricalProjections(xml, cfg);
    parseContinuousProjections(xml, cfg);

    if (!mod_dir.empty() && std::filesystem::is_directory(mod_dir)) {
        NEURONLoader::parseModDir(mod_dir, cfg);
    }
}

} // namespace wormsim2
