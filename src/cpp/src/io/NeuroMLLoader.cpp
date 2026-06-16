#include "io/NeuroMLLoader.h"
#include "io/NEURONLoader.h"

#include <algorithm>
#include <cctype>
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

/// Resolve a c302 NeuroML2 cell reference to a neuron id.
///
/// Handles two formats:
///   - Direct name: "AVBL[0]"
///   - Path form:   "../AVBL/0/GenericNeuronCell"  (c302 C2 network files)
///
/// Returns -1 when the reference cannot be resolved.
int resolveCellRef(const std::string& ref, const NetworkConfig& cfg) {
    // Direct match first
    if (const NeuronDef* nd = cfg.find_neuron(ref)) return nd->id;

    // Parse path form — split by '/', skip ".." and "Generic*" tokens,
    // treat a digit-only token as the instance index.
    std::string pop_id;
    std::string instance_str = "0";
    std::istringstream ss(ref);
    std::string tok;
    while (std::getline(ss, tok, '/')) {
        if (tok.empty() || tok == "..") continue;
        if (tok.starts_with("Generic")) continue;   // C++20
        const bool is_num = std::ranges::all_of(tok, ::isdigit);
        if (is_num && !pop_id.empty()) {
            instance_str = tok;
        } else if (!is_num) {
            pop_id = tok;
        }
    }
    if (!pop_id.empty()) {
        const std::string name = pop_id + "[" + instance_str + "]";
        if (const NeuronDef* nd = cfg.find_neuron(name)) return nd->id;
    }
    return -1;
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
    // Only parse populations inside <network>...</network> to avoid matching
    // cell-definition <population> elements in other NeuroML2 contexts.
    const auto net_start = xml.find("<network");
    const auto net_end   = xml.rfind("</network>");
    const std::string network_xml =
        (net_start != std::string::npos && net_end != std::string::npos)
        ? xml.substr(net_start, net_end - net_start + 10)
        : xml;

    // <population id="..." component="..." size="N" type="populationList">
    const std::regex pop_re(R"(<population\s+([^>]*)>)", std::regex::icase);

    std::sregex_iterator pop_it(network_xml.begin(), network_xml.end(), pop_re);
    const std::sregex_iterator end;
    for (; pop_it != end; ++pop_it) {
        const std::string pop_text  = (*pop_it)[1].str();
        const std::string pop_id    = getAttribute(pop_text, "id");
        const std::string component = getAttribute(pop_text, "component");
        const std::string size_str  = getAttribute(pop_text, "size");

        if (pop_id.empty() || size_str.empty()) continue;

        const int size = toInt(size_str);
        for (int i = 0; i < size; ++i) {
            NeuronDef nd;
            nd.id        = static_cast<int>(cfg.neurons.size());
            nd.name      = pop_id + "[" + std::to_string(i) + "]";
            nd.cell_type = component;
            cfg.neurons.push_back(std::move(nd));
        }
    }
}

// ---------------------------------------------------------------------------
// Biophysics parser — channel densities per cell group / per neuron
// ---------------------------------------------------------------------------

void NeuroMLLoader::parseBiophysics(const std::string& xml, NetworkConfig& cfg) {
    // Extract substring for one <cell id="cell_id">...</cell> block.
    auto extractCellBlock = [&](const std::string& cell_id) -> std::string {
        const std::regex cell_open(
            R"(<cell\b[^>]*\bid\s*=\s*["'])" + cell_id + R"(["'][^>]*>)",
            std::regex::icase);
        std::smatch m;
        if (!std::regex_search(xml, m, cell_open)) return {};
        const auto tag_end   = static_cast<std::size_t>(m.suffix().first - xml.begin());
        const auto close_pos = xml.find("</cell>", tag_end);
        if (close_pos == std::string::npos) return {};
        return xml.substr(tag_end, close_pos - tag_end);
    };

    // Parse channel densities + capacitance from a cell block and apply them
    // to neurons whose cell_type matches `component_type`.
    auto applyCellBiophysics = [&](const std::string& block,
                                    const std::string& component_type) {
        if (block.empty()) return;

        const std::regex cd_re(
            R"(<channelDensity\s+((?:[^/>]|/[^>])*)\s*/>)",
            std::regex::icase);
        const std::regex cap_re(
            R"(<specificCapacitance\s+value\s*=\s*["']([^"']*)["'])",
            std::regex::icase);

        std::smatch cap_m;
        const float cap = std::regex_search(block, cap_m, cap_re)
                          ? getFloatAttr(cap_m[0].str(), "value") : 0.0f;

        struct Entry { std::string id; float g; float erev; };
        std::vector<Entry> entries;
        std::sregex_iterator it(block.begin(), block.end(), cd_re);
        for (const std::sregex_iterator end_it; it != end_it; ++it) {
            const std::string elem = (*it)[1].str();
            Entry e;
            e.id   = getAttribute(elem, "ionChannel");
            e.g    = getFloatAttr(elem, "condDensity");
            e.erev = getFloatAttr(elem, "erev");
            if (!e.id.empty()) entries.push_back(std::move(e));
        }

        for (const auto& e : entries) {
            if (cfg.channels.find(e.id) == cfg.channels.end()) {
                ChannelDef ch; ch.id = e.id; ch.gbar = e.g; ch.e_rev_mV = e.erev;
                cfg.channels[e.id] = std::move(ch);
            } else if (e.erev != 0.0f) {
                cfg.channels[e.id].e_rev_mV = e.erev;
            }
        }

        for (auto& neuron : cfg.neurons) {
            if (!component_type.empty() && neuron.cell_type != component_type) continue;
            if (cap > 0.0f) neuron.capacitance_nF = cap;
            for (const auto& e : entries) {
                bool already = false;
                for (const auto& ca : neuron.channels)
                    if (ca.channel_id == e.id) { already = true; break; }
                if (!already)
                    neuron.channels.push_back(ChannelAssignment{e.id, e.g});
            }
        }
    };

    // Apply biophysics per cell type so neuron and muscle channels stay separate.
    applyCellBiophysics(extractCellBlock("GenericNeuronCell"), "GenericNeuronCell");
    applyCellBiophysics(extractCellBlock("GenericMuscleCell"), "GenericMuscleCell");

    // Fallback: if no <cell> blocks found, parse globally (older NML2 files).
    if (cfg.channels.empty()) {
        const std::regex cd_re(
            R"(<channelDensity\s+((?:[^/>]|/[^>])*)\s*/>)",
            std::regex::icase);
        std::sregex_iterator it(xml.begin(), xml.end(), cd_re);
        for (const std::sregex_iterator end_it; it != end_it; ++it) {
            const std::string elem       = (*it)[1].str();
            const std::string channel_id = getAttribute(elem, "ionChannel");
            const float cond             = getFloatAttr(elem, "condDensity");
            const float erev             = getFloatAttr(elem, "erev");
            if (channel_id.empty()) continue;
            if (cfg.channels.find(channel_id) == cfg.channels.end()) {
                ChannelDef ch; ch.id = channel_id; ch.gbar = cond; ch.e_rev_mV = erev;
                cfg.channels[channel_id] = std::move(ch);
            }
            for (auto& n : cfg.neurons)
                n.channels.push_back(ChannelAssignment{channel_id, cond});
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

    // Allow '/' inside attribute values (c302 uses "../POP/0/Component" paths).
    // Pattern: any char that is not '/' or '>', OR a '/' not followed by '>'.
    const std::regex conn_re(
        R"(<electricalConnectionInstance(?:W)?\s+((?:[^/>]|/[^>])*)\s*/>)",
        std::regex::icase);

    std::sregex_iterator it(xml.begin(), xml.end(), conn_re);
    const std::sregex_iterator end;
    for (; it != end; ++it) {
        const std::string elem     = (*it)[1].str();
        const std::string pre_cell = getAttribute(elem, "preCell");
        const std::string post_cell= getAttribute(elem, "postCell");
        const float weight         = getFloatAttr(elem, "weight", 1.0f);

        if (pre_cell.empty() || post_cell.empty()) continue;

        const int a = resolveCellRef(pre_cell,  cfg);
        const int b = resolveCellRef(post_cell, cfg);
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
        R"(<continuousConnectionInstance(?:W)?\s+((?:[^/>]|/[^>])*)\s*/>)",
        std::regex::icase);

    std::sregex_iterator it(xml.begin(), xml.end(), conn_re);
    const std::sregex_iterator end;
    for (; it != end; ++it) {
        const std::string elem      = (*it)[1].str();
        const std::string pre_cell  = getAttribute(elem, "preCell");
        const std::string post_cell = getAttribute(elem, "postCell");
        const float weight          = getFloatAttr(elem, "weight", 1.0f);

        if (pre_cell.empty() || post_cell.empty()) continue;

        const int pre_id  = resolveCellRef(pre_cell,  cfg);
        const int post_id = resolveCellRef(post_cell, cfg);
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
            nmj.post_neuron_id  = post_id;  // resolved index in cfg.neurons
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
