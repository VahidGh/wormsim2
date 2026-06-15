#include "io/NEURONLoader.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace wormsim2 {

// ---------------------------------------------------------------------------
// String utilities (file-local)
// ---------------------------------------------------------------------------

namespace {

/// Strip leading/trailing whitespace from a string_view.
std::string trim(std::string_view sv) {
    const auto first = sv.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return {};
    const auto last = sv.find_last_not_of(" \t\r\n");
    return std::string(sv.substr(first, last - first + 1));
}

/// Convert string to lowercase in-place.
std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

/// Remove single-line NMODL comments (: ... to end of line) and strip blank lines.
std::string stripNmodlComments(const std::string& src) {
    std::ostringstream out;
    std::istringstream in(src);
    std::string line;
    while (std::getline(in, line)) {
        const auto pos = line.find(':');
        if (pos != std::string::npos) line = line.substr(0, pos);
        const auto trimmed = trim(line);
        if (!trimmed.empty()) out << trimmed << '\n';
    }
    return out.str();
}

/// Extract the text content of the first NMODL block named @p block_name.
/// Returns empty string if the block is not present.
std::string extractBlock(const std::string& src, std::string_view block_name) {
    // Regex: BLOCK_NAME { ... } (non-greedy, handles nested-brace-free content)
    const std::string pat =
        std::string(block_name) + R"(\s*\{([^}]*)\})";
    const std::regex re(pat, std::regex::icase);
    std::smatch m;
    if (std::regex_search(src, m, re)) return m[1].str();
    return {};
}

/// Parse a float from a string; returns fallback on failure.
float parseFloat(const std::string& s, float fallback = 0.0f) {
    try { return std::stof(s); }
    catch (...) { return fallback; }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// NEURON block  →  channel name, ion, basic metadata
// ---------------------------------------------------------------------------

void NEURONLoader::parseNeuronBlock(const std::string& block, ChannelDef& ch) {
    std::istringstream ss(block);
    std::string line;
    while (std::getline(ss, line)) {
        line = trim(line);
        if (line.empty()) continue;

        std::istringstream ls(line);
        std::string keyword;
        ls >> keyword;
        keyword = toLower(keyword);

        if (keyword == "suffix") {
            ls >> ch.id;
        } else if (keyword == "useion") {
            // USEION k READ ek WRITE ik  →  ion = "k"
            ls >> ch.ion;
            ch.ion = toLower(ch.ion);
        } else if (keyword == "nonspecific_current") {
            ch.ion = "non_specific";
        }
    }
}

// ---------------------------------------------------------------------------
// PARAMETER block  →  default gbar / conductance, reversal potential
// ---------------------------------------------------------------------------

void NEURONLoader::parseParameterBlock(const std::string& block, ChannelDef& ch) {
    // Match lines like:  gbar = 0.006 (S/cm2)  or  e = -14.0 (mV)
    const std::regex param_re(R"((\w+)\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?))");
    std::sregex_iterator it(block.begin(), block.end(), param_re);
    const std::sregex_iterator end;
    for (; it != end; ++it) {
        const std::string name  = toLower((*it)[1].str());
        const float       value = parseFloat((*it)[2].str());
        // Recognise common conductance parameter names
        if (name == "gbar" || name == "gkbar" || name == "gnabar" ||
            name == "gcabar" || name == "gncabar" || name == "gmbar") {
            ch.gbar = value;
        }
        // Recognise reversal potential names
        if (name == "e" || name == "erev" || name == "el") {
            ch.e_rev_mV = value;
        }
    }
}

// ---------------------------------------------------------------------------
// STATE block  →  gate variable names
// ---------------------------------------------------------------------------

void NEURONLoader::parseStateBlock(const std::string& block, ChannelDef& ch) {
    std::istringstream ss(block);
    std::string token;
    while (ss >> token) {
        // Skip units tokens that start with '('
        if (token.front() == '(') continue;
        GateVar g;
        g.name          = token;
        g.initial_value = 0.0f; // will be set from INITIAL block if needed
        ch.gates.push_back(std::move(g));
    }
}

// ---------------------------------------------------------------------------
// Main .mod source parser
// ---------------------------------------------------------------------------

ChannelDef NEURONLoader::parseModSource(const std::string& source,
                                         const std::string& filename) {
    ChannelDef ch;
    ch.id          = filename; // fallback; overwritten by SUFFIX
    ch.mod_source  = source;

    const std::string clean = stripNmodlComments(source);

    const std::string neuron_block    = extractBlock(clean, "NEURON");
    const std::string parameter_block = extractBlock(clean, "PARAMETER");
    const std::string state_block     = extractBlock(clean, "STATE");

    if (!neuron_block.empty())    parseNeuronBlock(neuron_block, ch);
    if (!parameter_block.empty()) parseParameterBlock(parameter_block, ch);
    if (!state_block.empty())     parseStateBlock(state_block, ch);

    return ch;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void NEURONLoader::parseMod(const std::filesystem::path& mod_path,
                             NetworkConfig&               cfg) {
    std::ifstream file(mod_path);
    if (!file) {
        throw std::runtime_error("NEURONLoader: cannot open file: " +
                                 mod_path.string());
    }
    const std::string source(std::istreambuf_iterator<char>{file},
                              std::istreambuf_iterator<char>{});

    const std::string stem = mod_path.stem().string();
    ChannelDef ch = parseModSource(source, stem);

    // Use filename stem as id fallback when SUFFIX was absent
    if (ch.id.empty() || ch.id == stem) ch.id = stem;

    cfg.channels[ch.id] = std::move(ch);
}

void NEURONLoader::parseModDir(const std::filesystem::path& mod_dir,
                                NetworkConfig&               cfg) {
    if (!std::filesystem::is_directory(mod_dir)) {
        throw std::runtime_error("NEURONLoader: not a directory: " +
                                 mod_dir.string());
    }
    for (const auto& entry : std::filesystem::directory_iterator(mod_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".mod") {
            parseMod(entry.path(), cfg);
        }
    }
}

bool NEURONLoader::loadHoc(const std::filesystem::path& hoc_path,
                            NetworkConfig&               cfg,
                            std::string&                 error_msg) {
    // HOC is a full scripting language; a complete HOC interpreter is needed
    // for connectivity extraction. Directing users to the recommended path.
    // (ISSUE-009: implement via Python jNeuroML pre-processing or a HOC parser)
    [[maybe_unused]] auto& unused = cfg; // populated by NeuroMLLoader instead
    error_msg =
        "NEURONLoader::loadHoc: HOC network-description parsing is not yet "
        "implemented (ISSUE-009). Recommended workaround: convert the HOC file "
        "to NeuroML2 using jNeuroML (`pynml " + hoc_path.string() + " -nml2`) "
        "and load the resulting .nml file with NeuroMLLoader. Channel kinetics "
        "from .mod files in the same directory can be loaded separately via "
        "NEURONLoader::parseModDir().";
    return false;
}

} // namespace wormsim2
