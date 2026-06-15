#include "io/NEURONLoader.h"
#include "io/NetworkConfig.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

// Minimal KD channel .mod source (mirrors c302 KD.mod structure)
const char* kKdModSource = R"NMODL(
TITLE K delayed-rectifier channel (KD)
: C. elegans KVS-1 delayed rectifier

NEURON {
    SUFFIX KD
    USEION k READ ek WRITE ik
    RANGE gkbar, ik
    GLOBAL minf, mtau
}

PARAMETER {
    gkbar = 0.04 (S/cm2)
}

STATE {
    m
}

ASSIGNED {
    v (mV)
    ek (mV)
    ik (mA/cm2)
}
)NMODL";

// NCA (leak) channel
const char* kNcaModSource = R"NMODL(
NEURON {
    SUFFIX NCA
    NONSPECIFIC_CURRENT il
    RANGE gcabar, el, il
}

PARAMETER {
    gcabar = 0.006 (S/cm2)
    el = -14 (mV)
}

STATE {
    l
}
)NMODL";

void writeTemp(const std::filesystem::path& path, const char* content) {
    std::ofstream f(path);
    assert(f.is_open());
    f << content;
}

// ---------------------------------------------------------------------------

void test_kd_mod_parsing() {
    namespace fs = std::filesystem;
    const fs::path tmp = fs::temp_directory_path() / "wormsim2_test_kd.mod";
    writeTemp(tmp, kKdModSource);

    wormsim2::NetworkConfig cfg;
    wormsim2::NEURONLoader::parseMod(tmp, cfg);

    assert(cfg.channels.count("KD") == 1);
    const auto& ch = cfg.channels.at("KD");
    assert(ch.id == "KD");
    assert(ch.ion == "k");
    assert(ch.gbar == 0.04f);
    assert(ch.gates.size() == 1);
    assert(ch.gates[0].name == "m");

    fs::remove(tmp);
    std::cout << "[PASS] test_kd_mod_parsing\n";
}

void test_nca_mod_parsing() {
    namespace fs = std::filesystem;
    const fs::path tmp = fs::temp_directory_path() / "wormsim2_test_nca.mod";
    writeTemp(tmp, kNcaModSource);

    wormsim2::NetworkConfig cfg;
    wormsim2::NEURONLoader::parseMod(tmp, cfg);

    assert(cfg.channels.count("NCA") == 1);
    const auto& ch = cfg.channels.at("NCA");
    assert(ch.id == "NCA");
    assert(ch.ion == "non_specific");
    assert(ch.gbar == 0.006f);
    assert(ch.e_rev_mV == -14.0f);
    assert(ch.gates.size() == 1);
    assert(ch.gates[0].name == "l");

    fs::remove(tmp);
    std::cout << "[PASS] test_nca_mod_parsing\n";
}

void test_mod_dir_scan() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "wormsim2_test_moddir";
    fs::create_directories(dir);
    writeTemp(dir / "KD.mod",  kKdModSource);
    writeTemp(dir / "NCA.mod", kNcaModSource);

    wormsim2::NetworkConfig cfg;
    wormsim2::NEURONLoader::parseModDir(dir, cfg);

    assert(cfg.channels.size() == 2);
    assert(cfg.channels.count("KD")  == 1);
    assert(cfg.channels.count("NCA") == 1);

    fs::remove_all(dir);
    std::cout << "[PASS] test_mod_dir_scan\n";
}

void test_hoc_stub_returns_false() {
    namespace fs = std::filesystem;
    const fs::path tmp = fs::temp_directory_path() / "wormsim2_test.hoc";
    writeTemp(tmp, "// dummy hoc\n");

    wormsim2::NetworkConfig cfg;
    std::string err;
    const bool ok = wormsim2::NEURONLoader::loadHoc(tmp, cfg, err);

    assert(!ok);
    assert(!err.empty());
    assert(err.find("ISSUE-009") != std::string::npos);

    fs::remove(tmp);
    std::cout << "[PASS] test_hoc_stub_returns_false\n";
}

void test_find_neuron() {
    wormsim2::NetworkConfig cfg;
    wormsim2::NeuronDef a; a.id = 0; a.name = "AVBL";
    wormsim2::NeuronDef b; b.id = 1; b.name = "AVAL";
    cfg.neurons.push_back(a);
    cfg.neurons.push_back(b);

    assert(cfg.find_neuron("AVBL") != nullptr);
    assert(cfg.find_neuron("AVBL")->id == 0);
    assert(cfg.find_neuron("AVAL") != nullptr);
    assert(cfg.find_neuron("NOSUCH") == nullptr);

    std::cout << "[PASS] test_find_neuron\n";
}

} // anonymous namespace

int main() {
    try {
        test_kd_mod_parsing();
        test_nca_mod_parsing();
        test_mod_dir_scan();
        test_hoc_stub_returns_false();
        test_find_neuron();
        std::cout << "\nAll NMODL parser tests passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
