#include "io/NetworkConfig.h"
#include "io/PerturbationConfig.h"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

const char* kSpecContent = R"SPEC(
# Test perturbation spec

# Scale KD conductance by 50%
channel.KD.conductance_scale = 0.5

# Knock out NCA
channel.NCA.conductance_scale = 0.0

# Reduce all gap junctions
gap_junction.conductance_scale = 0.8

# Override AVBL initial voltage
neuron.AVBL.v_initial_mV = -55.0

# Override AVAL capacitance
neuron.AVAL.capacitance_nF = 12.5
)SPEC";

wormsim2::NetworkConfig buildBaseCfg() {
    wormsim2::NetworkConfig cfg;

    wormsim2::ChannelDef kd; kd.id = "KD"; kd.gbar = 0.04f;
    wormsim2::ChannelDef nca; nca.id = "NCA"; nca.gbar = 0.006f;
    cfg.channels["KD"]  = kd;
    cfg.channels["NCA"] = nca;

    wormsim2::NeuronDef avbl; avbl.id = 0; avbl.name = "AVBL";
    avbl.v_initial_mV = -65.0f;
    wormsim2::ChannelAssignment ca_kd;  ca_kd.channel_id = "KD";  ca_kd.conductance_density = 0.04f;
    wormsim2::ChannelAssignment ca_nca; ca_nca.channel_id = "NCA"; ca_nca.conductance_density = 0.006f;
    avbl.channels.push_back(ca_kd);
    avbl.channels.push_back(ca_nca);
    cfg.neurons.push_back(avbl);

    wormsim2::NeuronDef aval; aval.id = 1; aval.name = "AVAL";
    aval.capacitance_nF = 10.0f;
    cfg.neurons.push_back(aval);

    wormsim2::GapJunctionDef gj; gj.neuron_a = 0; gj.neuron_b = 1; gj.conductance_nS = 1.0f;
    cfg.gap_junctions.push_back(gj);

    return cfg;
}

constexpr float kEps = 1e-5f;
bool approx(float a, float b) { return std::fabs(a - b) < kEps; }

// ---------------------------------------------------------------------------

void test_channel_scale() {
    namespace fs = std::filesystem;
    const fs::path spec = fs::temp_directory_path() / "wormsim2_test_perturb.spec";
    { std::ofstream f(spec); f << kSpecContent; }

    wormsim2::NetworkConfig cfg = buildBaseCfg();
    wormsim2::PerturbationConfig::apply(spec, cfg);

    // KD scaled by 0.5
    assert(approx(cfg.channels.at("KD").gbar, 0.02f));
    assert(approx(cfg.neurons[0].channels[0].conductance_density, 0.02f));

    // NCA knocked out
    assert(approx(cfg.channels.at("NCA").gbar, 0.0f));
    assert(approx(cfg.neurons[0].channels[1].conductance_density, 0.0f));

    fs::remove(spec);
    std::cout << "[PASS] test_channel_scale\n";
}

void test_gap_junction_scale() {
    namespace fs = std::filesystem;
    const fs::path spec = fs::temp_directory_path() / "wormsim2_test_perturb_gj.spec";
    { std::ofstream f(spec); f << kSpecContent; }

    wormsim2::NetworkConfig cfg = buildBaseCfg();
    wormsim2::PerturbationConfig::apply(spec, cfg);

    assert(approx(cfg.gap_junctions[0].conductance_nS, 0.8f));

    fs::remove(spec);
    std::cout << "[PASS] test_gap_junction_scale\n";
}

void test_neuron_overrides() {
    namespace fs = std::filesystem;
    const fs::path spec = fs::temp_directory_path() / "wormsim2_test_perturb_n.spec";
    { std::ofstream f(spec); f << kSpecContent; }

    wormsim2::NetworkConfig cfg = buildBaseCfg();
    wormsim2::PerturbationConfig::apply(spec, cfg);

    assert(approx(cfg.neurons[0].v_initial_mV,  -55.0f));
    assert(approx(cfg.neurons[1].capacitance_nF, 12.5f));

    fs::remove(spec);
    std::cout << "[PASS] test_neuron_overrides\n";
}

void test_missing_file_throws() {
    wormsim2::NetworkConfig cfg;
    bool threw = false;
    try {
        wormsim2::PerturbationConfig::apply("/no/such/file.spec", cfg);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    std::cout << "[PASS] test_missing_file_throws\n";
}

void test_unknown_key_warns_not_throws() {
    namespace fs = std::filesystem;
    const fs::path spec = fs::temp_directory_path() / "wormsim2_test_unknown.spec";
    { std::ofstream f(spec); f << "unknown.key = 42\n"; }

    wormsim2::NetworkConfig cfg = buildBaseCfg();
    // Should NOT throw — just print a warning
    wormsim2::PerturbationConfig::apply(spec, cfg);

    fs::remove(spec);
    std::cout << "[PASS] test_unknown_key_warns_not_throws\n";
}

} // anonymous namespace

int main() {
    try {
        test_channel_scale();
        test_gap_junction_scale();
        test_neuron_overrides();
        test_missing_file_throws();
        test_unknown_key_warns_not_throws();
        std::cout << "\nAll perturbation config tests passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
