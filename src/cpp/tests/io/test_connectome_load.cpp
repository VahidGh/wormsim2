#include "io/NetworkInputParser.h"
#include "io/NetworkConfig.h"
#include "neural/NeuralIntegrator.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

using namespace wormsim2;

// Release-mode-safe check (assert is a no-op under NDEBUG).
static int g_failures = 0;
#define CHECK(cond, msg)                                                  \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAIL  %s : %s\n", __func__, (msg));    \
            ++g_failures;                                                  \
        }                                                                  \
    } while (false)

// ---------------------------------------------------------------------------
// Locate the c302 data file.
// ---------------------------------------------------------------------------
static std::filesystem::path nml_path() {
    const char* src = std::getenv("WORMSIM2_SOURCE_DIR");
    if (src)
        return std::filesystem::path(src) / "data/c302/c302_C2_Full.net.nml";
    std::filesystem::path p = std::filesystem::current_path();
    for (int i = 0; i < 6; ++i) {
        if (std::filesystem::exists(p / "data/c302/c302_C2_Full.net.nml"))
            return p / "data/c302/c302_C2_Full.net.nml";
        p = p.parent_path();
    }
    return {};
}

// ---------------------------------------------------------------------------
// Test 1: population counts — 302 neurons + 95 muscles = 397 cells
// ---------------------------------------------------------------------------
static void test_population_counts(const NetworkConfig& cfg) {
    CHECK(cfg.neuron_count() == 397,
          "c302_C2_Full must contain 302 neurons + 95 muscles = 397 populations");
    std::puts("PASS  test_population_counts  (397 cells)");
}

// ---------------------------------------------------------------------------
// Test 2: connectivity — gap junctions and chemical synapses resolved
// ---------------------------------------------------------------------------
static void test_connectivity(const NetworkConfig& cfg) {
    CHECK(cfg.gap_junction_count() >= 1000,
          "expected ≥1000 gap junctions in c302_C2_Full");
    CHECK(cfg.synapse_count() >= 1500,
          "expected ≥1500 chemical synapses in c302_C2_Full");

    const int N = cfg.neuron_count();
    for (const auto& gj : cfg.gap_junctions) {
        [[maybe_unused]] const bool ok =
            gj.neuron_a >= 0 && gj.neuron_a < N &&
            gj.neuron_b >= 0 && gj.neuron_b < N;
        CHECK(ok, "gap junction ids out of range");
    }
    for (const auto& syn : cfg.chem_synapses) {
        [[maybe_unused]] const bool ok =
            syn.pre_id  >= 0 && syn.pre_id  < N &&
            syn.post_id >= 0 && syn.post_id < N;
        CHECK(ok, "chem synapse ids out of range");
    }
    std::printf("PASS  test_connectivity  (%d GJ, %d chem synapses, %d NMJs)\n",
                cfg.gap_junction_count(), cfg.synapse_count(),
                static_cast<int>(cfg.nmj_connections.size()));
}

// ---------------------------------------------------------------------------
// Test 3: channel assignments — AVAL[0] gets 4 neuron channels, correct C_m
// ---------------------------------------------------------------------------
static void test_biophysics(const NetworkConfig& cfg) {
    const NeuronDef* const aval = cfg.find_neuron("AVAL[0]");
    CHECK(aval != nullptr, "AVAL[0] must exist");
    if (!aval) return;
    CHECK(aval->channels.size() == 4,
          "AVAL[0] should have 4 channels: Leak k_slow k_fast ca_boyle");
    CHECK(aval->capacitance_nF >= 4.0f && aval->capacitance_nF <= 6.0f,
          "AVAL[0] capacitance should be ~5 pF");
    CHECK(aval->cell_type == "GenericNeuronCell",
          "AVAL[0] cell_type should be GenericNeuronCell");
    std::printf("PASS  test_biophysics  (AVAL C_m=%.1f ch=%zu type=%s)\n",
                aval->capacitance_nF, aval->channels.size(),
                aval->cell_type.c_str());
}

// ---------------------------------------------------------------------------
// Test 4: key neurons present with distinct ids
// ---------------------------------------------------------------------------
static void test_key_neurons(const NetworkConfig& cfg) {
    const NeuronDef* const aval = cfg.find_neuron("AVAL[0]");
    const NeuronDef* const avbl = cfg.find_neuron("AVBL[0]");
    CHECK(aval != nullptr, "AVAL[0] must exist");
    CHECK(avbl != nullptr, "AVBL[0] must exist");
    if (aval && avbl)
        CHECK(aval->id != avbl->id, "AVAL[0] and AVBL[0] must have distinct ids");
    std::puts("PASS  test_key_neurons  (AVAL[0], AVBL[0] present)");
}

// ---------------------------------------------------------------------------
// Test 5: NeuralIntegrator runs 100 steps without NaN
// ---------------------------------------------------------------------------
static void test_integrator_runs(const NetworkConfig& cfg) {
    NeuralIntegrator ni(cfg);
    constexpr float dt = 0.025f;
    for (int i = 0; i < 100; ++i) ni.step(dt);

    const auto vv = ni.voltages();
    int nan_count = 0;
    for (int i = 0; i < cfg.neuron_count(); ++i)
        if (std::isnan(vv[static_cast<std::size_t>(i)])) ++nan_count;
    CHECK(nan_count == 0, "NaN voltages after 100 steps");
    std::printf("PASS  test_integrator_runs  (V[AVAL[0]]=%.2f mV after 2.5 ms)\n",
                static_cast<double>(vv[0]));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    const auto path = nml_path();
    if (path.empty() || !std::filesystem::exists(path)) {
        std::printf("SKIP  test_connectome_load: c302_C2_Full.net.nml not found "
                    "(set WORMSIM2_SOURCE_DIR or run from repo root)\n");
        return 0;
    }

    std::printf("Loading: %s\n", path.string().c_str());
    const NetworkConfig cfg = NetworkInputParser::load(path.string());

    test_population_counts(cfg);
    test_connectivity(cfg);
    test_biophysics(cfg);
    test_key_neurons(cfg);
    test_integrator_runs(cfg);

    if (g_failures > 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("ALL PASS  test_connectome_load");
    return 0;
}
