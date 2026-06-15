#include "neural/NeuralIntegrator.h"
#include "io/NetworkConfig.h"

#include <cassert>
#include <cmath>
#include <cstdio>

using namespace wormsim2;

// Build a minimal single-neuron NetworkConfig.
static NetworkConfig make_single(float C_pF, float v0_mV,
                                  const char* ch_id, float g_nS) {
    NetworkConfig cfg;
    cfg.source_format = "test";
    NeuronDef n;
    n.id             = 0;
    n.name           = "N0";
    n.capacitance_nF = C_pF;   // treated as pF in NeuralIntegrator
    n.v_initial_mV   = v0_mV;
    n.channels.push_back(ChannelAssignment{ ch_id, g_nS });
    cfg.neurons.push_back(std::move(n));
    return cfg;
}

// ---------------------------------------------------------------------------
// Test 1: NCA leak at reversal — voltage should not change
// ---------------------------------------------------------------------------
static void test_leak_at_reversal() {
    // NCA: E_rev = -20 mV. Set V_init = E_rev → I_NCA = 0 → V stays constant.
    constexpr float E_NCA = -20.0f;
    auto cfg = make_single(10.0f, E_NCA, "NCA", 0.8f);
    NeuralIntegrator ni(cfg);

    for (int step = 0; step < 1000; ++step)
        ni.step(0.025f);

    const float V = ni.voltages()[0];
    assert(std::abs(V - E_NCA) < 1e-3f &&
           "test_leak_at_reversal: V should stay at E_rev");
    std::puts("PASS  test_leak_at_reversal");
}

// ---------------------------------------------------------------------------
// Test 2: NCA leak — exponential decay toward reversal potential
//   Analytic: V(t) = E + (V0 - E) * exp(-g/C * t)
//   with C=10, g=0.8, E=-20, V0=-65 → tau=12.5 ms
// ---------------------------------------------------------------------------
static void test_leak_decay() {
    constexpr float C    = 10.0f;
    constexpr float g    = 0.8f;
    constexpr float E    = -20.0f;
    constexpr float V0   = -65.0f;
    constexpr float tau  = C / g;          // 12.5 ms
    constexpr float T    = 25.0f;          // simulate 25 ms (= 2τ)
    constexpr float dt   = 0.025f;
    constexpr int   N    = static_cast<int>(T / dt);

    auto cfg = make_single(C, V0, "NCA", g);
    NeuralIntegrator ni(cfg);

    for (int i = 0; i < N; ++i)
        ni.step(dt);

    const float V_sim    = ni.voltages()[0];
    const float V_exact  = E + (V0 - E) * std::exp(-T / tau);
    // Conductance method is exact for a pure leak → tolerance is float rounding only
    assert(std::abs(V_sim - V_exact) < 0.05f &&
           "test_leak_decay: simulated V diverges from analytic");
    std::puts("PASS  test_leak_decay");
}

// ---------------------------------------------------------------------------
// Test 3: KD gate activation direction (short window, V barely moves)
//   At V = -40 mV, n_inf ≈ 0.109, initial n = 0.05 < n_inf → gate must rise.
//   Check over 2 ms: V changes < 0.1 mV so n_inf stays above n_initial.
// ---------------------------------------------------------------------------
static void test_kd_gate_direction() {
    auto cfg = make_single(10.0f, -40.0f, "KD", 5.0f);
    NeuralIntegrator ni(cfg);

    const float n0 = ni.state().gate[0];   // KD has 1 gate at stride offset 0

    // 2 ms: V barely drifts (< 0.1 mV), gate rises from 0.05 toward n_inf ≈ 0.109
    for (int i = 0; i < 80; ++i)
        ni.step(0.025f);

    const float n1 = ni.state().gate[0];
    assert(n1 > n0 && "test_kd_gate_direction: n should increase toward n_inf");
    std::puts("PASS  test_kd_gate_direction");
}

// ---------------------------------------------------------------------------
// Test 4: External current depolarises neuron
// ---------------------------------------------------------------------------
static void test_external_current() {
    auto cfg = make_single(10.0f, -65.0f, "NCA", 0.8f);
    NeuralIntegrator ni(cfg);

    const float V_before = ni.voltages()[0];
    const std::vector<float> i_ext = { 200.0f };  // 200 pA depolarising

    for (int i = 0; i < 400; ++i)
        ni.step(0.025f, i_ext);

    const float V_after = ni.voltages()[0];
    assert(V_after > V_before && "test_external_current: depolarising I_ext must raise V");
    std::puts("PASS  test_external_current");
}

int main() {
    test_leak_at_reversal();
    test_leak_decay();
    test_kd_gate_direction();
    test_external_current();
    std::puts("\nAll HH single-neuron tests passed.");
    return 0;
}
