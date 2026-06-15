#include "neural/NeuralIntegrator.h"
#include "io/NetworkConfig.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>

using namespace wormsim2;

// Two bare neurons (no channels) joined by one gap junction.
static NetworkConfig make_two_coupled(float C_pF, float V0, float V1, float g_gap_nS) {
    NetworkConfig cfg;
    cfg.source_format = "test";
    for (int id = 0; id < 2; ++id) {
        NeuronDef n;
        n.id             = id;
        n.name           = (id == 0) ? "N0" : "N1";
        n.capacitance_nF = C_pF;
        n.v_initial_mV   = (id == 0) ? V0 : V1;
        cfg.neurons.push_back(std::move(n));
    }
    cfg.gap_junctions.push_back(GapJunctionDef{ .neuron_a = 0, .neuron_b = 1,
                                                 .conductance_nS = g_gap_nS });
    return cfg;
}

// ---------------------------------------------------------------------------
// Test 1: Mean voltage conserved when no channels present
//   With equal C_m and no leak:
//     d(V0+V1)/dt = 0  (gap current cancels)
//   So V_avg must be constant to float tolerance.
// ---------------------------------------------------------------------------
static void test_mean_conserved() {
    constexpr float V0 = -80.0f, V1 = -40.0f;
    constexpr float V_avg = (V0 + V1) * 0.5f;
    auto cfg = make_two_coupled(10.0f, V0, V1, 1.0f);
    NeuralIntegrator ni(cfg);

    for (int i = 0; i < 4000; ++i)
        ni.step(0.025f);

    const auto V = ni.voltages();
    const float mean = (V[0] + V[1]) * 0.5f;
    assert(std::abs(mean - V_avg) < 0.01f &&
           "test_mean_conserved: (V0+V1)/2 must be constant with no channels");
    std::puts("PASS  test_mean_conserved");
}

// ---------------------------------------------------------------------------
// Test 2: Voltage difference decays exponentially
//   d(V0-V1)/dt = -2*g_gap/C_m * (V0-V1)
//   V_diff(T) = V_diff_0 * exp(-2*g/C * T)
//   Conductance method is implicit → first-order accurate in dt
// ---------------------------------------------------------------------------
static void test_diff_decay() {
    constexpr float C    = 10.0f;
    constexpr float g    = 1.0f;
    constexpr float V0   = -80.0f, V1 = -40.0f;
    constexpr float T    = 25.0f;         // ms  (≈ 2.5 × tau = 2.5 × C/(2g) = 12.5)
    constexpr float dt   = 0.025f;
    constexpr int   N    = static_cast<int>(T / dt);

    auto cfg = make_two_coupled(C, V0, V1, g);
    NeuralIntegrator ni(cfg);

    for (int i = 0; i < N; ++i)
        ni.step(dt);

    const auto V       = ni.voltages();
    const float V_diff = V[0] - V[1];
    const float V_exact = (V0 - V1) * std::exp(-2.0f * g / C * T);

    // Conductance method is O(dt) accurate — allow 2 % tolerance
    assert(std::abs(V_diff - V_exact) < std::abs(V_exact) * 0.02f + 0.01f &&
           "test_diff_decay: voltage difference doesn't match analytic");
    std::puts("PASS  test_diff_decay");
}

// ---------------------------------------------------------------------------
// Test 3: Voltages converge (diff → 0) over long run
// ---------------------------------------------------------------------------
static void test_convergence() {
    auto cfg = make_two_coupled(10.0f, -80.0f, -40.0f, 1.0f);
    NeuralIntegrator ni(cfg);

    for (int i = 0; i < 20000; ++i)   // 500 ms >> tau
        ni.step(0.025f);

    const auto V = ni.voltages();
    assert(std::abs(V[0] - V[1]) < 0.5f &&
           "test_convergence: voltages should equilibrate over 500 ms");
    std::puts("PASS  test_convergence");
}

int main() {
    test_mean_conserved();
    test_diff_decay();
    test_convergence();
    std::puts("\nAll gap junction tests passed.");
    return 0;
}
