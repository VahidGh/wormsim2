#include "neural/NeuralIntegrator.h"
#include "io/NetworkConfig.h"

#include <cassert>
#include <cstdio>
#include <span>
#include <vector>

using namespace wormsim2;

// Build a minimal motoneuron→muscle NetworkConfig with one NMJ.
// Motor neuron (id=0): NCA channel, starts at -65 mV.
// Muscle cell (id=1): KSLOW_BC + KFAST_BC + CA_BOYLE + LEAK_BC, starts at -75 mV.
static NetworkConfig make_nmj_network() {
    NetworkConfig cfg;
    cfg.source_format = "test";

    // Motor neuron
    NeuronDef mn;
    mn.id             = 0;
    mn.name           = "DB1";
    mn.cell_type      = "GenericNeuronCell";
    mn.capacitance_nF = 5.0f;
    mn.v_initial_mV   = -65.0f;
    mn.channels.push_back(ChannelAssignment{"NCA", 0.8f});
    cfg.neurons.push_back(std::move(mn));

    // Muscle cell
    NeuronDef mc;
    mc.id             = 1;
    mc.name           = "BWM_01";
    mc.cell_type      = "GenericMuscleCell";
    mc.capacitance_nF = 72.38f;
    mc.v_initial_mV   = -75.0f;
    mc.channels.push_back(ChannelAssignment{"KSLOW_BC", 31.543f});
    mc.channels.push_back(ChannelAssignment{"KFAST_BC", 28.950f});
    mc.channels.push_back(ChannelAssignment{"CA_BOYLE", 15.938f});
    mc.channels.push_back(ChannelAssignment{"LEAK_BC",  1.399f});
    cfg.neurons.push_back(std::move(mc));

    // E_rev overrides for muscle channels
    auto ch = [&](const char* id, float erev) {
        ChannelDef d; d.id = id; d.e_rev_mV = erev; cfg.channels[id] = d;
    };
    ch("KSLOW_BC", -64.3461f);
    ch("KFAST_BC", -54.9998f);
    ch("CA_BOYLE",  49.11f);
    ch("LEAK_BC",   10.0f);

    // NMJ: motor neuron 0 → muscle 1
    NMJDef nmj;
    nmj.motor_neuron_id = 0;
    nmj.muscle_id       = 0;
    nmj.post_neuron_id  = 1;
    nmj.weight          = 1.0f;
    cfg.nmj_connections.push_back(nmj);

    return cfg;
}

// ---------------------------------------------------------------------------
// Test 1: NMJ depolarizes muscle — motor neuron spike → muscle V rises
// ---------------------------------------------------------------------------
static void test_nmj_drives_depolarization() {
    auto cfg = make_nmj_network();
    NeuralIntegrator ni(cfg);

    const float dt      = 0.025f;
    const int   steps   = static_cast<int>(200.0f / dt);  // 200 ms

    // Depolarize motoneuron with strong current (force it above spike threshold)
    const std::vector<float> I_ext_stim{400.0f, -120.0f};  // 400 pA to MN, -120 pA to muscle
    const std::vector<float> I_ext_hold{0.0f,   -120.0f};  // hold after stimulus

    const float V_muscle_init = ni.voltages()[1];

    // Drive motoneuron for 50 ms, then observe
    const int stim_steps = static_cast<int>(50.0f / dt);
    for (int i = 0; i < stim_steps; ++i)
        ni.step(dt, std::span{I_ext_stim});
    for (int i = stim_steps; i < steps; ++i)
        ni.step(dt, std::span{I_ext_hold});

    const float V_motor_final  = ni.voltages()[0];
    const float V_muscle_final = ni.voltages()[1];

    std::printf("test_nmj_drives_depolarization:\n");
    std::printf("  V_motor_initial  = %.2f mV\n", -65.0f);
    std::printf("  V_motor_final    = %.2f mV\n", V_motor_final);
    std::printf("  V_muscle_initial = %.2f mV\n", V_muscle_init);
    std::printf("  V_muscle_final   = %.2f mV\n", V_muscle_final);
    std::printf("  ΔV_muscle        = %.2f mV\n", V_muscle_final - V_muscle_init);

    // Muscle must depolarize: at least +2 mV above its initial value
    assert(V_muscle_final > V_muscle_init + 2.0f &&
           "NMJ must depolarize muscle by at least 2 mV");
    std::printf("  PASS\n\n");
}

// ---------------------------------------------------------------------------
// Test 2: No NMJ → muscle stays at rest (control)
// ---------------------------------------------------------------------------
static void test_no_nmj_muscle_stays_at_rest() {
    auto cfg = make_nmj_network();
    cfg.nmj_connections.clear();  // remove NMJ
    NeuralIntegrator ni(cfg);

    const float dt    = 0.025f;
    const int   steps = static_cast<int>(200.0f / dt);

    const std::vector<float> I_ext{400.0f, -120.0f};
    for (int i = 0; i < steps; ++i)
        ni.step(dt, std::span{I_ext});

    const float V_muscle_final = ni.voltages()[1];

    std::printf("test_no_nmj_muscle_stays_at_rest:\n");
    std::printf("  V_muscle_final = %.2f mV  (no NMJ — should stay near -75 mV)\n",
                V_muscle_final);
    // Without NMJ, muscle should remain within 5 mV of its initial -75 mV
    assert(V_muscle_final > -80.0f && V_muscle_final < -70.0f &&
           "Without NMJ, muscle voltage should remain near rest (-75 mV ± 5 mV)");
    std::printf("  PASS\n\n");
}

// ---------------------------------------------------------------------------
// Test 3: s_nmj allocation matches nmj_connections size
// ---------------------------------------------------------------------------
static void test_nmj_state_allocation() {
    auto cfg = make_nmj_network();
    NeuralIntegrator ni(cfg);

    assert(ni.state().n_nmj() == cfg.nmj_connections.size() &&
           "s_nmj size must match nmj_connections count");
    assert(ni.state().n_nmj() == 1 && "Expected exactly 1 NMJ in test network");

    std::printf("test_nmj_state_allocation:\n");
    std::printf("  s_nmj.size() = %zu  PASS\n\n", ni.state().n_nmj());
}

int main() {
    test_nmj_state_allocation();
    test_nmj_drives_depolarization();
    test_no_nmj_muscle_stays_at_rest();
    std::printf("All NMJ tests PASS\n");
    return 0;
}
