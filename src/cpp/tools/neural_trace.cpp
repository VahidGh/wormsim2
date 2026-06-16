// neural_trace — C++ data runner for notebook demonstrations
//
// Runs NeuralIntegrator scenarios and emits CSV to stdout.
// Python notebook reads CSV → plots → compares against reference.
//
// Usage: neural_trace <scenario> [params...]
//   nca_decay    C_m V0 T dt
//   kd_gate      C_m V0 T dt
//   gap_junc     C_m g_gj V0_A V0_B T dt
//   multi_ch     C_m V0 T dt I_pulse t_on t_off
//   chan_kinetics channel_id V_min V_max n_points
// Units: mV · ms · pA · pF · nS

#include "neural/NeuralIntegrator.h"
#include "neural/ChannelKinetics.h"
#include "io/NetworkConfig.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

using namespace wormsim2;

static NeuronDef make_neuron(int id, const char* name, float C_pF, float V0,
                              std::initializer_list<std::pair<const char*, float>> chs)
{
    NeuronDef n;
    n.id             = id;
    n.name           = name;
    n.capacitance_nF = C_pF;
    n.v_initial_mV   = V0;
    for (auto [ch, g] : chs)
        n.channels.push_back(ChannelAssignment{ch, g});
    return n;
}

// ---------------------------------------------------------------------------
// nca_decay — NCA leak neuron: exponential decay to E_rev
//   Input:  C_m(pF)  V0(mV)  T(ms)  dt(ms)
//   Output: t,V
// ---------------------------------------------------------------------------
static int nca_decay(int argc, char** argv)
{
    if (argc < 4) { std::fputs("nca_decay: C_m V0 T dt\n", stderr); return 1; }
    const float C_m = std::atof(argv[0]);
    const float V0  = std::atof(argv[1]);
    const float T   = std::atof(argv[2]);
    const float dt  = std::atof(argv[3]);

    NetworkConfig cfg;
    cfg.source_format = "tool";
    cfg.neurons.push_back(make_neuron(0, "N0", C_m, V0,
        {{"NCA", channel_spec("NCA").g_bar}}));
    NeuralIntegrator ni(cfg);

    std::printf("t,V\n%.6f,%.6f\n", 0.0f, V0);
    const int steps = static_cast<int>(T / dt);
    for (int i = 0; i < steps; ++i) {
        ni.step(dt);
        std::printf("%.6f,%.6f\n", (i + 1) * dt, ni.voltages()[0]);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// kd_gate — KD-only neuron: gate activation toward n_inf
//   Input:  C_m(pF)  V0(mV)  T(ms)  dt(ms)
//   Output: t,V,n   (n = KD gate variable; g_eff = g_bar * n^2)
// ---------------------------------------------------------------------------
static int kd_gate(int argc, char** argv)
{
    if (argc < 4) { std::fputs("kd_gate: C_m V0 T dt\n", stderr); return 1; }
    const float C_m = std::atof(argv[0]);
    const float V0  = std::atof(argv[1]);
    const float T   = std::atof(argv[2]);
    const float dt  = std::atof(argv[3]);

    NetworkConfig cfg;
    cfg.source_format = "tool";
    cfg.neurons.push_back(make_neuron(0, "N0", C_m, V0,
        {{"KD", channel_spec("KD").g_bar}}));
    NeuralIntegrator ni(cfg);

    std::printf("t,V,n\n%.6f,%.6f,%.6f\n", 0.0f, V0, ni.state().gate[0]);
    const int steps = static_cast<int>(T / dt);
    for (int i = 0; i < steps; ++i) {
        ni.step(dt);
        std::printf("%.6f,%.6f,%.6f\n",
                    (i + 1) * dt, ni.voltages()[0], ni.state().gate[0]);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// gap_junc — two neurons (no channels) coupled by one gap junction
//   Input:  C_m(pF)  g_gj(nS)  V0_A(mV)  V0_B(mV)  T(ms)  dt(ms)
//   Output: t,V_A,V_B
// ---------------------------------------------------------------------------
static int gap_junc(int argc, char** argv)
{
    if (argc < 6) { std::fputs("gap_junc: C_m g_gj V0_A V0_B T dt\n", stderr); return 1; }
    const float C_m  = std::atof(argv[0]);
    const float g_gj = std::atof(argv[1]);
    const float V0_A = std::atof(argv[2]);
    const float V0_B = std::atof(argv[3]);
    const float T    = std::atof(argv[4]);
    const float dt   = std::atof(argv[5]);

    NetworkConfig cfg;
    cfg.source_format = "tool";
    cfg.neurons.push_back(make_neuron(0, "A", C_m, V0_A, {}));
    cfg.neurons.push_back(make_neuron(1, "B", C_m, V0_B, {}));
    cfg.gap_junctions.push_back(GapJunctionDef{0, 1, g_gj});
    NeuralIntegrator ni(cfg);

    std::printf("t,V_A,V_B\n%.6f,%.6f,%.6f\n", 0.0f, V0_A, V0_B);
    const int steps = static_cast<int>(T / dt);
    for (int i = 0; i < steps; ++i) {
        ni.step(dt);
        const auto V = ni.voltages();
        std::printf("%.6f,%.6f,%.6f\n", (i + 1) * dt, V[0], V[1]);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// multi_ch — NCA + KD with external current pulse
//   Input:  C_m(pF)  V0(mV)  T(ms)  dt(ms)  I_pulse(pA)  t_on(ms)  t_off(ms)
//   Output: t,V,n   (n = KD gate; NCA has 0 gates so gate[0] belongs to KD)
// ---------------------------------------------------------------------------
static int multi_ch(int argc, char** argv)
{
    if (argc < 7) {
        std::fputs("multi_ch: C_m V0 T dt I_pulse t_on t_off\n", stderr); return 1;
    }
    const float C_m     = std::atof(argv[0]);
    const float V0      = std::atof(argv[1]);
    const float T       = std::atof(argv[2]);
    const float dt      = std::atof(argv[3]);
    const float I_pulse = std::atof(argv[4]);
    const float t_on    = std::atof(argv[5]);
    const float t_off   = std::atof(argv[6]);

    NetworkConfig cfg;
    cfg.source_format = "tool";
    cfg.neurons.push_back(make_neuron(0, "N0", C_m, V0,
        {{"NCA", channel_spec("NCA").g_bar},
         {"KD",  channel_spec("KD").g_bar}}));
    NeuralIntegrator ni(cfg);

    const std::vector<float> zero_i{0.0f};
    const std::vector<float> pulse_i{I_pulse};

    std::printf("t,V,n\n%.6f,%.6f,%.6f\n", 0.0f, V0, ni.state().gate[0]);
    const int steps = static_cast<int>(T / dt);
    for (int i = 0; i < steps; ++i) {
        const float t_now = (i + 1) * dt;
        const bool in_pulse = (t_now >= t_on && t_now < t_off);
        ni.step(dt, in_pulse ? std::span{pulse_i} : std::span{zero_i});
        std::printf("%.6f,%.6f,%.6f\n",
                    t_now, ni.voltages()[0], ni.state().gate[0]);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// boyle2008 — single-compartment neuron with k_slow_bc + k_fast_bc + leak_bc
//   Parameters from openworm/CElegansNeuroML k_slow.mod / k_fast.mod (Boyle 2008)
//   Input:  C_m(pF)  g_kslow(nS)  g_kfast(nS)  g_leak(nS)  V0(mV)
//           T(ms)  dt(ms)  I_pulse(pA)  t_on(ms)  t_off(ms)
//   Output: t,V,n,p,q   (n=k_slow gate; p,q=k_fast gates)
// ---------------------------------------------------------------------------
static int boyle2008(int argc, char** argv)
{
    if (argc < 10) {
        std::fputs("boyle2008: C_m g_kslow g_kfast g_leak V0 T dt I_pulse t_on t_off\n",
                   stderr);
        return 1;
    }
    const float C_m     = std::atof(argv[0]);
    const float g_kslow = std::atof(argv[1]);
    const float g_kfast = std::atof(argv[2]);
    const float g_leak  = std::atof(argv[3]);
    const float V0      = std::atof(argv[4]);
    const float T       = std::atof(argv[5]);
    const float dt      = std::atof(argv[6]);
    const float I_pulse = std::atof(argv[7]);
    const float t_on    = std::atof(argv[8]);
    const float t_off   = std::atof(argv[9]);

    NetworkConfig cfg;
    cfg.source_format = "tool";
    cfg.neurons.push_back(make_neuron(0, "N0", C_m, V0,
        {{"KSLOW_BC", g_kslow},
         {"KFAST_BC", g_kfast},
         {"LEAK_BC",  g_leak}}));
    NeuralIntegrator ni(cfg);

    const std::vector<float> zero_i{0.0f};
    const std::vector<float> pulse_i{I_pulse};

    // Gate layout: KSLOW_BC has 1 gate (index 0 = n),
    //              KFAST_BC has 2 gates (index 1 = p, index 2 = q)
    std::printf("t,V,n,p,q\n");
    std::printf("%.6f,%.6f,%.6f,%.6f,%.6f\n",
                0.0f, V0,
                ni.state().gate[0], ni.state().gate[1], ni.state().gate[2]);

    const int steps = static_cast<int>(T / dt);
    for (int i = 0; i < steps; ++i) {
        const float t_now = (i + 1) * dt;
        const bool  pulse = (t_now >= t_on && t_now < t_off);
        ni.step(dt, pulse ? std::span{pulse_i} : std::span{zero_i});
        const auto& s = ni.state();
        std::printf("%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    t_now, ni.voltages()[0],
                    s.gate[0], s.gate[1], s.gate[2]);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// chan_kinetics — voltage sweep: x_inf(V) and tau_x(V) for each gate
//   Input:  channel_id  V_min(mV)  V_max(mV)  n_points
//   Output: V,gate_idx,x_inf,tau_ms
// ---------------------------------------------------------------------------
static int chan_kinetics(int argc, char** argv)
{
    if (argc < 4) {
        std::fputs("chan_kinetics: channel_id V_min V_max n_points\n", stderr); return 1;
    }
    const std::string ch_id = argv[0];
    const float V_min = std::atof(argv[1]);
    const float V_max = std::atof(argv[2]);
    const int   n_pts = std::atoi(argv[3]);

    const ChannelSpec& spec = channel_spec(ch_id);

    std::printf("V,gate_idx,x_inf,tau_ms\n");
    for (int k = 0; k < n_pts; ++k) {
        const float V = V_min + k * (V_max - V_min) / static_cast<float>(n_pts - 1);
        if (spec.gates.empty()) {
            // Passive/leak: no voltage-dependent gates
            std::printf("%.4f,0,1.000000,0.000000\n", V);
        } else {
            for (int gi = 0; gi < static_cast<int>(spec.gates.size()); ++gi) {
                const float xi  = x_inf(V, spec.gates[gi]);
                const float tau = tau_x(V, spec.gates[gi]);
                std::printf("%.4f,%d,%.6f,%.6f\n", V, gi, xi, tau);
            }
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fputs(
            "Usage: neural_trace <scenario> [params...]\n"
            "Scenarios:\n"
            "  nca_decay    C_m V0 T dt\n"
            "  kd_gate      C_m V0 T dt\n"
            "  gap_junc     C_m g_gj V0_A V0_B T dt\n"
            "  multi_ch     C_m V0 T dt I_pulse t_on t_off\n"
            "  boyle2008    C_m g_kslow g_kfast g_leak V0 T dt I_pulse t_on t_off\n"
            "  chan_kinetics channel_id V_min V_max n_points\n"
            "Units: mV · ms · pA · pF · nS\n",
            stderr);
        return 1;
    }

    const std::string scenario = argv[1];
    argc -= 2;
    argv += 2;

    if (scenario == "nca_decay")    return nca_decay(argc, argv);
    if (scenario == "kd_gate")      return kd_gate(argc, argv);
    if (scenario == "gap_junc")     return gap_junc(argc, argv);
    if (scenario == "multi_ch")     return multi_ch(argc, argv);
    if (scenario == "boyle2008")    return boyle2008(argc, argv);
    if (scenario == "chan_kinetics") return chan_kinetics(argc, argv);

    std::fprintf(stderr, "Unknown scenario: %s\n", scenario.c_str());
    return 1;
}
