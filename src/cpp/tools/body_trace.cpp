// body_trace — eigenworm body simulation driven by the full c302 connectome.
//
// Usage:
//   body_trace <nml_file> <T_ms> <dt_ms> [options]
//
// Loads the c302 NeuroML2 connectome, runs the Hodgkin–Huxley neural integrator
// and the eigenworm reduced body, and emits one CSV row per timestep.
//
// Output (stdout): CSV  t_ms,a0,a1,a2,a3
// Stats   (stderr): load summary, valid NMJ count, any missing neurons
//
// Options:
//   --avb-drive pA           Tonic current to AVBL+AVBR+PVCL+PVCR (forward drive).
//   --sine-avb pA freq_Hz    Sinusoidal: +half → AVB/PVC, -half → AVA/PVD.
//   --db-vb-sine pA freq_Hz  Anti-phase sine to DB vs VB motor neurons directly.
//                            +half-cycle inhibits VB (ventral quiets),
//                            -half-cycle inhibits DB (dorsal quiets).
//                            Best approach to generate body undulation without
//                            proprioceptive feedback (v0.7.0 CPG proxy).
//   --wcon path              Also write a WCON JSON file with body centerline.

#include "io/NetworkInputParser.h"
#include "io/NetworkConfig.h"
#include "neural/NeuralIntegrator.h"
#include "body/EigenwormBody.h"
#include "output/WCONExporter.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

using namespace wormsim2;

// Parse --flag value pairs; returns empty string if flag not found.
static std::string flag_value(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return {};
}
int main(int argc, char** argv)
{
    if (argc < 4) {
        std::fputs(
            "Usage: body_trace <nml_file> <T_ms> <dt_ms> "
            "[--avb-drive <pA>] [--sine-avb <pA> <freq_Hz>] [--wcon <output.wcon>]\n",
            stderr);
        return 1;
    }

    const std::string nml_path  = argv[1];
    const float       T_ms      = std::atof(argv[2]);
    const float       dt_ms     = std::atof(argv[3]);
    const float       avb_drive = [&]() -> float {
        const auto s = flag_value(argc, argv, "--avb-drive");
        return s.empty() ? 0.0f : std::atof(s.c_str());
    }();
    // Sinusoidal alternating drive (mimics CPG without body feedback)
    float sine_amp  = 0.0f;
    float sine_freq = 0.5f;
    bool  use_db_vb = false;   // true → drive DB/VB directly (better for undulation)
    {
        const auto sa  = flag_value(argc, argv, "--sine-avb");
        const auto sdb = flag_value(argc, argv, "--db-vb-sine");
        const auto& picked = sdb.empty() ? sa : sdb;
        use_db_vb = !sdb.empty();
        if (!picked.empty()) {
            sine_amp = std::atof(picked.c_str());
            const char* flag = use_db_vb ? "--db-vb-sine" : "--sine-avb";
            for (int i = 1; i < argc - 2; ++i)
                if (std::strcmp(argv[i], flag) == 0)
                    sine_freq = std::atof(argv[i + 2]);
        }
    }
    const std::string wcon_path = flag_value(argc, argv, "--wcon");

    // ── Load connectome ───────────────────────────────────────────────────────
    NetworkConfig cfg = NetworkInputParser::load(nml_path);
    std::fprintf(stderr,
        "Loaded: %d cells, %d gap junctions, %d chem synapses, %d NMJs\n",
        cfg.neuron_count(), cfg.gap_junction_count(), cfg.synapse_count(),
        static_cast<int>(cfg.nmj_connections.size()));

    // ── Build body model ──────────────────────────────────────────────────────
    EigenwormBody body(cfg);
    std::fprintf(stderr, "EigenwormBody: %d / %d NMJs with valid muscle assignment\n",
        body.valid_nmj_count(),
        static_cast<int>(cfg.nmj_connections.size()));

    // ── External current indices ──────────────────────────────────────────────
    std::vector<float> i_ext(static_cast<std::size_t>(cfg.neuron_count()), 0.0f);

    std::vector<int> avb_indices, ava_indices;
    std::vector<int> db_indices,  vb_indices;

    {
        static const char* const kFwdNeurons[] = {"AVBL[0]", "AVBR[0]", "PVCL[0]", "PVCR[0]"};
        static const char* const kBwdNeurons[] = {"AVAL[0]", "AVAR[0]", "PVDL[0]", "PVDR[0]"};
        // DB1-7 and VB1-11 (c302 forward motor neurons, bilateral)
        static const char* const kDB[] = {
            "DB1[0]","DB2[0]","DB3[0]","DB4[0]","DB5[0]","DB6[0]","DB7[0]"};
        static const char* const kVB[] = {
            "VB1[0]","VB2[0]","VB3[0]","VB4[0]","VB5[0]","VB6[0]",
            "VB7[0]","VB8[0]","VB9[0]","VB10[0]","VB11[0]"};
        for (const char* name : kFwdNeurons) {
            const NeuronDef* nd = cfg.find_neuron(name);
            if (nd) avb_indices.push_back(nd->id);
        }
        for (const char* name : kBwdNeurons) {
            const NeuronDef* nd = cfg.find_neuron(name);
            if (nd) ava_indices.push_back(nd->id);
        }
        for (const char* name : kDB) {
            const NeuronDef* nd = cfg.find_neuron(name);
            if (nd) db_indices.push_back(nd->id);
        }
        for (const char* name : kVB) {
            const NeuronDef* nd = cfg.find_neuron(name);
            if (nd) vb_indices.push_back(nd->id);
        }
    }

    if (avb_drive != 0.0f) {
        for (int idx : avb_indices)
            i_ext[static_cast<std::size_t>(idx)] = avb_drive;
        std::fprintf(stderr, "AVB/PVC tonic drive: %.1f pA\n", avb_drive);
    }
    if (sine_amp != 0.0f) {
        if (use_db_vb)
            std::fprintf(stderr, "DB/VB anti-phase sine: %.1f pA @ %.3f Hz  "
                         "(%zu DB, %zu VB neurons)\n",
                         sine_amp, sine_freq, db_indices.size(), vb_indices.size());
        else
            std::fprintf(stderr, "Sine AVB/AVA drive: %.1f pA @ %.3f Hz\n",
                         sine_amp, sine_freq);
    }

    // ── Run integrator ────────────────────────────────────────────────────────
    NeuralIntegrator ni(cfg);
    WCONExporter     wcon;

    const bool write_wcon = !wcon_path.empty();
    const int  steps      = static_cast<int>(T_ms / dt_ms);

    // WCON: every 10 steps (every 10×dt_ms ms) to keep file size manageable.
    constexpr int kWconStride = 10;

    // CSV header
    std::puts("t_ms,a0,a1,a2,a3");

    // t=0 initial body state
    body.step(ni.state());
    {
        const float* a = body.mode_amplitudes();
        std::printf("%.4f,%.7f,%.7f,%.7f,%.7f\n", 0.0f, a[0], a[1], a[2], a[3]);
        if (write_wcon && 0 % kWconStride == 0)
            wcon.add_frame(0.0f, body.centerline_x(), body.centerline_y(),
                           EigenwormBasis::kNPoints);
    }

    for (int i = 0; i < steps; ++i) {
        // Update time-varying sine drive
        if (sine_amp != 0.0f) {
            const float t_s = (i + 1) * dt_ms * 1e-3f;
            const float s   = sine_amp * std::sin(2.0f * static_cast<float>(M_PI) * sine_freq * t_s);
            if (use_db_vb) {
                // Anti-phase: positive s inhibits VB (ventral quiets), negative inhibits DB
                for (int idx : db_indices)
                    i_ext[static_cast<std::size_t>(idx)] = std::min(0.0f,  s); // inhibit when s<0
                for (int idx : vb_indices)
                    i_ext[static_cast<std::size_t>(idx)] = std::min(0.0f, -s); // inhibit when s>0
            } else {
                for (int idx : avb_indices)
                    i_ext[static_cast<std::size_t>(idx)] = std::max(0.0f,  s);
                for (int idx : ava_indices)
                    i_ext[static_cast<std::size_t>(idx)] = std::max(0.0f, -s);
            }
        }
        ni.step(dt_ms, std::span<const float>{i_ext});
        body.step(ni.state());

        const float t_now = (i + 1) * dt_ms;
        const float* a    = body.mode_amplitudes();
        std::printf("%.4f,%.7f,%.7f,%.7f,%.7f\n", t_now, a[0], a[1], a[2], a[3]);

        if (write_wcon && (i + 1) % kWconStride == 0)
            wcon.add_frame(t_now, body.centerline_x(), body.centerline_y(),
                           EigenwormBasis::kNPoints);
    }

    if (write_wcon) {
        if (wcon.write(wcon_path)) {
            std::fprintf(stderr, "WCON written: %s  (%d frames, %d pts/frame)\n",
                wcon_path.c_str(), wcon.frame_count(), wcon.points_per_frame());
        } else {
            std::fprintf(stderr, "Error: could not write WCON to '%s'\n",
                wcon_path.c_str());
            return 1;
        }
    }

    return 0;
}
