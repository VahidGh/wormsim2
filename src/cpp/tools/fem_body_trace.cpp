// fem_body_trace — open-loop FEM body, sinusoidal or CSV-driven activation.
//
// Sine mode (default):
//   fem_body_trace [--freq Hz] [--T_s s] [--dt_ms ms] [--amp 0-1]
//   a_dorsal = amp*(1 + sin(2π f t))  (DL 0..23, DR 24..47)
//   a_ventral= amp*(1 - sin(2π f t))  (VL 48..71, VR 72..94)
//
// CSV mode (NeuromuscularTuner output):
//   fem_body_trace --activation-csv <path> [--dt_ms ms]
//   CSV format: t_ms, seg00, seg01, ..., seg23   (signed net D-V activation)
//   Positive seg_i → dorsal bend; negative → ventral bend.
//   Activation is linearly interpolated to the simulation timestep.
//
// Output (stdout): CSV  t_ms,x0,y0,x1,y1,...,x24,y24
//   25 centerline points (head=0 → tail=24); x,y in mm; every kStride steps.

#include "body/FEMBody.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace wormsim2;

static std::string flag_val(int argc, char** argv, const char* flag, const char* def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return def;
}

// ── CSV activation loader ─────────────────────────────────────────────────────
struct ActFrame { double t_ms; std::array<double, 24> seg{}; };

static std::vector<ActFrame> load_activation_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "[fem_body_trace] cannot open %s\n", path.c_str());
        std::exit(1);
    }
    std::vector<ActFrame> frames;
    std::string line;
    std::getline(f, line);  // skip header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        ActFrame fr{}; std::string tok;
        std::getline(ss, tok, ','); fr.t_ms = std::stod(tok);
        for (int i = 0; i < 24 && std::getline(ss, tok, ','); ++i)
            fr.seg[i] = std::stod(tok);
        frames.push_back(fr);
    }
    std::fprintf(stderr, "[fem_body_trace] loaded %zu frames from %s\n",
                 frames.size(), path.c_str());
    return frames;
}

// Linear interpolation of net activation at time t_ms.
static std::array<double, 24> interp_act(
        const std::vector<ActFrame>& frames, double t_ms) {
    if (frames.empty()) return {};
    if (t_ms <= frames.front().t_ms) return frames.front().seg;
    if (t_ms >= frames.back().t_ms)  return frames.back().seg;
    auto it = std::lower_bound(frames.begin(), frames.end(), t_ms,
        [](const ActFrame& a, double v){ return a.t_ms < v; });
    const auto& hi = *it; const auto& lo = *(it - 1);
    double alpha = (t_ms - lo.t_ms) / (hi.t_ms - lo.t_ms);
    std::array<double, 24> r{};
    for (int i = 0; i < 24; ++i)
        r[i] = lo.seg[i] + alpha * (hi.seg[i] - lo.seg[i]);
    return r;
}

// Signed net D-V (24 segs) → 95-muscle activation array.
// seg[i] > 0 → dorsal bend; seg[i] < 0 → ventral bend.
static void net_to_muscles(const std::array<double, 24>& net,
                            std::array<double, kNMuscles>& act) {
    act.fill(0.0);
    for (int i = 0; i < 24; ++i) {
        const double d = std::max(0.0,  net[i]);
        const double v = std::max(0.0, -net[i]);
        act[i]      = d;   // DL
        act[24 + i] = d;   // DR
        act[48 + i] = v;   // VL
        if (72 + i < static_cast<int>(kNMuscles)) act[72 + i] = v;  // VR
    }
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    const std::string act_csv = flag_val(argc, argv, "--activation-csv", "");
    const bool csv_mode = !act_csv.empty();

    const double freq  = std::atof(flag_val(argc, argv, "--freq",  "0.5").c_str());
    const double T_s   = std::atof(flag_val(argc, argv, "--T_s",   "4.0").c_str());
    const double dt_ms = std::atof(flag_val(argc, argv, "--dt_ms", "0.5").c_str());
    const double amp   = std::atof(flag_val(argc, argv, "--amp",   "0.5").c_str());
    const int    stride = 10;

    std::vector<ActFrame> csv_frames;
    double run_T_s = T_s;

    if (csv_mode) {
        csv_frames = load_activation_csv(act_csv);
        run_T_s = csv_frames.back().t_ms * 1.0e-3;
        std::fprintf(stderr, "[fem_body_trace] CSV mode  T=%.3f s\n", run_T_s);
    } else {
        std::fprintf(stderr,
            "[fem_body_trace] sine mode  freq=%.2f Hz  T=%.1f s  dt=%.3f ms  amp=%.2f\n",
            freq, run_T_s, dt_ms, amp);
    }

    const double dt     = dt_ms * 1.0e-3;
    const int    nsteps = static_cast<int>(run_T_s / dt);
    const int    ns     = FEMBodyMesh::kNSegments;

    FEMBodyParams p; p.dt = dt;
    FEMBody body; body.init(p);
    std::fprintf(stderr, "[fem_body_trace] n_vertices=%d\n", body.n_vertices());

    std::printf("t_ms");
    for (int j = 0; j <= ns; ++j) std::printf(",x%d,y%d,z%d", j, j, j);
    std::printf("\n");

    std::array<double, kNMuscles> act{};
    for (int step = 0; step < nsteps; ++step) {
        const double t     = step * dt;
        const double t_ms_ = t * 1000.0;

        if (csv_mode) {
            auto net = interp_act(csv_frames, t_ms_);
            net_to_muscles(net, act);
        } else {
            const double a_d = amp * (1.0 + std::sin(2.0 * M_PI * freq * t));
            const double a_v = amp * (1.0 - std::sin(2.0 * M_PI * freq * t));
            for (int m = 0;  m < 48;        ++m) act[m] = a_d;
            for (int m = 48; m < kNMuscles; ++m) act[m] = a_v;
        }
        body.step(act);

        if (step % stride == 0) {
            auto cl = body.centerline();
            std::printf("%d", static_cast<int>(t_ms_));
            for (int j = 0; j <= ns; ++j)
                std::printf(",%.5f,%.5f,%.5f",
                            cl[j].x * 1000.0, cl[j].y * 1000.0, cl[j].z * 1000.0);
            std::printf("\n");
        }
    }
    return 0;
}
