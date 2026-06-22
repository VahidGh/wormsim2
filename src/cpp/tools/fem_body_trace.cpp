// fem_body_trace — open-loop FEM body driven by sinusoidal anti-phase D/V activation.
//
// Usage: fem_body_trace [--freq Hz] [--T_s seconds] [--dt_ms ms] [--amp 0-1]
//
// Output (stdout): CSV  t_ms,x0,y0,x1,y1,...,x24,y24
//   25 centerline points (head=0 → tail=24); x,y in mm; every kStride steps.
// Stats   (stderr): mesh info, timing.
//
// Default: 0.5 Hz, 4 s, dt=0.5 ms, amplitude=0.5.
// Driven by: a_dorsal = amp*(1 + sin(2π f t))  (DL 0..23, DR 24..47)
//            a_ventral = amp*(1 - sin(2π f t))  (VL 48..71, VR 72..94)

#include "body/FEMBody.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <string>

using namespace wormsim2;

static std::string flag_val(int argc, char** argv, const char* flag, const char* def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return def;
}

int main(int argc, char** argv)
{
    const double freq   = std::atof(flag_val(argc, argv, "--freq",  "0.5").c_str());
    const double T_s    = std::atof(flag_val(argc, argv, "--T_s",   "4.0").c_str());
    const double dt_ms  = std::atof(flag_val(argc, argv, "--dt_ms", "0.5").c_str());
    const double amp    = std::atof(flag_val(argc, argv, "--amp",   "0.5").c_str());
    const int    stride = 10;  // output every 10 steps (50 Hz @ dt=0.5 ms)

    const double dt    = dt_ms * 1.0e-3;
    const int    nsteps = static_cast<int>(T_s / dt);
    const int    ns     = FEMBodyMesh::kNSegments;  // 24

    std::fprintf(stderr, "[fem_body_trace] freq=%.2f Hz  T=%.1f s  dt=%.3f ms  amp=%.2f\n",
                 freq, T_s, dt_ms, amp);

    FEMBodyParams p;
    p.dt = dt;
    FEMBody body;
    body.init(p);

    std::fprintf(stderr, "[fem_body_trace] n_vertices=%d\n", body.n_vertices());

    // Header: t_ms, then x0,y0, x1,y1, ..., x24,y24
    std::printf("t_ms");
    for (int j = 0; j <= ns; ++j)
        std::printf(",x%d,y%d", j, j);
    std::printf("\n");

    std::array<double, kNMuscles> act{};
    for (int step = 0; step < nsteps; ++step) {
        const double t     = step * dt;
        const double a_d   = amp * (1.0 + std::sin(2.0 * M_PI * freq * t));
        const double a_v   = amp * (1.0 - std::sin(2.0 * M_PI * freq * t));
        for (int m = 0;  m < 48;           ++m) act[m] = a_d;
        for (int m = 48; m < kNMuscles;    ++m) act[m] = a_v;
        body.step(act);

        if (step % stride == 0) {
            auto cl = body.centerline();
            std::printf("%d", static_cast<int>(t * 1000.0));
            for (int j = 0; j <= ns; ++j)
                std::printf(",%.5f,%.5f", cl[j].x * 1000.0, cl[j].y * 1000.0);
            std::printf("\n");
        }
    }
    return 0;
}
