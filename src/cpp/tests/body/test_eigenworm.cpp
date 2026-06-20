// test_eigenworm — CTest suite for the eigenworm body module.
//
// Assertions:
//   1. Basis unit-norm: each Stephens eigenvector has L2-norm ≈ 1 (tolerance 0.01).
//   2. Zero NMJ activation → zero mode amplitudes (|a[m]| < 1e-6).
//   3. Pure dorsal activation → kappa > 0 at all body segments AND |a[0]| > 0.01.

#include "body/EigenwormBasis.h"
#include "body/EigenwormBody.h"
#include "io/NetworkInputParser.h"
#include "neural/NeuralState.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace wormsim2;
namespace fs = std::filesystem;

// Locate the c302 NML file by walking up from the binary's directory.
static std::string find_nml() {
#ifdef WORMSIM2_SOURCE_DIR
    fs::path p = fs::path(WORMSIM2_SOURCE_DIR) / "data/c302/c302_C2_Full.net.nml";
    if (fs::exists(p)) return p.string();
#endif
    // Walk up from CWD
    fs::path cwd = fs::current_path();
    for (int depth = 0; depth < 8; ++depth) {
        auto p2 = cwd / "data/c302/c302_C2_Full.net.nml";
        if (fs::exists(p2)) return p2.string();
        cwd = cwd.parent_path();
    }
    return {};
}

int main() {
    // ── Test 1: Basis unit-norm ───────────────────────────────────────────────
    {
        using B = EigenwormBasis;
        for (int m = 0; m < B::kNModes; ++m) {
            float norm2 = 0.0f;
            for (int j = 0; j < B::kNSeg; ++j)
                norm2 += B::kBasis[m][j] * B::kBasis[m][j];
            const float norm = std::sqrt(norm2);
            const bool  ok   = std::fabs(norm - 1.0f) < 0.01f;
            std::printf("Test 1 / mode %d: norm = %.6f  %s\n", m, norm,
                        ok ? "(PASS)" : "(FAIL)");
            assert(ok);
        }
    }

    // ── Load connectome ───────────────────────────────────────────────────────
    const std::string nml = find_nml();
    if (nml.empty()) {
        std::fputs("SKIP: c302 NML not found — skipping tests 2+3\n", stdout);
        return 0;
    }
    const NetworkConfig cfg = NetworkInputParser::load(nml);

    // Construct a minimal NeuralState with only s_nmj populated (no gate/v arrays
    // needed — EigenwormBody only reads s_nmj).
    NeuralState st;
    st.s_nmj.assign(cfg.nmj_connections.size(), 0.0f);

    EigenwormBody body(cfg);

    // ── Test 2: Zero NMJ → zero mode amplitudes ───────────────────────────────
    {
        // s_nmj is zero-initialised by NeuralState::allocate; just call step().
        body.step(st);
        const float* a = body.mode_amplitudes();
        for (int m = 0; m < EigenwormBasis::kNModes; ++m) {
            const bool ok = std::fabs(a[m]) < 1e-6f;
            std::printf("Test 2 / mode %d: a = %.2e  %s\n", m, a[m],
                        ok ? "(PASS)" : "(FAIL)");
            assert(ok);
        }
    }

    // ── Test 3: Pure dorsal NMJ → positive curvature and non-zero a[0] ───────
    {
        // Force all dorsal (MDL/MDR) NMJ activations to 1.0, ventral to 0.0.
        const auto& conns = cfg.nmj_connections;
        for (std::size_t ki = 0; ki < conns.size(); ++ki) {
            const int pid = conns[ki].post_neuron_id;
            if (pid < 0 || pid >= (int)cfg.neurons.size()) continue;
            const std::string& name = cfg.neurons[pid].name;
            if (name.size() > 1 && name[1] == 'D')
                st.s_nmj[ki] = 1.0f;  // MDL / MDR → dorsal
            else
                st.s_nmj[ki] = 0.0f;  // MVL / MVR → ventral
        }

        body.step(st);
        const float* kappa = body.curvature();
        const float* a     = body.mode_amplitudes();

        // All curvatures must be >= 0 (dorsal bend).
        bool all_non_neg = true;
        for (int j = 0; j < EigenwormBasis::kNSeg; ++j)
            if (kappa[j] < 0.0f) { all_non_neg = false; break; }
        std::printf("Test 3a: all kappa >= 0  %s\n", all_non_neg ? "(PASS)" : "(FAIL)");
        assert(all_non_neg);

        // Dominant mode must have significant amplitude.
        const bool big_a0 = std::fabs(a[0]) > 0.01f;
        std::printf("Test 3b: |a[0]| = %.4f > 0.01  %s\n", std::fabs(a[0]),
                    big_a0 ? "(PASS)" : "(FAIL)");
        assert(big_a0);
    }

    std::puts("All test_eigenworm assertions PASS");
    return 0;
}
