// test_wcon — CTest suite for the WCON exporter.
//
// Assertions:
//   1. Frame count: add_frame increments frame_count() correctly.
//   2. Points per frame: points_per_frame() reflects n_pts passed to first add_frame.
//   3. Write round-trip: written file is valid JSON (starts with '{', contains "data").

#include "output/WCONExporter.h"

#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace wormsim2;
namespace fs = std::filesystem;

int main() {
    // ── Test 1: Frame count ───────────────────────────────────────────────────
    {
        WCONExporter exp;
        assert(exp.frame_count() == 0);

        std::array<float, 5> x{0.0f, 0.1f, 0.2f, 0.3f, 0.4f};
        std::array<float, 5> y{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        exp.add_frame(0.0f, x.data(), y.data(), 5);
        exp.add_frame(1.0f, x.data(), y.data(), 5);
        assert(exp.frame_count() == 2);
        std::puts("Test 1: frame_count  (PASS)");
    }

    // ── Test 2: Points per frame ──────────────────────────────────────────────
    {
        WCONExporter exp;
        std::array<float, 49> x{}, y{};
        exp.add_frame(0.0f, x.data(), y.data(), 49);
        const bool ok = (exp.points_per_frame() == 49);
        std::printf("Test 2: points_per_frame = %d  %s\n",
                    exp.points_per_frame(), ok ? "(PASS)" : "(FAIL)");
        assert(ok);
    }

    // ── Test 3: Write round-trip ─────────────────────────────────────────────
    {
        WCONExporter exp;
        constexpr int kN = 49;
        std::array<float, kN> x{}, y{};
        for (int i = 0; i < kN; ++i) {
            x[i] = 0.02f * i;
            y[i] = 0.0f;
        }
        exp.add_frame(0.0f,  x.data(), y.data(), kN);
        exp.add_frame(10.0f, x.data(), y.data(), kN);

        const fs::path tmp = fs::temp_directory_path() / "test_wcon_output.wcon";
        assert(exp.write(tmp.string()));

        std::ifstream f(tmp.string());
        std::stringstream buf;
        buf << f.rdbuf();
        const std::string content = buf.str();

        const bool starts_brace = (!content.empty() && content[0] == '{');
        const bool has_data     = content.find("\"data\"")    != std::string::npos;
        const bool has_units    = content.find("\"units\"")   != std::string::npos;
        const bool has_version  = content.find("tracker-commons") != std::string::npos;

        std::printf("Test 3: valid JSON structure  %s\n",
                    (starts_brace && has_data && has_units && has_version)
                    ? "(PASS)" : "(FAIL)");
        assert(starts_brace && has_data && has_units && has_version);

        fs::remove(tmp);
    }

    std::puts("All test_wcon assertions PASS");
    return 0;
}
