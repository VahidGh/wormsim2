#include "output/WCONExporter.h"

#include <cstdio>
#include <cstring>

namespace wormsim2 {

void WCONExporter::add_frame(float t_ms, const float* x, const float* y,
                              int n_pts) {
    if (n_pts_ == 0) n_pts_ = n_pts;
    Frame f;
    f.t_s = t_ms * 1e-3f;
    f.x.assign(x, x + n_pts);
    f.y.assign(y, y + n_pts);
    frames_.push_back(std::move(f));
}

// Hand-written JSON — avoids any external library dependency.
bool WCONExporter::write(const std::string& file_path) const {
    FILE* fp = std::fopen(file_path.c_str(), "w");
    if (!fp) return false;

    std::fputs("{\n", fp);
    std::fputs("  \"tracker-commons-version\": \"1.3\",\n", fp);
    std::fputs("  \"units\": {\"t\": \"s\", \"x\": \"mm\", \"y\": \"mm\"},\n", fp);
    std::fputs("  \"data\": [{\n", fp);
    std::fputs("    \"id\": \"0\",\n", fp);

    // t array
    std::fputs("    \"t\": [", fp);
    for (std::size_t i = 0; i < frames_.size(); ++i) {
        if (i) std::fputs(", ", fp);
        std::fprintf(fp, "%.6f", frames_[i].t_s);
    }
    std::fputs("],\n", fp);

    // x array of arrays
    std::fputs("    \"x\": [", fp);
    for (std::size_t i = 0; i < frames_.size(); ++i) {
        if (i) std::fputs(", ", fp);
        std::fputs("[", fp);
        for (int j = 0; j < n_pts_; ++j) {
            if (j) std::fputs(", ", fp);
            std::fprintf(fp, "%.5f", frames_[i].x[j]);
        }
        std::fputs("]", fp);
    }
    std::fputs("],\n", fp);

    // y array of arrays
    std::fputs("    \"y\": [", fp);
    for (std::size_t i = 0; i < frames_.size(); ++i) {
        if (i) std::fputs(", ", fp);
        std::fputs("[", fp);
        for (int j = 0; j < n_pts_; ++j) {
            if (j) std::fputs(", ", fp);
            std::fprintf(fp, "%.5f", frames_[i].y[j]);
        }
        std::fputs("]", fp);
    }
    std::fputs("]\n", fp);

    std::fputs("  }]\n", fp);
    std::fputs("}\n", fp);

    return std::fclose(fp) == 0;
}

} // namespace wormsim2
