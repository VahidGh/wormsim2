#pragma once
// WCON (Worm tracker Commons Open Number) JSON exporter.
//
// Spec: https://github.com/openworm/tracker-commons/blob/master/WCON_format.md
// Writes a minimal WCON 1.3 JSON file for a single simulated worm:
//   units: t = s, x = mm, y = mm
//   data[0].t: array of frame times
//   data[0].x: array of centerline x-vectors (one per frame)
//   data[0].y: array of centerline y-vectors (one per frame)
//
// Usage:
//   WCONExporter exp;
//   for each simulation step:
//       exp.add_frame(t_ms, body.centerline_x(), body.centerline_y(), npts);
//   exp.write("output.wcon");

#include <string>
#include <vector>

namespace wormsim2 {

class WCONExporter {
public:
    /// Add one frame. t_ms is simulation time in milliseconds (stored as seconds).
    /// x and y are centerline positions in mm (n_pts values each).
    void add_frame(float t_ms, const float* x, const float* y, int n_pts);

    /// Write WCON JSON to file_path. Returns true on success.
    [[nodiscard]] bool write(const std::string& file_path) const;

    /// Number of frames accumulated so far.
    [[nodiscard]] int frame_count() const noexcept {
        return static_cast<int>(frames_.size());
    }

    /// Number of centerline points per frame (set from first add_frame call; 0 if empty).
    [[nodiscard]] int points_per_frame() const noexcept { return n_pts_; }

private:
    struct Frame {
        float              t_s;
        std::vector<float> x;
        std::vector<float> y;
    };

    std::vector<Frame> frames_;
    int                n_pts_{0};
};

} // namespace wormsim2
