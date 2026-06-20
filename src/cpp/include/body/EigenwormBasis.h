#pragma once
// Stephens et al. 2008 (PLoS CB doi:10.1371/journal.pcbi.1000028) eigenworm basis.
//
// Source: openworm/open-worm-analysis-toolbox
//   features/master_eigen_worms_N2.mat, key "eigenWorms", shape (48, 7).
//   First four columns are the dominant modes; unit L2-norm over 48 body segments.
//
// Physical interpretation:
//   theta[j] = tangent angle at normalized arc position j/N_SEG (in radians).
//   A body shape is represented as theta = sum_m a_m * phi_m.
//   Curvature kappa[j] = d(theta)/ds ≈ (theta[j] - theta[j-1]) * N_SEG.
//
// Modes 0-1 form the dominant undulation pair (forward/backward locomotion).
// Modes 2-3 capture turning and second-harmonic body shapes.

#include <cmath>

namespace wormsim2 {

struct EigenwormBasis {
    static constexpr int   kNSeg      = 48;       ///< body segments (arc points)
    static constexpr int   kNModes    = 4;         ///< dominant modes from Stephens 2008
    static constexpr float kBodyLenMm = 1.0f;      ///< adult N2 body length (mm)
    static constexpr int   kNPoints   = kNSeg + 1; ///< centerline points (head + segments)

    // Stephens 2008 eigenvectors: kBasis[m][j] = φ_m(j/N_SEG), unit L2-norm.
    // clang-format off
    static constexpr float kBasis[kNModes][kNSeg] = {
        // φ_0 — first mode (dominant forward/backward component)
        {-0.274402f, -0.272072f, -0.276189f, -0.276801f, -0.272097f, -0.262398f,
         -0.246854f, -0.226784f, -0.202628f, -0.174033f, -0.141613f, -0.107635f,
         -0.072683f, -0.039088f, -0.007426f,  0.020558f,  0.044295f,  0.063010f,
          0.076923f,  0.085648f,  0.089665f,  0.089279f,  0.085942f,  0.079909f,
          0.071473f,  0.060232f,  0.049459f,  0.038631f,  0.030589f,  0.025821f,
          0.024156f,  0.026573f,  0.031525f,  0.038868f,  0.048870f,  0.060376f,
          0.074126f,  0.089052f,  0.104539f,  0.121042f,  0.136676f,  0.149843f,
          0.159728f,  0.168119f,  0.172804f,  0.176767f,  0.179766f,  0.178441f},
        // φ_1 — second mode (90° phase partner of φ_0)
        { 0.003748f,  0.008651f,  0.023561f,  0.038526f,  0.052781f,  0.063597f,
          0.070928f,  0.073323f,  0.071164f,  0.063485f,  0.050241f,  0.031751f,
          0.009007f, -0.018450f, -0.048287f, -0.079790f, -0.111901f, -0.142512f,
         -0.172101f, -0.198020f, -0.219495f, -0.236177f, -0.245573f, -0.248097f,
         -0.243095f, -0.229457f, -0.208542f, -0.180849f, -0.147970f, -0.109837f,
         -0.068150f, -0.026140f,  0.015109f,  0.054557f,  0.091199f,  0.124666f,
          0.153900f,  0.178260f,  0.196883f,  0.208981f,  0.214154f,  0.211400f,
          0.201618f,  0.185332f,  0.163568f,  0.138237f,  0.118765f,  0.117052f},
        // φ_2 — third mode (second harmonic)
        {-0.097917f, -0.097615f, -0.082722f, -0.062694f, -0.037192f, -0.006509f,
          0.026702f,  0.061857f,  0.096843f,  0.131636f,  0.164357f,  0.192941f,
          0.215227f,  0.230549f,  0.237735f,  0.236466f,  0.225665f,  0.206142f,
          0.177861f,  0.143186f,  0.103429f,  0.060522f,  0.014366f, -0.031990f,
         -0.077502f, -0.120199f, -0.157338f, -0.188785f, -0.212103f, -0.227786f,
         -0.234155f, -0.232135f, -0.223161f, -0.206631f, -0.185416f, -0.159472f,
         -0.130120f, -0.097712f, -0.063982f, -0.030384f, -0.000126f,  0.025496f,
          0.046027f,  0.060518f,  0.071131f,  0.076563f,  0.078879f,  0.079550f},
        // φ_3 — fourth mode (second harmonic 90° partner)
        {-0.279275f, -0.271567f, -0.227064f, -0.175897f, -0.116878f, -0.054271f,
          0.006722f,  0.061596f,  0.108651f,  0.146116f,  0.172733f,  0.186508f,
          0.186779f,  0.173738f,  0.148993f,  0.116435f,  0.078569f,  0.039287f,
          0.000169f, -0.034916f, -0.062238f, -0.080166f, -0.088587f, -0.087187f,
         -0.075762f, -0.053521f, -0.025169f,  0.009904f,  0.047295f,  0.084442f,
          0.118292f,  0.145138f,  0.164774f,  0.175307f,  0.177461f,  0.170336f,
          0.153270f,  0.126657f,  0.090878f,  0.046223f, -0.002652f, -0.051927f,
         -0.100738f, -0.150378f, -0.195648f, -0.237000f, -0.273342f, -0.292090f},
    };
    // clang-format on

    /// Project tangent-angle profile theta[kNSeg] onto mode m.
    static float project(int m, const float* theta) noexcept {
        float s = 0.0f;
        for (int j = 0; j < kNSeg; ++j) s += kBasis[m][j] * theta[j];
        return s;
    }

    /// Reconstruct tangent-angle profile from mode amplitudes a[kNModes].
    static void reconstruct(const float* a, float* theta) noexcept {
        for (int j = 0; j < kNSeg; ++j) {
            theta[j] = 0.0f;
            for (int m = 0; m < kNModes; ++m) theta[j] += a[m] * kBasis[m][j];
        }
    }

    /// Integrate tangent angles → centerline (x, y) in mm.
    /// x and y must have capacity kNPoints = kNSeg + 1.
    /// x[0]=y[0]=0 (head at origin, body extends from head to tail).
    static void integrate_centerline(const float* theta, float* x, float* y,
                                     float body_len = kBodyLenMm) noexcept {
        const float ds    = body_len / kNSeg;
        float       cumth = 0.0f;
        x[0] = 0.0f;
        y[0] = 0.0f;
        for (int j = 0; j < kNSeg; ++j) {
            cumth += theta[j];
            x[j + 1] = x[j] + ds * std::cos(cumth);
            y[j + 1] = y[j] + ds * std::sin(cumth);
        }
    }
};

} // namespace wormsim2
