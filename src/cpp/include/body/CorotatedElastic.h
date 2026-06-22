#pragma once
// CorotatedElastic — per-tetrahedron corotated linear-elastic constitutive law.
// Follows MetaWorm (BAAIWorm, Nat. Comput. Sci. 2024) and Irving et al. 2004.
// For each tet: compute deformation gradient F from current vs. rest positions,
// polar-decompose F = R·S, compute first Piola-Kirchhoff stress P = R·(2µ(S-I) +
// λ tr(S-I) I), scatter forces to the 4 tet vertices.
// Material: C. elegans body-wall tissue — E = 100 Pa, ν = 0.30 (soft tissue,
// Petzold et al. 2011; consistent with Sibernetic & MetaWorm defaults).

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <array>

namespace wormsim2 {

// Lamé parameters derived from Young's modulus and Poisson ratio.
struct LameParams {
    double mu;     // shear modulus  = E / (2*(1+ν))
    double lambda; // bulk correction = E*ν / ((1+ν)*(1-2ν))

    static LameParams from_E_nu(double E, double nu) {
        return { E / (2.0 * (1.0 + nu)),
                 E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) };
    }
};

// Default C. elegans body-wall material.
inline constexpr double kBodyE  = 100.0; // Pa
inline constexpr double kBodyNu = 0.30;

// Corotated elastic element: stores rest-pose data, computes forces.
class CorotatedElasticElement {
public:
    using Vec3  = dealii::Tensor<1, 3>;
    using Mat33 = dealii::Tensor<2, 3>;

    // Initialise from the four rest-pose vertex positions (indices 0..3).
    // Must be called before any force computation.
    void init(const std::array<Vec3, 4>& rest_pos, const LameParams& mat);

    // Compute internal elastic force contributions on each of the 4 vertices
    // from current deformed positions.
    // f_out[i] += elastic force on vertex i (accumulates into caller's array).
    void add_forces(const std::array<Vec3, 4>& cur_pos,
                    std::array<Vec3, 4>&        f_out) const;

    // Volume of the element in the rest configuration.
    double rest_volume() const { return vol0_; }

private:
    // Dm_inv: inverse of the rest-pose edge matrix (3×3), used to compute F.
    Mat33       Dm_inv_;
    double      vol0_;    // rest volume (positive)
    LameParams  mat_;

    // Polar decomposition: F = R * S.  Returns R (rotation) and S (stretch).
    static void polar_decomp(const Mat33& F, Mat33& R, Mat33& S);
};

} // namespace wormsim2
