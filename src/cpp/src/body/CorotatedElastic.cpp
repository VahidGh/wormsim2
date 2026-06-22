#include "body/CorotatedElastic.h"

#include <deal.II/base/tensor.h>
#include <cmath>
#include <stdexcept>
#include <array>

namespace wormsim2 {

using Vec3  = dealii::Tensor<1, 3>;
using Mat33 = dealii::Tensor<2, 3>;

// ------------------------------------------------------------
// Polar decomposition: F = R * S via SVD (Higham 1988 / Irving 2004).
// For small-to-moderate deformations of a biological body, Newton-Schulz
// is fast and sufficient (≤5 iterations to double-precision tolerance).
// ------------------------------------------------------------
void CorotatedElasticElement::polar_decomp(const Mat33& F, Mat33& R, Mat33& S)
{
    // Newton-Schulz iteration: X_{k+1} = 0.5 * X_k * (3*I - X_k^T * X_k)
    // Converges to the orthogonal polar factor of F.
    R = F;
    for (int iter = 0; iter < 20; ++iter) {
        const Mat33 Rt     = dealii::transpose(R);
        const Mat33 RtR    = Rt * R; // should converge to I
        // Check convergence: ||R^T R - I||_F < tol
        double err = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double d = RtR[i][j] - (i == j ? 1.0 : 0.0);
                err += d * d;
            }
        if (err < 1.0e-24) break;

        // X_{k+1} = 0.5 * (X_k + X_k^{-T})
        // For non-degenerate F, R^{-T} = (R^T)^{-1}; use R_new = 0.5*(R + R^{-T}).
        // R^{-T}: invert Rt analytically (3×3).
        const double det = Rt[0][0]*(Rt[1][1]*Rt[2][2]-Rt[1][2]*Rt[2][1])
                         - Rt[0][1]*(Rt[1][0]*Rt[2][2]-Rt[1][2]*Rt[2][0])
                         + Rt[0][2]*(Rt[1][0]*Rt[2][1]-Rt[1][1]*Rt[2][0]);
        if (std::abs(det) < 1.0e-14)
            throw std::runtime_error("CorotatedElastic: degenerate element (det≈0)");

        Mat33 inv_Rt;
        const double id = 1.0 / det;
        inv_Rt[0][0] = id*(Rt[1][1]*Rt[2][2]-Rt[1][2]*Rt[2][1]);
        inv_Rt[0][1] = id*(Rt[0][2]*Rt[2][1]-Rt[0][1]*Rt[2][2]);
        inv_Rt[0][2] = id*(Rt[0][1]*Rt[1][2]-Rt[0][2]*Rt[1][1]);
        inv_Rt[1][0] = id*(Rt[1][2]*Rt[2][0]-Rt[1][0]*Rt[2][2]);
        inv_Rt[1][1] = id*(Rt[0][0]*Rt[2][2]-Rt[0][2]*Rt[2][0]);
        inv_Rt[1][2] = id*(Rt[0][2]*Rt[1][0]-Rt[0][0]*Rt[1][2]);
        inv_Rt[2][0] = id*(Rt[1][0]*Rt[2][1]-Rt[1][1]*Rt[2][0]);
        inv_Rt[2][1] = id*(Rt[0][1]*Rt[2][0]-Rt[0][0]*Rt[2][1]);
        inv_Rt[2][2] = id*(Rt[0][0]*Rt[1][1]-Rt[0][1]*Rt[1][0]);

        R = 0.5 * (R + inv_Rt);
    }
    // S = R^T * F (symmetric stretch tensor)
    S = dealii::transpose(R) * F;
}

// ------------------------------------------------------------
// Initialise element from rest-pose vertex positions.
// ------------------------------------------------------------
void CorotatedElasticElement::init(const std::array<Vec3, 4>& rest_pos,
                                    const LameParams&           mat)
{
    mat_  = mat;

    // Edge matrix Dm: columns are (X1-X0), (X2-X0), (X3-X0).
    Mat33 Dm;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Dm[i][j] = rest_pos[j+1][i] - rest_pos[0][i];

    // Volume = det(Dm) / 6  (signed — keep positive).
    const double det = Dm[0][0]*(Dm[1][1]*Dm[2][2]-Dm[1][2]*Dm[2][1])
                     - Dm[0][1]*(Dm[1][0]*Dm[2][2]-Dm[1][2]*Dm[2][0])
                     + Dm[0][2]*(Dm[1][0]*Dm[2][1]-Dm[1][1]*Dm[2][0]);
    vol0_ = std::abs(det) / 6.0;
    if (vol0_ < 1.0e-20)
        throw std::runtime_error("CorotatedElastic: zero-volume element");

    // Precompute Dm_inv = Dm^{-1}  (standard inverse, NOT transpose).
    // The deformation gradient F = Ds * Dm_inv requires the true inverse.
    Dm_inv_ = dealii::invert(Dm);
}

// ------------------------------------------------------------
// Compute corotated elastic force contributions on 4 vertices.
// ------------------------------------------------------------
void CorotatedElasticElement::add_forces(const std::array<Vec3, 4>& cur_pos,
                                          std::array<Vec3, 4>&        f_out) const
{
    // Deformed edge matrix Ds.
    Mat33 Ds;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Ds[i][j] = cur_pos[j+1][i] - cur_pos[0][i];

    // Deformation gradient F = Ds * Dm_inv.
    const Mat33 F = Ds * Dm_inv_;

    // Polar decomposition: F = R * S.
    Mat33 R, S;
    polar_decomp(F, R, S);

    // Corotated strain: E_c = S - I.
    Mat33 Ec;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Ec[i][j] = S[i][j] - (i == j ? 1.0 : 0.0);

    // Trace of E_c.
    const double trE = Ec[0][0] + Ec[1][1] + Ec[2][2];

    // 2nd Piola-Kirchhoff stress in corotated frame: P_c = λ*tr(Ec)*I + 2µ*Ec.
    Mat33 Pc;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Pc[i][j] = mat_.lambda * trE * (i==j ? 1.0 : 0.0)
                      + 2.0 * mat_.mu * Ec[i][j];

    // First Piola-Kirchhoff stress: P = R * Pc.
    const Mat33 P = R * Pc;

    // Forces on vertices 1..3: f_i = -V0 * P * Dm_inv^T * e_i
    // f_0 = -(f_1 + f_2 + f_3)  [Newton's 3rd law]
    const Mat33 PDt = P * dealii::transpose(Dm_inv_);

    for (int j = 0; j < 3; ++j) {
        Vec3 fj;
        for (int i = 0; i < 3; ++i)
            fj[i] = -vol0_ * PDt[i][j];
        f_out[j+1] += fj;
        f_out[0]   -= fj;
    }
}

} // namespace wormsim2
