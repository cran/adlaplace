#ifndef ADLAPLACE_PROFILE_DERIV_HPP
#define ADLAPLACE_PROFILE_DERIV_HPP

#include <Eigen/SparseCore>
#include <cstddef>
#include <vector>

// Profile Laplace derivative pieces matching R log_lik_deriv():
//   dU = -H_inv %*% H_outer[gamma, psi]
//   d_det_upart = -trace3[gamma] %*% dU
//   d_det_tpart = -trace3[psi]
//   grad_theta  = -grad[psi]
//   grad_u      = -grad[gamma] %*% dU
//   d_neg_log_lik = -grad_theta + d_det - grad_u
struct ProfileDerivResult {
  Eigen::MatrixXd dU; // Ngamma x Npsi
  Eigen::VectorXd d_det_upart;
  Eigen::VectorXd d_det_tpart;
  Eigen::VectorXd d_det;
  Eigen::VectorXd grad_theta;
  Eigen::VectorXd grad_u;
  Eigen::VectorXd d_neg_log_lik;
  Eigen::VectorXd d_log_lik;
};

// H_inv is upper-triangle CSC (dsCMatrix uplo=U pattern); treated as symmetric.
// H_outer is full symmetric Eigen CSC over (beta, gamma, theta).
// grad_outer and trace3 are length Nparams = Nbeta + Ngamma + Ntheta.
ProfileDerivResult profile_deriv(
    const Eigen::SparseMatrix<double> &H_outer,
    const Eigen::VectorXd &grad_outer, const std::vector<int> &H_inv_p,
    const std::vector<int> &H_inv_i, const std::vector<double> &H_inv_x,
    const std::vector<double> &trace3, std::size_t n_beta, std::size_t n_gamma,
    std::size_t n_theta);

#endif
