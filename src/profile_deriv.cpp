#include "adlaplace/profile_deriv.hpp"

#include <Rcpp.h>
#include <Eigen/SparseCore>

#include <cstddef>
#include <vector>

ProfileDerivResult profile_deriv(
    const Eigen::SparseMatrix<double> &H_outer,
    const Eigen::VectorXd &grad_outer, const std::vector<int> &H_inv_p,
    const std::vector<int> &H_inv_i, const std::vector<double> &H_inv_x,
    const std::vector<double> &trace3, std::size_t n_beta, std::size_t n_gamma,
    std::size_t n_theta) {
  const std::size_t n_params = n_beta + n_gamma + n_theta;
  const std::size_t n_psi = n_beta + n_theta;
  const Eigen::Index n_gamma_i = static_cast<Eigen::Index>(n_gamma);
  const Eigen::Index n_psi_i = static_cast<Eigen::Index>(n_psi);
  const Eigen::Index n_params_i = static_cast<Eigen::Index>(n_params);

  if (H_inv_p.size() != n_gamma + 1) {
    Rcpp::stop("profile_deriv: H_inv_p length must be Ngamma + 1");
  }
  if (H_inv_i.size() != H_inv_x.size()) {
    Rcpp::stop("profile_deriv: H_inv_i and H_inv_x length must match");
  }
  if (static_cast<std::size_t>(grad_outer.size()) != n_params) {
    Rcpp::stop("profile_deriv: grad_outer length must equal Nparams");
  }
  if (trace3.size() != n_params) {
    Rcpp::stop("profile_deriv: trace3 length must equal Nparams");
  }
  if (H_outer.rows() != n_params_i || H_outer.cols() != n_params_i) {
    Rcpp::stop("profile_deriv: H_outer must be Nparams x Nparams");
  }

  // Dense H_gamma_psi: gamma rows, psi = (beta, theta) columns.
  Eigen::MatrixXd H_gamma_psi =
      Eigen::MatrixXd::Zero(n_gamma_i, n_psi_i);
  for (Eigen::Index col = 0; col < n_params_i; ++col) {
    Eigen::Index psi_col = -1;
    if (col < static_cast<Eigen::Index>(n_beta)) {
      psi_col = col;
    } else if (col >= static_cast<Eigen::Index>(n_beta + n_gamma)) {
      psi_col = col - static_cast<Eigen::Index>(n_gamma);
    } else {
      continue;
    }
    for (Eigen::SparseMatrix<double>::InnerIterator it(H_outer, col); it;
         ++it) {
      const Eigen::Index row = it.row();
      if (row < static_cast<Eigen::Index>(n_beta) ||
          row >= static_cast<Eigen::Index>(n_beta + n_gamma)) {
        continue;
      }
      const Eigen::Index gamma_row = row - static_cast<Eigen::Index>(n_beta);
      H_gamma_psi(gamma_row, psi_col) = it.value();
    }
  }

  // Map upper-triangle CSC H_inv as symmetric (matches R dsCMatrix %*%).
  Eigen::Map<const Eigen::SparseMatrix<double>> H_inv_upper(
      n_gamma_i, n_gamma_i, static_cast<Eigen::Index>(H_inv_i.size()),
      H_inv_p.data(), H_inv_i.data(), H_inv_x.data());
  Eigen::SparseMatrix<double> H_inv =
      H_inv_upper.selfadjointView<Eigen::Upper>();
  H_inv.makeCompressed();

  ProfileDerivResult out;
  out.dU = -(H_inv * H_gamma_psi);

  Eigen::VectorXd trace_gamma(n_gamma_i);
  Eigen::VectorXd trace_psi(n_psi_i);
  Eigen::VectorXd grad_gamma(n_gamma_i);
  Eigen::VectorXd grad_psi(n_psi_i);
  for (std::size_t i = 0; i < n_gamma; ++i) {
    trace_gamma[static_cast<Eigen::Index>(i)] = trace3[n_beta + i];
    grad_gamma[static_cast<Eigen::Index>(i)] =
        grad_outer[static_cast<Eigen::Index>(n_beta + i)];
  }
  for (std::size_t i = 0; i < n_beta; ++i) {
    trace_psi[static_cast<Eigen::Index>(i)] = trace3[i];
    grad_psi[static_cast<Eigen::Index>(i)] =
        grad_outer[static_cast<Eigen::Index>(i)];
  }
  for (std::size_t i = 0; i < n_theta; ++i) {
    trace_psi[static_cast<Eigen::Index>(n_beta + i)] =
        trace3[n_beta + n_gamma + i];
    grad_psi[static_cast<Eigen::Index>(n_beta + i)] =
        grad_outer[static_cast<Eigen::Index>(n_beta + n_gamma + i)];
  }

  out.d_det_upart = -(trace_gamma.transpose() * out.dU).transpose();
  out.d_det_tpart = -trace_psi;
  out.d_det = out.d_det_upart + out.d_det_tpart;
  out.grad_theta = -grad_psi;
  out.grad_u = -(grad_gamma.transpose() * out.dU).transpose();
  out.d_neg_log_lik = -out.grad_theta + out.d_det - out.grad_u;
  out.d_log_lik = -out.d_neg_log_lik;
  return out;
}
