#ifndef ADLAPLACE_CHOL_UPDATE_HPP
#define ADLAPLACE_CHOL_UPDATE_HPP

#include <Rcpp.h>
#include <Eigen/SparseCore>

#include "adlaplace/backend.hpp"
#include "adlaplace/chol_update_impl.hpp"

#include <vector>

// Returns log(det(H_perm)) = sum(log(D)); NA_REAL if any D is non-positive or non-finite.
double chol_update(
	const Eigen::SparseMatrix<double>& H,
	const std::vector<int>& perm,
	const std::vector<int>& p_out,
	const std::vector<int>& i_out,
	std::vector<double>& x_out,
	std::vector<double>& d_out
);

// Fill Linv_x on fixed CSC pattern given numeric unit-lower L (same permuted ordering).
// nrow inferred from L_p.size() - 1.
void linv_update(
	const std::vector<int>& L_p,
	const std::vector<int>& L_i,
	const std::vector<double>& L_x,
	const std::vector<int>& Linv_p,
	const std::vector<int>& Linv_i,
	std::vector<double>& Linv_x
);

// half_H_inv[r,j] = Linv[j, perm_inv[r]] * D[j]^(-1/2) in original gamma ordering.
// nrow inferred from Linv_p.size() - 1.
void half_h_inv_update(
	const std::vector<int>& Linv_p,
	const std::vector<int>& Linv_i,
	const std::vector<double>& Linv_x,
	const std::vector<double>& d,
	const std::vector<int>& perm_inv,
	const std::vector<int>& half_H_inv_p,
	const std::vector<int>& half_H_inv_i,
	std::vector<double>& half_H_inv_x
);

// H_inv = half_H_inv %*% t(half_H_inv) on fixed upper-triangle CSC pattern.
// Only entries with row <= col are filled; exported as dsCMatrix (uplo = U).
// nrow inferred from half_H_inv_p.size() - 1.
void h_inv_update(
	const std::vector<int>& half_H_inv_p,
	const std::vector<int>& half_H_inv_i,
	const std::vector<double>& half_H_inv_x,
	const std::vector<int>& H_inv_p,
	const std::vector<int>& H_inv_i,
	std::vector<double>& H_inv_x
);

void ad_fun_attach_chol_pattern_from_list(ad_pack& shards, const Rcpp::List& hessian_pack);

#endif
