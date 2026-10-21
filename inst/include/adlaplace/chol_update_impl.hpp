#ifndef ADLAPLACE_CHOL_UPDATE_IMPL_HPP
#define ADLAPLACE_CHOL_UPDATE_IMPL_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Rcpp.h>

namespace adlaplace {
namespace chol {

// Look up symmetric entry: stored in column max(row,col), row min(row,col).
template <typename Scalar>
Scalar csc_sym_entry(
	std::size_t row,
	std::size_t col,
	const std::vector<int>& p,
	const std::vector<int>& i,
	const std::vector<Scalar>& x)
{
	const std::size_t c = std::max(row, col);
	const std::size_t r = std::min(row, col);
	for (int pos = p[c]; pos < p[c + 1]; ++pos) {
		if (static_cast<std::size_t>(i[static_cast<std::size_t>(pos)]) == r) {
			return x[static_cast<std::size_t>(pos)];
		}
	}
	return Scalar(0);
}

template <typename Scalar>
struct CholLogDetTraits {
	static bool invalid_diag(const Scalar&) { return false; }
	static Scalar invalid_result() { return Scalar(0); }
	static Scalar log_of(const Scalar& d) { return log(d); }
};

template <>
struct CholLogDetTraits<double> {
	static bool invalid_diag(const double& d) {
		return d <= 0.0 || !std::isfinite(d);
	}
	static double invalid_result() { return NA_REAL; }
	static double log_of(const double& d) { return std::log(d); }
};

// Numeric LDL of P H P' with H given as CSC (upper or full) in original order.
// Fills unit-lower L (pattern p_out/i_out) and diagonal D; returns sum(log(D)).
template <typename Scalar>
Scalar chol_update_csc(
	const std::vector<int>& H_p,
	const std::vector<int>& H_i,
	const std::vector<Scalar>& H_x,
	const std::vector<int>& perm,
	const std::vector<int>& p_out,
	const std::vector<int>& i_out,
	std::vector<Scalar>& x_out,
	std::vector<Scalar>& d_out)
{
	if (H_p.size() < 2) {
		Rcpp::stop("chol_update_csc: H_p is empty");
	}
	const std::size_t n = H_p.size() - 1;
	if (perm.size() != n) {
		Rcpp::stop(
			"chol_update_csc: perm length (%d) must match nrow (%d)",
			static_cast<int>(perm.size()),
			static_cast<int>(n)
		);
	}
	if (p_out.size() != n + 1) {
		Rcpp::stop("chol_update_csc: p_out length must be nrow + 1");
	}
	if (d_out.size() != n) {
		d_out.assign(n, Scalar(0));
	}
	if (x_out.size() != i_out.size()) {
		x_out.assign(i_out.size(), Scalar(0));
	}

	std::vector<Scalar> y(n, Scalar(0));

	for (std::size_t j = 0; j < n; ++j) {
		for (std::size_t r = 0; r < n; ++r) {
			y[r] = csc_sym_entry(
				static_cast<std::size_t>(perm[r]),
				static_cast<std::size_t>(perm[j]),
				H_p,
				H_i,
				H_x
			);
		}

		for (std::size_t k = 0; k < j; ++k) {
			bool has_l_jk = false;
			Scalar l_jk = Scalar(0);
			for (std::size_t pos = static_cast<std::size_t>(p_out[k]);
			     pos < static_cast<std::size_t>(p_out[k + 1]);
			     ++pos) {
				if (static_cast<std::size_t>(i_out[pos]) == j) {
					l_jk = x_out[pos];
					has_l_jk = true;
					break;
				}
			}
			if (!has_l_jk) {
				continue;
			}
			const Scalar alpha = l_jk * d_out[k];
			for (std::size_t pos = static_cast<std::size_t>(p_out[k]);
			     pos < static_cast<std::size_t>(p_out[k + 1]);
			     ++pos) {
				const std::size_t row = static_cast<std::size_t>(i_out[pos]);
				if (row < j) {
					continue;
				}
				y[row] -= alpha * x_out[pos];
			}
		}

		d_out[j] = y[j];
		for (std::size_t pos = static_cast<std::size_t>(p_out[j]);
		     pos < static_cast<std::size_t>(p_out[j + 1]);
		     ++pos) {
			const std::size_t row = static_cast<std::size_t>(i_out[pos]);
			if (row == j) {
				x_out[pos] = Scalar(1);
			} else if (row > j) {
				x_out[pos] = y[row] / d_out[j];
			}
		}
	}

	Scalar log_det = Scalar(0);
	for (std::size_t j = 0; j < n; ++j) {
		if (CholLogDetTraits<Scalar>::invalid_diag(d_out[j])) {
			return CholLogDetTraits<Scalar>::invalid_result();
		}
		log_det += CholLogDetTraits<Scalar>::log_of(d_out[j]);
	}
	return log_det;
}

} // namespace chol
} // namespace adlaplace

#endif
