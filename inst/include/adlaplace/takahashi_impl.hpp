#ifndef ADLAPLACE_TAKAHASHI_IMPL_HPP
#define ADLAPLACE_TAKAHASHI_IMPL_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "adlaplace/chol_update_impl.hpp"

namespace adlaplace {
namespace chol {

// Forward-mode dual number so chol_update_csc / takahashi_davis can
// deliver directional derivatives (used for second-order atomic passes).
struct Dual {
	double val;
	double dot;
	Dual() : val(0.0), dot(0.0) {}
	Dual(double v) : val(v), dot(0.0) {}
	Dual(double v, double d) : val(v), dot(d) {}
};

inline Dual operator+(const Dual& a, const Dual& b) {
	return Dual(a.val + b.val, a.dot + b.dot);
}
inline Dual operator-(const Dual& a, const Dual& b) {
	return Dual(a.val - b.val, a.dot - b.dot);
}
inline Dual operator-(const Dual& a) { return Dual(-a.val, -a.dot); }
inline Dual operator*(const Dual& a, const Dual& b) {
	return Dual(a.val * b.val, a.dot * b.val + a.val * b.dot);
}
inline Dual operator/(const Dual& a, const Dual& b) {
	const double v = a.val / b.val;
	return Dual(v, (a.dot - v * b.dot) / b.val);
}
inline Dual& operator+=(Dual& a, const Dual& b) {
	a.val += b.val;
	a.dot += b.dot;
	return a;
}
inline Dual& operator-=(Dual& a, const Dual& b) {
	a.val -= b.val;
	a.dot -= b.dot;
	return a;
}
inline Dual log(const Dual& a) { return Dual(std::log(a.val), a.dot / a.val); }

template <>
struct CholLogDetTraits<Dual> {
	static bool invalid_diag(const Dual& d) {
		return d.val <= 0.0 || !std::isfinite(d.val);
	}
	static Dual invalid_result() {
		const double nan = std::numeric_limits<double>::quiet_NaN();
		return Dual(nan, nan);
	}
	static Dual log_of(const Dual& d) { return log(d); }
};

// Look up a symmetric entry stored on a lower-triangular CSC pattern
// (column = min index, row = max index). Returns 0 when absent.
template <typename Scalar>
Scalar csc_sym_lower_entry(
	std::size_t r,
	std::size_t c,
	const std::vector<int>& p,
	const std::vector<int>& i,
	const std::vector<Scalar>& x)
{
	const std::size_t col = std::min(r, c);
	const std::size_t row = std::max(r, c);
	for (int pos = p[col]; pos < p[col + 1]; ++pos) {
		if (static_cast<std::size_t>(i[static_cast<std::size_t>(pos)]) == row) {
			return x[static_cast<std::size_t>(pos)];
		}
	}
	return Scalar(0);
}

// Takahashi–Davis selected inverse.
//
// Given the unit-lower CSC factor L and the LDL diagonal D of a symmetric
// positive-definite matrix (diagonal of L explicit and equal to one; row
// indices in each column sorted), write (L D L')^{-1} on the sparsity
// pattern of L. Columns run from last to first. Already computed entries
// are scattered into a work vector of length n and the Takahashi update
// reads that vector by row index.
//
// Header-only. A package that lists LinkingTo: adlaplace compiles this
// template into its own shared library:
//
//   #include <adlaplace/takahashi_impl.hpp>
//   adlaplace::chol::takahashi_davis(Lp, Li, Lx, D, sigma);
//
// A caller that has only Q factors it first (chol_update_csc).
template <typename Scalar>
void takahashi_davis(
	const std::vector<int>& Lp,
	const std::vector<int>& Li,
	const std::vector<Scalar>& Lx,
	const std::vector<Scalar>& D,
	std::vector<Scalar>& sigma)
{
	if (Lp.size() < 2) {
		throw std::invalid_argument("takahashi_davis: L has no columns");
	}
	const std::size_t n = Lp.size() - 1;
	if (D.size() != n) {
		throw std::invalid_argument("takahashi_davis: D length must equal nrow(L)");
	}
	if (Lx.size() != Li.size()) {
		throw std::invalid_argument("takahashi_davis: Lx and Li lengths differ");
	}
	sigma.assign(Li.size(), Scalar(0));

	std::vector<int> diag_pos(n, -1);
	// Off-diagonal L(row, col) positions, grouped by row, so column j can
	// walk the k < j with L(j, k) != 0 without scanning earlier columns.
	struct Link {
		int col;
		int pos;
	};
	std::vector<std::vector<Link>> links(n);
	for (std::size_t k = 0; k < n; ++k) {
		for (int pos = Lp[k]; pos < Lp[k + 1]; ++pos) {
			const int row = Li[static_cast<std::size_t>(pos)];
			if (row == static_cast<int>(k)) {
				diag_pos[k] = pos;
				sigma[static_cast<std::size_t>(pos)] = Scalar(1) / D[k];
			} else {
				links[static_cast<std::size_t>(row)].push_back(
				    Link{static_cast<int>(k), pos});
			}
		}
		if (diag_pos[k] < 0) {
			throw std::invalid_argument(
			    "takahashi_davis: missing diagonal in L");
		}
	}
	for (std::size_t row = 0; row < n; ++row) {
		auto& hit = links[row];
		// Decreasing column index: z[i] for i > k is already known.
		std::sort(hit.begin(), hit.end(), [](const Link& a, const Link& b) {
			return a.col > b.col;
		});
	}

	std::vector<Scalar> z(n, Scalar(0));
	for (std::size_t jj = n; jj-- > 0;) {
		const std::size_t j = jj;
		for (int pos = Lp[j]; pos < Lp[j + 1]; ++pos) {
			z[static_cast<std::size_t>(Li[static_cast<std::size_t>(pos)])] =
			    sigma[static_cast<std::size_t>(pos)];
		}

		const std::vector<Link>& hit = links[j];
		for (std::size_t t = 0; t < hit.size(); ++t) {
			const int k = hit[t].col;
			Scalar zkj(0);
			for (int p = Lp[static_cast<std::size_t>(k)];
			     p < Lp[static_cast<std::size_t>(k) + 1]; ++p) {
				const int i = Li[static_cast<std::size_t>(p)];
				if (i > static_cast<int>(k)) {
					zkj -= Lx[static_cast<std::size_t>(p)] *
					       z[static_cast<std::size_t>(i)];
				}
			}
			z[static_cast<std::size_t>(k)] = zkj;
		}

		for (std::size_t t = 0; t < hit.size(); ++t) {
			const int k = hit[t].col;
			const Scalar ljk = Lx[static_cast<std::size_t>(hit[t].pos)];
			const std::size_t ks = static_cast<std::size_t>(k);
			for (int p = Lp[ks]; p < Lp[ks + 1]; ++p) {
				const std::size_t i =
				    static_cast<std::size_t>(Li[static_cast<std::size_t>(p)]);
				sigma[static_cast<std::size_t>(p)] -= z[i] * ljk;
			}
		}

		for (std::size_t t = 0; t < hit.size(); ++t) {
			z[static_cast<std::size_t>(hit[t].col)] = Scalar(0);
		}
		for (int pos = Lp[j]; pos < Lp[j + 1]; ++pos) {
			const std::size_t i =
			    static_cast<std::size_t>(Li[static_cast<std::size_t>(pos)]);
			sigma[static_cast<std::size_t>(pos)] = z[i];
			z[i] = Scalar(0);
		}
	}
}

// Scatter the selected inverse (permuted ordering, pattern of L) onto the
// original-order CSC pattern of H: S_x[pos] = (H^{-1})_{Q_i[pos], col}.
// perm_inv maps original indices to permuted indices.
template <typename Scalar>
void selected_inv_scatter(
	const std::vector<int>& H_p,
	const std::vector<int>& H_i,
	const std::vector<int>& perm_inv,
	const std::vector<int>& p_out,
	const std::vector<int>& i_out,
	const std::vector<Scalar>& sigma,
	std::vector<Scalar>& S_x)
{
	const std::size_t n = H_p.size() - 1;
	S_x.assign(H_i.size(), Scalar(0));
	for (std::size_t c = 0; c < n; ++c) {
		for (int pos = H_p[c]; pos < H_p[c + 1]; ++pos) {
			const std::size_t r = static_cast<std::size_t>(
				H_i[static_cast<std::size_t>(pos)]);
			const std::size_t pr = static_cast<std::size_t>(perm_inv[r]);
			const std::size_t pc = static_cast<std::size_t>(perm_inv[c]);
			S_x[static_cast<std::size_t>(pos)] =
				csc_sym_lower_entry(pr, pc, p_out, i_out, sigma);
		}
	}
}

} // namespace chol
} // namespace adlaplace

#endif
