#include <Rcpp.h>

#include "adlaplace/chol_update_impl.hpp"
#include "adlaplace/rviews.hpp"
#include "adlaplace/takahashi_impl.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

// Numeric LDL of Q with the supplied symbolic factor, then the
// Takahashi–Davis selected inverse. Q is upper-triangle CSC in the original
// order. sigma is (P Q P')^{-1} on the pattern of L; the returned dgCMatrix
// is Q^{-1} on that pattern, permuted back.
// [[Rcpp::export]]
Rcpp::S4 takahashi_davis_matrix(Rcpp::S4 Q, Rcpp::IntegerVector perm,
                                Rcpp::S4 L1) {
	const CscMatrix Qu(Q);
	const CscMatrix Lpat(L1);
	const std::vector<int> perm_v = Rcpp::as<std::vector<int>>(perm);
	const std::size_t n = static_cast<std::size_t>(Qu.nrow());
	if (n == 0 || Qu.ncol() != Qu.nrow()) {
		Rcpp::stop("takahashi_davis: Q must be square");
	}
	if (perm_v.size() != n) {
		Rcpp::stop("takahashi_davis: perm length must equal nrow(Q)");
	}
	if (static_cast<std::size_t>(Lpat.nrow()) != n || Lpat.p.size() != n + 1) {
		Rcpp::stop("takahashi_davis: L1 dimensions do not match Q");
	}
	if (!Qu.has_x() || Qu.x.size() != Qu.nnz()) {
		Rcpp::stop("takahashi_davis: Q has no numeric values");
	}

	std::vector<double> Lx(Lpat.nnz(), 0.0);
	std::vector<double> D(n, 0.0);
	const double log_det = adlaplace::chol::chol_update_csc(
	    Qu.p, Qu.i, Qu.x, perm_v, Lpat.p, Lpat.i, Lx, D);
	if (!std::isfinite(log_det)) {
		Rcpp::stop("takahashi_davis: Q is not positive definite");
	}

	std::vector<double> sigma;
	try {
		adlaplace::chol::takahashi_davis(Lpat.p, Lpat.i, Lx, D, sigma);
	} catch (const std::invalid_argument& err) {
		Rcpp::stop("%s", err.what());
	}

	std::vector<std::vector<std::pair<int, double>>> cols(n);
	for (std::size_t c = 0; c < n; ++c) {
		for (int pos = Lpat.p[c]; pos < Lpat.p[c + 1]; ++pos) {
			const std::size_t r =
			    static_cast<std::size_t>(Lpat.i[static_cast<std::size_t>(pos)]);
			const int oi = perm_v[r];
			const int oj = perm_v[c];
			const double val = sigma[static_cast<std::size_t>(pos)];
			cols[static_cast<std::size_t>(oj)].push_back(std::make_pair(oi, val));
			if (oi != oj) {
				cols[static_cast<std::size_t>(oi)].push_back(
				    std::make_pair(oj, val));
			}
		}
	}

	std::vector<int> ip;
	std::vector<int> ii;
	std::vector<double> xx;
	ip.reserve(n + 1);
	ip.push_back(0);
	for (std::size_t c = 0; c < n; ++c) {
		auto& col = cols[c];
		std::sort(col.begin(), col.end(),
		          [](const std::pair<int, double>& a,
		             const std::pair<int, double>& b) { return a.first < b.first; });
		for (std::size_t t = 0; t < col.size(); ++t) {
			ii.push_back(col[t].first);
			xx.push_back(col[t].second);
		}
		ip.push_back(static_cast<int>(ii.size()));
	}

	Rcpp::S4 out("dgCMatrix");
	out.slot("Dim") = Rcpp::IntegerVector::create(static_cast<int>(n),
	                                               static_cast<int>(n));
	out.slot("i") = Rcpp::IntegerVector(ii.begin(), ii.end());
	out.slot("p") = Rcpp::IntegerVector(ip.begin(), ip.end());
	out.slot("x") = Rcpp::NumericVector(xx.begin(), xx.end());
	out.slot("Dimnames") = Rcpp::List::create(R_NilValue, R_NilValue);
	if (out.hasSlot("factors")) {
		out.slot("factors") = Rcpp::List();
	}
	return out;
}
