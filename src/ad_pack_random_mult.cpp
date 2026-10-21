#include <Rcpp.h>
#include <Rinternals.h>

#include "adlaplace/density_data.hpp"
#include "adlaplace/ad_pack.hpp"
#include "adlaplace/ad_pack_random.hpp"
#include "adlaplace/atomics.hpp"
#include "adlaplace/quadform_atomic.hpp"
#include "adlaplace/register.hpp"
#include "adlaplace/rviews.hpp"

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

extern ad_shard *adlaplace_make_ad_shard(AdTape &&);

namespace {

CppAD::vector<CppAD::AD<double>>
random_diagonal_impl(const CppAD::vector<CppAD::AD<double>> &x,
                     const density_data &model, const NumVecView &Q,
                     const std::vector<std::size_t> &gamma_indices,
                     const Config &config) {

  const std::size_t Ngamma = gamma_indices.size();
  if (Q.size() != Ngamma) {
    Rcpp::warning("precision length (%d) differs from gamma_map columns (%d)",
                  static_cast<int>(Q.size()), static_cast<int>(Ngamma));
  }

  // Empty theta_map: known-SD prior; Q already holds 1/sd^2 (treat theta = 1).
  const bool has_theta = !model.theta_global.empty();
  CppAD::AD<double> logSd = 0.0;
  if (has_theta) {
    const std::size_t t_global = model.theta_index(0);
    const std::size_t t_row = model.theta_row(0);
    if (transform_theta_at(config, t_row)) {
      logSd = x[t_global];
    } else {
      logSd = CppAD::log(x[t_global]);
    }
  }

  CppAD::AD<double> precision = CppAD::exp(-2 * logSd);
  double logQsum = 0.0;

  const std::size_t nq = std::min(Ngamma, Q.size());
  std::vector<double> diag_vals(nq);
  CppAD::vector<CppAD::AD<double>> ax(nq);
  for (std::size_t k = 0; k < nq; ++k) {
    const double qk = Q[k];
    diag_vals[k] = qk;
    ax[k] = x[gamma_indices[k]];
    logQsum += std::log(qk);
  }
  const std::size_t call_id = adlaplace::quadform::register_diagonal(diag_vals);
  CppAD::vector<CppAD::AD<double>> ay(1);
  adlaplace::quadform::call_quadform(call_id, ax, ay);
  const CppAD::AD<double> qpart =
      CppAD::AD<double>(0.5) * precision * ay[0];

  CppAD::AD<double> qDet = logSd * CppAD::AD<double>(Ngamma) +
                           CppAD::AD<double>(Ngamma * ONEHALFLOGTWOPI);

  if (config.verbose) {
    if (has_theta) {
      Rcpp::Rcout << "theta index " << model.theta_index(0) << " logVariance "
                  << logSd << " precision " << precision << "\n";
    } else {
      Rcpp::Rcout << "random_diagonal known-SD (no theta) precision "
                  << precision << "\n";
    }
    Rcpp::Rcout << "random_diagonal n_gamma " << Ngamma << " qDet " << qDet
                << " qpart " << qpart << "\n";
  }

  CppAD::vector<CppAD::AD<double>> result(1);
  result[0] = -qpart - qDet + CppAD::AD<double>(0.5 * logQsum);
  return result;
}

CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)>
random_diagonal_sparsity(const density_data &model) {

  const std::vector<std::size_t> gamma_indices =
      model.all_gamma_global_indices();
  const std::size_t n_gamma = gamma_indices.size();
  const std::size_t n_params = model.n_tape;
  const bool has_theta = !model.theta_global.empty();

  CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> hessian;
  if (has_theta) {
    const std::size_t theta_index = model.theta_index(0);
    hessian.resize(n_params, n_params, 3 * n_gamma + 1);
    hessian.set(0, theta_index, theta_index);
    std::size_t t = 1;
    for (std::size_t k = 0; k < n_gamma; ++k) {
      const std::size_t gi = gamma_indices[k];
      hessian.set(t++, gi, gi);
      hessian.set(t++, gi, theta_index);
      hessian.set(t++, theta_index, gi);
    }
  } else {
    hessian.resize(n_params, n_params, n_gamma);
    for (std::size_t k = 0; k < n_gamma; ++k) {
      hessian.set(k, gamma_indices[k], gamma_indices[k]);
    }
  }
  return hessian;
}

CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)>
random_mult_sparsity(const density_data &model) {

  const CscMatrix &Q = model.mult_precision_csc();
  const std::size_t theta_index = model.theta_index(0);
  const std::size_t n_params = model.n_tape;
  const int n_term = model.gamma_map.ncol();

  std::set<std::pair<std::size_t, std::size_t>> pairs;

  // gamma-gamma block: structural nonzeros of upper-triangle Q, mapped to
  // global indices. Insert both triangle orders for cppad.symmetric coloring.
  for (int col = 0; col < n_term; ++col) {
    const std::vector<std::size_t> gj_idx = model.gamma_global_indices(col);
    if (gj_idx.empty()) {
      continue;
    }
    const std::size_t gj = gj_idx[0];

    for (int k = Q.p[static_cast<std::size_t>(col)];
         k < Q.p[static_cast<std::size_t>(col) + 1]; ++k) {
      const std::vector<std::size_t> gi_idx =
          model.gamma_global_indices(Q.i[static_cast<std::size_t>(k)]);
      if (gi_idx.empty()) {
        continue;
      }
      const std::size_t gi = gi_idx[0];
      pairs.insert({gi, gj});
      pairs.insert({gj, gi});
    }
  }

  // gamma-theta cross terms (both orders) and the theta diagonal.
  for (int col = 0; col < n_term; ++col) {
    const std::vector<std::size_t> g_idx = model.gamma_global_indices(col);
    if (g_idx.empty()) {
      continue;
    }
    pairs.insert({g_idx[0], theta_index});
    pairs.insert({theta_index, g_idx[0]});
  }
  pairs.insert({theta_index, theta_index});

  CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> hessian;
  set_sparse_rc_pairs(hessian, n_params, pairs);
  return hessian;
}

} // namespace

CppAD::vector<CppAD::AD<double>>
random_diagonal(const CppAD::vector<CppAD::AD<double>> &x, const density_data &model,
                const Config &config) {

  if (Rf_isNull(model.precision)) {
    Rcpp::stop("precision is required for random_diagonal");
  }
  const NumVecView Q(model.precision);
  const std::vector<std::size_t> gamma_indices =
      model.all_gamma_global_indices();
  return random_diagonal_impl(x, model, Q, gamma_indices, config);
}

CppAD::vector<CppAD::AD<double>>
random_mult(const CppAD::vector<CppAD::AD<double>> &x, const density_data &model,
            const Config &config) {

  const CscMatrix &Q = model.mult_precision_csc();
  const int n_term = model.gamma_map.ncol();
  if (Q.nrow() != n_term || Q.ncol() != n_term) {
    Rcpp::stop("random_mult Q is %d x %d but gamma_map has %d columns",
               Q.nrow(), Q.ncol(), n_term);
  }
  if (!Q.has_x()) {
    Rcpp::stop("random_mult Q has no numeric values");
  }

  const double rank = model.mult_precision_rank();
  const double log_det = model.mult_precision_log_det();

  const std::size_t t_global = model.theta_index(0);
  const std::size_t t_row = model.theta_row(0);
  CppAD::AD<double> logSd;
  if (transform_theta_at(config, t_row)) {
    logSd = x[t_global];
  } else {
    logSd = CppAD::log(x[t_global]);
  }
  const CppAD::AD<double> tau = CppAD::exp(-2 * logSd);

  // Register a copy of the shard-build cached CSC into the atomic payload.
  const std::size_t n = static_cast<std::size_t>(n_term);
  CppAD::vector<CppAD::AD<double>> ax(n);
  for (int j = 0; j < n_term; ++j) {
    const std::vector<std::size_t> gj_idx = model.gamma_global_indices(j);
    if (gj_idx.empty()) {
      Rcpp::stop("gamma_map column %d has no structural nonzero", j + 1);
    }
    ax[static_cast<std::size_t>(j)] = x[gj_idx[0]];
  }
  // Validate that every row index of Q also has a gamma mapping.
  for (std::size_t k = 0; k < Q.i.size(); ++k) {
    const int row = Q.i[k];
    if (model.gamma_global_indices(row).empty()) {
      Rcpp::stop("gamma_map column %d has no structural nonzero", row + 1);
    }
  }

  const std::size_t call_id =
      adlaplace::quadform::register_csc(CscMatrix(Q));
  CppAD::vector<CppAD::AD<double>> ay(1);
  adlaplace::quadform::call_quadform(call_id, ax, ay);
  const CppAD::AD<double> qpart = CppAD::AD<double>(0.5) * tau * ay[0];

  if (config.verbose) {
    Rcpp::Rcout << "random_mult n_gamma " << n_term << " rank " << rank
                << " log_det " << log_det << " theta index " << t_global
                << "\n";
  }

  CppAD::vector<CppAD::AD<double>> result(1);
  result[0] = -qpart - logSd * CppAD::AD<double>(rank) +
              CppAD::AD<double>(0.5 * log_det - rank * ONEHALFLOGTWOPI);
  return result;
}

//' Build raw AD handle for a random_diagonal shard
//'
//' @param model An \code{density_data} S4 object with \code{precision} slot set.
//' @param config Model configuration list.
//' @return External pointer of class \code{ad_pack_ptr}.
//' @keywords internal
// [[Rcpp::export]]
SEXP create_ad_shard_random_diagonal(SEXP model, Rcpp::List config) {
  const density_data ad_model(model);
  AdTape pack = build_ad_fun_random_with_pattern(
      ad_model, config, random_diagonal, random_diagonal_sparsity);
  std::vector<AdTape> packs;
  packs.push_back(std::move(pack));
  ad_pack *groups = packs_to_ad_fun(std::move(packs), ad_model.num_beta,
                                   ad_model.num_theta, adlaplace_make_ad_shard);
  return make_ad_pack_ptr(groups);
}

//' Build raw AD handle for a random_mult shard
//'
//' @param model An \code{density_data} S4 object with \code{precision} slot set.
//' @param config Model configuration list.
//' @return External pointer of class \code{ad_pack_ptr}.
//' @keywords internal
// [[Rcpp::export]]
SEXP create_ad_shard_random_mult(SEXP model, Rcpp::List config) {
  const density_data ad_model(model);
  // Cache Q on a local compacted model, then tape with the shared builder.
  density_data model_cached = ad_model;
  const Config cfg(config);
  model_cached.apply_tape_domain(cfg, "all", 0);
  model_cached.cache_mult_precision_csc();
  AdTape pack = build_ad_fun_random_with_pattern(
      model_cached, config, random_mult, random_mult_sparsity);
  std::vector<AdTape> packs;
  packs.push_back(std::move(pack));
  ad_pack *groups = packs_to_ad_fun(std::move(packs), ad_model.num_beta,
                                   ad_model.num_theta, adlaplace_make_ad_shard);
  return make_ad_pack_ptr(groups);
}
