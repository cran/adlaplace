#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "adlaplace/density_data.hpp"
#include "adlaplace/register.hpp"
#include "adlaplace/runtime.hpp"

namespace {

Rcpp::S4 make_obs_groups_identity_batch(
  const std::vector<int>& unit_ids,
  int n_domain) {
  const int n_batch = static_cast<int>(unit_ids.size());
  if (n_batch < 1) {
    Rcpp::stop("make_obs_groups_identity_batch: empty batch");
  }
  if (n_domain < 1) {
    Rcpp::stop("make_obs_groups_identity_batch: n_domain must be positive");
  }

  Rcpp::IntegerVector i(n_batch);
  Rcpp::IntegerVector p(n_batch + 1);
  p[0] = 0;
  for (int j = 0; j < n_batch; ++j) {
    const int u = unit_ids[static_cast<std::size_t>(j)];
    if (u < 0 || u >= n_domain) {
      Rcpp::stop(
        "unit index %d out of range [0, %d)",
        u,
        n_domain);
    }
    i[j] = u;
    p[j + 1] = j + 1;
  }

  Rcpp::S4 mat("ngCMatrix");
  mat.slot("i") = i;
  mat.slot("p") = p;
  mat.slot("Dim") = Rcpp::IntegerVector::create(n_domain, n_batch);
  return mat;
}

Rcpp::S4 csc_from_triplets(
  int nrow,
  int ncol,
  std::vector<int>& ti,
  std::vector<int>& tj,
  std::vector<double>& tx) {
  const std::size_t nnz = ti.size();
  if (tj.size() != nnz || tx.size() != nnz) {
    Rcpp::stop("csc_from_triplets: triplet length mismatch");
  }

  if (nnz == 0) {
    Rcpp::S4 mat("dgCMatrix");
    mat.slot("i") = Rcpp::IntegerVector();
    mat.slot("p") = Rcpp::IntegerVector(ncol + 1, 0);
    mat.slot("x") = Rcpp::NumericVector();
    mat.slot("Dim") = Rcpp::IntegerVector::create(nrow, ncol);
    return mat;
  }

  std::vector<std::size_t> ord(nnz);
  for (std::size_t k = 0; k < nnz; ++k) {
    ord[k] = k;
  }
  std::sort(ord.begin(), ord.end(), [&](std::size_t a, std::size_t b) {
    if (tj[a] != tj[b]) {
      return tj[a] < tj[b];
    }
    return ti[a] < ti[b];
  });

  Rcpp::IntegerVector i(static_cast<R_xlen_t>(nnz));
  Rcpp::NumericVector x(static_cast<R_xlen_t>(nnz));
  Rcpp::IntegerVector p(ncol + 1);
  int col = 0;
  p[0] = 0;
  for (std::size_t pos = 0; pos < nnz; ++pos) {
    const std::size_t k = ord[pos];
    while (col < tj[k]) {
      ++col;
      p[col] = static_cast<int>(pos);
    }
    i[static_cast<R_xlen_t>(pos)] = ti[k];
    x[static_cast<R_xlen_t>(pos)] = tx[k];
  }
  while (col < ncol) {
    ++col;
    p[col] = static_cast<int>(nnz);
  }

  Rcpp::S4 mat("dgCMatrix");
  mat.slot("i") = i;
  mat.slot("p") = p;
  mat.slot("x") = x;
  mat.slot("Dim") = Rcpp::IntegerVector::create(nrow, ncol);
  return mat;
}

}  // namespace

//' Stream per-unit observation gradients (grad-only tapes)
//'
//' For each observation unit (ELGM stratum column, or row of \code{y} when
//' there is no ELGM map), records a temporary one-unit observation tape with
//' \code{hessian_sparsity = FALSE}, evaluates the sparse gradient at
//' \code{x}, then destroys the tape. Does not run \code{inner_opt}.
//'
//' @param model Observation \code{density_data} S4 object.
//' @param x Full parameter vector \code{c(beta, gamma, theta)}.
//' @param config Config list (merged with grad-only defaults upstream).
//' @param units Optional 0-based unit indices; empty means all units.
//' @param batch_size Positive integer batch size for temporary tapes.
//' @param inner If \code{TRUE}, return inner-\eqn{\gamma} gradients only
//'   (rows are gamma). If \code{FALSE}, rows are the full parameter vector.
//' @param negative If \code{TRUE}, negate the density gradient (minimization
//'   sign, matching \code{\link{grad}}).
//' @return A \code{dgCMatrix} with one column per unit.
//' @keywords internal
// [[Rcpp::export(name = ".grad_obs_units_cpp")]]
Rcpp::S4 grad_obs_units_cpp(
  SEXP model,
  const Rcpp::NumericVector& x,
  Rcpp::List config,
  Rcpp::IntegerVector units,
  int batch_size,
  bool inner,
  bool negative) {

  const density_data data(model);
  const std::size_t n_beta = data.num_beta;
  const std::size_t n_gamma = data.num_gamma;
  const std::size_t n_full = data.num_full;

  if (static_cast<std::size_t>(x.size()) != n_full) {
    Rcpp::stop(
      "x has length %d but expected n_full=%d",
      static_cast<int>(x.size()),
      static_cast<int>(n_full));
  }
  if (batch_size < 1) {
    Rcpp::stop("batch_size must be a positive integer");
  }

  SEXP dens_slot = R_do_slot(model, Rf_mkString("density"));
  const std::string dens_name = Rcpp::as<std::string>(dens_slot);
  if (dens_name.empty()) {
    Rcpp::stop("model@density is required");
  }

  const bool use_elgm = data.elgm_matrix.ncol() > 0;
  const int n_domain = use_elgm
    ? data.elgm_matrix.ncol()
    : static_cast<int>(data.y.size());
  if (n_domain < 1) {
    Rcpp::stop("grad_obs_units: no observation units found");
  }

  std::vector<int> unit_ids;
  if (units.size() == 0) {
    unit_ids.resize(static_cast<std::size_t>(n_domain));
    for (int u = 0; u < n_domain; ++u) {
      unit_ids[static_cast<std::size_t>(u)] = u;
    }
  } else {
    unit_ids.assign(units.begin(), units.end());
  }
  const int n_units = static_cast<int>(unit_ids.size());
  if (n_units < 1) {
    Rcpp::stop("grad_obs_units: units is empty");
  }

  const int n_rows = inner
    ? static_cast<int>(n_gamma)
    : static_cast<int>(n_full);

  std::vector<int> ti;
  std::vector<int> tj;
  std::vector<double> tx;
  ti.reserve(static_cast<std::size_t>(n_units) * 8U);
  tj.reserve(static_cast<std::size_t>(n_units) * 8U);
  tx.reserve(static_cast<std::size_t>(n_units) * 8U);

  std::vector<double> x_vec(x.begin(), x.end());
  std::vector<double> g_buf(n_full, 0.0);

  // Force grad-only / compact defaults even if caller forgot.
  config["hessian_sparsity"] = false;
  if (!config.containsElementNamed("compact_tape") ||
      Rf_isNull(config["compact_tape"])) {
    config["compact_tape"] = true;
  }

  for (int start = 0; start < n_units; start += batch_size) {
    const int end = std::min(start + batch_size, n_units);
    const int n_batch = end - start;
    std::vector<int> batch(static_cast<std::size_t>(n_batch));
    for (int j = 0; j < n_batch; ++j) {
      batch[static_cast<std::size_t>(j)] =
        unit_ids[static_cast<std::size_t>(start + j)];
    }

    Rcpp::List cfg_batch = Rcpp::clone(config);
    cfg_batch["obs_groups"] = make_obs_groups_identity_batch(batch, n_domain);

    ad_pack* groups = get_ad_pack_raw_obs_h(model, cfg_batch, dens_name);
    if (!groups || groups->fun.size() != static_cast<std::size_t>(n_batch)) {
      if (groups) {
        ad_fun_destroy(groups);
      }
      Rcpp::stop(
        "grad_obs_units: expected %d shards, got %d",
        n_batch,
        groups ? static_cast<int>(groups->fun.size()) : 0);
    }

    try {
      for (int j = 0; j < n_batch; ++j) {
        ad_shard* shard = shard_handle(groups, static_cast<std::size_t>(j));
        std::fill(g_buf.begin(), g_buf.end(), 0.0);
        double f_dummy = 0.0;
        if (shard->f_grad(x_vec.data(), inner, &f_dummy, g_buf.data()) != 0) {
          Rcpp::stop("grad_obs_units: f_grad failed for unit batch index %d", j);
        }
        if (negative) {
          for (double& v : g_buf) {
            v = -v;
          }
        }

        const int out_col = start + j;
        if (inner) {
          for (std::size_t gi = 0; gi < n_gamma; ++gi) {
            const double v = g_buf[n_beta + gi];
            if (v != 0.0 && R_finite(v)) {
              ti.push_back(static_cast<int>(gi));
              tj.push_back(out_col);
              tx.push_back(v);
            }
          }
        } else {
          for (std::size_t gi = 0; gi < n_full; ++gi) {
            const double v = g_buf[gi];
            if (v != 0.0 && R_finite(v)) {
              ti.push_back(static_cast<int>(gi));
              tj.push_back(out_col);
              tx.push_back(v);
            }
          }
        }
      }
    } catch (...) {
      ad_fun_destroy(groups);
      throw;
    }
    ad_fun_destroy(groups);
  }

  return csc_from_triplets(n_rows, n_units, ti, tj, tx);
}
