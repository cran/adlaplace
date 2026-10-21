#include <Rcpp.h>

#include <vector>

#include "adlaplace/runtime.hpp"

namespace {

Rcpp::IntegerVector shards_vector(
  const Rcpp::Nullable<Rcpp::IntegerVector>& ad_shards) {
  if (ad_shards.isNull()) {
    return Rcpp::IntegerVector();
  }
  return ad_shards.get();
}

}  // namespace

// [[Rcpp::export(name = ".joint_log_dens_cpp")]]
double joint_log_dens(
  SEXP ad_pack,
  const Rcpp::NumericVector& x,
  Rcpp::Nullable<Rcpp::IntegerVector> ad_shards = R_NilValue,
  bool negative = true) {

  ::ad_pack* backend = resolve_ad_pack_eval(ad_pack);
  adlaplace_require_serial_dens_handle(*backend);
  const size_t n_shards = backend->fun.size();
  const size_t Nparams = ad_tape_n_global(backend->fun[0]->pack);
  if (static_cast<size_t>(x.size()) != Nparams) {
    Rcpp::stop("x has length %d but expected Nparams=%d", x.size(), (int)Nparams);
  }

  const std::vector<size_t> shard_idx =
    resolve_shard_indices(n_shards, shards_vector(ad_shards));

  double total = 0.0;
  for (size_t s : shard_idx) {
    ad_shard* shard = shard_handle(backend, s);
    double fg = 0.0;
    const int rc = shard->f(x.begin(), &fg);
    if (rc != 0) {
      Rcpp::stop("shard f failed for shard %d with code %d", (int)s, rc);
    }
    total += fg;
  }
  return negative ? -total : total;
}

// [[Rcpp::export(name = ".grad_cpp")]]
Rcpp::NumericVector grad(
  SEXP ad_pack,
  const Rcpp::NumericVector& x,
  Rcpp::Nullable<Rcpp::IntegerVector> ad_shards = R_NilValue,
  bool inner = false,
  bool negative = true) {

  ::ad_pack* backend = resolve_ad_pack_eval(ad_pack);
  adlaplace_require_serial_dens_handle(*backend);
  const size_t Nparams = x.size();
  const size_t n_shards = backend->fun.size();
  const std::vector<size_t> shard_idx =
    resolve_shard_indices(n_shards, shards_vector(ad_shards));

  Rcpp::NumericVector grad_out(Nparams, 0.0);
  double f_dummy = 0.0;

  for (size_t s : shard_idx) {
    ad_shard* shard = shard_handle(backend, s);
    if (shard->f_grad(x.begin(), inner, &f_dummy, grad_out.begin()) != 0)
      Rcpp::stop("shard f_grad failed for shard %d", (int)s);
  }
  if (negative) {
    grad_out = -grad_out;
  }
  return grad_out;
}

// [[Rcpp::export(name = ".hessian_cpp")]]
Rcpp::S4 hessian(
  SEXP ad_pack,
  const Rcpp::NumericVector& x,
  Rcpp::Nullable<Rcpp::IntegerVector> ad_shards = R_NilValue,
  bool inner = false,
  const bool verbose = false,
  bool negative = true) {
  ::ad_pack* backend = resolve_ad_pack_eval(ad_pack);
  adlaplace_require_serial_dens_handle(*backend);

  const size_t Nparams = x.size();
  const size_t n_shards = backend->fun.size();
  const std::vector<size_t> shard_idx =
    resolve_shard_indices(n_shards, shards_vector(ad_shards));

  std::vector<int> tri_i;
  std::vector<int> tri_j;
  std::vector<double> tri_x;

  if (verbose) Rcpp::Rcout << "Starting Hessian computation..." << std::endl;

  for (size_t s : shard_idx) {
    ad_shard* shard = shard_handle(backend, s);
    int n_inner, n_outer, n_beta, n_theta, nnz_grad_i, nnz_grad_o, nnz_hes_i, nnz_hes_o;
    if (shard->get_sizes(&n_inner, &n_outer, &n_beta, &n_theta,
                          &nnz_grad_i, &nnz_grad_o, &nnz_hes_i, &nnz_hes_o) != 0)
      Rcpp::stop("shard get_sizes failed for shard %d", (int)s);

    int nnz = inner ? nnz_hes_i : nnz_hes_o;
    if (nnz == 0) continue;

    std::vector<int> p_gi(nnz_grad_i), p_go(nnz_grad_o);
    std::vector<int> p_hir(nnz_hes_i), p_hic(nnz_hes_i);
    std::vector<int> p_hor(nnz_hes_o), p_hoc(nnz_hes_o);

    if (shard->get_sparse_pattern(
        p_gi.data(), p_go.data(),
        p_hir.data(), p_hic.data(),
        p_hor.data(), p_hoc.data()) != 0)
      Rcpp::stop("shard get_sparse_pattern failed for shard %d", (int)s);

    const int* rows = inner ? p_hir.data() : p_hor.data();
    const int* cols = inner ? p_hic.data() : p_hoc.data();

    std::vector<double> vals(nnz);
    std::vector<int> map(nnz);
    for (int i = 0; i < nnz; ++i) map[i] = i;

    std::vector<double> grad_scratch(Nparams, 0.0);
    double f_dummy = 0.0;
    if (shard->f_grad_hess(x.begin(), inner, &f_dummy, grad_scratch.data(), vals.data(), map.data()) != 0)
      Rcpp::stop("shard f_grad_hess failed for shard %d", (int)s);

    const double sign = negative ? -1.0 : 1.0;
    for (int k = 0; k < nnz; ++k) {
      tri_i.push_back(rows[k]);
      tri_j.push_back(cols[k]);
      tri_x.push_back(sign * vals[k]);
    }
  }

  if (verbose) Rcpp::Rcout << "Hessian computation completed successfully" << std::endl;

  static Rcpp::Function stats_aggregate =
    Rcpp::Environment::namespace_env("stats")["aggregate"];
  static Rcpp::Function sparseMatrix =
    Rcpp::Environment::namespace_env("Matrix")["sparseMatrix"];
  const int n = static_cast<int>(Nparams);

  Rcpp::IntegerVector agg_i;
  Rcpp::IntegerVector agg_j;
  Rcpp::NumericVector agg_x;
  if (tri_x.empty()) {
    agg_i = Rcpp::IntegerVector(0);
    agg_j = Rcpp::IntegerVector(0);
    agg_x = Rcpp::NumericVector(0);
  } else {
    Rcpp::DataFrame df = Rcpp::DataFrame::create(
      Rcpp::Named("i") = Rcpp::wrap(tri_i),
      Rcpp::Named("j") = Rcpp::wrap(tri_j),
      Rcpp::Named("x") = Rcpp::wrap(tri_x)
    );
    Rcpp::DataFrame out = stats_aggregate(
      Rcpp::Formula("x ~ i + j"),
      Rcpp::Named("data") = df,
      Rcpp::Named("FUN") = Rcpp::Function("sum")
    );
    agg_i = out["i"];
    agg_j = out["j"];
    agg_x = out["x"];
  }

  return sparseMatrix(
    Rcpp::Named("i") = agg_i,
    Rcpp::Named("j") = agg_j,
    Rcpp::Named("x") = agg_x,
    Rcpp::Named("index1") = false,
    Rcpp::Named("symmetric") = true,
    Rcpp::Named("dims") = Rcpp::IntegerVector::create(n, n)
  );
}
