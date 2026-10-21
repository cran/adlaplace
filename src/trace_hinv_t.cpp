#include <Rcpp.h>

#include <chrono>
#include <vector>

#include "adlaplace/eval.hpp"
#include "adlaplace/runtime.hpp"
#include "adlaplace/trace_hinv_t_runtime.hpp"

namespace {

std::vector<double> ones_from_nnz(std::size_t nnz) {
  return std::vector<double>(nnz, 1.0);
}

double time_shard_trace3_once(
  ad_shard* shard,
  const std::vector<double>& x_vec,
  const std::vector<int>& LinvPt_p,
  const std::vector<int>& LinvPt_i,
  const std::vector<double>& LinvPt_x,
  const std::size_t LinvPt_ncol,
  const std::vector<int>& LinvPtColumns_p,
  const std::vector<int>& LinvPtColumns_i
) {
  const std::size_t Nparams = x_vec.size();
  std::vector<double> out_trace(Nparams, 0.0);
  shard->assign_memory();
  const auto t0 = std::chrono::steady_clock::now();
  const int rc = shard->trace_hinv_t(
    x_vec.data(),
    LinvPt_p.data(),
    LinvPt_i.data(),
    LinvPt_x.data(),
    LinvPt_ncol,
    LinvPt_p.size(),
    LinvPt_i.size(),
    LinvPt_x.size(),
    LinvPtColumns_p.data(),
    LinvPtColumns_i.data(),
    LinvPtColumns_p.size(),
    LinvPtColumns_i.size(),
    out_trace.data()
  );
  const auto t1 = std::chrono::steady_clock::now();
  if (rc != 0) {
    Rcpp::stop(
      "profile_shard_trace3_times: trace_hinv_t failed for shard %d: %s (%d)",
      static_cast<int>(shard->pack.shard_index),
      trace_hinv_t_strerror(rc),
      rc
    );
  }
  return std::chrono::duration<double>(t1 - t0).count();
}

}  // namespace

// [[Rcpp::export(name = ".trace_hinv_t_cpp")]]
Rcpp::NumericVector trace_hinv_t(
  SEXP ad_pack,
  const Rcpp::NumericVector& x,
  const Rcpp::S4& LinvPt,
  const Rcpp::S4& LinvPtColumns,
  bool verbose = false,
  Rcpp::Nullable<Rcpp::NumericVector> H_inv_x = R_NilValue
) {
  ::ad_pack* backend = resolve_ad_pack_eval(ad_pack);

  Rcpp::IntegerVector LinvPt_p = LinvPt.slot("p");
  Rcpp::IntegerVector LinvPt_i = LinvPt.slot("i");
  Rcpp::NumericVector LinvPt_x = LinvPt.slot("x");
  Rcpp::IntegerVector LinvPt_Dim = LinvPt.slot("Dim");

  Rcpp::IntegerVector LinvPtColumns_p = LinvPtColumns.slot("p");
  Rcpp::IntegerVector LinvPtColumns_i = LinvPtColumns.slot("i");

  const std::size_t LinvPt_ncol = static_cast<std::size_t>(LinvPt_Dim[1]);
  const std::vector<double> x_vec(x.begin(), x.end());
  const std::vector<int> LinvPt_p_vec(LinvPt_p.begin(), LinvPt_p.end());
  const std::vector<int> LinvPt_i_vec(LinvPt_i.begin(), LinvPt_i.end());
  const std::vector<double> LinvPt_x_vec(LinvPt_x.begin(), LinvPt_x.end());
  const std::vector<int> LinvPtColumns_p_vec(LinvPtColumns_p.begin(), LinvPtColumns_p.end());
  const std::vector<int> LinvPtColumns_i_vec(LinvPtColumns_i.begin(), LinvPtColumns_i.end());

  std::vector<double> hinv_vec;
  const double* hinv_ptr = nullptr;
  std::size_t hinv_len = 0;
  if (H_inv_x.isNotNull()) {
    const Rcpp::NumericVector v(H_inv_x.get());
    hinv_vec.assign(v.begin(), v.end());
    hinv_ptr = hinv_vec.data();
    hinv_len = hinv_vec.size();
  }

  const std::vector<double> trace_accum = adlaplace_trace::trace_hinv_t_impl(
    *backend,
    x_vec,
    LinvPt_p_vec,
    LinvPt_i_vec,
    LinvPt_x_vec,
    LinvPt_ncol,
    LinvPtColumns_p_vec,
    LinvPtColumns_i_vec,
    verbose,
    hinv_ptr,
    hinv_len
  );

  return Rcpp::NumericVector(trace_accum.begin(), trace_accum.end());
}

//' Profile per-shard Reverse3 cost with dummy LinvPt directions
//'
//' Serial timing helper for \code{ad_pack(..., reorder_shards = "third")}.
//' Uses the symbolic \code{half_H_inv} CSC pattern (values forced to 1) and
//' \code{trace_columns} as \code{LinvPtColumns}. Does not require owner
//' threads or a numeric Cholesky.
//'
//' @param handle External pointer of class \code{ad_pack_ptr}.
//' @param x Parameter vector of length \code{Nparams}.
//' @param half_H_inv Sparse Matrix (\code{dgCMatrix}) symbolic pattern.
//' @param trace_columns Sparse Matrix (\code{ngCMatrix}/\code{dgCMatrix})
//'   shard-to-column map (\code{dims = c(n_gamma, n_shards)}).
//' @param n_rep Number of timed repetitions per shard (mean stored).
//' @param n_warmup Number of untimed warmup sweeps per shard.
//' @return Numeric vector of length \code{n_shards} (mean elapsed seconds).
//' @keywords internal
// [[Rcpp::export]]
Rcpp::NumericVector profile_shard_trace3_times(
  SEXP handle,
  const Rcpp::NumericVector& x,
  const Rcpp::S4& half_H_inv,
  const Rcpp::S4& trace_columns,
  int n_rep = 4,
  int n_warmup = 1
) {
  if (n_rep < 1) {
    Rcpp::stop("n_rep must be a positive integer");
  }
  if (n_warmup < 0) {
    Rcpp::stop("n_warmup must be non-negative");
  }

  ::ad_pack* backend = ad_fun_from_handle(handle);
  const std::size_t n_shards = backend->fun.size();
  if (n_shards == 0) {
    return Rcpp::NumericVector();
  }

  Rcpp::IntegerVector LinvPt_p = half_H_inv.slot("p");
  Rcpp::IntegerVector LinvPt_i = half_H_inv.slot("i");
  Rcpp::IntegerVector LinvPt_Dim = half_H_inv.slot("Dim");
  const std::size_t LinvPt_ncol = static_cast<std::size_t>(LinvPt_Dim[1]);
  const std::size_t nnz = static_cast<std::size_t>(LinvPt_i.size());

  Rcpp::IntegerVector LinvPtColumns_p = trace_columns.slot("p");
  Rcpp::IntegerVector LinvPtColumns_i = trace_columns.slot("i");

  const std::vector<double> x_vec(x.begin(), x.end());
  const std::vector<int> LinvPt_p_vec(LinvPt_p.begin(), LinvPt_p.end());
  const std::vector<int> LinvPt_i_vec(LinvPt_i.begin(), LinvPt_i.end());
  const std::vector<double> LinvPt_x_vec = ones_from_nnz(nnz);
  const std::vector<int> LinvPtColumns_p_vec(
    LinvPtColumns_p.begin(), LinvPtColumns_p.end()
  );
  const std::vector<int> LinvPtColumns_i_vec(
    LinvPtColumns_i.begin(), LinvPtColumns_i.end()
  );

  const std::size_t Nparams = ad_tape_n_global(backend->fun[0]->pack);
  if (x_vec.size() != Nparams) {
    Rcpp::stop(
      "x has length %d but expected Nparams=%d",
      x.size(),
      static_cast<int>(Nparams)
    );
  }

  Rcpp::NumericVector times(static_cast<R_xlen_t>(n_shards));
  for (std::size_t s = 0; s < n_shards; ++s) {
    ad_shard* shard = shard_handle(backend, s);
    for (int w = 0; w < n_warmup; ++w) {
      (void)time_shard_trace3_once(
        shard,
        x_vec,
        LinvPt_p_vec,
        LinvPt_i_vec,
        LinvPt_x_vec,
        LinvPt_ncol,
        LinvPtColumns_p_vec,
        LinvPtColumns_i_vec
      );
    }
    double sum = 0.0;
    for (int r = 0; r < n_rep; ++r) {
      sum += time_shard_trace3_once(
        shard,
        x_vec,
        LinvPt_p_vec,
        LinvPt_i_vec,
        LinvPt_x_vec,
        LinvPt_ncol,
        LinvPtColumns_p_vec,
        LinvPtColumns_i_vec
      );
    }
    times[static_cast<R_xlen_t>(s)] = sum / static_cast<double>(n_rep);
  }
  return times;
}
