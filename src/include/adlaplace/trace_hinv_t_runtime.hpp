#ifndef ADLAPLACE_TRACE_HINV_T_RUNTIME_HPP
#define ADLAPLACE_TRACE_HINV_T_RUNTIME_HPP

#include <Rcpp.h>
#include <cppad/cppad.hpp>
#include <vector>

#include "adlaplace/backend.hpp"
#include "adlaplace/omp_compat.hpp"
#include "adlaplace/ompad.hpp"
#include "adlaplace/runtime.hpp"

namespace adlaplace_trace {

// Alias: sequential drain of ADFun taylor + TraceWorkspace (see ompad.hpp).
inline void trace_hinv_t_reset_shards(ad_pack& backend) {
  adlaplace_release_shard_eval_buffers(backend);
}

inline std::vector<double> trace_hinv_t_parallel(
  ad_pack& backend,
  const std::vector<double>& x,
  const std::vector<int>& LinvPt_p,
  const std::vector<int>& LinvPt_i,
  const std::vector<double>& LinvPt_x,
  const std::size_t LinvPt_ncol,
  const std::vector<int>& LinvPtColumns_p,
  const std::vector<int>& LinvPtColumns_i,
  const bool verbose,
  const double* H_inv_x = nullptr,
  std::size_t H_inv_x_len = 0) {

  const std::size_t Nparams = ad_tape_n_global(backend.fun[0]->pack);
  if (x.size() != Nparams) {
    Rcpp::stop(
      "x has length %d but expected Nparams=%d",
      static_cast<int>(x.size()),
      static_cast<int>(Nparams)
    );
  }

  const std::size_t LinvPt_p_len = LinvPt_p.size();
  const std::size_t LinvPt_i_len = LinvPt_i.size();
  const std::size_t LinvPt_x_len = LinvPt_x.size();
  const std::size_t LinvPtColumns_p_len = LinvPtColumns_p.size();
  const std::size_t LinvPtColumns_i_len = LinvPtColumns_i.size();

  std::vector<double> trace_accum(Nparams, 0.0);
  if (backend.fun.empty()) {
    return trace_accum;
  }

  adlaplace_require_owner_threads_assigned(backend);
  const std::vector<std::vector<std::size_t>> thread_groups =
    thread_groups_for_parallel_eval(backend);
  const int num_threads = static_cast<int>(thread_groups.size());

  if (verbose) {
    Rcpp::Rcout << "trace_hinv_t: threads = " << num_threads
                << ", shards = " << backend.fun.size()
                << ", params = " << Nparams << "\n";
  }

#pragma omp parallel num_threads(num_threads)
  {
    const int tid = omp_get_thread_num();
    std::vector<double> trace_local(Nparams, 0.0);
    const std::vector<std::size_t>& shard_group =
      thread_groups[static_cast<std::size_t>(tid)];
    const std::vector<int>& q_p = backend.chol_pattern.hinv_q_index_p;
    const std::vector<int>& q_i = backend.chol_pattern.hinv_q_index_i;
    const bool q_ok = q_p.size() == backend.fun.size() + 1;
    for (std::size_t s : shard_group) {
      ad_shard* shard = shard_handle(&backend, s);
      shard->assign_memory();
      int rc = 1;
      if (H_inv_x != nullptr && q_ok) {
        const int q0 = q_p[s];
        const int q1 = q_p[s + 1];
        if (q1 > q0 && q0 >= 0 &&
            static_cast<std::size_t>(q1) <= q_i.size()) {
          rc = shard->trace_hinv_from_h(
            x.data(),
            H_inv_x,
            q_i.data() + q0,
            H_inv_x_len,
            static_cast<std::size_t>(q1 - q0),
            trace_local.data()
          );
        }
      }
      if (rc != 0) {
        shard->trace_hinv_t(
          x.data(),
          LinvPt_p.data(),
          LinvPt_i.data(),
          LinvPt_x.data(),
          LinvPt_ncol,
          LinvPt_p_len,
          LinvPt_i_len,
          LinvPt_x_len,
          LinvPtColumns_p.data(),
          LinvPtColumns_i.data(),
          LinvPtColumns_p_len,
          LinvPtColumns_i_len,
          trace_local.data()
        );
      }
    }
#pragma omp critical(trace_hinv_merge)
    {
      for (std::size_t d = 0; d < Nparams; ++d) {
        trace_accum[d] += trace_local[d];
      }
    }
  }

  if (verbose) {
    Rcpp::Rcout << "trace_hinv_t: finished\n";
  }

  return trace_accum;
}

inline std::vector<double> trace_hinv_t_impl(
  ad_pack& backend,
  const std::vector<double>& x,
  const std::vector<int>& LinvPt_p,
  const std::vector<int>& LinvPt_i,
  const std::vector<double>& LinvPt_x,
  const std::size_t LinvPt_ncol,
  const std::vector<int>& LinvPtColumns_p,
  const std::vector<int>& LinvPtColumns_i,
  const bool verbose,
  const double* H_inv_x = nullptr,
  std::size_t H_inv_x_len = 0) {
  if (backend.fun.empty()) {
    return std::vector<double>();
  }

  adlaplace_require_owner_threads_assigned(backend);
  const std::vector<std::vector<std::size_t>> thread_groups =
    thread_groups_for_parallel_eval(backend);
  const int num_threads = static_cast<int>(thread_groups.size());

  if (verbose && adlaplace_debug_enabled()) {
    Rcpp::Rcout << "trace_hinv_t: entering CppAD parallel session ("
                << num_threads << " threads)\n";
  }

  std::vector<double> trace_accum;
  {
    // Entry drain (via scope) replaces the former manual pre-reset.
    CppadParallelScope parallel_scope(
        static_cast<std::size_t>(num_threads), verbose, &backend);
    trace_accum = trace_hinv_t_parallel(
      backend,
      x,
      LinvPt_p,
      LinvPt_i,
      LinvPt_x,
      LinvPt_ncol,
      LinvPtColumns_p,
      LinvPtColumns_i,
      verbose,
      H_inv_x,
      H_inv_x_len
    );
    if (verbose && adlaplace_debug_enabled()) {
      Rcpp::Rcout << "trace_hinv_t: shard eval done\n";
    }
  }

  if (verbose && adlaplace_debug_enabled()) {
    Rcpp::Rcout << "trace_hinv_t: parallel block ended\n";
  }
  adlaplace_debug_raise_if_any("trace_hinv_t after parallel block");

  return trace_accum;
}

}  // namespace adlaplace_trace

#endif
