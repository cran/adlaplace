#ifndef ADLAPLACE_RUNTIME_HPP
#define ADLAPLACE_RUNTIME_HPP

#include <Rcpp.h>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "adlaplace/backend.hpp"
#include "adlaplace/ompad.hpp"
#include "adlaplace/rviews.hpp"

// Flushable verbose breadcrumb (endl flushes so last line survives aborts).
inline void adlaplace_verbose_msg(bool verbose, const std::string& msg) {
  if (!verbose) {
    return;
  }
  Rcpp::Rcout << msg << std::endl;
}

ad_pack* clone_ad_pack(const ad_pack* src);

hessian_template hessian_template_from_dgc(
  const DgCView& tpl,
  const char* label);

void ad_fun_attach_hessians_from_list(
  ad_pack& shards,
  const Rcpp::List& ad_pack);

ad_shard* shard_handle(ad_pack* backend, size_t shard);

SEXP ad_fun_handle_sexp(const Rcpp::List& ad_fun_list);

ad_pack* ad_fun_from_list(const Rcpp::List& ad_fun_list);

ad_pack* ad_fun_from_handle(SEXP handle);

ad_pack* resolve_ad_pack_eval(SEXP ad_pack_ptr);

ad_pack* resolve_ad_pack_laplace(const Rcpp::S4& ad_fun_s4);

std::vector<size_t> resolve_shard_indices(
  size_t n_shards,
  const Rcpp::IntegerVector& shards);

Rcpp::List sparsity_shard_from_handle(ad_shard* shard);

inline void adlaplace_require_owner_threads_assigned(const ad_pack& backend) {
  for (std::size_t s = 0; s < backend.fun.size(); ++s) {
    ad_shard* shard = shard_handle(const_cast<ad_pack*>(&backend), s);
    if (!shard->pack.owner_thread_assigned) {
      Rcpp::stop(
        "OpenMP thread assignment missing; call ad_pack(..., num_threads=) before inner_opt or trace_hinv_t"
      );
    }
  }
}

// Serial debug dens/grad/hessian: reject multi-thread handles from ad_pack(N>1).
inline void adlaplace_require_serial_dens_handle(const ad_pack& backend) {
  if (backend.configured_num_threads > 1) {
    Rcpp::stop(
      "joint_log_dens/grad/hessian are serial debug APIs; this handle has "
      "num_threads=%d from ad_pack(). Use clone_ad_pack_ptr() before ad_pack(), "
      "or call dens on that plain ptr (or ad_pack with num_threads=1).",
      static_cast<int>(backend.configured_num_threads)
    );
  }
}

inline std::vector<std::vector<std::size_t>> thread_groups_from_backend(ad_pack& backend) {
  const std::size_t n = backend.fun.size();
  std::vector<std::size_t> owners(n);
  std::size_t max_t = 0;
  for (std::size_t s = 0; s < n; ++s) {
    const std::size_t t = backend.fun[s]->pack.owner_thread;
    owners[s] = t;
    if (t > max_t) max_t = t;
  }
  std::vector<std::vector<std::size_t>> groups(max_t + 1);
  for (std::size_t s = 0; s < n; ++s) {
    groups[owners[s]].push_back(s);
  }
  return groups;
}

// Owner-derived groups, padded with empty trailing groups up to the process
// parallel-team high-water mark (Windows clamp-up). Callers must use
// groups.size() as the OpenMP / CppadParallelScope / AD_Func_Opt width so the
// team never shrinks after a larger session, while shard owners still follow
// the requested num_threads from ad_pack().
inline std::vector<std::vector<std::size_t>>
thread_groups_for_parallel_eval(ad_pack& backend) {
  std::vector<std::vector<std::size_t>> groups =
      thread_groups_from_backend(backend);
  const std::size_t team =
      adlaplace_latch_parallel_threads(groups.empty() ? 1 : groups.size());
  if (team > groups.size()) {
    groups.resize(team);
  }
  return groups;
}

namespace adlaplace_debug {

inline constexpr double kThreadMismatchSentinel =
  std::numeric_limits<double>::quiet_NaN();

}  // namespace adlaplace_debug

bool adlaplace_debug_enabled();

bool adlaplace_shard_thread_ok(const AdTape& pack);

void adlaplace_debug_note_grad_mismatch(double* grad_local, std::size_t n);

void adlaplace_debug_note_trace_mismatch(double* trace_accum, std::size_t n);

void adlaplace_debug_record_mismatch(
  std::size_t shard,
  std::size_t owner_thread,
  std::size_t actual_thread,
  const char* phase);

void adlaplace_debug_raise_if_any(const char* context);

void adlaplace_debug_print_load_banner();

#endif
