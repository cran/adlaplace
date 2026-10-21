#include "adlaplace/omp_compat.hpp"
#include "adlaplace/ompad.hpp"
#include "adlaplace/runtime.hpp"
#include "adlaplace/ad_pack_registry.hpp"

#include <Rcpp.h>
#include <cppad/cppad.hpp>
#include <cppad/utility/thread_alloc.hpp>
#include <cstdlib>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::size_t cppad_team_num_threads = 1;

// High-water mark of cppad_team_num_threads across the process lifetime.
// cppad_parallel_teardown() flushes thread_alloc free-lists for every thread
// that ever participated in a parallel team, not just the current team size.
// Rationale: when a caller does setup(4) -> teardown -> setup(2), the guard
// inside setup(2) fires teardown while cppad_team_num_threads is already 1
// (reset by the prior teardown). Without this mark, the 4-thread free-lists
// would never be flushed and the next team would inherit stale per-thread
// allocator state, which on Windows libomp aborts the process.
std::size_t max_team_num_threads = 1;

// Live ad_pack handles created by this DSO (adlaplace.so). Used by
// cppad_parallel_teardown to clear stale per-shard owner_thread assignments
// when the team size changes between independent callers. All access is on
// OpenMP thread 0 (R main thread / R GC); the mutex is belt-and-braces for
// the rare case of a finalizer racing with teardown on the same thread.
std::set<ad_pack *>& live_ad_packs() {
  static std::set<ad_pack *> s;
  return s;
}

std::mutex& registry_mutex() {
  static std::mutex m;
  return m;
}

void registry_add(ad_pack *p) {
  if (!p) return;
  std::lock_guard<std::mutex> lk(registry_mutex());
  live_ad_packs().insert(p);
}

void registry_remove(ad_pack *p) {
  if (!p) return;
  std::lock_guard<std::mutex> lk(registry_mutex());
  live_ad_packs().erase(p);
}

// Clear owner_thread / owner_thread_assigned on every shard of every live
// ad_pack. Called from cppad_parallel_teardown so the next team cannot
// reference a stale thread id from the torn-down team.
void registry_reset_owner_threads() {
  std::lock_guard<std::mutex> lk(registry_mutex());
  for (ad_pack *groups : live_ad_packs()) {
    if (!groups) continue;
    for (ad_shard *shard : groups->fun) {
      if (!shard) continue;
      shard->pack.owner_thread = 0;
      shard->pack.owner_thread_assigned = false;
    }
    groups->configured_num_threads = 1;
    groups->num_threads_configured = false;
  }
}

bool in_parallel_wrapper() { return omp_in_parallel() != 0; }

std::size_t thread_num_wrapper() {
  return static_cast<std::size_t>(omp_get_thread_num());
}

void set_num_threads_wrapper(std::size_t n) {
  omp_set_dynamic(0);
  omp_set_num_threads(static_cast<int>(n));
}

void require_serial_main_thread(const char *phase) {
  if (omp_in_parallel() != 0) {
    Rcpp::stop("%s: must not run inside an OpenMP parallel region", phase);
  }
  if (omp_get_thread_num() != 0) {
    Rcpp::stop("%s: must run on OpenMP thread 0", phase);
  }
}

// ADLAPLACE_HOLD_MEMORY=0 disables hold_memory(true) so return_memory deletes
// via the system allocator (thread-safe). Default is hold (1). Mitigation only;
// prefer draining shard buffers at parallel boundaries.
bool hold_memory_enabled() {
  const char *env = std::getenv("ADLAPLACE_HOLD_MEMORY");
  if (env == nullptr || env[0] == '\0') {
    return true;
  }
  return !(env[0] == '0' && env[1] == '\0');
}

#ifdef DEBUG
void debug_teardown_flush(std::size_t n_flush) {
  Rcpp::Rcout << "cppad_parallel_teardown: flushing " << n_flush
              << " thread pools\n";
}
void debug_teardown_restored() {
  Rcpp::Rcout << "cppad_parallel_teardown: serial mode restored\n";
}
#else
void debug_teardown_flush(std::size_t) {}
void debug_teardown_restored() {}
#endif

bool clamp_team_threads_enabled() {
  const char *env = std::getenv("ADLAPLACE_CLAMP_TEAM_THREADS");
  // Default on. Escape hatch: ADLAPLACE_CLAMP_TEAM_THREADS=0.
  return !(env != nullptr && env[0] == '0' && env[1] == '\0');
}

} // namespace

std::size_t adlaplace_latch_parallel_threads(std::size_t requested) {
  if (requested < 1)
    requested = 1;
#ifndef _OPENMP
  return 1;
#else
  // Serial (1) is the normal CppadParallelScope teardown / dens path and must
  // not latch or clamp. Only parallel requests (>1) raise / obey the mark.
  if (requested == 1)
    return 1;

  if (requested > max_team_num_threads)
    max_team_num_threads = requested;

  // Clamp-up only on Windows: decreasing the OpenMP/CppAD team size within one
  // R process aborts there (matrix: A4->B2 = 127, A4->B4 = 0). Other platforms
  // still track the high-water mark for teardown flush, but honor the request
  // so tests / users can change thread counts in-process.
#if defined(_WIN32)
  if (clamp_team_threads_enabled() && max_team_num_threads > requested)
    return max_team_num_threads;
#endif
  return requested;
#endif
}

//' Latch (and optionally raise) the process-wide parallel team size.
//'
//' For \code{requested > 1}, updates the process high-water mark. On Windows,
//' returns that mark after raising it so OpenMP/CppAD team setup never shrinks
//' within one R process (a decrease aborts under rtools/libomp). At eval time
//' empty trailing thread groups are padded to this width; shard owners from
//' \code{ad_pack()} still follow the requested \code{num_threads}. On other
//' platforms the requested count is returned unchanged. Serial
//' \code{requested = 1} is always returned unchanged.
//'
//' @param requested Positive integer thread count.
//' @return Integer effective OpenMP/CppAD team size (may exceed requested on
//'   Windows after a larger parallel team was used in this process).
//' @keywords internal
// [[Rcpp::export(rng = false)]]
int latch_parallel_threads(int requested) {
  if (requested < 1) {
    Rcpp::stop("requested must be a positive integer");
  }
  return static_cast<int>(
      adlaplace_latch_parallel_threads(static_cast<std::size_t>(requested)));
}

void cppad_parallel_setup(std::size_t num_threads) {
  require_serial_main_thread("cppad_parallel_setup");
  if (num_threads < 1)
    num_threads = 1;
#ifndef _OPENMP
  num_threads = 1;
#endif

  // Never shrink the CppAD/OpenMP team within a process: decreasing the team
  // size (e.g. 4 -> 2) leaves CppAD thread_alloc / libomp global state
  // inconsistent on Windows. Latch raises/clamps parallel requests for the
  // team width only; shard owners stay at the requested count from ad_pack().
  // Serial (1) is left alone for normal CppadParallelScope teardown.
  num_threads = adlaplace_latch_parallel_threads(num_threads);

  if (num_threads != cppad_team_num_threads) {
    cppad_parallel_teardown();
  }

  cppad_team_num_threads = num_threads;
  if (num_threads > max_team_num_threads) {
    max_team_num_threads = num_threads;
  }
  set_num_threads_wrapper(num_threads);

  if (num_threads == 1) {
    CppAD::thread_alloc::parallel_setup(1, nullptr, nullptr);
    CppAD::thread_alloc::hold_memory(false);
  } else {
    CppAD::thread_alloc::parallel_setup(num_threads, &in_parallel_wrapper,
                                        &thread_num_wrapper);
    CppAD::thread_alloc::hold_memory(hold_memory_enabled());
  }
  CppAD::parallel_ad<double>();
}

//' Whether this build was compiled with OpenMP support.
//'
//' @return \code{TRUE} if OpenMP was enabled at compile time, otherwise
//'   \code{FALSE}.
//' @export
// [[Rcpp::export(rng = false)]]
bool has_openmp() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

//' Touch OpenMP from the main thread after \code{dyn.load}.
//'
//' Intended for fresh R processes (e.g. \code{R CMD check} vignette re-builds)
//' so Homebrew libomp TLS is initialized before the first multi-thread
//' CppAD session.
//'
//' @keywords internal
// [[Rcpp::export(rng = false)]]
void warm_openmp_runtime() {
#ifdef _OPENMP
  omp_set_dynamic(0);
  (void)omp_get_max_threads();
#pragma omp parallel num_threads(1)
  {
    (void)omp_get_thread_num();
  }
#endif
}

void cppad_parallel_teardown() {
  require_serial_main_thread("cppad_parallel_teardown");

  // Flush thread_alloc free-lists for every thread that ever participated in
  // a parallel team in this process, not just the current team size. The
  // current cppad_team_num_threads may already be 1 (e.g. when teardown is
  // re-entered via the guard inside cppad_parallel_setup), in which case the
  // old code skipped the flush entirely and left stale per-thread allocator
  // state behind. max_team_num_threads remembers the high-water mark.
  const std::size_t n_flush = max_team_num_threads;
  if (n_flush > 1) {
    debug_teardown_flush(n_flush);
    set_num_threads_wrapper(n_flush);
#pragma omp parallel num_threads(static_cast<int>(n_flush))
    {
      CppAD::thread_alloc::free_available(
          static_cast<std::size_t>(omp_get_thread_num()));
    }
    for (std::size_t t = 0; t < n_flush; ++t) {
      CppAD::thread_alloc::free_available(t);
    }
    const bool all_freed = CppAD::thread_alloc::free_all();
#ifdef DEBUG
    if (!all_freed) {
      Rcpp::Rcout
          << "cppad_parallel_teardown: free_all() returned false "
             "(in-use blocks remain; check cross-thread thread_alloc frees)\n";
    }
#else
    (void)all_freed;
#endif
  }

  // Clear stale per-shard OpenMP owner_thread assignments on every live
  // ad_pack so the next team cannot reference a thread id from the team we
  // just tore down. Also drops configured_num_threads so a subsequent
  // ad_pack() call must reassign owner threads before any parallel eval.
  adlaplace_registry::reset_owner_threads();

  set_num_threads_wrapper(1);
  CppAD::thread_alloc::parallel_setup(1, nullptr, nullptr);
  CppAD::thread_alloc::hold_memory(false);
  CppAD::parallel_ad<double>();
  cppad_team_num_threads = 1;
  // Keep process-wide high-water mark across teardowns so a later request for
  // a smaller parallel team can be clamped back up. Resetting this here would
  // disable the clamp after every CppadParallelScope exit (the Windows 4->2
  // abort path).
  debug_teardown_restored();
}

#ifdef DEBUG

namespace {

struct ThreadMismatch {
  std::size_t shard = 0;
  std::size_t owner_thread = 0;
  std::size_t actual_thread = 0;
  std::string phase;
};

std::vector<ThreadMismatch> &mismatch_log() {
  static std::vector<ThreadMismatch> log;
  return log;
}

} // namespace

bool adlaplace_debug_enabled() { return true; }

bool adlaplace_shard_thread_ok(const AdTape &pack) {
  if (!pack.owner_thread_assigned) {
    return true;
  }
  return static_cast<std::size_t>(omp_get_thread_num()) == pack.owner_thread;
}

void adlaplace_debug_note_grad_mismatch(double *grad_local, std::size_t n) {
  if (n == 0 || grad_local == nullptr)
    return;
#pragma omp critical(adlaplace_debug_grad)
  {
    grad_local[0] = adlaplace_debug::kThreadMismatchSentinel;
  }
}

void adlaplace_debug_note_trace_mismatch(double *trace_accum, std::size_t n) {
  if (n == 0 || trace_accum == nullptr)
    return;
#pragma omp critical(adlaplace_debug_trace)
  {
    trace_accum[0] = adlaplace_debug::kThreadMismatchSentinel;
  }
}

void adlaplace_debug_record_mismatch(std::size_t shard,
                                     std::size_t owner_thread,
                                     std::size_t actual_thread,
                                     const char *phase) {
  const char *phase_str = phase ? phase : "(unknown)";
#pragma omp critical(adlaplace_debug_mismatch_log)
  {
    mismatch_log().push_back(ThreadMismatch{shard, owner_thread, actual_thread,
                                            std::string(phase_str)});
  }
}

void adlaplace_debug_raise_if_any(const char *context) {
  std::vector<ThreadMismatch> local;
#pragma omp critical(adlaplace_debug_mismatch_log)
  {
    local.swap(mismatch_log());
  }
  if (local.empty()) {
    return;
  }

  std::ostringstream os;
  os << "OpenMP thread affinity mismatch";
  if (context && context[0] != '\0') {
    os << " (" << context << ")";
  }
  os << ":\n";
  for (const ThreadMismatch &m : local) {
    os << "  shard " << m.shard << ": owner_thread=" << m.owner_thread
       << ", omp_get_thread_num()=" << m.actual_thread << ", phase=" << m.phase
       << "\n";
  }
  Rcpp::stop("%s", os.str().c_str());
}

void adlaplace_debug_print_load_banner() {
  static bool shown = false;
  if (shown)
    return;
  shown = true;
  Rcpp::Rcout << "adlaplace: DEBUG build (thread-affinity checks)\n";
}

#else

bool adlaplace_debug_enabled() { return false; }

bool adlaplace_shard_thread_ok(const AdTape &) { return true; }

void adlaplace_debug_note_grad_mismatch(double *, std::size_t) {}

void adlaplace_debug_note_trace_mismatch(double *, std::size_t) {}

void adlaplace_debug_record_mismatch(std::size_t, std::size_t, std::size_t,
                                     const char *) {}

void adlaplace_debug_raise_if_any(const char *) {}

void adlaplace_debug_print_load_banner() {}

#endif

// --- Live-ad_pack registry hook installation (adlaplace.so only) ----------
//
// registry_add / registry_remove / registry_reset_owner_threads live in the
// anonymous namespace above. Expose them through stable names and install
// into the ad_pack_registry hook slots so register_impl.hpp (included by
// this DSO) can register/unregister ad_pack handles and cppad_parallel_teardown
// can reset live shards' owner_thread affinity. Called once from
// R_init_adlaplace. Backend DSOs never call this; their hook slots stay null
// and the register_/unregister wrappers are no-ops there.

namespace adlaplace_detail {

void registry_add_hook(ad_pack *p) { registry_add(p); }
void registry_remove_hook(ad_pack *p) { registry_remove(p); }
void registry_reset_owner_threads_hook() { registry_reset_owner_threads(); }

}  // namespace adlaplace_detail

void adlaplace_install_registry_hooks() {
  adlaplace_registry::add_hook() = &adlaplace_detail::registry_add_hook;
  adlaplace_registry::remove_hook() = &adlaplace_detail::registry_remove_hook;
  adlaplace_registry::reset_hook() =
      &adlaplace_detail::registry_reset_owner_threads_hook;
}
