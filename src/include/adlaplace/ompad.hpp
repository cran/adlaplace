#ifndef ADLAPLACE_OMPAD_HPP
#define ADLAPLACE_OMPAD_HPP

#include <Rcpp.h>
#include <cstddef>
#include <string>

// Setup/teardown must run on OpenMP thread 0, outside any #pragma omp parallel
// region. Tape build stays in default sequential CppAD mode; CppadParallelScope(N)
// at eval time (inner_opt, trace_hinv_t, etc.) owns parallel setup/teardown.
// cppad_parallel_setup() tears down first when the thread count changes.
//
// Before/after each parallel session, drain per-shard ADFun taylor + trace
// buffers in sequential mode so worker threads never free thread-0-owned
// thread_alloc blocks (hold_memory(true) free-list race).

struct ad_pack;

// Process-wide parallel-team latch. For requested > 1, raises the high-water
// mark. On Windows, returns max(requested, high-water) so OpenMP/CppAD team
// setup never shrinks after a larger team was used (that decrease aborts under
// rtools/libomp). Other platforms return requested unchanged. Serial requests
// (0/1) are always unchanged. Used by cppad_parallel_setup and by
// thread_groups_for_parallel_eval (pads empty groups to the team width).
// Owner maps from ad_pack()/assign_owner_threads always follow the requested
// count; they must not use the clamped return.
std::size_t adlaplace_latch_parallel_threads(std::size_t requested);

void cppad_parallel_setup(std::size_t num_threads);
void cppad_parallel_teardown();

// Must run with omp_in_parallel() == false. Frees ADFun taylor capacity and
// clears TraceWorkspace vectors so subsequent parallel eval reallocates on
// the owning OpenMP thread.
void adlaplace_release_shard_eval_buffers(ad_pack& backend);

struct CppadParallelScope {
  explicit CppadParallelScope(std::size_t num_threads, bool verbose = false,
                              ad_pack* backend = nullptr)
      : verbose_(verbose), backend_(backend) {
    if (backend_ != nullptr) {
      if (verbose_) {
        Rcpp::Rcout << "CppadParallelScope: drain shard eval buffers (entry)..."
                    << std::endl;
      }
      adlaplace_release_shard_eval_buffers(*backend_);
    }
    if (verbose_) {
      Rcpp::Rcout << "CppadParallelScope: setup num_threads=" << num_threads
                  << std::endl;
    }
    cppad_parallel_setup(num_threads);
    if (verbose_) {
      Rcpp::Rcout << "CppadParallelScope: setup done" << std::endl;
    }
  }
  ~CppadParallelScope() {
    if (backend_ != nullptr) {
      if (verbose_) {
        Rcpp::Rcout << "CppadParallelScope: drain shard eval buffers (exit)..."
                    << std::endl;
      }
      adlaplace_release_shard_eval_buffers(*backend_);
    }
    if (verbose_) {
      Rcpp::Rcout
          << "CppadParallelScope: teardown (free_available + free_all)..."
          << std::endl;
    }
    cppad_parallel_teardown();
    if (verbose_) {
      Rcpp::Rcout << "CppadParallelScope: teardown done" << std::endl;
    }
  }

  CppadParallelScope(const CppadParallelScope&) = delete;
  CppadParallelScope& operator=(const CppadParallelScope&) = delete;

private:
  bool verbose_;
  ad_pack* backend_;
};

#endif
