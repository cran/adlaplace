#ifndef ADLAPLACE_AD_PACK_REGISTRY_HPP
#define ADLAPLACE_AD_PACK_REGISTRY_HPP

// Cross-DSO live-ad_pack registry, used by cppad_parallel_teardown() to reset
// stale per-shard OpenMP owner_thread assignments when the CppAD team size
// changes between independent callers (e.g. between testthat files running
// in one R process).
//
// adlaplace.so owns the registry and sets the hooks below at load time
// (R_init_adlaplace). Backend shared libraries include this header but do
// NOT link adlaplace.so; their per-DSO hook slots stay null, so the
// register_/unregister wrappers below are no-ops there. That is intentional:
// cppad_parallel_teardown lives in adlaplace.so and only needs to see
// adlaplace.so-owned ad_pack handles. Backends still reset their own shards'
// owner_thread inside ad_fun_destroy() (see register_impl.hpp), so backend
// handles are cleaned up locally at finalization time.
//
// All registry access happens on OpenMP thread 0 (R main thread / R GC):
// make_ad_pack_ptr runs from R, adfun_finalizer runs from R's single-threaded
// GC, and cppad_parallel_teardown requires omp_in_parallel()==false on thread
// 0. No locking needed.

struct ad_pack;

namespace adlaplace_registry {

using add_fn = void (*)(ad_pack*);
using remove_fn = void (*)(ad_pack*);
using reset_owner_threads_fn = void (*)();

// Per-DSO hook slots. adlaplace.so populates its own slots in R_init_adlaplace;
// backend DSOs leave them null.
inline add_fn& add_hook() {
  static add_fn f = nullptr;
  return f;
}
inline remove_fn& remove_hook() {
  static remove_fn f = nullptr;
  return f;
}
inline reset_owner_threads_fn& reset_hook() {
  static reset_owner_threads_fn f = nullptr;
  return f;
}

// Safe wrappers used by register_impl.hpp. No-ops when adlaplace.so is not
// loaded in this DSO.
inline void register_(ad_pack* p) {
  if (add_hook()) add_hook()(p);
}
inline void unregister(ad_pack* p) {
  if (remove_hook()) remove_hook()(p);
}

// Walk all live ad_pack handles registered with this DSO and clear
// owner_thread / owner_thread_assigned on every shard. Called by
// cppad_parallel_teardown. Lives in adlaplace.so; null elsewhere.
inline void reset_owner_threads() {
  if (reset_hook()) reset_hook();
}

}  // namespace adlaplace_registry

#endif
