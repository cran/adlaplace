// Include in exactly one .cpp per shared library (adlaplace.so or a backend .so),
// alongside eval_impl.hpp. Provides packs_to_ad_fun / make_ad_pack_ptr so backends
// do not need to link adlaplace.so.

#include "adlaplace/extension.hpp"
#include "adlaplace/ad_pack_registry.hpp"

#include <R.h>
#include <Rinternals.h>

// Reset per-shard OpenMP owner_thread affinity before destroying shards.
// Called from ad_fun_destroy (finalizer + combine_ad_fun). Ensures stale
// owner_thread ids from a previous, now-torn-down CppAD team do not survive
// into ADFun / sparse-work destructors (which free thread_alloc blocks and
// must not reference a thread id that no longer exists in the active team).
// Also unregisters the handle from the live-ad_pack registry (adlaplace.so).
static void ad_fun_reset_shard_affinity(ad_pack* groups) {
  if (!groups) return;
  for (ad_shard* shard : groups->fun) {
    if (!shard) continue;
    shard->pack.owner_thread = 0;
    shard->pack.owner_thread_assigned = false;
  }
}

void ad_fun_destroy(ad_pack* groups) {
  if (!groups) return;
  adlaplace_registry::unregister(groups);
  ad_fun_reset_shard_affinity(groups);
  for (ad_shard* shard : groups->fun) {
    delete shard;
  }
  delete groups;
}

void adfun_finalizer(SEXP ext) {
  ad_pack* groups = static_cast<ad_pack*>(R_ExternalPtrAddr(ext));
  ad_fun_destroy(groups);
  R_ClearExternalPtr(ext);
}

SEXP make_ad_pack_ptr(ad_pack* groups) {
  SEXP handle = R_MakeExternalPtr(static_cast<void*>(groups), R_NilValue, R_NilValue);
  R_RegisterCFinalizerEx(handle, adfun_finalizer, TRUE);
  Rf_setAttrib(handle, R_ClassSymbol, Rf_mkString("ad_pack_ptr"));
  adlaplace_registry::register_(groups);
  return handle;
}

ad_pack* packs_to_ad_fun(
  std::vector<AdTape>&& packs,
  std::size_t n_beta,
  std::size_t n_theta,
  ShardFactory factory) {

  if (!factory) {
    Rcpp::stop("packs_to_ad_fun: factory is NULL");
  }

  auto* groups = new ad_pack();
  groups->abi_version = ADLAPLACE_ABI_VERSION;
  groups->fun.reserve(packs.size());
  for (size_t g = 0; g < packs.size(); ++g) {
    packs[g].shard_index = g;
    packs[g].n_beta = n_beta;
    packs[g].n_theta = n_theta;
    groups->fun.push_back(factory(std::move(packs[g])));
  }
  return groups;
}
