#ifndef ADLAPLACE_EXTENSION_HPP
#define ADLAPLACE_EXTENSION_HPP

// Public API for backend packages that
// compile custom log densities and CppAD tape construction in their own shared
// library.
//
// On macOS, tape recording and evaluation must live in the same .so as the
// density functions. In exactly one backend .cpp:
//   #include "adlaplace/eval_impl.hpp"
//   #include "adlaplace/register_impl.hpp"  // packs_to_ad_fun / make_ad_pack_ptr
//   ADLAPLACE_DEFINE_BACKEND(my_make_shard)
// Backends do not link adlaplace.so; registration helpers are header-only.
// Observation/parameter densities export get_ad_pack_raw_obs /
// get_ad_pack_raw_parameters; random densities export create_ad_shard_<name>
// looked up via data@package.

#include <cstddef>
#include <vector>

#include <Rcpp.h>

#include "adlaplace/backend.hpp"
#include "adlaplace/density_data.hpp"
#include "adlaplace/ad_pack.hpp"
#include "adlaplace/eta.hpp"
#include "adlaplace/rviews.hpp"
#include "adlaplace/eval.hpp"

#if defined(__GNUC__)
#define ADLAPLACE_EXTENSION_EXPORT __attribute__((visibility("default")))
#else
#define ADLAPLACE_EXTENSION_EXPORT
#endif

SEXP make_ad_pack_ptr(ad_pack* groups);

ad_pack* packs_to_ad_fun(
  std::vector<AdTape>&& packs,
  std::size_t n_beta,
  std::size_t n_theta,
  ShardFactory factory);

// Include eval_impl.hpp exactly once in the backend .cpp before this macro.
#define ADLAPLACE_DEFINE_BACKEND(factory_sym) \
  ADLAPLACE_EXTENSION_EXPORT ad_shard* factory_sym(AdTape&& pack) { \
    return new EvalShard(std::move(pack), factory_sym); \
  }

#endif
