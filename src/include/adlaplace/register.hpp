#ifndef ADLAPLACE_REGISTER_HPP
#define ADLAPLACE_REGISTER_HPP

#include <Rcpp.h>
#include <string>
#include <vector>

#include "adlaplace/rviews.hpp"
#include "adlaplace/density_data.hpp"
#include "adlaplace/backend.hpp"
#include "adlaplace/extension.hpp"

void adlaplace_init_atomics();

void ad_fun_destroy(ad_pack* groups);
void adfun_finalizer(SEXP ext);
SEXP make_ad_pack_ptr(ad_pack* groups);

ad_pack* packs_to_ad_fun(
  std::vector<AdTape>&& packs,
  std::size_t n_beta,
  std::size_t n_theta,
  ShardFactory factory);

ad_pack* combine_ad_fun(const std::vector<ad_pack*>& parts);

ad_pack* get_ad_pack_raw_obs_h(
  SEXP model,
  const Rcpp::List& config,
  const std::string& obs_name);

ad_pack* get_ad_pack_raw_parameters_h(
  SEXP model,
  const Rcpp::List& config,
  const std::string& single_name);

#endif
