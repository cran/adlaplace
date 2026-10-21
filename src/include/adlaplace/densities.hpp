#ifndef ADLAPLACE_DENSITIES_HPP
#define ADLAPLACE_DENSITIES_HPP

#include <Rcpp.h>
#include <cppad/cppad.hpp>
#include "adlaplace/density_data.hpp"
#include "adlaplace/rviews.hpp"

CppAD::vector<CppAD::AD<double>> binomial_obs(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config,
  size_t Dgroup);

CppAD::vector<CppAD::AD<double>> gaussian_obs(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config,
  size_t Dgroup);

CppAD::vector<CppAD::AD<double>> gaussian_extra(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config);

CppAD::vector<CppAD::AD<double>> nbinom_obs(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config,
  size_t Dgroup);

CppAD::vector<CppAD::AD<double>> nbinom_extra(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config);

CppAD::vector<CppAD::AD<double>> poisson_obs(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config,
  size_t Dgroup);

CppAD::vector<CppAD::AD<double>> dirichlet_multinomial(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config,
  size_t Dgroup);

CppAD::vector<CppAD::AD<double>> dirichlet_multinomial_extra(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config);

CppAD::vector<CppAD::AD<double>> exp_prior(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config);

#endif
