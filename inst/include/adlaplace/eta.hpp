#ifndef ADLAPLACE_ETA_HPP
#define ADLAPLACE_ETA_HPP

#include "adlaplace/density_data.hpp"

#include <cppad/cppad.hpp>
#include <cstddef>

// Linear predictor at observation Dobs using beta_map / gamma_map lookups.
template <class ADScalar>
inline ADScalar eta_at(
  const CppAD::vector<ADScalar>& x,
  const density_data& model,
  std::size_t Dobs) {

  ADScalar eta = ADScalar(0.0);
  const std::size_t p0x = static_cast<std::size_t>(model.XTp.p[Dobs]);
  const std::size_t p1x = static_cast<std::size_t>(model.XTp.p[Dobs + 1]);
  for (std::size_t D = p0x; D < p1x; ++D) {
    const std::size_t local_col = static_cast<std::size_t>(model.XTp.i[D]);
    eta += model.XTp.x[D] * x[model.beta_global[local_col]];
  }
  const std::size_t p0a = static_cast<std::size_t>(model.ATp.p[Dobs]);
  const std::size_t p1a = static_cast<std::size_t>(model.ATp.p[Dobs + 1]);
  for (std::size_t D = p0a; D < p1a; ++D) {
    const std::size_t local_col = static_cast<std::size_t>(model.ATp.i[D]);
    eta += model.ATp.x[D] * x[model.gamma_global[local_col]];
  }
  return eta;
}

#endif
