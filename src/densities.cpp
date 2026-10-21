#include "adlaplace/density_data.hpp"
#include "adlaplace/atomics.hpp"
#include "adlaplace/densities.hpp"
#include "adlaplace/eta.hpp"
#include "adlaplace/rviews.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace {

std::pair<std::size_t, std::size_t> obs_range(
  const Config& config,
  std::size_t ny,
  std::size_t Dgroup) {

  const bool have_shards = config.obs_groups.ncol() > 0;
  std::size_t startP = 0;
  std::size_t endP = 0;
  if (have_shards) {
    startP = static_cast<std::size_t>(config.obs_groups.p[Dgroup]);
    endP = static_cast<std::size_t>(config.obs_groups.p[Dgroup + 1]);
  } else if (Dgroup == 0) {
    endP = ny;
  }
  return {startP, endP};
}

CppAD::AD<double> softplus_ad(const CppAD::AD<double>& z) {
  return CppAD::CondExpGt(
    z, CppAD::AD<double>(0.0),
    z + CppAD::log1p(CppAD::exp(-z)),
    CppAD::log1p(CppAD::exp(z)));
}

CppAD::AD<double> log_theta_ad(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config) {

  const std::size_t t_global = model.theta_index(0);
  const std::size_t t_row = model.theta_row(0);
  const CppAD::AD<double> thetaIn = x[t_global];
  return transform_theta_at(config, t_row) ? thetaIn : CppAD::log(thetaIn);
}

}  // namespace

CppAD::vector<CppAD::AD<double>> binomial_obs(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config,
  const size_t Dgroup) {

  const size_t ny = model.y.size();
  const auto range = obs_range(config, ny, Dgroup);
  const bool have_shards = config.obs_groups.ncol() > 0;
  const bool have_weights = model.weights.size() == ny;

  CppAD::AD<double> result = 0.0;
  for (size_t DI = range.first; DI < range.second; ++DI) {
    const size_t Dobs = have_shards ? static_cast<size_t>(config.obs_groups.i[DI]) : DI;
    const CppAD::AD<double> eta = eta_at(x, model, Dobs);
    const CppAD::AD<double> ntrials = have_weights
      ? CppAD::AD<double>(model.weights[Dobs])
      : CppAD::AD<double>(1.0);
    result += CppAD::AD<double>(model.y[Dobs]) * eta - ntrials * softplus_ad(eta);
  }

  CppAD::vector<CppAD::AD<double>> out(1);
  out[0] = result;
  return out;
}

CppAD::vector<CppAD::AD<double>> gaussian_obs(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config,
  const size_t Dgroup) {

  const CppAD::AD<double> logSigma = log_theta_ad(x, model, config);
  const CppAD::AD<double> inv2var = CppAD::AD<double>(0.5) * CppAD::exp(-2 * logSigma);

  const size_t ny = model.y.size();
  const auto range = obs_range(config, ny, Dgroup);
  const bool have_shards = config.obs_groups.ncol() > 0;

  CppAD::AD<double> ss = 0.0;
  for (size_t DI = range.first; DI < range.second; ++DI) {
    const size_t Dobs = have_shards ? static_cast<size_t>(config.obs_groups.i[DI]) : DI;
    const CppAD::AD<double> eta = eta_at(x, model, Dobs);
    const CppAD::AD<double> resid = CppAD::AD<double>(model.y[Dobs]) - eta;
    ss += resid * resid;
  }

  CppAD::vector<CppAD::AD<double>> result(1);
  result[0] = -inv2var * ss;
  return result;
}

CppAD::vector<CppAD::AD<double>> gaussian_extra(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config) {

  const CppAD::AD<double> logSigma = log_theta_ad(x, model, config);
  const double ny = static_cast<double>(model.y.size());

  CppAD::vector<CppAD::AD<double>> result(1);
  result[0] = -ny * logSigma - CppAD::AD<double>(ny * ONEHALFLOGTWOPI);
  return result;
}

CppAD::vector<CppAD::AD<double>> nbinom_obs(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config,
  const size_t Dgroup) {

  CppAD::AD<double> logDens1 = 0.0;
  CppAD::AD<double> logDens2 = 0.0;

  const CppAD::AD<double> logTheta = log_theta_ad(x, model, config);
  const CppAD::AD<double> logNbSize = -2 * logTheta;
  const CppAD::AD<double> nbSize = CppAD::exp(logNbSize);

  const size_t ny = model.y.size();
  const auto range = obs_range(config, ny, Dgroup);
  const bool have_shards = config.obs_groups.ncol() > 0;

  for (size_t DI = range.first; DI < range.second; ++DI) {
    const size_t Dobs = have_shards ? static_cast<size_t>(config.obs_groups.i[DI]) : DI;
    const CppAD::AD<double> eta = eta_at(x, model, Dobs);
    const CppAD::AD<double> diff = eta - logNbSize;
    const CppAD::AD<double> logRplusMu = logNbSize + softplus_ad(diff);

    logDens1 += model.y[Dobs] * eta;
    logDens2 += -logRplusMu * (model.y[Dobs] + nbSize);
  }

  CppAD::vector<CppAD::AD<double>> result(1);
  result[0] = logDens1 + logDens2;
  return result;
}

CppAD::vector<CppAD::AD<double>> nbinom_extra(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config) {

  const CppAD::AD<double> logTheta = log_theta_ad(x, model, config);
  const CppAD::AD<double> logNbSize = -2 * logTheta;
  const CppAD::AD<double> nbSize = CppAD::exp(logNbSize);
  const CppAD::AD<double> lgammaNbSize = lgamma_ad(nbSize);
  const CppAD::AD<double> sizeLogSize = nbSize * logNbSize;

  const std::size_t ny = model.y.size();
  CppAD::AD<double> logDens1 = 0.0;
  for (std::size_t D = 0; D < ny; ++D) {
    logDens1 += lgamma_ad(model.y[D] + nbSize);
  }

  CppAD::AD<double> logDens2 = static_cast<double>(ny) * (sizeLogSize - lgammaNbSize);

  double constants = 0.0;
  for (std::size_t D = 0; D < ny; ++D) {
    constants += std::lgamma(model.y[D] + 1.0);
  }
  logDens2 -= constants;

  CppAD::vector<CppAD::AD<double>> result(1);
  result[0] = logDens1 + logDens2;
  return result;
}

CppAD::vector<CppAD::AD<double>> poisson_obs(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config,
  const size_t Dgroup) {

  const bool have_offset = !config.offset.empty();
  if (have_offset &&
      config.offset.size() != 1 &&
      config.offset.size() != model.y.size()) {
    Rcpp::stop(
      "config$offset must have length 1 or length(y) (%d), got %d",
      static_cast<int>(model.y.size()), static_cast<int>(config.offset.size()));
  }

  const size_t ny = model.y.size();
  const auto range = obs_range(config, ny, Dgroup);
  const bool have_shards = config.obs_groups.ncol() > 0;

  CppAD::AD<double> linear = 0.0;
  CppAD::AD<double> expsum = 0.0;
  double constants = 0.0;

  for (size_t DI = range.first; DI < range.second; ++DI) {
    const size_t Dobs = have_shards ? static_cast<size_t>(config.obs_groups.i[DI]) : DI;
    const CppAD::AD<double> eta = eta_at(x, model, Dobs);

    const double off = have_offset
      ? (config.offset.size() == 1 ? config.offset[0] : config.offset[Dobs])
      : 0.0;
    const double y = model.y[Dobs];

    linear += y * eta;
    expsum += CppAD::exp(eta + off);
    constants += y * off - std::lgamma(y + 1.0);
  }

  CppAD::vector<CppAD::AD<double>> result(1);
  result[0] = linear - expsum + CppAD::AD<double>(constants);
  return result;
}

namespace {

CppAD::AD<double> stable_logsumexp(
  const CppAD::vector<CppAD::AD<double>>& eta) {

  CppAD::AD<double> max_eta = eta[0];
  for (size_t Deta = 1; Deta < eta.size(); ++Deta) {
    max_eta = CppAD::CondExpGt(eta[Deta], max_eta, eta[Deta], max_eta);
  }

  CppAD::AD<double> sumexp = 0.0;
  for (size_t Deta = 0; Deta < eta.size(); ++Deta) {
    sumexp += CppAD::exp(eta[Deta] - max_eta);
  }

  return CppAD::log(sumexp) + max_eta;
}

CppAD::vector<CppAD::AD<double>> compute_eta_for_stratum(
  const size_t Dstrata,
  const density_data& model,
  const CppAD::vector<CppAD::AD<double>>& x) {

  const size_t n_strata = static_cast<size_t>(model.elgm_matrix.ncol());
  if (Dstrata >= n_strata) {
    Rcpp::stop(
      "dirichlet_multinomial stratum index %d out of range [0, %d)",
      static_cast<int>(Dstrata),
      static_cast<int>(n_strata)
    );
  }
  if (Dstrata + 1 >= static_cast<size_t>(model.elgm_matrix.p.size())) {
    Rcpp::stop(
      "elgm_matrix column pointer missing for stratum %d",
      static_cast<int>(Dstrata)
    );
  }

  const size_t startHere = model.elgm_matrix.p[Dstrata];
  const size_t endHere = model.elgm_matrix.p[Dstrata + 1];
  const size_t NinStrata = endHere - startHere;

  CppAD::vector<CppAD::AD<double>> etaHere(NinStrata);
  for (size_t j = 0, k = startHere; j < NinStrata; ++j, ++k) {
    const size_t Deta = static_cast<size_t>(model.elgm_matrix.i[k]);
    etaHere[j] = eta_at(x, model, Deta);
  }
  return etaHere;
}

CppAD::AD<double> accumulate_contrib_for_stratum(
  const size_t Dstrata,
  const density_data& model,
  const CppAD::vector<CppAD::AD<double>>& etaHere,
  const CppAD::AD<double>& tauSq) {

  const size_t startHere = model.elgm_matrix.p[Dstrata];
  const size_t endHere = model.elgm_matrix.p[Dstrata + 1];
  const size_t NinStrata = endHere - startHere;

  const CppAD::AD<double> etaLogSum = stable_logsumexp(etaHere);
  CppAD::AD<double> contrib = 0.0;

  for (size_t j = 0, k = startHere; j < NinStrata; ++j, ++k) {
    const size_t idx = static_cast<size_t>(model.elgm_matrix.i[k]);
    const double yhere = model.y[idx];

    if (yhere > 0) {
      const CppAD::AD<double> logMuKhere = etaHere[j] - etaLogSum;
      contrib += logMuKhere;

      if (yhere > 1) {
        const CppAD::AD<double> expMuKhere = CppAD::exp(logMuKhere);
        for (int i = 1; i < yhere; ++i) {
          const CppAD::AD<double> to_log = expMuKhere + static_cast<double>(i) * tauSq;
          contrib += CppAD::log(to_log);
        }
      }
    }
  }

  return contrib;
}

CppAD::AD<double> tau_sq_from_x(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config) {

  const std::size_t t_global = model.theta_index(0);
  const std::size_t t_row = model.theta_row(0);
  const CppAD::AD<double> tau_in = x[t_global];
  const CppAD::AD<double> tau =
    transform_theta_at(config, t_row) ? CppAD::exp(tau_in) : tau_in;
  return tau * tau;
}

}  // namespace

CppAD::vector<CppAD::AD<double>> dirichlet_multinomial(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config,
  const size_t Dgroup) {

  CppAD::AD<double> result1 = 0.0;
  CppAD::vector<CppAD::AD<double>> result(1);

  const bool have_shards = config.obs_groups.ncol() > 0;
  const size_t startP = have_shards ? config.obs_groups.p[Dgroup] : Dgroup;
  const size_t endP = have_shards ? config.obs_groups.p[Dgroup + 1] : Dgroup + 1;

  const CppAD::AD<double> tauSq = tau_sq_from_x(x, model, config);
  const bool verbose_here = config.verbose && (Dgroup < 1);

  if (verbose_here) {
    Rcpp::Rcout << "group " << Dgroup << " startP " << startP << " endP " << endP << "\n";
  }

  for (size_t DstrataI = startP; DstrataI < endP; ++DstrataI) {
    const size_t Dstrata = have_shards ? config.obs_groups.i[DstrataI] : DstrataI;
    const auto etaHere = compute_eta_for_stratum(Dstrata, model, x);
    const auto contrib = accumulate_contrib_for_stratum(Dstrata, model, etaHere, tauSq);

    if (verbose_here) {
      Rcpp::Rcout << "strata " << Dstrata << " contrib " << contrib << "\n";
    }

    result1 += contrib;
  }

  if (verbose_here) {
    Rcpp::Rcout << " result " << result1 << "\n";
  }

  result[0] = result1;
  return result;
}

CppAD::vector<CppAD::AD<double>> dirichlet_multinomial_extra(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config) {

  const CppAD::AD<double> tauSq = tau_sq_from_x(x, model, config);
  const size_t Nstrata = static_cast<size_t>(model.elgm_matrix.ncol());

  CppAD::AD<double> contribLog1pjTauSq = 0.0;
  double contribLogYkFact = 0.0;
  double contribLogNfact = 0.0;

  for (size_t Dstrata = 0; Dstrata < Nstrata; ++Dstrata) {
    int sumYhere = 0;
    const size_t endHere = model.elgm_matrix.p[Dstrata + 1];

    for (size_t DobsI = model.elgm_matrix.p[Dstrata]; DobsI < endHere; ++DobsI) {
      const size_t Dobs = static_cast<size_t>(model.elgm_matrix.i[DobsI]);
      sumYhere += static_cast<int>(model.y[Dobs]);
      contribLogYkFact += std::lgamma(1.0 + model.y[Dobs]);
    }

    contribLogNfact += std::lgamma(1.0 + sumYhere);
    const size_t sumYhereS = static_cast<size_t>(sumYhere);
    for (size_t Dsum = 1; Dsum < sumYhereS; ++Dsum) {
      contribLog1pjTauSq += CppAD::log1p(static_cast<double>(Dsum) * tauSq);
    }
  }

  const CppAD::AD<double> contrib = contribLogNfact - contribLogYkFact - contribLog1pjTauSq;

  if (config.verbose) {
    Rcpp::Rcout << "logDensExtra " << contrib << " tauSq " << tauSq
      << " contribLogNfact " << contribLogNfact
      << " contribLogYkFact " << contribLogYkFact
      << " contribLog1pjTauSq " << contribLog1pjTauSq << "\n";
  }

  CppAD::vector<CppAD::AD<double>> out(1);
  out[0] = contrib;
  return out;
}

CppAD::vector<CppAD::AD<double>> exp_prior(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config) {

  if (model.y.size() < 1) {
    Rcpp::stop("exp_prior requires y[0] = rate");
  }
  const double rate = model.y[0];
  if (!(rate > 0.0)) {
    Rcpp::stop("exp_prior rate must be positive");
  }

  const CppAD::AD<double> logSd = log_theta_ad(x, model, config);
  const CppAD::AD<double> sd = CppAD::exp(logSd);
  // Exp(rate) on SD plus Jacobian from u = log(sd): log λ - λ e^u + u
  CppAD::vector<CppAD::AD<double>> out(1);
  out[0] = CppAD::AD<double>(std::log(rate)) - CppAD::AD<double>(rate) * sd + logSd;

  if (config.verbose) {
    Rcpp::Rcout << "exp_prior rate " << rate << " logSd " << logSd
                << " contrib " << out[0] << "\n";
  }
  return out;
}
