#ifndef ADLAPLACE_ADFUN_HPP
#define ADLAPLACE_ADFUN_HPP

#include <Rcpp.h>
#include <cppad/cppad.hpp>

#include <algorithm>
#include <string>
#include <vector>

#include "adlaplace/density_data.hpp"
#include "adlaplace/backend.hpp"
#include "adlaplace/rviews.hpp"

// Seed vector over the current tape domain (model.n_tape / tape_to_global).
inline CPPAD_TESTVECTOR(double)
    make_ad_params_seed(const Config &cfg, const density_data &model) {

  CPPAD_TESTVECTOR(double) ad_params_G(model.n_tape);
  for (std::size_t k = 0; k < model.n_tape; ++k) {
    const std::size_t g = model.tape_to_global[k];
    if (g < model.num_beta) {
      ad_params_G[k] = cfg.beta[g];
    } else if (g < model.num_beta + model.num_gamma) {
      ad_params_G[k] = cfg.gamma[g - model.num_beta];
    } else {
      ad_params_G[k] = cfg.theta[g - model.num_beta - model.num_gamma];
    }
  }
  return ad_params_G;
}

static const std::string JAC_COLOR = "cppad";
static const std::string HESS_COLOR = "cppad.symmetric";

template <class SizeVec, class ValVec>
inline ValVec &rcv_val(CppAD::sparse_rcv<SizeVec, ValVec> &rcv) {
  return const_cast<ValVec &>(rcv.val());
}

inline const CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> &empty_sparse_rc() {
  static CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> empty;
  return empty;
}

inline void
adpack_discover_grad(AdTape &gp,
                     CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> &grad,
                     const std::size_t n_params) {

  CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> pattern_in;
  pattern_in.resize(n_params, n_params, n_params);
  for (std::size_t j = 0; j < n_params; ++j) {
    pattern_in.set(j, j, j);
  }

  const bool transpose = false;
  const bool dependency = false;
  const bool internal_bool = false;

  gp.fun.for_jac_sparsity(pattern_in, transpose, dependency, internal_bool,
                          grad);
}

inline void
adpack_discover_hessian(AdTape &gp,
                        CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> &hessian,
                        const std::size_t n_params) {

  CppAD::vectorBool select_domain(n_params), select_range(1);
  for (std::size_t j = 0; j < n_params; ++j) {
    select_domain[j] = true;
  }
  select_range[0] = true;

  const bool internal_bool = false;
  gp.fun.for_hes_sparsity(select_domain, select_range, internal_bool, hessian);
}

inline void adpack_attach_tape_maps(AdTape &gp, const density_data &model) {
  gp.n_global = model.num_full;
  gp.tape_to_global = model.tape_to_global;
  gp.gamma_row_to_tape.assign(model.num_gamma, model.n_tape);
  for (std::size_t k = 0; k < model.n_tape; ++k) {
    const std::size_t g = model.tape_to_global[k];
    if (g >= model.num_beta && g < model.num_beta + model.num_gamma) {
      gp.gamma_row_to_tape[g - model.num_beta] = k;
    }
  }
}

inline void adpack_fill_global_patterns(AdTape &gp) {
  const auto &map = gp.tape_to_global;
  const std::size_t n_tape = map.size();

  auto to_global = [&](std::size_t local) -> std::size_t {
    if (local >= n_tape) {
      Rcpp::stop("adpack_fill_global_patterns: local index out of range");
    }
    return map[local];
  };

  {
    const std::size_t nnz = gp.pattern_grad.nnz();
    const auto &cols = gp.pattern_grad.col();
    gp.grad_cols_global.resize(nnz);
    for (std::size_t d = 0; d < nnz; ++d) {
      gp.grad_cols_global[d] = to_global(cols[d]);
    }
  }
  {
    const std::size_t nnz = gp.pattern_grad_inner.nnz();
    const auto &cols = gp.pattern_grad_inner.col();
    gp.grad_inner_cols_global.resize(nnz);
    for (std::size_t d = 0; d < nnz; ++d) {
      gp.grad_inner_cols_global[d] = to_global(cols[d]);
    }
  }
  {
    const std::size_t nnz = gp.pattern_hessian.nnz();
    const auto &rows = gp.pattern_hessian.row();
    const auto &cols = gp.pattern_hessian.col();
    gp.hes_rows_global.resize(nnz);
    gp.hes_cols_global.resize(nnz);
    for (std::size_t d = 0; d < nnz; ++d) {
      gp.hes_rows_global[d] = to_global(rows[d]);
      gp.hes_cols_global[d] = to_global(cols[d]);
    }
  }
  {
    const std::size_t nnz = gp.pattern_hessian_inner.nnz();
    const auto &rows = gp.pattern_hessian_inner.row();
    const auto &cols = gp.pattern_hessian_inner.col();
    gp.hes_inner_rows_global.resize(nnz);
    gp.hes_inner_cols_global.resize(nnz);
    for (std::size_t d = 0; d < nnz; ++d) {
      gp.hes_inner_rows_global[d] = to_global(rows[d]);
      gp.hes_inner_cols_global[d] = to_global(cols[d]);
    }
  }
}

// Carrier for the patterns the calibration sweeps need. Hessian members keep
// both triangles; the upper-triangle views are what AdTape stores.
struct AdpackPatterns {
  CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> grad;
  CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> grad_inner;
  CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> hessian;
  CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> hessian_inner;
  CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> hessian_upper;
  CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> hessian_inner_upper;
};

// Pure index work: inner/upper subsets of the given patterns. Touches no tape.
inline AdpackPatterns
adpack_build_patterns(AdTape &gp, const std::size_t n_params,
                      const std::vector<int> &subset,
                      const CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> &grad,
                      const CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)>
                          &hessian_full,
                      const bool verbose = false) {

  gp.w.resize(1);
  gp.w[0] = 1.0;
  gp.x.resize(n_params);

  AdpackPatterns pats;
  pats.grad = grad;
  pats.hessian = hessian_full;

  std::vector<unsigned char> insubset(n_params, 0);
  for (std::size_t v : subset) {
    if (v < n_params) {
      insubset[v] = 1;
    }
  }

  const auto &full_cols = pats.grad.col();
  {
    std::size_t k = 0;
    for (std::size_t j : full_cols) {
      k += insubset[j];
    }
    pats.grad_inner.resize(1, n_params, k);

    std::size_t t = 0;
    for (std::size_t j : full_cols) {
      if (insubset[j])
        pats.grad_inner.set(t++, 0, j);
    }
  }

  {
    const auto &r = pats.hessian.row();
    const auto &c = pats.hessian.col();
    const std::size_t k = pats.hessian.nnz();

    std::size_t k_upper = 0, k_inner = 0, k_inner_upper = 0;

    for (std::size_t idx = 0; idx < k; ++idx) {
      const std::size_t i = r[idx], j = c[idx];
      const bool is_upper = (i <= j);
      const bool is_inner = insubset[i] && insubset[j];

      if (is_upper)
        ++k_upper;
      if (is_inner)
        ++k_inner;
      if (is_upper && is_inner)
        ++k_inner_upper;
    }

    pats.hessian_upper.resize(n_params, n_params, k_upper);
    pats.hessian_inner.resize(n_params, n_params, k_inner);
    pats.hessian_inner_upper.resize(n_params, n_params, k_inner_upper);

    std::size_t tu = 0, ti = 0, tiu = 0;
    for (std::size_t idx = 0; idx < k; ++idx) {
      const std::size_t i = r[idx], j = c[idx];
      const bool is_upper = (i <= j);
      const bool is_inner = insubset[i] && insubset[j];

      if (is_upper)
        pats.hessian_upper.set(tu++, i, j);
      if (is_inner)
        pats.hessian_inner.set(ti++, i, j);
      if (is_upper && is_inner)
        pats.hessian_inner_upper.set(tiu++, i, j);
    }
  }

  if (verbose) {
    Rcpp::Rcout << "  sparsity: grad " << pats.grad.nnz() << ", grad_inner "
                << pats.grad_inner.nnz() << ", hes full" << pats.hessian.nnz()
                << ", hes upper" << pats.hessian_upper.nnz() << ", hes_inner "
                << pats.hessian_inner_upper.nnz() << "\n";
  }

  return pats;
}

// Assign the four rcv patterns onto the pack and map them to global indices.
// Safe with or without a recorded tape.
inline void adpack_install_patterns(AdTape &gp, const AdpackPatterns &pats) {
  gp.pattern_grad =
      CppAD::sparse_rcv<CPPAD_TESTVECTOR(size_t), CPPAD_TESTVECTOR(double)>(
          pats.grad);
  gp.pattern_grad_inner =
      CppAD::sparse_rcv<CPPAD_TESTVECTOR(size_t), CPPAD_TESTVECTOR(double)>(
          pats.grad_inner);
  gp.pattern_hessian =
      CppAD::sparse_rcv<CPPAD_TESTVECTOR(size_t), CPPAD_TESTVECTOR(double)>(
          pats.hessian_upper);
  gp.pattern_hessian_inner =
      CppAD::sparse_rcv<CPPAD_TESTVECTOR(size_t), CPPAD_TESTVECTOR(double)>(
          pats.hessian_inner_upper);
  adpack_fill_global_patterns(gp);
}

// One-time CppAD coloring / reverse sweeps that fill the work_* objects
// EvalShard later reuses. Requires a recorded tape. Interleaves rcv
// construction with each sweep, matching the original adpack_sparsity order.
inline void adpack_calibrate(const CPPAD_TESTVECTOR(double) & x, AdTape &gp,
                             AdpackPatterns &pats) {
  gp.fun.Forward(0, x);

  gp.pattern_grad =
      CppAD::sparse_rcv<CPPAD_TESTVECTOR(size_t), CPPAD_TESTVECTOR(double)>(
          pats.grad);
  gp.fun.sparse_jac_rev(x, gp.pattern_grad, pats.grad, JAC_COLOR,
                        gp.work_grad);

  gp.pattern_grad_inner =
      CppAD::sparse_rcv<CPPAD_TESTVECTOR(size_t), CPPAD_TESTVECTOR(double)>(
          pats.grad_inner);
  gp.fun.sparse_jac_rev(x, gp.pattern_grad_inner, pats.grad_inner, JAC_COLOR,
                        gp.work_inner_grad);

  gp.pattern_hessian =
      CppAD::sparse_rcv<CPPAD_TESTVECTOR(size_t), CPPAD_TESTVECTOR(double)>(
          pats.hessian_upper);
  gp.pattern_hessian_inner =
      CppAD::sparse_rcv<CPPAD_TESTVECTOR(size_t), CPPAD_TESTVECTOR(double)>(
          pats.hessian_inner_upper);

  // Grad-only packs leave Hessian patterns empty; skip sparse_hes calibration.
  if (pats.hessian.nnz() > 0 || pats.hessian_upper.nnz() > 0) {
    gp.fun.sparse_hes(x, gp.w, gp.pattern_hessian, pats.hessian, HESS_COLOR,
                      gp.work_hess);
    gp.fun.sparse_hes(x, gp.w, gp.pattern_hessian_inner, pats.hessian_inner,
                      HESS_COLOR, gp.work_inner_hess);
  }

  adpack_fill_global_patterns(gp);
  // Drop the calibration Taylor buffer in the same DSO that recorded the
  // tape. Later drain from adlaplace.so then sees cap_order 0 and is a no-op.
  gp.fun.capacity_order(0);
}

inline void adpack_sparsity(const CPPAD_TESTVECTOR(double) & x,
                            const std::vector<int> &subset, AdTape &gp,
                            const bool verbose = false,
                            const CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)>
                                &hessian = empty_sparse_rc(),
                            const bool hessian_sparsity = true) {

  const std::size_t n_params = x.size();

  CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> grad;
  adpack_discover_grad(gp, grad, n_params);

  CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> hessian_here;
  if (hessian.nnz() > 0) {
    if (verbose) {
      Rcpp::Rcout << "using external Hessian sparsity pattern\n";
    }
    hessian_here = hessian;
  } else if (hessian_sparsity) {
    if (verbose) {
      Rcpp::Rcout << "discovering Hessian sparsity pattern\n";
    }
    adpack_discover_hessian(gp, hessian_here, n_params);
  } else {
    if (verbose) {
      Rcpp::Rcout << "skipping Hessian sparsity (grad-only)\n";
    }
    hessian_here = empty_sparse_rc();
  }

  AdpackPatterns pats =
      adpack_build_patterns(gp, n_params, subset, grad, hessian_here, verbose);
  adpack_calibrate(x, gp, pats);
}

inline size_t count_obs_shards(const density_data &model, const Rcpp::List &config) {
  const Config cfg(config);
  size_t ng = cfg.obs_groups.ncol();
  if (ng == 0) {
    if (model.y.size() > 0) {
      ng = 1;
    } else if (model.elgm_matrix.p.size() > 0) {
      ng = static_cast<size_t>(model.elgm_matrix.ncol());
    }
  }
  return ng > 0 ? ng : 1;
}

// Defined in obs_sparsity.hpp (included below). Forward-declared so
// build_ad_fun_obs compiles if that header's include is skipped by
// include-order (obs_sparsity.hpp -> ad_pack_random.hpp -> ad_pack.hpp).
inline CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)>
obs_hessian_sparsity(const density_data &model, const Config &cfg,
                     std::size_t Dgroup);

#include "adlaplace/obs_sparsity.hpp"

inline std::vector<AdTape> build_ad_fun_obs(const density_data &model_in,
                                               const Rcpp::List &config,
                                               LogDensObsFn log_dens) {

  const Config cfg(config);
  validate_config_matches_model(cfg, model_in);
  const size_t ng = count_obs_shards(model_in, config);

  if (cfg.verbose) {
    Rcpp::Rcout << "build_ad_fun_obs groups " << ng << "\n";
  }

  std::vector<AdTape> result(ng);
  density_data model = model_in;

  for (size_t d = 0; d < ng; ++d) {
    if (cfg.verbose) {
      Rcpp::Rcout << "  taping observation group " << (d + 1) << " / " << ng
                  << "\n";
    }
    model.apply_tape_domain(cfg, "obs", d);
    const CPPAD_TESTVECTOR(double) ad_params_G = make_ad_params_seed(cfg, model);
    CppAD::vector<CppAD::AD<double>> ad_params(model.n_tape);
    for (size_t j = 0; j < model.n_tape; ++j) {
      ad_params[j] = ad_params_G[j];
    }
    CppAD::Independent(ad_params);
    auto result_here = log_dens(ad_params, model, cfg, d);
    CppAD::ADFun<double> fun(ad_params, result_here);
    result[d].fun = std::move(fun);
    adpack_attach_tape_maps(result[d], model);
  }

  for (size_t d = 0; d < ng; ++d) {
    if (cfg.verbose) {
      Rcpp::Rcout << "  sparsity observation group " << (d + 1) << " / " << ng
                  << "\n";
    }
    model.apply_tape_domain(cfg, "obs", d);
    const CPPAD_TESTVECTOR(double) ad_params_G = make_ad_params_seed(cfg, model);
    CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> hess = empty_sparse_rc();
    if (cfg.hessian_sparsity && use_analytic_obs_hessian(cfg)) {
      hess = obs_hessian_sparsity(model, cfg, d);
    }
    adpack_sparsity(ad_params_G, model.seq_gamma, result[d], cfg.verbose,
                    hess, cfg.hessian_sparsity);
  }

  return result;
}

inline AdTape build_ad_fun_parameters(
    const density_data &model_in, const Rcpp::List &config, LogDensSingleDataFn log_dens,
    const CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> &hessian = empty_sparse_rc()) {

  const Config cfg(config);
  validate_config_matches_model(cfg, model_in);

  if (cfg.verbose) {
    Rcpp::Rcout << "build_ad_fun_parameters: taping...\n";
  }

  density_data model = model_in;
  model.apply_tape_domain(cfg, "all", 0);

  const CPPAD_TESTVECTOR(double) ad_params_G = make_ad_params_seed(cfg, model);

  CppAD::vector<CppAD::AD<double>> ad_params(model.n_tape);
  for (size_t d = 0; d < model.n_tape; ++d) {
    ad_params[d] = ad_params_G[d];
  }

  CppAD::Independent(ad_params);

  CppAD::vector<CppAD::AD<double>> result_here =
      log_dens(ad_params, model, cfg);

  CppAD::ADFun<double> fun(ad_params, result_here);

  AdTape pack;
  pack.fun = std::move(fun);
  pack.owner_thread_assigned = false;
  adpack_attach_tape_maps(pack, model);
  if (cfg.verbose) {
    Rcpp::Rcout << "build_ad_fun_parameters: computing sparsity...\n";
  }
  adpack_sparsity(ad_params_G, model.seq_gamma, pack, cfg.verbose, hessian,
                  cfg.hessian_sparsity);
  return pack;
}

#endif
