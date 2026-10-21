#ifndef ADLAPLACE_OBS_SPARSITY_HPP
#define ADLAPLACE_OBS_SPARSITY_HPP

// Analytic observation Hessian sparsity from design (and ELGM) co-occurrence.
// Call after apply_tape_domain(cfg, "obs", Dgroup). Indices are tape-local.

#include "adlaplace/ad_pack_random.hpp"
#include "adlaplace/density_data.hpp"
#include "adlaplace/rviews.hpp"

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

// Tape-local design column indices for observation Dobs (skip n_tape sentinels).
// Appends to cols; caller clears when starting a new clique.
inline void collect_obs_design_tape_indices(
    const density_data& model, std::size_t Dobs,
    std::vector<std::size_t>& cols) {

  if (static_cast<std::size_t>(model.XTp.p.size()) > Dobs + 1) {
    const std::size_t p0 = static_cast<std::size_t>(model.XTp.p[Dobs]);
    const std::size_t p1 = static_cast<std::size_t>(model.XTp.p[Dobs + 1]);
    for (std::size_t k = p0; k < p1; ++k) {
      const std::size_t local_col = static_cast<std::size_t>(model.XTp.i[k]);
      if (local_col < model.beta_global.size()) {
        const std::size_t t = model.beta_global[local_col];
        if (t < model.n_tape) {
          cols.push_back(t);
        }
      }
    }
  }
  if (static_cast<std::size_t>(model.ATp.p.size()) > Dobs + 1) {
    const std::size_t p0 = static_cast<std::size_t>(model.ATp.p[Dobs]);
    const std::size_t p1 = static_cast<std::size_t>(model.ATp.p[Dobs + 1]);
    for (std::size_t k = p0; k < p1; ++k) {
      const std::size_t local_col = static_cast<std::size_t>(model.ATp.i[k]);
      if (local_col < model.gamma_global.size()) {
        const std::size_t t = model.gamma_global[local_col];
        if (t < model.n_tape) {
          cols.push_back(t);
        }
      }
    }
  }
}

// Clique on cols: insert all pairs including diagonal, both triangles.
inline void clique_pairs(
    const std::vector<std::size_t>& cols_in,
    std::set<std::pair<std::size_t, std::size_t>>& pairs) {

  std::vector<std::size_t> cols = cols_in;
  std::sort(cols.begin(), cols.end());
  cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
  for (std::size_t a = 0; a < cols.size(); ++a) {
    for (std::size_t b = 0; b < cols.size(); ++b) {
      pairs.insert({cols[a], cols[b]});
    }
  }
}

namespace adlaplace_detail {

inline std::pair<std::size_t, std::size_t> obs_glm_range(
    const Config& cfg, std::size_t ny, std::size_t Dgroup) {

  const bool have_shards = cfg.obs_groups.ncol() > 0;
  std::size_t startP = 0;
  std::size_t endP = 0;
  if (have_shards) {
    if (Dgroup + 1 >= cfg.obs_groups.p.size()) {
      Rcpp::stop("obs_groups column %d out of range", static_cast<int>(Dgroup));
    }
    startP = static_cast<std::size_t>(cfg.obs_groups.p[Dgroup]);
    endP = static_cast<std::size_t>(cfg.obs_groups.p[Dgroup + 1]);
  } else if (Dgroup == 0) {
    endP = ny;
  }
  return {startP, endP};
}

inline void add_glm_row_cliques(
    const density_data& model, const Config& cfg, std::size_t Dgroup,
    bool with_theta,
    std::set<std::pair<std::size_t, std::size_t>>& pairs) {

  const bool have_shards = cfg.obs_groups.ncol() > 0;
  const auto range = obs_glm_range(cfg, model.y.size(), Dgroup);
  const std::size_t theta = model.theta_index(0);
  const bool have_theta = with_theta && theta < model.n_tape;

  std::vector<std::size_t> cols;
  for (std::size_t DI = range.first; DI < range.second; ++DI) {
    const std::size_t Dobs = have_shards
      ? static_cast<std::size_t>(cfg.obs_groups.i[DI])
      : DI;
    cols.clear();
    collect_obs_design_tape_indices(model, Dobs, cols);
    if (have_theta) {
      cols.push_back(theta);
    }
    clique_pairs(cols, pairs);
  }
}

inline void add_dirichlet_stratum_cliques(
    const density_data& model, const Config& cfg, std::size_t Dgroup,
    std::set<std::pair<std::size_t, std::size_t>>& pairs) {

  if (model.elgm_matrix.ncol() <= 0) {
    return;
  }

  const bool have_shards = cfg.obs_groups.ncol() > 0;
  std::size_t startP = 0;
  std::size_t endP = 0;
  if (have_shards) {
    if (Dgroup + 1 >= cfg.obs_groups.p.size()) {
      Rcpp::stop("obs_groups column %d out of range", static_cast<int>(Dgroup));
    }
    startP = static_cast<std::size_t>(cfg.obs_groups.p[Dgroup]);
    endP = static_cast<std::size_t>(cfg.obs_groups.p[Dgroup + 1]);
  } else {
    startP = Dgroup;
    endP = Dgroup + 1;
  }

  const std::size_t n_strata =
    static_cast<std::size_t>(model.elgm_matrix.ncol());
  const std::size_t theta = model.theta_index(0);
  const bool have_theta = theta < model.n_tape;

  std::vector<std::size_t> cols;
  for (std::size_t DI = startP; DI < endP; ++DI) {
    const std::size_t Dstrata = have_shards
      ? static_cast<std::size_t>(cfg.obs_groups.i[DI])
      : DI;
    if (Dstrata >= n_strata) {
      Rcpp::stop(
        "obs_groups stratum index %d out of range [0, %d)",
        static_cast<int>(Dstrata),
        static_cast<int>(n_strata));
    }
    if (Dstrata + 1 >= static_cast<std::size_t>(model.elgm_matrix.p.size())) {
      Rcpp::stop(
        "elgm_matrix column pointer missing for stratum %d",
        static_cast<int>(Dstrata));
    }
    cols.clear();
    const std::size_t p0 =
      static_cast<std::size_t>(model.elgm_matrix.p[Dstrata]);
    const std::size_t p1 =
      static_cast<std::size_t>(model.elgm_matrix.p[Dstrata + 1]);
    bool any_y_gt_1 = false;
    for (std::size_t k = p0; k < p1; ++k) {
      const std::size_t Dobs =
        static_cast<std::size_t>(model.elgm_matrix.i[k]);
      collect_obs_design_tape_indices(model, Dobs, cols);
      if (Dobs < model.y.size() && model.y[Dobs] > 1.0) {
        any_y_gt_1 = true;
      }
    }
    // tau^2 only enters when some y > 1 (C++ control flow on data).
    if (have_theta && any_y_gt_1) {
      cols.push_back(theta);
    }
    clique_pairs(cols, pairs);
  }
}

}  // namespace adlaplace_detail

// After apply_tape_domain(cfg, "obs", Dgroup). Empty rc => discover.
inline CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)>
obs_hessian_sparsity(
    const density_data& model, const Config& cfg, std::size_t Dgroup) {

  const std::string& dens = model.density;
  const bool is_glm =
    dens == "poisson_obs" || dens == "binomial_obs";
  const bool is_theta_glm =
    dens == "gaussian_obs" || dens == "nbinom_obs";
  const bool is_dm = dens == "dirichlet_multinomial";

  if (!is_glm && !is_theta_glm && !is_dm) {
    return empty_sparse_rc();
  }

  std::set<std::pair<std::size_t, std::size_t>> pairs;
  if (is_dm) {
    adlaplace_detail::add_dirichlet_stratum_cliques(model, cfg, Dgroup, pairs);
  } else {
    adlaplace_detail::add_glm_row_cliques(
      model, cfg, Dgroup, is_theta_glm, pairs);
  }

  CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> hessian;
  set_sparse_rc_pairs(hessian, model.n_tape, pairs);
  return hessian;
}

#endif
