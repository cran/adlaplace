#ifndef ADLAPLACE_AD_DATA_HPP
#define ADLAPLACE_AD_DATA_HPP

#include "adlaplace/rviews.hpp"

#include <Rinternals.h>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include <string>
#include <vector>

struct density_data {
  DgCView beta_map;
  DgCView gamma_map;
  DgCView theta_map;

  DgCView ATp;
  DgCView XTp;
  DgCView elgm_matrix;
  NumVecView y;
  NumVecView weights;

  SEXP precision = R_NilValue;
  std::string density;

  std::size_t num_beta = 0;
  std::size_t num_gamma = 0;
  std::size_t num_theta = 0;
  std::size_t num_full = 0;

  // Canonical map lookups: design column -> GLOBAL x index (fixed at construct).
  std::vector<std::size_t> beta_map_global;
  std::vector<std::size_t> gamma_map_global;
  std::vector<std::size_t> theta_map_global;

  // Active for indexing the tape Independent vector (tape positions after
  // apply_tape_domain). Kernels (eta_at, densities) use these.
  std::vector<std::size_t> beta_global;
  std::vector<std::size_t> gamma_global;
  std::vector<std::size_t> theta_global;

  std::size_t n_tape = 0;
  std::vector<std::size_t> tape_to_global;

  // Tape positions of gamma parameters present on the tape (inner subset).
  std::vector<int> seq_gamma;

  // Owned copy of random_mult precision Q (dsCMatrix uplo=U), filled by
  // cache_mult_precision_csc() at shard-build time.
  CscMatrix mult_Q_;
  bool has_mult_Q_ = false;

  explicit density_data(SEXP data_sexp);

  std::size_t theta_index(int col = 0) const;
  std::size_t theta_row(int col = 0) const;
  std::vector<std::size_t> gamma_global_indices(int col = 0) const;
  std::vector<std::size_t> all_gamma_global_indices() const;
  DgCView mult_precision_Q() const;
  void cache_mult_precision_csc();
  const CscMatrix& mult_precision_csc() const;
  double mult_precision_rank() const;
  double mult_precision_log_det() const;

  // Set tape domain: identity (full) or compact betas/gammas (+ all thetas).
  // mode: "obs" uses obs_groups column Dgroup; "all" uses all design columns /
  // map columns (parameters / random shards).
  void apply_tape_domain(
    const Config& cfg,
    const char* mode,
    std::size_t Dgroup = 0);
};

inline SEXP ad_data_slot(SEXP obj, const char* name) {
  return R_do_slot(obj, Rf_mkString(name));
}

namespace adlaplace_detail {

inline std::vector<std::size_t> build_map_global_lookup(
  const DgCView& map,
  std::size_t block_offset,
  std::size_t empty_sentinel,
  const char* what,
  bool require_nonempty_columns) {

  const int ncols = map.ncol();
  std::vector<std::size_t> out(static_cast<std::size_t>(ncols), empty_sentinel);
  for (int col = 0; col < ncols; ++col) {
    const int start = map.p[col];
    const int end = map.p[col + 1];
    const int nnz_col = end - start;
    if (nnz_col == 0) {
      if (require_nonempty_columns) {
        Rcpp::stop(
          "%s column %d must have exactly one nonzero (got none)",
          what,
          col + 1
        );
      }
      continue;
    }
    if (nnz_col != 1) {
      Rcpp::stop(
        "%s column %d must have exactly one nonzero (got %d)",
        what,
        col + 1,
        nnz_col
      );
    }
    const int local_row = map.i[start];
    if (local_row < 0 || local_row >= map.nrow()) {
      Rcpp::stop(
        "%s column %d has row index %d out of range [0, %d)",
        what,
        col + 1,
        local_row,
        map.nrow()
      );
    }
    out[static_cast<std::size_t>(col)] =
      block_offset + static_cast<std::size_t>(local_row);
  }
  return out;
}

inline void mark_design_active_for_obs(
  std::vector<unsigned char>& active,
  const density_data& model,
  std::size_t Dobs) {

  if (static_cast<std::size_t>(model.XTp.p.size()) > Dobs + 1) {
    const std::size_t p0 = static_cast<std::size_t>(model.XTp.p[Dobs]);
    const std::size_t p1 = static_cast<std::size_t>(model.XTp.p[Dobs + 1]);
    for (std::size_t k = p0; k < p1; ++k) {
      const std::size_t local_col = static_cast<std::size_t>(model.XTp.i[k]);
      if (local_col < model.beta_map_global.size()) {
        const std::size_t g = model.beta_map_global[local_col];
        if (g < model.num_full) {
          active[g] = 1;
        }
      }
    }
  }
  if (static_cast<std::size_t>(model.ATp.p.size()) > Dobs + 1) {
    const std::size_t p0 = static_cast<std::size_t>(model.ATp.p[Dobs]);
    const std::size_t p1 = static_cast<std::size_t>(model.ATp.p[Dobs + 1]);
    for (std::size_t k = p0; k < p1; ++k) {
      const std::size_t local_col = static_cast<std::size_t>(model.ATp.i[k]);
      if (local_col < model.gamma_map_global.size()) {
        const std::size_t g = model.gamma_map_global[local_col];
        if (g < model.num_full) {
          active[g] = 1;
        }
      }
    }
  }
}

inline void mark_obs_active(
  std::vector<unsigned char>& active,
  const density_data& model,
  const Config& cfg,
  std::size_t Dgroup) {

  const std::size_t ny = model.y.size();
  const bool have_shards = cfg.obs_groups.ncol() > 0;
  const bool have_elgm = model.elgm_matrix.ncol() > 0;
  std::size_t startP = 0;
  std::size_t endP = 0;
  if (have_shards) {
    if (Dgroup + 1 >= cfg.obs_groups.p.size()) {
      Rcpp::stop("obs_groups column %d out of range", static_cast<int>(Dgroup));
    }
    startP = static_cast<std::size_t>(cfg.obs_groups.p[Dgroup]);
    endP = static_cast<std::size_t>(cfg.obs_groups.p[Dgroup + 1]);
  } else if (Dgroup == 0) {
    endP = have_elgm
      ? static_cast<std::size_t>(model.elgm_matrix.ncol())
      : ny;
  }

  for (std::size_t DI = startP; DI < endP; ++DI) {
    const std::size_t Did = have_shards
      ? static_cast<std::size_t>(cfg.obs_groups.i[DI])
      : DI;

    if (have_elgm) {
      // obs_groups indexes ELGM strata (columns), not observation rows.
      const std::size_t n_strata =
        static_cast<std::size_t>(model.elgm_matrix.ncol());
      if (Did >= n_strata) {
        Rcpp::stop(
          "obs_groups stratum index %d out of range [0, %d)",
          static_cast<int>(Did),
          static_cast<int>(n_strata)
        );
      }
      if (Did + 1 >= static_cast<std::size_t>(model.elgm_matrix.p.size())) {
        Rcpp::stop(
          "elgm_matrix column pointer missing for stratum %d",
          static_cast<int>(Did)
        );
      }
      const std::size_t p0 =
        static_cast<std::size_t>(model.elgm_matrix.p[Did]);
      const std::size_t p1 =
        static_cast<std::size_t>(model.elgm_matrix.p[Did + 1]);
      for (std::size_t k = p0; k < p1; ++k) {
        const std::size_t Dobs =
          static_cast<std::size_t>(model.elgm_matrix.i[k]);
        mark_design_active_for_obs(active, model, Dobs);
      }
    } else {
      mark_design_active_for_obs(active, model, Did);
    }
  }
}

inline void mark_all_map_active(
  std::vector<unsigned char>& active,
  const density_data& model) {

  for (std::size_t g : model.beta_map_global) {
    if (g < model.num_full) {
      active[g] = 1;
    }
  }
  for (std::size_t g : model.gamma_map_global) {
    if (g < model.num_full) {
      active[g] = 1;
    }
  }
}

}  // namespace adlaplace_detail

inline density_data::density_data(SEXP data_sexp) {
  if (!Rf_inherits(data_sexp, "density_data")) {
    Rcpp::stop("expected S4 object of class density_data");
  }

  beta_map = DgCView(Rcpp::as<Rcpp::S4>(ad_data_slot(data_sexp, "beta_map")));
  gamma_map = DgCView(Rcpp::as<Rcpp::S4>(ad_data_slot(data_sexp, "gamma_map")));
  theta_map = DgCView(Rcpp::as<Rcpp::S4>(ad_data_slot(data_sexp, "theta_map")));

  ATp = DgCView(Rcpp::as<Rcpp::S4>(ad_data_slot(data_sexp, "ATp")));
  XTp = DgCView(Rcpp::as<Rcpp::S4>(ad_data_slot(data_sexp, "XTp")));
  elgm_matrix = DgCView(Rcpp::as<Rcpp::S4>(ad_data_slot(data_sexp, "elgm_matrix")));
  y = NumVecView(ad_data_slot(data_sexp, "y"));
  weights = NumVecView(ad_data_slot(data_sexp, "weights"));
  precision = ad_data_slot(data_sexp, "precision");

  SEXP dens_slot = ad_data_slot(data_sexp, "density");
  if (TYPEOF(dens_slot) == STRSXP && XLENGTH(dens_slot) >= 1) {
    SEXP s = STRING_ELT(dens_slot, 0);
    if (s != NA_STRING) {
      density = CHAR(s);
    }
  }

  num_beta = static_cast<std::size_t>(beta_map.nrow());
  num_gamma = static_cast<std::size_t>(gamma_map.nrow());
  num_theta = static_cast<std::size_t>(theta_map.nrow());
  num_full = num_beta + num_gamma + num_theta;

  const std::size_t theta_sentinel = num_beta + num_gamma;
  const bool have_beta_design = XTp.nrow() > 0;
  const bool have_gamma_design = ATp.nrow() > 0;

  if (have_beta_design && beta_map.ncol() != XTp.nrow()) {
    Rcpp::stop(
      "beta_map ncol (%d) must match ncol(X) (%d)",
      beta_map.ncol(),
      XTp.nrow()
    );
  }
  if (have_gamma_design && gamma_map.ncol() != ATp.nrow()) {
    Rcpp::stop(
      "gamma_map ncol (%d) must match ncol(A) (%d)",
      gamma_map.ncol(),
      ATp.nrow()
    );
  }

  beta_map_global = adlaplace_detail::build_map_global_lookup(
    beta_map,
    0,
    num_full,
    "beta_map",
    have_beta_design
  );
  gamma_map_global = adlaplace_detail::build_map_global_lookup(
    gamma_map,
    num_beta,
    num_full,
    "gamma_map",
    have_gamma_design
  );
  theta_map_global = adlaplace_detail::build_map_global_lookup(
    theta_map,
    num_beta + num_gamma,
    theta_sentinel,
    "theta_map",
    false
  );

  // Default identity tape domain until apply_tape_domain is called.
  n_tape = num_full;
  tape_to_global.resize(num_full);
  for (std::size_t g = 0; g < num_full; ++g) {
    tape_to_global[g] = g;
  }
  beta_global = beta_map_global;
  gamma_global = gamma_map_global;
  theta_global.resize(theta_map_global.size());
  for (std::size_t j = 0; j < theta_map_global.size(); ++j) {
    const std::size_t g = theta_map_global[j];
    theta_global[j] =
      (g < num_beta + num_gamma || g >= num_full) ? n_tape : g;
  }
  seq_gamma.resize(num_gamma);
  for (std::size_t d = 0; d < num_gamma; ++d) {
    seq_gamma[d] = static_cast<int>(num_beta + d);
  }
}

inline void density_data::apply_tape_domain(
  const Config& cfg,
  const char* mode,
  std::size_t Dgroup) {

  std::vector<unsigned char> active(num_full, 0);

  // Always keep every theta on the tape.
  for (std::size_t t = 0; t < num_theta; ++t) {
    active[num_beta + num_gamma + t] = 1;
  }

  if (!cfg.compact_tape) {
    std::fill(active.begin(), active.end(), 1);
  } else if (std::strcmp(mode, "obs") == 0 &&
             (y.size() > 0 || elgm_matrix.ncol() > 0)) {
    // Plain obs densities use y rows; ELGM (dirichlet_multinom) expands
    // obs_groups strata through elgm_matrix inside mark_obs_active.
    adlaplace_detail::mark_obs_active(active, *this, cfg, Dgroup);
  } else {
    // parameters/random shards: keep every mapped beta/gamma.
    adlaplace_detail::mark_all_map_active(active, *this);
  }

  tape_to_global.clear();
  tape_to_global.reserve(num_full);
  for (std::size_t g = 0; g < num_full; ++g) {
    if (active[g]) {
      tape_to_global.push_back(g);
    }
  }
  n_tape = tape_to_global.size();

  std::vector<std::size_t> global_to_tape(num_full, n_tape);
  for (std::size_t k = 0; k < n_tape; ++k) {
    global_to_tape[tape_to_global[k]] = k;
  }

  beta_global.resize(beta_map_global.size());
  for (std::size_t j = 0; j < beta_map_global.size(); ++j) {
    const std::size_t g = beta_map_global[j];
    beta_global[j] = (g < num_full) ? global_to_tape[g] : n_tape;
  }
  gamma_global.resize(gamma_map_global.size());
  for (std::size_t j = 0; j < gamma_map_global.size(); ++j) {
    const std::size_t g = gamma_map_global[j];
    gamma_global[j] = (g < num_full) ? global_to_tape[g] : n_tape;
  }
  theta_global.resize(theta_map_global.size());
  for (std::size_t j = 0; j < theta_map_global.size(); ++j) {
    const std::size_t g = theta_map_global[j];
    // theta_map sentinel is num_beta+num_gamma (not num_full).
    if (g < num_beta + num_gamma || g >= num_full) {
      theta_global[j] = n_tape;
    } else {
      theta_global[j] = global_to_tape[g];
    }
  }

  seq_gamma.clear();
  seq_gamma.reserve(num_gamma);
  for (std::size_t k = 0; k < n_tape; ++k) {
    const std::size_t g = tape_to_global[k];
    if (g >= num_beta && g < num_beta + num_gamma) {
      seq_gamma.push_back(static_cast<int>(k));
    }
  }
}

inline std::size_t density_data::theta_index(int col) const {
  if (col < 0 || col >= static_cast<int>(theta_global.size())) {
    return n_tape;
  }
  return theta_global[static_cast<std::size_t>(col)];
}

inline std::size_t density_data::theta_row(int col) const {
  if (col < 0 || col >= static_cast<int>(theta_map_global.size())) {
    return num_theta;
  }
  const std::size_t g = theta_map_global[static_cast<std::size_t>(col)];
  if (g < num_beta + num_gamma || g >= num_full) {
    return num_theta;
  }
  return g - num_beta - num_gamma;
}

inline std::vector<std::size_t> density_data::gamma_global_indices(int col) const {
  std::vector<std::size_t> out;
  if (col < 0 || col >= static_cast<int>(gamma_global.size())) {
    return out;
  }
  const std::size_t g = gamma_global[static_cast<std::size_t>(col)];
  if (g >= n_tape) {
    return out;
  }
  out.push_back(g);
  return out;
}

inline std::vector<std::size_t> density_data::all_gamma_global_indices() const {
  std::vector<std::size_t> out;
  out.reserve(gamma_global.size());
  for (std::size_t g : gamma_global) {
    if (g < n_tape) {
      out.push_back(g);
    }
  }
  return out;
}

inline DgCView density_data::mult_precision_Q() const {
  if (Rf_isNull(precision) || TYPEOF(precision) != VECSXP) {
    Rcpp::stop(
        "random_mult precision must be list(Q = <dsCMatrix>, log_det, rank)");
  }
  Rcpp::List prec(precision);
  if (!prec.containsElementNamed("Q")) {
    Rcpp::stop("random_mult precision list must contain Q");
  }
  SEXP q_sexp = prec["Q"];
  if (Rf_inherits(q_sexp, "dgCMatrix")) {
    Rcpp::stop(
        "random_mult precision Q must be a dsCMatrix with uplo = \"U\"; "
        "coerce in R with adlaplace:::as_dsC_upper(Q) "
        "(C++ does not convert dgCMatrix)");
  }
  if (!Rf_inherits(q_sexp, "dsCMatrix")) {
    Rcpp::stop(
        "random_mult precision Q must be a dsCMatrix with uplo = \"U\"");
  }
  Rcpp::S4 q_s4(q_sexp);
  const std::string uplo = Rcpp::as<std::string>(q_s4.slot("uplo"));
  if (uplo != "U") {
    Rcpp::stop(
        "random_mult precision Q must have uplo = \"U\" (got \"%s\"); "
        "coerce in R with adlaplace:::as_dsC_upper(Q)",
        uplo.c_str());
  }
  return DgCView(q_s4);
}

inline void density_data::cache_mult_precision_csc() {
  mult_Q_ = CscMatrix(mult_precision_Q());
  has_mult_Q_ = true;
}

inline const CscMatrix& density_data::mult_precision_csc() const {
  if (!has_mult_Q_) {
    Rcpp::stop(
        "mult_precision_csc: call cache_mult_precision_csc() before use");
  }
  return mult_Q_;
}

inline double density_data::mult_precision_rank() const {
  const int n_term = gamma_map.ncol();
  if (Rf_isNull(precision) || TYPEOF(precision) != VECSXP) {
    return static_cast<double>(n_term);
  }
  Rcpp::List prec(precision);
  return prec.containsElementNamed("rank")
    ? Rcpp::as<double>(prec["rank"])
    : static_cast<double>(n_term);
}

inline double density_data::mult_precision_log_det() const {
  if (Rf_isNull(precision) || TYPEOF(precision) != VECSXP) {
    return 0.0;
  }
  Rcpp::List prec(precision);
  return prec.containsElementNamed("log_det")
    ? Rcpp::as<double>(prec["log_det"])
    : 0.0;
}

inline void validate_config_matches_model(
  const Config& cfg,
  const density_data& data,
  bool check_gamma = true) {
  if (cfg.beta.size() != data.num_beta) {
    Rcpp::stop(
      "length(config$beta) (%d) must match nrow(beta_map) (%d)",
      static_cast<int>(cfg.beta.size()),
      static_cast<int>(data.num_beta)
    );
  }
  if (check_gamma && cfg.gamma.size() != data.num_gamma) {
    Rcpp::stop(
      "length(config$gamma) (%d) must match nrow(gamma_map) (%d)",
      static_cast<int>(cfg.gamma.size()),
      static_cast<int>(data.num_gamma)
    );
  }
  if (cfg.theta.size() != data.num_theta) {
    Rcpp::stop(
      "length(config$theta) (%d) must match nrow(theta_map) (%d)",
      static_cast<int>(cfg.theta.size()),
      static_cast<int>(data.num_theta)
    );
  }
}

#endif
