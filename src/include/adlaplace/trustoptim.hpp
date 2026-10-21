#ifndef ADLAPLACE_TRUSTOPTIM_HPP
#define ADLAPLACE_TRUSTOPTIM_HPP

#include<Rcpp.h>

// Small helpers with defaults
inline double get_double_ctrl(const Rcpp::List& ctl,
  const char* key,
  double def) {
  if (ctl.containsElementNamed(key) && !Rf_isNull(ctl[key])) {
    return Rcpp::as<double>(ctl[key]);
  }
  return def;
}

inline int get_int_ctrl(const Rcpp::List& ctl,
  const char* key,
  int def) {
  if (ctl.containsElementNamed(key) && !Rf_isNull(ctl[key])) {
    return Rcpp::as<int>(ctl[key]);
  }
  return def;
}



// Bundle of trust-region control parameters
struct TrustControl {
  // Scalars
  double rad;
  double min_rad;
  double tol;
  double prec;

  int    report_freq;
  int    report_level;
  int    header_freq;
  int    report_precision;
  int    maxit;

  double contract_factor;
  double expand_factor;
  double contract_threshold;
  double expand_threshold_rad;
  double expand_threshold_ap;
  double function_scale_factor;

  int    precond_refresh_freq;
  int    precond_ID;
//  int    quasi_newton_method;  // 1 = SR1, 2 = BFGS
  int    trust_iter;
  // Half-width for optional large-gradient restart clamp of gamma into [-x, x].
  // Non-finite (e.g. Inf, the default) disables the restart.
  double restart_gamma_clamp;

  // Construct from an Rcpp::List with defaults
  explicit TrustControl(const Rcpp::List& control)
  : rad(                  get_double_ctrl(control, "step.size",              1.0))
  , min_rad(              get_double_ctrl(control, "min.step.size",          1e-8))
  , tol(                  get_double_ctrl(control, "cg.tol",                 1e-4))
  , prec(                 get_double_ctrl(control, "grad.tol",               1e-3))
  , report_freq(          get_int_ctrl   (control, "report.freq",            0))
  , report_level(         get_int_ctrl   (control, "report.level",           0))
  , header_freq(          get_int_ctrl   (control, "header.freq",            10))
  , report_precision(     get_int_ctrl   (control, "report.precision",       6))
  , maxit(                get_int_ctrl   (control, "maxit",                  100))
  , contract_factor(      get_double_ctrl(control, "contract.factor",        0.5))
  , expand_factor(        get_double_ctrl(control, "expand.factor",          3.0))
  , contract_threshold(   get_double_ctrl(control, "contract.threshold",     0.25))
  , expand_threshold_rad( get_double_ctrl(control, "expand.threshold.rad",   0.8))
  , expand_threshold_ap(  get_double_ctrl(control, "expand.threshold.ap",    0.75))
  , function_scale_factor(get_double_ctrl(control, "function.scale.factor",  1.0))
  , precond_refresh_freq( get_int_ctrl   (control, "precond.refresh",        1))
  , precond_ID(           get_int_ctrl   (control, "precond.ID",             1))
//    , quasi_newton_method(  get_int_ctrl   (control, "quasi.newton.method",    1))  // 1 = SR1
  , trust_iter(           get_int_ctrl   (control, "trust.iter",             4000))
  , restart_gamma_clamp(  get_double_ctrl(control, "restart.gamma.clamp",    R_PosInf))
  {}
};

#include <Rcpp.h>
#include <Rinternals.h>

#include <Eigen/SparseCore>
#include <numeric>
#include <string>
#include <vector>

#include "adlaplace/omp_compat.hpp"
#include "adlaplace/backend.hpp"
#include "adlaplace/rviews.hpp"
#include "adlaplace/runtime.hpp"

struct AD_Func_Opt {
  ad_pack *const shards;
  const std::vector<std::vector<size_t>> *const thread_groups;
  const bool inner;
  const int num_threads;
  const bool verbose;

  const size_t Nparams;
  const size_t n_shards_backend;
  const size_t Nbeta_backend;
  const size_t Ngamma_backend;
  const size_t Ntheta_backend;
  const size_t nvars_opt;
  const size_t var_offset;

  const Eigen::SparseMatrix<int, Eigen::ColMajor, int> Htemplate;
  const std::vector<std::vector<int>> hess_maps;
  const std::vector<size_t> all_shards;

  std::vector<double> parameters;
  std::vector<double> hess_upper_accum;

  AD_Func_Opt(
      ad_pack &ag, const std::vector<double> &params_init, bool innerIn = true,
      int num_threads_in = 1,
      const std::vector<std::vector<size_t>> *thread_groups_in = nullptr,
      bool verbose_in = false)
      : AD_Func_Opt(ag, params_init, make_layout(ag, innerIn), innerIn,
                    num_threads_in, thread_groups_in, verbose_in) {}

  int get_nvars() const { return static_cast<int>(nvars_opt); }
  int get_nnz() const { return static_cast<int>(Htemplate.nonZeros()); }
  bool openmp_enabled() const { return num_threads > 1; }
  int openmp_threads() const { return num_threads; }

  template <class DerivedX>
  void get_f(const Eigen::MatrixBase<DerivedX> &x, double &f) {
    update_gamma_from_x(x);

    double f_sum = 0.0;
    adlaplace_verbose_msg(
        verbose,
        std::string(inner ? "inner_opt get_f" : "outer get_f") +
            ": before OpenMP threads=" + std::to_string(num_threads) +
            " shards=" + std::to_string(n_shards_backend));

#pragma omp parallel num_threads(num_threads)
    {
      double f_local = 0.0;
      std::vector<double> params_local(parameters.begin(), parameters.end());
      const std::vector<size_t> &shard_group = shard_group_for_thread();
      if (verbose && adlaplace_debug_enabled()) {
#pragma omp critical(adlaplace_verbose)
        {
          Rcpp::Rcout << "  get_f thread=" << omp_get_thread_num()
                      << " n_shards=" << shard_group.size() << std::endl;
        }
      }

      for (size_t s : shard_group) {
        ad_shard *shard = shards->fun[s];
        AdTape &pack = shard->pack;
        if (!adlaplace_shard_thread_ok(pack)) {
          adlaplace_debug_record_mismatch(
              s, pack.owner_thread,
              static_cast<std::size_t>(omp_get_thread_num()),
              inner ? "inner_opt get_f" : "inner_opt get_f");
          continue;
        }
        (void)shard->f(params_local.data(), &f_local);
      }

#pragma omp critical(f_sum)
      {
        f_sum -= f_local;
      }
    }

    adlaplace_debug_raise_if_any(inner ? "inner_opt get_f" : "inner_opt get_f");
    adlaplace_verbose_msg(
        verbose,
        std::string(inner ? "inner_opt get_f" : "outer get_f") +
            ": after OpenMP");
    f = f_sum;
  }

  template <class DerivedX, class DerivedG>
  void get_fdf(const Eigen::MatrixBase<DerivedX> &x, double &f,
               Eigen::MatrixBase<DerivedG> &g) {
    update_gamma_from_x(x);

    f = 0.0;
    auto &gout = g.derived();
    gout.setZero();

    std::vector<double> grad_full(Nparams, 0.0);
    const bool inner_flag = inner;
    const char *phase = inner ? "inner_opt get_fdf" : "outer get_fdf";
    adlaplace_verbose_msg(
        verbose,
        std::string(phase) + ": before OpenMP threads=" +
            std::to_string(num_threads) + " Nparams=" + std::to_string(Nparams) +
            " shards=" + std::to_string(n_shards_backend));

#pragma omp parallel num_threads(num_threads)
    {
      double f_local = 0.0;
      std::vector<double> grad_local(Nparams, 0.0);
      std::vector<double> params_local(parameters.begin(), parameters.end());
      const std::vector<size_t> &shard_group = shard_group_for_thread();
      if (verbose && adlaplace_debug_enabled()) {
#pragma omp critical(adlaplace_verbose)
        {
          Rcpp::Rcout << "  " << phase << " thread=" << omp_get_thread_num()
                      << " n_shards=" << shard_group.size() << std::endl;
        }
      }

      for (size_t s : shard_group) {
        ad_shard *shard = shards->fun[s];
        AdTape &pack = shard->pack;
        if (!adlaplace_shard_thread_ok(pack)) {
          adlaplace_debug_record_mismatch(
              s, pack.owner_thread,
              static_cast<std::size_t>(omp_get_thread_num()), phase);
          adlaplace_debug_note_grad_mismatch(grad_local.data(), Nparams);
          continue;
        }
        (void)shard->f_grad(params_local.data(), inner_flag, &f_local,
                            grad_local.data());
      }

#pragma omp critical(grad_sum)
      {
        f -= f_local;
        for (size_t k = 0; k < Nparams; ++k) {
          grad_full[k] -= grad_local[k];
        }
      }
    }

    adlaplace_debug_raise_if_any(phase);
    adlaplace_verbose_msg(verbose, std::string(phase) + ": after OpenMP");
    const size_t gsize = static_cast<size_t>(gout.size());
    const size_t ncopy = gsize < nvars_opt ? gsize : nvars_opt;
    for (size_t k = 0; k < ncopy; ++k) {
      gout[static_cast<Eigen::Index>(k)] = grad_full[var_offset + k];
    }
  }

  template <class DerivedX, class DerivedG>
  void get_fdfh(const Eigen::MatrixBase<DerivedX> &x, double &f,
                Eigen::MatrixBase<DerivedG> &g,
                Eigen::SparseMatrix<double> &H) {
    update_gamma_from_x(x);

    f = 0.0;
    auto &gout = g.derived();
    if (gout.size() > 0)
      gout.setZero();

    std::fill(hess_upper_accum.begin(), hess_upper_accum.end(), 0.0);
    std::vector<double> grad_full(Nparams, 0.0);
    const bool inner_flag = inner;
    const char *phase = inner ? "inner_opt get_fdfh" : "outer get_fdfh";
    int api_err_shard = -1;
    int api_err_rc = 0;
    const size_t hess_size = hess_upper_accum.size();

    adlaplace_verbose_msg(
        verbose,
        std::string(phase) + ": before OpenMP threads=" +
            std::to_string(num_threads) + " Nparams=" + std::to_string(Nparams) +
            " hess_size=" + std::to_string(hess_size) +
            " shards=" + std::to_string(n_shards_backend));

#ifdef DEBUG_EXTRA
    std::vector<double> f_local_log(static_cast<size_t>(num_threads), -99.0);
#endif

    int parallel_team = 1;
#pragma omp parallel num_threads(num_threads)
    {
#pragma omp single
      parallel_team = omp_get_num_threads();

      double f_local = 0.0;
      std::vector<double> grad_local(Nparams, 0.0);
      std::vector<double> hess_local(hess_size, 0.0);
      std::vector<double> hess_thread(hess_size, 0.0);
      std::vector<double> params_local(parameters.begin(), parameters.end());
      const std::vector<size_t> &shard_group = shard_group_for_thread();
      if (verbose && adlaplace_debug_enabled()) {
#pragma omp critical(adlaplace_verbose)
        {
          Rcpp::Rcout << "  " << phase << " thread=" << omp_get_thread_num()
                      << " n_shards=" << shard_group.size() << std::endl;
        }
      }

      for (size_t s : shard_group) {
        ad_shard *shard = shards->fun[s];
        AdTape &pack = shard->pack;
        if (!adlaplace_shard_thread_ok(pack)) {
          adlaplace_debug_record_mismatch(
              s, pack.owner_thread,
              static_cast<std::size_t>(omp_get_thread_num()), phase);
          adlaplace_debug_note_grad_mismatch(grad_local.data(), Nparams);
          continue;
        }
        int *map_ptr = const_cast<int *>(hess_maps[s].data());
        std::fill(hess_local.begin(), hess_local.end(), 0.0);
        const int rc = shard->f_grad_hess(
            params_local.data(), inner_flag, &f_local,
            grad_local.data(), hess_local.data(), map_ptr);
        if (rc != 0) {
#pragma omp critical(adlaplace_api_err)
          {
            if (api_err_shard < 0) {
              api_err_shard = static_cast<int>(s);
              api_err_rc = rc;
            }
          }
          continue;
        }
        for (size_t k = 0; k < hess_size; ++k) {
          hess_thread[k] += hess_local[k];
        }
      }

#pragma omp critical(hess_sum)
      {
        f -= f_local;
        for (size_t k = 0; k < Nparams; ++k) {
          grad_full[k] -= grad_local[k];
        }
        for (size_t k = 0; k < hess_size; ++k) {
          hess_upper_accum[k] -= hess_thread[k];
        }
      }

#ifdef DEBUG_EXTRA
      f_local_log[static_cast<size_t>(omp_get_thread_num())] = f_local;
#endif
    }

#ifdef DEBUG_EXTRA
    Rcpp::Rcout << phase << " f_local:\n";
    for (int t = 0; t < num_threads; ++t) {
      Rcpp::Rcout << "  thread=" << t
                  << " f_local=" << f_local_log[static_cast<size_t>(t)] << "\n";
    }
#endif

    if (api_err_shard >= 0) {
      Rcpp::stop("%s: f_grad_hess failed for shard %d with code %d", phase,
                 api_err_shard, api_err_rc);
    }

    adlaplace_debug_raise_if_any(phase);
    if (verbose) {
      Rcpp::Rcout << phase << " parallel_team=" << parallel_team << "\n";
    }
    adlaplace_verbose_msg(
        verbose,
        std::string(phase) + ": after OpenMP api_err_shard=" +
            std::to_string(api_err_shard));
    if (gout.size() > 0) {
      const size_t gsize = static_cast<size_t>(gout.size());
      const size_t ncopy = gsize < nvars_opt ? gsize : nvars_opt;
      for (size_t k = 0; k < ncopy; ++k) {
        gout[static_cast<Eigen::Index>(k)] = grad_full[var_offset + k];
      }
    }

    adlaplace_verbose_msg(verbose, std::string(phase) + ": scatter Hessian...");
    if (H.rows() != Htemplate.rows() || H.cols() != Htemplate.cols() ||
        H.nonZeros() != Htemplate.nonZeros()) {
      H = Htemplate.cast<double>();
      H.makeCompressed();
    }

    double *Hx = H.valuePtr();
    const int *cell_id = Htemplate.valuePtr();
    const Eigen::Index nz = H.nonZeros();
    for (Eigen::Index t = 0; t < nz; ++t) {
      const int cid = cell_id[t];
      if (cid < 0 || static_cast<size_t>(cid) >= hess_size) {
        Rcpp::stop(
            "%s: Hessian cell_id OOB t=%d cell_id=%d hess_size=%d", phase,
            static_cast<int>(t), cid, static_cast<int>(hess_size));
      }
      Hx[t] = hess_upper_accum[static_cast<size_t>(cid)];
    }
    adlaplace_verbose_msg(verbose, std::string(phase) + ": scatter done");
  }

  template <class DerivedX>
  void get_hessian(const Eigen::MatrixBase<DerivedX> &x,
                   Eigen::SparseMatrix<double> &H) {
    double f_dummy = 0.0;
    Eigen::VectorXd g_dummy(static_cast<Eigen::Index>(nvars_opt));
    get_fdfh(x, f_dummy, g_dummy, H);
  }

  template <class DerivedX, class DerivedH>
  void get_hessian(const Eigen::MatrixBase<DerivedX> &x,
                   Eigen::SparseMatrixBase<DerivedH> &Hbase) {
    DerivedH &H = Hbase.derived();
    if (H.rows() != Htemplate.rows() || H.cols() != Htemplate.cols() ||
        H.nonZeros() != Htemplate.nonZeros()) {
      H = Htemplate.cast<double>();
      H.makeCompressed();
    }
    double f_dummy = 0.0;
    Eigen::VectorXd g_dummy(static_cast<Eigen::Index>(nvars_opt));
    get_fdfh(x, f_dummy, g_dummy, H);
  }

  template <class DerivedX, class DerivedG>
  void get_df(const Eigen::MatrixBase<DerivedX> &x,
              Eigen::MatrixBase<DerivedG> &g) {
    double f_dummy = 0.0;
    get_fdf(x, f_dummy, g);
  }

private:
  struct Layout {
    size_t Nparams;
    size_t n_shards_backend;
    size_t nvars_opt;
    size_t var_offset;
    hessian_template Htemplate;
    std::vector<std::vector<int>> hess_maps;
  };

  static std::vector<std::vector<int>> build_hess_maps(ad_pack &ag, bool innerIn,
                                                       size_t n_shards) {

    const hessian_map_view &hv = innerIn ? ag.map_inner : ag.map_outer;
    const R_xlen_t p_len = hv.p.size();
    if (p_len != static_cast<R_xlen_t>(n_shards + 1)) {
      Rcpp::stop("hessian map p length %d but expected n_shards+1=%d",
                 static_cast<int>(p_len), static_cast<int>(n_shards + 1));
    }

    std::vector<std::vector<int>> maps(n_shards);
    for (size_t s = 0; s < n_shards; ++s) {
      ad_shard *shard = ag.fun[s];
      if (!shard) {
        Rcpp::stop("ad_pack.fun[%d] is NULL", static_cast<int>(s));
      }

      int n_inner = 0;
      int n_outer = 0;
      int n_beta = 0;
      int n_theta = 0;
      int nnz_grad_inner = 0;
      int nnz_grad_outer = 0;
      int nnz_hes_inner = 0;
      int nnz_hes_outer = 0;
      if (shard->get_sizes(&n_inner, &n_outer, &n_beta, &n_theta,
                            &nnz_grad_inner, &nnz_grad_outer, &nnz_hes_inner,
                            &nnz_hes_outer) != 0) {
        Rcpp::stop("get_sizes failed for shard %d", static_cast<int>(s));
      }

      const int nnz_hes = innerIn ? nnz_hes_inner : nnz_hes_outer;
      maps[s].assign(static_cast<size_t>(nnz_hes), -1);

      const int col_start = hv.p[static_cast<R_xlen_t>(s)];
      const int col_end = hv.p[static_cast<R_xlen_t>(s + 1)];
      for (int k = col_start; k < col_end; ++k) {
        const int loc = hv.local[k];
        const int glob = hv.global[k];
        if (loc < 0 || loc >= nnz_hes) {
          Rcpp::stop(
              "hessian map local index %d out of range [0, %d) for shard %d",
              loc, nnz_hes, static_cast<int>(s));
        }
        maps[s][static_cast<size_t>(loc)] = glob;
      }

      for (int loc = 0; loc < nnz_hes; ++loc) {
        if (maps[s][static_cast<size_t>(loc)] < 0) {
          Rcpp::stop("hessian map missing entry for shard %d local index %d",
                     static_cast<int>(s), loc);
        }
      }
    }
    return maps;
  }

  static Layout make_layout(ad_pack &ag, bool innerIn) {
    if (ag.fun.empty())
      Rcpp::stop("ad_pack.fun is empty");

    const size_t Nbeta = static_cast<size_t>(ag.sizes.named("beta"));
    const size_t Ngamma = static_cast<size_t>(ag.sizes.named("gamma"));
    const size_t n_shards = ag.fun.size();
    ad_shard *shard0 = ag.fun[0];
    const size_t Nparams = ad_tape_n_global(shard0->pack);

    const size_t nvars_opt = innerIn ? Ngamma : Nparams;
    const size_t var_offset = innerIn ? Nbeta : 0;

    const hessian_template &tpl = innerIn ? ag.hessian_inner : ag.hessian_outer;
    if (tpl.nonZeros() == 0) {
      Rcpp::stop("%s Hessian template missing on ad_pack; attach hessian_map "
                 "result first",
                 innerIn ? "inner" : "outer");
    }
    if (static_cast<size_t>(tpl.rows()) != nvars_opt) {
      Rcpp::stop("%s Hessian nrow=%d but expected nvars=%d",
                 innerIn ? "inner" : "outer", static_cast<int>(tpl.rows()),
                 static_cast<int>(nvars_opt));
    }

    Layout out;
    out.Nparams = Nparams;
    out.n_shards_backend = n_shards;
    out.nvars_opt = nvars_opt;
    out.var_offset = var_offset;
    out.Htemplate = tpl;
    out.hess_maps = build_hess_maps(ag, innerIn, n_shards);
    return out;
  }

  static std::vector<size_t> make_all_shards(size_t n_shards) {
    std::vector<size_t> out(n_shards);
    for (size_t s = 0; s < n_shards; ++s) {
      out[s] = s;
    }
    return out;
  }

  AD_Func_Opt(ad_pack &ag, const std::vector<double> &params_init, Layout layout,
              bool innerIn, int num_threads_in,
              const std::vector<std::vector<size_t>> *thread_groups_in,
              bool verbose_in)
      : shards(&ag), thread_groups(thread_groups_in), inner(innerIn),
        num_threads(num_threads_in > 0 ? num_threads_in : 1),
        verbose(verbose_in),
        Nparams(layout.Nparams), n_shards_backend(layout.n_shards_backend),
        Nbeta_backend(static_cast<size_t>(ag.sizes.named("beta"))),
        Ngamma_backend(static_cast<size_t>(ag.sizes.named("gamma"))),
        Ntheta_backend(static_cast<size_t>(ag.sizes.named("theta"))),
        nvars_opt(layout.nvars_opt), var_offset(layout.var_offset),
        Htemplate(layout.Htemplate), hess_maps(std::move(layout.hess_maps)),
        all_shards(make_all_shards(layout.n_shards_backend)), parameters(),
        hess_upper_accum(static_cast<size_t>(Htemplate.nonZeros()), 0.0) {
    if (thread_groups &&
        static_cast<int>(thread_groups->size()) != num_threads) {
      Rcpp::stop("thread affinity group count %d does not match num_threads %d",
                 static_cast<int>(thread_groups->size()), num_threads);
    }
    parameters.resize(Nparams, 0.0);
    for (size_t j = 0; j < params_init.size() && j < Nparams; ++j) {
      parameters[j] = params_init[j];
    }
    // Once-per-construction check (avoids rescanning every get_fdfh call).
    const size_t hess_size = hess_upper_accum.size();
    const char *phase = inner ? "inner_opt" : "outer";
    for (size_t s = 0; s < hess_maps.size(); ++s) {
      const auto &map = hess_maps[s];
      for (size_t Di = 0; Di < map.size(); ++Di) {
        const int glob = map[Di];
        if (glob < 0 || static_cast<size_t>(glob) >= hess_size) {
          Rcpp::stop(
              "%s: hess map OOB shard=%d Di=%d glob=%d hess_size=%d", phase,
              static_cast<int>(s), static_cast<int>(Di), glob,
              static_cast<int>(hess_size));
        }
      }
    }
  }

  inline const std::vector<size_t> &shard_group_for_thread() const {
    if (!thread_groups) {
      return all_shards;
    }
    const int tid = omp_get_thread_num();
    if (tid < 0 || tid >= static_cast<int>(thread_groups->size())) {
      Rcpp::stop(
          "OpenMP thread %d out of range for thread affinity groups (%d)", tid,
          static_cast<int>(thread_groups->size()));
    }
    return (*thread_groups)[static_cast<size_t>(tid)];
  }

  template <class DerivedX>
  inline void update_gamma_from_x(const Eigen::MatrixBase<DerivedX> &x) {
    if (static_cast<size_t>(x.size()) != nvars_opt) {
      return;
    }
    if (inner) {
      for (size_t k = 0; k < Ngamma_backend; ++k) {
        parameters[Nbeta_backend + k] = x[static_cast<Eigen::Index>(k)];
      }
    } else {
      for (size_t k = 0; k < Nparams; ++k) {
        parameters[k] = x[static_cast<Eigen::Index>(k)];
      }
    }
  }
};

#endif
