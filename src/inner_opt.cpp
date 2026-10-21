
// Standard C
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
// Eigen (SparseCholesky pulls SparseCore + OrderingMethods)
#include <Eigen/SparseCholesky>
// Rcpp
#include <Rcpp.h>

// trustOptim
#include <CG-sparse.h>

// Local
#include "adlaplace/rviews.hpp"
#include "adlaplace/atomics.hpp"
#include "adlaplace/omp_compat.hpp"
#include "adlaplace/ompad.hpp"
#include "adlaplace/runtime.hpp"
#include "adlaplace/chol_update.hpp"
#include "adlaplace/profile_deriv.hpp"
#include "adlaplace/trace_hinv_t_runtime.hpp"
#include "adlaplace/trustoptim.hpp"

// Convert Eigen sparse matrix to Matrix::dgCMatrix S4 object directly
Rcpp::S4 eigen_to_dgCMatrix(const Eigen::SparseMatrix<double> &M) {
  const Eigen::Index nrow = M.rows();
  const Eigen::Index ncol = M.cols();
  if (nrow == 0 || ncol == 0) {
    Rcpp::S4 mat("dgCMatrix");
    mat.slot("p") = Rcpp::IntegerVector::create(0);
    mat.slot("i") = Rcpp::IntegerVector();
    mat.slot("x") = Rcpp::NumericVector();
    mat.slot("Dim") = Rcpp::IntegerVector::create(0, 0);
    return mat;
  }
  const Eigen::Index nnz = M.nonZeros();

  Rcpp::IntegerVector i(nnz);
  Rcpp::IntegerVector p(ncol + 1);
  Rcpp::NumericVector x(nnz);

  // Copy data from Eigen (already in CSC format, 0-based)
  std::copy(M.innerIndexPtr(), M.innerIndexPtr() + nnz, i.begin());
  std::copy(M.outerIndexPtr(), M.outerIndexPtr() + ncol + 1, p.begin());
  std::copy(M.valuePtr(), M.valuePtr() + nnz, x.begin());

  Rcpp::S4 mat("dgCMatrix");
  mat.slot("i") = i;
  mat.slot("p") = p;
  mat.slot("x") = x;
  mat.slot("Dim") = Rcpp::IntegerVector::create(static_cast<int>(nrow),
                                                static_cast<int>(ncol));
  return mat;
}

// Unit-lower triangular factor L from chol_inner CSC pattern (0-based p, i).
Rcpp::S4 csc_to_dtCMatrix_lower(const std::vector<int> &p,
                                const std::vector<int> &i,
                                const std::vector<double> &x, int n) {
  if (static_cast<size_t>(n + 1) != p.size()) {
    Rcpp::stop("csc_to_dtCMatrix_lower: p length must be nrow + 1");
  }
  if (i.size() != x.size()) {
    Rcpp::stop("csc_to_dtCMatrix_lower: i and x length must match");
  }

  Rcpp::S4 mat("dtCMatrix");
  mat.slot("p") = Rcpp::IntegerVector(p.begin(), p.end());
  mat.slot("i") = Rcpp::IntegerVector(i.begin(), i.end());
  mat.slot("x") = Rcpp::NumericVector(x.begin(), x.end());
  mat.slot("Dim") = Rcpp::IntegerVector::create(n, n);
  mat.slot("uplo") = Rcpp::String("L");
  mat.slot("diag") = Rcpp::String("N");
  return mat;
}

Rcpp::S4 csc_to_dsCMatrix_upper(const std::vector<int> &p,
                                const std::vector<int> &i,
                                const std::vector<double> &x, int n) {
  if (static_cast<size_t>(n + 1) != p.size()) {
    Rcpp::stop("csc_to_dsCMatrix_upper: p length must be nrow + 1");
  }
  if (i.size() != x.size()) {
    Rcpp::stop("csc_to_dsCMatrix_upper: i and x length must match");
  }

  Rcpp::S4 mat("dsCMatrix");
  mat.slot("p") = Rcpp::IntegerVector(p.begin(), p.end());
  mat.slot("i") = Rcpp::IntegerVector(i.begin(), i.end());
  mat.slot("x") = Rcpp::NumericVector(x.begin(), x.end());
  mat.slot("Dim") = Rcpp::IntegerVector::create(n, n);
  mat.slot("uplo") = Rcpp::String("U");
  return mat;
}

Rcpp::S4 csc_to_dgCMatrix(const std::vector<int> &p, const std::vector<int> &i,
                          const std::vector<double> &x, int n) {
  if (static_cast<size_t>(n + 1) != p.size()) {
    Rcpp::stop("csc_to_dgCMatrix: p length must be nrow + 1");
  }
  if (i.size() != x.size()) {
    Rcpp::stop("csc_to_dgCMatrix: i and x length must match");
  }

  Rcpp::S4 mat("dgCMatrix");
  mat.slot("p") = Rcpp::IntegerVector(p.begin(), p.end());
  mat.slot("i") = Rcpp::IntegerVector(i.begin(), i.end());
  mat.slot("x") = Rcpp::NumericVector(x.begin(), x.end());
  mat.slot("Dim") = Rcpp::IntegerVector::create(n, n);
  return mat;
}

// Numeric LDL from chol_update (x on L1 pattern, d on diagonal); pattern from
// chol_inner_list.
Rcpp::List chol_inner_numeric(const ad_pack &backend,
                              const std::vector<double> &x,
                              const std::vector<double> &d,
                              const std::vector<double> &linv_x,
                              bool include_linv) {
  const CholPattern &pat = backend.chol_pattern;

  Rcpp::List out = Rcpp::List::create(
      Rcpp::Named("L1") = csc_to_dtCMatrix_lower(pat.L1_p, pat.L1_i, x, pat.n),
      Rcpp::Named("D") = Rcpp::NumericVector(d.begin(), d.end()),
      Rcpp::Named("perm") =
          Rcpp::IntegerVector(pat.perm.begin(), pat.perm.end()),
      Rcpp::Named("perm_inv") =
          Rcpp::IntegerVector(pat.perm_inv.begin(), pat.perm_inv.end()));

  if (include_linv) {
    out["Linv"] = csc_to_dgCMatrix(pat.Linv_p, pat.Linv_i, linv_x, pat.n);
  }

  return out;
}

struct InnerOptResult {
  double fval = NA_REAL;
  double log_lik = NA_REAL;
  double neg_log_lik = NA_REAL;
  Eigen::VectorXd solution;
  Eigen::VectorXd parameters;
  Eigen::VectorXd full_parameters;
  Eigen::VectorXd grad_inner;
  Eigen::VectorXd grad_outer;
  Eigen::SparseMatrix<double> hessian_inner;
  Eigen::SparseMatrix<double> hessian_outer;
  std::vector<double> x_out;
  std::vector<double> d_out;
  std::vector<double> linv_x_out;
  std::vector<double> half_h_inv_x_out;
  std::vector<double> h_inv_x_out;
  std::vector<double> trace3;
  ProfileDerivResult profile;
  bool has_profile = false;
  int iterations = NA_INTEGER;
  MB_Status status = SUCCESS;
  double trust_radius = NA_REAL;
  bool chol_ok = false;
  double half_log_det = NA_REAL;
};

InnerOptResult inner_opt(const std::vector<double> &parameters,
                         const std::vector<double> &gamma, ad_pack &backend,
                         const TrustControl &control, bool deriv,
                         bool verbose) {
  adlaplace_require_owner_threads_assigned(backend);
  const std::vector<std::vector<size_t>> thread_groups =
      thread_groups_for_parallel_eval(backend);
  const int num_threads = static_cast<int>(thread_groups.size());
  using Tvec = Eigen::VectorXd;
  using THess = Eigen::SparseMatrix<double>;
  using TPreLLt = Eigen::SimplicialLLT<THess>;

  const std::size_t n_beta =
      static_cast<std::size_t>(backend.sizes.named("beta"));
  const std::size_t n_gamma =
      static_cast<std::size_t>(backend.sizes.named("gamma"));
  const std::size_t n_theta =
      static_cast<std::size_t>(backend.sizes.named("theta"));
  const std::size_t n_params = n_beta + n_gamma + n_theta;
  const std::size_t gamma_begin = n_beta;
  const std::size_t theta_begin = n_beta + n_gamma;

  if (parameters.size() != n_beta + n_theta) {
    Rcpp::stop("parameters has length %d but expected Nbeta+Ntheta=%d",
               static_cast<int>(parameters.size()),
               static_cast<int>(n_beta + n_theta));
  }
  if (gamma.size() != n_gamma) {
    Rcpp::stop("gamma has length %d but expected Ngamma=%d",
               static_cast<int>(gamma.size()), static_cast<int>(n_gamma));
  }
  if (n_gamma == 0) {
    Rcpp::stop("Ngamma must be > 0");
  }

  if (verbose) {
    Rcpp::Rcout << "inner_opt: threads = " << num_threads
                << ", shards = " << backend.fun.size()
                << ", params = " << n_params << " (beta = " << n_beta
                << ", gamma = " << n_gamma << ", theta = " << n_theta << ")"
                << std::endl;
  }
  if (verbose && adlaplace_debug_enabled()) {
    Rcpp::Rcout << "inner_opt: DEBUG build (thread-affinity checks active)"
                << std::endl;
  }

  Tvec gamma_start(static_cast<Eigen::Index>(n_gamma));
  Tvec solution(static_cast<Eigen::Index>(n_gamma));
  Tvec fullParams(static_cast<Eigen::Index>(n_params));
  Tvec grad(static_cast<Eigen::Index>(n_gamma));

  std::copy(gamma.begin(), gamma.end(), gamma_start.data());

  std::vector<double> params_init(n_params);
  for (std::size_t d = 0; d < n_beta; ++d) {
    params_init[d] = parameters[d];
    fullParams[static_cast<Eigen::Index>(d)] = parameters[d];
  }
  for (std::size_t d = 0; d < n_gamma; ++d) {
    params_init[gamma_begin + d] = gamma[d];
  }
  for (std::size_t d = 0; d < n_theta; ++d) {
    params_init[theta_begin + d] = parameters[n_beta + d];
    fullParams[static_cast<Eigen::Index>(theta_begin + d)] =
        parameters[n_beta + d];
  }

  adlaplace_verbose_msg(verbose, "inner_opt: constructing AD_Func_Opt (hess maps)...");
  AD_Func_Opt funObj(backend, params_init, true, num_threads, &thread_groups,
                     verbose);
  Eigen::SparseMatrix<double> H = funObj.Htemplate.cast<double>();
  adlaplace_verbose_msg(
      verbose,
      "inner_opt: AD_Func_Opt done nnz=" + std::to_string(H.nonZeros()));

  Eigen::SparseMatrix<double> Houter;
  Tvec gradOuter;

  double fval = NA_REAL;
  double radius = NA_REAL;
  int iterations = NA_INTEGER;
  MB_Status status = SUCCESS;

  InnerOptResult out;
  out.x_out.assign(backend.chol_pattern.L1_i.size(), 0.0);
  out.d_out.assign(H.rows(), 0.0);
  if (deriv) {
    out.linv_x_out.assign(backend.chol_pattern.Linv_i.size(), 0.0);
  }

  double log_det = NA_REAL;

  {
    adlaplace_verbose_msg(verbose, "inner_opt: entering CppadParallelScope...");
    CppadParallelScope parallel_scope(static_cast<std::size_t>(num_threads),
                                      verbose, &backend);

    Trust_CG_Sparse<Tvec, AD_Func_Opt, THess, TPreLLt> opt(
        funObj, gamma_start, control.rad, control.min_rad, control.tol,
        control.prec, control.report_freq, control.report_level,
        control.header_freq, control.report_precision, control.maxit,
        control.contract_factor, control.expand_factor,
        control.contract_threshold, control.expand_threshold_rad,
        control.expand_threshold_ap, control.function_scale_factor,
        control.precond_refresh_freq, control.precond_ID, control.trust_iter);

    adlaplace_verbose_msg(verbose, "inner_opt: opt.run() starting (first get_fdfh)...");
    opt.run();
    status = opt.get_current_state(solution, fval, grad, H, iterations, radius);
    adlaplace_verbose_msg(
        verbose,
        "inner_opt: opt.run() done iters=" + std::to_string(iterations) +
            " fval=" + std::to_string(fval));

    const double grad_l2_sq = grad.squaredNorm();
    // Optional restart: clamp gamma into [-x, x] and re-run. Disabled when
    // restart.gamma.clamp is non-finite (e.g. Inf).
    if (R_finite(control.restart_gamma_clamp) && grad_l2_sq > 10.0) {
      const double clamp = std::fabs(control.restart_gamma_clamp);
      if (verbose) {
        Rcpp::Rcout << "restarting with gamma clamped to [-" << clamp << ", "
                    << clamp << "]" << std::endl;
      }

      gamma_start = gamma_start.cwiseMax(-clamp).cwiseMin(clamp);
      Trust_CG_Sparse<Tvec, AD_Func_Opt, THess, TPreLLt> opt_retry(
          funObj, gamma_start, control.rad, control.min_rad, control.tol,
          control.prec, control.report_freq, control.report_level,
          control.header_freq, control.report_precision, control.maxit,
          control.contract_factor, control.expand_factor,
          control.contract_threshold, control.expand_threshold_rad,
          control.expand_threshold_ap, control.function_scale_factor,
          control.precond_refresh_freq, control.precond_ID, control.trust_iter);
      opt_retry.run();
      status = opt_retry.get_current_state(solution, fval, grad, H, iterations,
                                           radius);
    }

    H.makeCompressed();
    adlaplace_verbose_msg(verbose, "inner_opt: chol_update starting...");
    log_det =
        chol_update(H, backend.chol_pattern.perm, backend.chol_pattern.L1_p,
                    backend.chol_pattern.L1_i, out.x_out, out.d_out);
    adlaplace_verbose_msg(
        verbose, "inner_opt: chol_update done log_det=" + std::to_string(log_det));

    for (std::size_t d = 0; d < n_gamma; ++d) {
      fullParams[static_cast<Eigen::Index>(gamma_begin + d)] =
          solution[static_cast<Eigen::Index>(d)];
    }

    if (deriv) {
      // Outer grad/Hessian
      if (verbose) {
        Rcpp::Rcout << "inner_opt: trust region done; outer get_fdfh next ("
                    << num_threads << " threads)" << std::endl;
      }
      AD_Func_Opt funObjOuter(backend, params_init, false, num_threads,
                              &thread_groups, verbose);

      gradOuter = Tvec(static_cast<Eigen::Index>(n_params));
      Houter = funObjOuter.Htemplate.cast<double>();
      funObjOuter.get_fdfh(fullParams, fval, gradOuter, Houter);

      if (verbose) {
        Rcpp::Rcout << "inner_opt: outer get_fdfh done" << std::endl;
      }
      adlaplace_debug_raise_if_any("inner_opt outer get_fdfh");

      linv_update(backend.chol_pattern.L1_p, backend.chol_pattern.L1_i,
                  out.x_out, backend.chol_pattern.Linv_p,
                  backend.chol_pattern.Linv_i, out.linv_x_out);

      half_h_inv_update(
          backend.chol_pattern.Linv_p, backend.chol_pattern.Linv_i,
          out.linv_x_out, out.d_out, backend.chol_pattern.perm_inv,
          backend.chol_pattern.half_H_inv_p, backend.chol_pattern.half_H_inv_i,
          out.half_h_inv_x_out);

      h_inv_update(backend.chol_pattern.half_H_inv_p,
                   backend.chol_pattern.half_H_inv_i, out.half_h_inv_x_out,
                   backend.chol_pattern.H_inv_p, backend.chol_pattern.H_inv_i,
                   out.h_inv_x_out);

      if (verbose) {
        Rcpp::Rcout << "inner_opt: trace_hinv_t next (" << num_threads
                    << " threads)" << std::endl;
      }

      // Drain before nested parallel assign_memory()/capacity_order so workers
      // do not free thread-0 taylor buffers allocated during get_fdfh.
      adlaplace_release_shard_eval_buffers(backend);

      const std::vector<double> x_vec(fullParams.data(),
                                      fullParams.data() +
                                          static_cast<Eigen::Index>(n_params));
      out.trace3 = adlaplace_trace::trace_hinv_t_parallel(
          backend, x_vec, backend.chol_pattern.half_H_inv_p,
          backend.chol_pattern.half_H_inv_i, out.half_h_inv_x_out, n_gamma,
          backend.chol_pattern.trace_columns_p,
          backend.chol_pattern.trace_columns_i, verbose,
          out.h_inv_x_out.data(), out.h_inv_x_out.size());

      // Profile gradient assembly (same algebra as R log_lik_deriv).
      if (R_finite(log_det)) {
        out.profile = profile_deriv(
            Houter, gradOuter, backend.chol_pattern.H_inv_p,
            backend.chol_pattern.H_inv_i, out.h_inv_x_out, out.trace3, n_beta,
            n_gamma, n_theta);
        out.has_profile = true;
      }
    }
    adlaplace_verbose_msg(
        verbose, "inner_opt: leaving CppadParallelScope (teardown next)...");
  } // parallel section

  if (verbose) {
    Rcpp::Rcout << "inner_opt: parallel block ended" << std::endl;
  }
  adlaplace_debug_raise_if_any("inner_opt after parallel block");

  out.fval = fval;
  out.solution = solution;
  out.parameters = Eigen::VectorXd(static_cast<Eigen::Index>(n_beta + n_theta));
  for (std::size_t d = 0; d < n_beta; ++d) {
    out.parameters[static_cast<Eigen::Index>(d)] = parameters[d];
  }
  for (std::size_t d = 0; d < n_theta; ++d) {
    out.parameters[static_cast<Eigen::Index>(n_beta + d)] =
        parameters[n_beta + d];
  }
  out.full_parameters = fullParams;
  out.iterations = iterations;
  out.status = status;
  out.trust_radius = radius;

  if (R_finite(log_det)) {
    out.half_log_det = 0.5 * log_det;
    out.log_lik = -out.fval + static_cast<double>(n_gamma) * ONEHALFLOGTWOPI -
                  out.half_log_det;
    out.neg_log_lik = -out.log_lik;
    out.chol_ok = true;
  } else {
    Rcpp::warning(
        "inner_opt: non-positive or non-finite D; half_log_det not computed");
  }

  out.grad_inner = grad;
  out.hessian_inner = H;
  if (deriv) {
    out.grad_outer = gradOuter;
    out.hessian_outer = Houter;
  } else {
    out.grad_outer.resize(0);
    out.hessian_outer.resize(0, 0);
  }

  if (verbose) {
    Rcpp::Rcout << "inner_opt: finished" << std::endl;
  }

  return out;
}

//' Inner optimization over gamma using trust-region CG (sparse)
//'
//' Runs the inner optimization over \eqn{\gamma} using the
//' \pkg{trustOptim} sparse trust-region CG solver on a pre-built
//' \code{\link{ad_pack}} handle.
//'
//' @param parameters Numeric vector of fixed outer parameters
//'   (\code{beta}, \code{theta}; length \code{Nbeta+Ntheta}).
//' @param gamma Numeric vector of starting values for inner parameters
//'   (\code{gamma}; length \code{Ngamma}).
//' @param ad_pack \code{ad_pack} S4 object from \code{\link{ad_pack}}.
//' @param control List of trust-region control parameters (see \pkg{trustOptim}).
//' @param deriv Logical; if \code{TRUE}, also return outer gradient/Hessian
//'   pieces, Cholesky-based quantities, and the profile gradient at the
//'   inner mode.
//' @param verbose Logical; if \code{TRUE}, print thread/shard diagnostics.
//' @param return_hessians Logical; if \code{FALSE} and \code{deriv=TRUE},
//'   skip converting sparse Hessians / inverses to R Matrix objects and
//'   return only the scalar objective, \code{inner_opt$solution}, and
//'   \code{deriv$d_neg_log_lik} (optimizer hot path).
//'
//' @return A list with \code{log_lik}, \code{neg_log_lik}, \code{parameters},
//'   \code{full_parameters}, \code{inner_opt}, \code{gradient}
//'   (list \code{inner}/\code{outer}), and \code{hessian}
//'   (list \code{inner}/\code{outer}/\code{chol_inner}/\code{half_log_det};
//'   with \code{deriv=TRUE} and \code{return_hessians=TRUE} also
//'   \code{half_H_inv}, \code{H_inv}, \code{trace3}).
//'   With \code{deriv=TRUE}, also \code{deriv} (profile pieces including
//'   \code{d_neg_log_lik}) and \code{dU}.
//'   Objective and derivatives use the negative log-density convention.
//'
//' @details
//' Negates tape log-density values in C++ for minimization via
//' \pkg{trustOptim}.
//'
//' @export
// [[Rcpp::export]]
Rcpp::List inner_opt(const Rcpp::NumericVector parameters,
                     const Rcpp::NumericVector gamma, const Rcpp::S4 &ad_pack,
                     SEXP control = R_NilValue, bool deriv = false,
                     bool verbose = false, bool return_hessians = true) {
  try {
    const Rcpp::List control_list(
        Rf_isNull(control) ? Rcpp::List() : Rcpp::as<Rcpp::List>(control));
    const TrustControl control_c(control_list);
    ::ad_pack *backend = resolve_ad_pack_laplace(ad_pack);

    std::vector<double> parameters_vec(parameters.begin(), parameters.end());
    std::vector<double> gamma_vec(gamma.begin(), gamma.end());

    const InnerOptResult result = inner_opt(parameters_vec, gamma_vec, *backend,
                                            control_c, deriv, verbose);

    Rcpp::List inner_opt_out = Rcpp::List::create(
        Rcpp::Named("fval") = Rcpp::wrap(result.fval),
        Rcpp::Named("solution") = Rcpp::wrap(result.solution),
        Rcpp::Named("iterations") = Rcpp::wrap(result.iterations),
        Rcpp::Named("status") =
            Rcpp::wrap(std::string(MB_strerror(result.status))),
        Rcpp::Named("trust.radius") = Rcpp::wrap(result.trust_radius),
        Rcpp::Named("method") = Rcpp::wrap("Sparse"));

    Rcpp::List out = Rcpp::List::create(
        Rcpp::Named("log_lik") = Rcpp::wrap(result.log_lik),
        Rcpp::Named("neg_log_lik") = Rcpp::wrap(result.neg_log_lik),
        Rcpp::Named("parameters") = Rcpp::wrap(result.parameters),
        Rcpp::Named("full_parameters") = Rcpp::wrap(result.full_parameters),
        Rcpp::Named("inner_opt") = inner_opt_out);

    if (deriv && result.has_profile) {
      const ProfileDerivResult &pd = result.profile;
      Rcpp::DataFrame deriv_df = Rcpp::DataFrame::create(
          Rcpp::Named("d_det_upart") = Rcpp::wrap(pd.d_det_upart),
          Rcpp::Named("d_det_tpart") = Rcpp::wrap(pd.d_det_tpart),
          Rcpp::Named("grad_theta") = Rcpp::wrap(pd.grad_theta),
          Rcpp::Named("grad_u") = Rcpp::wrap(pd.grad_u),
          Rcpp::Named("d_det") = Rcpp::wrap(pd.d_det),
          Rcpp::Named("d_neg_log_lik") = Rcpp::wrap(pd.d_neg_log_lik),
          Rcpp::Named("d_log_lik") = Rcpp::wrap(pd.d_log_lik));
      out["deriv"] = deriv_df;
    }

    // Optimizer hot path: scalar + gamma + short gradient only.
    if (deriv && !return_hessians) {
      out["gradient"] = Rcpp::List::create(
          Rcpp::Named("inner") = R_NilValue, Rcpp::Named("outer") = R_NilValue);
      out["hessian"] = Rcpp::List::create(
          Rcpp::Named("half_log_det") = Rcpp::wrap(result.half_log_det));
      return out;
    }

    const Rcpp::List chol_inner = chol_inner_numeric(
        *backend, result.x_out, result.d_out, result.linv_x_out, deriv);

    Rcpp::List hessian_out = Rcpp::List::create(
        Rcpp::Named("inner") = eigen_to_dgCMatrix(result.hessian_inner),
        Rcpp::Named("outer") = eigen_to_dgCMatrix(result.hessian_outer),
        Rcpp::Named("chol_inner") = chol_inner,
        Rcpp::Named("half_log_det") = Rcpp::wrap(result.half_log_det));
    if (deriv && result.chol_ok && !result.half_h_inv_x_out.empty()) {
      const CholPattern &pat = backend->chol_pattern;
      hessian_out["half_H_inv"] = csc_to_dgCMatrix(
          pat.half_H_inv_p, pat.half_H_inv_i, result.half_h_inv_x_out, pat.n);
      hessian_out["H_inv"] = csc_to_dsCMatrix_upper(pat.H_inv_p, pat.H_inv_i,
                                                    result.h_inv_x_out, pat.n);
    }
    if (deriv && result.chol_ok && !result.trace3.empty()) {
      hessian_out["trace3"] = Rcpp::wrap(result.trace3);
    }

    out["gradient"] = Rcpp::List::create(
        Rcpp::Named("inner") = Rcpp::wrap(result.grad_inner),
        Rcpp::Named("outer") = Rcpp::wrap(result.grad_outer));
    out["hessian"] = hessian_out;

    if (deriv && result.has_profile) {
      out["dU"] = Rcpp::wrap(result.profile.dU);
    }

    return out;
  } catch (const Rcpp::exception &e) {
    Rcpp::stop("inner_opt failed: %s", e.what());
  } catch (...) {
    Rcpp::stop("inner_opt failed with unknown error");
  }
}

// [[Rcpp::export(name = ".fun_obj_fdfh_cpp")]]
Rcpp::List fun_obj_fdfh(const Rcpp::NumericVector &parameters,
                        const Rcpp::NumericVector &gamma,
                        const Rcpp::S4 &ad_pack, bool inner = true,
                        bool verbose = false) {

  ::ad_pack *backend = resolve_ad_pack_laplace(ad_pack);
  adlaplace_require_owner_threads_assigned(*backend);

  const std::vector<std::vector<std::size_t>> thread_groups =
      thread_groups_for_parallel_eval(*backend);
  const int num_threads = static_cast<int>(thread_groups.size());

  const std::size_t n_beta =
      static_cast<std::size_t>(backend->sizes.named("beta"));
  const std::size_t n_gamma =
      static_cast<std::size_t>(backend->sizes.named("gamma"));
  const std::size_t n_theta =
      static_cast<std::size_t>(backend->sizes.named("theta"));
  const std::size_t n_params = n_beta + n_gamma + n_theta;
  const std::size_t gamma_begin = n_beta;
  const std::size_t theta_begin = n_beta + n_gamma;

  if (static_cast<std::size_t>(parameters.size()) != n_beta + n_theta) {
    Rcpp::stop("parameters has length %d but expected Nbeta+Ntheta=%d",
               static_cast<int>(parameters.size()),
               static_cast<int>(n_beta + n_theta));
  }
  if (static_cast<std::size_t>(gamma.size()) != n_gamma) {
    Rcpp::stop("gamma has length %d but expected Ngamma=%d",
               static_cast<int>(gamma.size()), static_cast<int>(n_gamma));
  }
  if (n_gamma == 0) {
    Rcpp::stop("Ngamma must be > 0");
  }

  std::vector<double> params_init(n_params);
  Eigen::VectorXd fullParams(static_cast<Eigen::Index>(n_params));
  for (std::size_t d = 0; d < n_beta; ++d) {
    params_init[d] = parameters[static_cast<R_xlen_t>(d)];
    fullParams[static_cast<Eigen::Index>(d)] =
        parameters[static_cast<R_xlen_t>(d)];
  }
  for (std::size_t d = 0; d < n_gamma; ++d) {
    params_init[gamma_begin + d] = gamma[static_cast<R_xlen_t>(d)];
    fullParams[static_cast<Eigen::Index>(gamma_begin + d)] =
        gamma[static_cast<R_xlen_t>(d)];
  }
  for (std::size_t d = 0; d < n_theta; ++d) {
    params_init[theta_begin + d] =
        parameters[static_cast<R_xlen_t>(n_beta + d)];
    fullParams[static_cast<Eigen::Index>(theta_begin + d)] =
        parameters[static_cast<R_xlen_t>(n_beta + d)];
  }

  const std::size_t nvars_opt = inner ? n_gamma : n_params;
  if (verbose) {
    auto env_or = [](const char *name) -> const char * {
      const char *v = std::getenv(name);
      return (v && v[0] != '\0') ? v : "(unset)";
    };
    Rcpp::Rcout << "fun_obj_fdfh: " << (inner ? "inner" : "outer")
                << ", threads = " << num_threads
                << ", shards = " << backend->fun.size()
                << ", nvars = " << nvars_opt << "\n";
    Rcpp::Rcout << "  ad_pack configured_num_threads = "
                << backend->configured_num_threads << "\n";
    Rcpp::Rcout << "  thread_groups:";
    for (int t = 0; t < num_threads; ++t) {
      Rcpp::Rcout << " [" << t
                  << "]=" << thread_groups[static_cast<std::size_t>(t)].size();
    }
    Rcpp::Rcout << "\n";
    Rcpp::Rcout << "  env OMP_NUM_THREADS=" << env_or("OMP_NUM_THREADS")
                << " OMP_THREAD_LIMIT=" << env_or("OMP_THREAD_LIMIT")
                << " OMP_DYNAMIC=" << env_or("OMP_DYNAMIC")
                << " OMP_MAX_ACTIVE_LEVELS=" << env_or("OMP_MAX_ACTIVE_LEVELS")
                << " OMP_NESTED=" << env_or("OMP_NESTED")
                << " OMP_PROC_BIND=" << env_or("OMP_PROC_BIND")
                << " OMP_PLACES=" << env_or("OMP_PLACES")
                << " OMP_WAIT_POLICY=" << env_or("OMP_WAIT_POLICY") << "\n";
    Rcpp::Rcout << "  env MKL_NUM_THREADS=" << env_or("MKL_NUM_THREADS")
                << " OPENBLAS_NUM_THREADS=" << env_or("OPENBLAS_NUM_THREADS")
                << " VECLIB_MAXIMUM_THREADS="
                << env_or("VECLIB_MAXIMUM_THREADS") << "\n";
    Rcpp::Rcout << "  omp (serial, pre-CppadParallelScope):"
                << " in_parallel=" << (omp_in_parallel() != 0)
                << " thread_num=" << omp_get_thread_num() << " num_threads="
                << omp_get_num_threads() // 1 outside parallel
                << " max_threads=" << omp_get_max_threads()
                << " num_procs=" << omp_get_num_procs()
#ifdef _OPENMP
                << " dynamic=" << omp_get_dynamic()
#if _OPENMP >= 200805
                << " max_active_levels=" << omp_get_max_active_levels()
#endif
#else
                << " (_OPENMP not defined at compile time)"
#endif
                << "\n";
  }

  AD_Func_Opt funObj(*backend, params_init, inner, num_threads, &thread_groups,
                     verbose);
  Eigen::VectorXd grad_vec(static_cast<Eigen::Index>(nvars_opt));
  Eigen::SparseMatrix<double> H = funObj.Htemplate.cast<double>();
  double fval = 0.0;

  {
    CppadParallelScope parallel_scope(static_cast<std::size_t>(num_threads),
                                      verbose, backend);
    if (inner) {
      Eigen::VectorXd x_gamma(static_cast<Eigen::Index>(n_gamma));
      for (std::size_t d = 0; d < n_gamma; ++d) {
        x_gamma[static_cast<Eigen::Index>(d)] = gamma[static_cast<R_xlen_t>(d)];
      }
      funObj.get_fdfh(x_gamma, fval, grad_vec, H);
    } else {
      funObj.get_fdfh(fullParams, fval, grad_vec, H);
    }
    H.makeCompressed();
  }

  if (verbose) {
    Rcpp::Rcout << "fun_obj_fdfh: done\n";
  }

  return Rcpp::List::create(Rcpp::Named("f") = fval,
                            Rcpp::Named("grad") = Rcpp::wrap(grad_vec),
                            Rcpp::Named("hessian") = eigen_to_dgCMatrix(H));
}
