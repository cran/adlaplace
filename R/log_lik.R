#' Outer objective and gradient wrappers
#'
#' Convenience wrappers around \code{\link{log_lik_laplace}} for use with outer optimizers
#' that expect \code{fn} / \code{gr} callbacks. Both functions solve (or warm-start)
#' the inner problem over \code{gamma} via \code{\link{log_lik_laplace}} and update a mutable
#' \code{cache} environment with the latest inner solution.
#'
#' L-BFGS-B requests \code{fn} then \code{gr} at the same parameter vector. Both wrappers
#' share one \code{inner_opt(..., deriv = TRUE, return_hessians = FALSE)} evaluation
#' at that point: the objective is free once the profile gradient has been computed.
#' Cached numerics are stored in \code{cache$fg_x}, \code{cache$neg_log_lik}, and
#' \code{cache$d_neg_log_lik} and reused until \code{x} changes.
#'
#' \describe{
#' \item{\code{outer_fn()}}{Returns the scalar objective negative log likelihood.}
#' \item{\code{outer_gr()}}{Returns the gradient.}
#' }
#'
#' @param x Numeric outer parameter vector \code{c(beta, theta)}.
#' @param config Configuration list passed to \code{\link{log_lik_laplace}}.
#' @param ad_pack \code{ad_pack} object from \code{\link{ad_pack}}.
#' @param ... Additional arguments forwarded to \code{\link{log_lik_laplace}}
#'   (for example \code{data} or \code{package}).
#' @param control_inner A list of control options forwarded to the \code{control}
#'   argument of \code{\link{log_lik_laplace}} for the inner optimization.
#' @param cache An \code{\link[base]{environment}} used to warm-start and store the inner
#'   \code{gamma} solution. If \code{cache$gamma} is missing or the wrong length,
#'   it is initialized from \code{config$gamma} (if present) or zeros of length
#'   \code{ad_pack@sizes["gamma"]}. Both functions update \code{cache$gamma} after
#'   each evaluation, and share \code{cache$fg_x} / \code{cache$neg_log_lik} /
#'   \code{cache$d_neg_log_lik} for the latest combined objective/gradient evaluation.
#'   For multi-pack / modular objectives, use one \code{cache} environment per
#'   \code{ad_pack}; \code{cache$gamma} is that pack's inner warm start.
#'
#' @return
#' \itemize{
#' \item \code{outer_fn}: a numeric scalar \code{neg_log_lik}.
#' \item \code{outer_gr}: gradient of \code{neg_log_lik} (same sign as \code{outer_fn}).
#' }
#'
#' @seealso \code{\link{log_lik_laplace}}
#'
#' @examples
#' set.seed(1)
#' n <- 80
#' g <- factor(sample(4, n, replace = TRUE))
#' x <- rbinom(n, 1, 0.5)
#' eta <- 0.5 + 0.4 * x + rnorm(4, sd = 0.3)[g]
#' nb_sd <- 0.3
#' z <- rgamma(n, nb_sd^(-2), nb_sd^(-2))
#' dat <- data.frame(y = rpois(n, exp(eta) * z), x = x, g = g)
#' md <- model_data(
#'   nbinom(y, lower = 1e-9, init = 0.15) ~ x + iid(g, init = 0.3),
#'   data = dat
#' )
#' config <- list(
#'   transform_theta = TRUE,
#'   obs_groups = obs_groups(md$term_data$A, num_shards = 5),
#'   verbose = FALSE
#' )
#' ap <- ad_pack(md, config)
#' cache <- new.env(parent = emptyenv())
#' x0 <- md$term_data$info$parameters$init
#' outer_fn(x = x0, config = config, cache = cache, ad_pack = ap)
#' outer_gr(x = x0, config = config, cache = cache, ad_pack = ap)
#'
#' @name outer_optim_wrappers
#' @rdname outer_optim_wrappers
#' @export
outer_fn <- function(
  x, config, cache, ad_pack, control_inner = list(), ...
) {
  assign("last_par_fn", x, cache)
  result <- .outer_fg(
    x = x,
    config = config,
    cache = cache,
    ad_pack = ad_pack,
    control_inner = control_inner,
    ...
  )
  result$neg_log_lik
}

#' Laplace-approximate log-likelihood (positive convention)
#'
#' Thin wrapper around \code{\link{log_lik_laplace}} that returns the
#' \emph{positive} profile log-likelihood and its gradient with respect to the
#' outer parameters \code{c(beta, theta)}. Prefer this over reading
#' \code{inner_opt(...)$deriv$d_neg_log_lik} when building custom outer
#' objectives (e.g. modular / multi-pack likelihoods).
#'
#' @param ad_pack An \code{ad_pack} object.
#' @param par Numeric outer parameter vector \code{c(beta, theta)} on the
#'   internal (log-theta) scale.
#' @param gamma Numeric vector of random-effect modes used as the inner start.
#' @param config Configuration list (may be empty).
#' @param control_inner Control list for the inner trust-region solve.
#' @param deriv Logical; if \code{TRUE} (default), return the profile gradient.
#'
#' @return A list with components
#' \describe{
#'   \item{log_lik}{Scalar Laplace-approximate marginal log-likelihood.}
#'   \item{gradient}{Numeric vector \eqn{\nabla_\psi \ell}, or \code{NULL} when
#'     \code{deriv = FALSE}.}
#'   \item{gamma}{Inner modes at the optimum.}
#' }
#'
#' @seealso \code{\link{log_lik_laplace}}, \code{\link{inner_opt}},
#'   \code{\link{outer_fn}}
#' @export
laplace_loglik <- function(
  ad_pack,
  par,
  gamma,
  config = list(),
  control_inner = list(),
  deriv = TRUE
) {
  if (!methods::is(ad_pack, "ad_pack")) {
    stop("ad_pack must be an ad_pack object", call. = FALSE)
  }
  res <- log_lik_laplace(
    x = as.numeric(par),
    gamma = gamma,
    ad_pack = ad_pack,
    config = config,
    control = control_inner,
    deriv = isTRUE(deriv),
    return_hessians = FALSE
  )
  gradient <- if (isTRUE(deriv)) {
    as.numeric(res$deriv$d_log_lik)
  } else {
    NULL
  }
  list(
    log_lik = as.numeric(res$log_lik),
    gradient = gradient,
    gamma = as.numeric(res$inner_opt$solution)
  )
}

#' @rdname outer_optim_wrappers
#' @export
outer_gr <- function(
  x, config, cache, ad_pack, control_inner = list(), ...
) {
  assign("last_par_gr", x, cache)
  result <- .outer_fg(
    x = x,
    config = config,
    cache = cache,
    ad_pack = ad_pack,
    control_inner = control_inner,
    ...
  )
  result$d_neg_log_lik
}

#' Combined outer Laplace evaluation with per-\code{x} cache
#'
#' Calls \code{inner_opt(..., deriv = TRUE, return_hessians = FALSE)} so the
#' optimizer path returns only a scalar and short gradient (no Matrix objects).
#' Cached fields: \code{gamma}, \code{fg_x}, \code{neg_log_lik},
#' \code{d_neg_log_lik}.
#'
#' Warn when the inner trust-region solve did not report success.
#'
#' @noRd
warn_if_inner_status <- function(result) {
  status <- result[["inner_opt"]][["status"]]
  if (is.null(status) || !length(status)) {
    return(invisible(NULL))
  }
  status <- as.character(status[[1L]])
  if (!identical(status, "Success")) {
    warning(warningCondition(
      paste0(
        "inner_opt status is ", status,
        "; the Laplace approximation used this inner solution anyway"
      ),
      class = "adlaplace_inner_status",
      call = NULL
    ))
  }
  invisible(status)
}

#' @noRd
.outer_fg <- function(
  x, config, cache, ad_pack, control_inner = list(), ...
) {
  x_num <- as.numeric(x)
  if (!is.null(cache$fg_x) &&
    length(cache$fg_x) == length(x_num) &&
    isTRUE(all(cache$fg_x == x_num)) &&
    !is.null(cache$neg_log_lik) &&
    !is.null(cache$d_neg_log_lik)) {
    return(list(
      neg_log_lik = cache$neg_log_lik,
      d_neg_log_lik = cache$d_neg_log_lik,
      inner_opt = list(solution = cache$gamma)
    ))
  }

  num_gamma <- as.integer(ad_pack@sizes["gamma"])
  cache$gamma <- resolve_gamma_start(config, cache, num_gamma)

  if (isTRUE(config$verbose)) {
    message("outer_fg: calling inner_opt(deriv=TRUE, return_hessians=FALSE)...")
    utils::flush.console()
  }
  result <- adlaplace::inner_opt(
    parameters = x_num,
    gamma = cache$gamma,
    control = control_inner,
    ad_pack = ad_pack,
    deriv = TRUE,
    return_hessians = FALSE,
    verbose = isTRUE(config[["verbose"]])
  )
  warn_if_inner_status(result)
  if (isTRUE(config$verbose)) {
    message(
      "outer_fg: done neg_log_lik=", format(result$neg_log_lik),
      " length(d_neg_log_lik)=", length(result$deriv$d_neg_log_lik)
    )
    utils::flush.console()
  }

  d_neg <- as.numeric(result$deriv$d_neg_log_lik)
  assign("gamma", result$inner_opt$solution, cache)
  assign("fg_x", x_num, cache)
  assign("neg_log_lik", result$neg_log_lik, cache)
  assign("d_neg_log_lik", d_neg, cache)
  # Clear any leftover full Laplace result so fit.R does not reuse a slim cache.
  if (exists("fg_result", envir = cache, inherits = FALSE)) {
    rm("fg_result", envir = cache)
  }
  if (is.null(cache$fg_evals)) {
    cache$fg_evals <- 0L
  }
  cache$fg_evals <- cache$fg_evals + 1L
  list(
    neg_log_lik = result$neg_log_lik,
    d_neg_log_lik = d_neg,
    inner_opt = list(solution = result$inner_opt$solution)
  )
}

#' Log-likelihood with inner Laplace optimization
#'
#' Evaluates the (profiled) log-likelihood for a hierarchical model by solving an
#' inner optimization problem for the latent vector \code{gamma} (e.g. random
#' effects) given outer parameters \code{x} (typically \code{beta} and \code{theta}).
#' The function delegates the inner optimization to \code{inner_opt()} from the
#' selected backend package (by default \pkg{adlaplace}).
#'
#' @param x Numeric vector of outer parameters (\code{beta}, then \code{theta}).
#'   Length must equal \code{num_beta + num_theta} from \code{ad_pack@sizes}.
#' @param config A list of backend options (\code{verbose}, \code{package}, etc.).
#'   When \code{ad_pack} is missing, must also include \code{beta}, \code{gamma},
#'   and \code{theta} so \code{ad_pack()} can be built from \code{data}.
#' @param gamma Optional numeric vector of starting values for the inner
#'   parameter \code{gamma}. If missing, uses \code{config$gamma} when present,
#'   otherwise a zero vector of length \code{ad_pack@sizes["gamma"]}.
#' @param control List of control parameters passed *as-is* to the backend inner
#'   optimizer (e.g., \code{report.level}, \code{report.freq}). See backend
#'   documentation (e.g., \pkg{trustOptim}) for supported options.
#' @param ad_pack Optional \code{ad_pack} object from \code{\link{ad_pack}}.
#'   This is a single backend handle (no separate inner/outer handles). If
#'   missing, it will be constructed automatically using \code{data}.
#' @param data Optional data list used to build \code{ad_pack} when \code{ad_pack}
#'   is not supplied.
#' @param package Character scalar naming the backend package to use for
#'   \code{ad_pack()} and \code{inner_opt()}. Defaults to \code{"adlaplace"}.
#'   Other backends must export a compatible \code{ad_pack()} builder and \code{inner_opt()}.
#' @param deriv Logical scalar. If \code{TRUE}, include derivative quantities in
#'   the output (gradient, intermediate derivatives).
#' @param return_hessians Logical. Forwarded to \code{\link{inner_opt}}. When
#'   \code{FALSE} with \code{deriv=TRUE}, skip Matrix Hessian objects (optimizer
#'   path). Default \code{TRUE} for diagnostic / post-fit evaluations.
#'
#' @details
#' The default \pkg{adlaplace} backend uses a single AD handle. This function
#' passes that handle to \code{inner_opt(..., ad_pack = ad_pack)}. Parameter block
#' sizes are read from \code{ad_pack@sizes}; \code{config} is not required to
#' contain \code{beta}, \code{gamma}, or \code{theta} when \code{ad_pack} is
#' supplied.
#'
#' When \code{deriv=FALSE}, the return value mirrors \code{\link{inner_opt}()}
#' with \code{inner_opt}, top-level \code{gradient}, and \code{hessian}.
#' When \code{deriv=TRUE}, the profile gradient is assembled in C++ inside
#' \code{inner_opt}; the same list is augmented with \code{deriv} and
#' profile-derivative pieces in \code{gradient$outer}.
#'
#' @return A list. With \code{deriv=FALSE}, same structure as
#' \code{\link{inner_opt}} (without \code{parameters}). With \code{deriv=TRUE},
#' additionally:
#' \describe{
#'   \item{deriv}{Data frame of profile derivative pieces.}
#'   \item{gradient$outer$dU, gradient$outer$trace3}{Intermediate profile objects.}
#' }
#'
#' @note The Laplace approximation assumes the inner Hessian is positive definite
#'   at the optimum. If \code{\link{inner_opt}} fails to converge or returns a
#'   non-invertible Hessian, \code{log_lik_laplace()} issues a warning or error.
#'
#' @seealso
#' \code{\link{ad_pack}}, \code{\link{inner_opt}}
#'
#' @export
log_lik_laplace <- function(
  x,
  config = list(),
  gamma,
  control = list(report.level = 4, report.freq = 1),
  ad_pack,
  data,
  package = c(config[["package"]], "adlaplace")[1L],
  deriv = FALSE,
  return_hessians = TRUE
) {
  if (missing(ad_pack)) {
    if (missing(data)) {
      stop("at least one of data and ad_pack must be supplied", call. = FALSE)
    }
    ad_pack <- adlaplace::ad_pack(data, config)
  }

  if (!is(ad_pack, "ad_pack")) {
    stop("ad_pack must be an ad_pack object", call. = FALSE)
  }
  sz <- ad_pack@sizes
  if (length(sz) < 3L || any(is.na(sz[c("beta", "gamma", "theta")]))) {
    stop(
      "ad_pack@sizes must contain finite beta, gamma, and theta",
      call. = FALSE
    )
  }
  num_beta <- as.integer(sz["beta"])
  num_theta <- as.integer(sz["theta"])
  num_gamma <- as.integer(sz["gamma"])

  n_outer <- length(x)
  n_outer_expected <- num_beta + num_theta
  if (n_outer != n_outer_expected) {
    stop(
      "length(x) (", n_outer, ") must equal num_beta + num_theta (",
      n_outer_expected, ")",
      call. = FALSE
    )
  }

  gamma_use <- if (!missing(gamma)) {
    gamma
  } else if (!is.null(config[["gamma"]]) && length(config[["gamma"]]) > 0L) {
    config[["gamma"]]
  } else {
    rep(0, num_gamma)
  }
  if (length(gamma_use) != num_gamma) {
    stop(
      "length(gamma) (", length(gamma_use), ") must equal num_gamma (",
      num_gamma, ")",
      call. = FALSE
    )
  }

  result_inner <- inner_opt(
    parameters = x,
    gamma = gamma_use,
    control = control,
    ad_pack = ad_pack,
    deriv = deriv,
    return_hessians = return_hessians,
    verbose = isTRUE(config[["verbose"]])
  )
  warn_if_inner_status(result_inner)
  result <- restructure_laplace_result(result_inner, control_inner = control)

  if (!deriv) {
    return(result)
  }

  # Profile gradient is assembled in C++ inside inner_opt.
  if (is.null(result_inner$deriv)) {
    stop(
      "inner_opt(..., deriv = TRUE) did not return deriv; rebuild adlaplace",
      call. = FALSE
    )
  }
  result$deriv <- result_inner$deriv

  if (isTRUE(return_hessians)) {
    if (!is.null(result_inner$dU)) {
      result$gradient$outer$dU <- result_inner$dU
    }
    if (!is.null(result$hessian$trace3)) {
      result$gradient$outer$trace3 <- result$hessian$trace3
      result$hessian$trace3 <- NULL
    }
  }

  result
}

#' @noRd
restructure_laplace_result <- function(result_inner, control_inner = NULL) {
  keep <- setdiff(
    names(result_inner),
    c("opt", "inner_opt", "gradient", "hessian", "deriv", "dU")
  )
  out <- result_inner[keep]
  inner <- result_inner$inner_opt
  if (is.null(inner) && !is.null(result_inner$opt)) {
    inner <- result_inner$opt
  }
  out$inner_opt <- inner
  if (!is.null(control_inner)) {
    out$inner_opt$control_inner <- control_inner
  }
  out$gradient <- list(
    inner = result_inner$gradient$inner,
    outer = list(grad = result_inner$gradient$outer)
  )
  out$hessian <- result_inner$hessian
  out
}

log_lik_deriv <- function(
  full_parameters,
  hessian_pack,
  grad,
  ad_pack,
  verbose = FALSE
) {
  if (!is(ad_pack, "ad_pack")) {
    stop("ad_pack must be an ad_pack object", call. = FALSE)
  }
  sz <- ad_pack@sizes
  if (length(sz) < 3L || any(is.na(sz[c("beta", "gamma", "theta")]))) {
    stop(
      "ad_pack@sizes must contain finite beta, gamma, and theta",
      call. = FALSE
    )
  }
  num_beta <- as.integer(sz["beta"])
  num_theta <- as.integer(sz["theta"])
  num_gamma <- as.integer(sz["gamma"])
  n_params <- num_beta + num_gamma + num_theta
  if (length(grad) != n_params) {
    stop(
      "length(grad) (", length(grad), ") must equal num_beta + num_gamma + ",
      "num_theta (", n_params, ")",
      call. = FALSE
    )
  }

  ldl <- as_ldl_list(hessian_pack$chol_inner)
  if (is.null(ldl$Linv)) {
    stop(
      "chol_inner$Linv required for log_lik_deriv (inner_opt with deriv = TRUE)",
      call. = FALSE
    )
  }
  half_H_inv <- hessian_pack$half_H_inv
  if (is.null(half_H_inv)) {
    half_H_inv <- half_H_inv_from_ldl(ldl)
  }
  H_inv <- hessian_pack$H_inv
  if (is.null(H_inv)) {
    H_inv <- Matrix::tcrossprod(half_H_inv)
  }
  hessian_inv <- list(
    half_H_inv = half_H_inv,
    H_inv = H_inv
  )

  seq_gamma1 <- seq.int(num_beta + 1L, length.out = num_gamma)

  the_trace <- hessian_pack$trace3
  if (is.null(the_trace)) {
    stop(
      "hessian$trace3 is missing; run inner_opt(..., deriv = TRUE) with ",
      "current adlaplace (trace is computed inside inner_opt)",
      call. = FALSE
    )
  }
  if (length(the_trace) != n_params) {
    stop(
      "length(hessian$trace3) (", length(the_trace), ") must equal num_beta + ",
      "num_gamma + num_theta (", n_params, ")",
      call. = FALSE
    )
  }

  dU <- -hessian_inv$H_inv %*% hessian_pack$outer[seq_gamma1, -seq_gamma1]

  result <- list(
    extra = c(
      list(dU = dU, trace3 = the_trace),
      hessian_inv
    )
  )

  result$deriv <- data.frame(
    d_det_upart = -as.vector(the_trace[seq_gamma1] %*% dU),
    d_det_tpart = -the_trace[-seq_gamma1]
  )
  result$deriv$grad_theta <- -grad[-seq_gamma1]
  result$deriv$grad_u <- as.vector(-grad[seq_gamma1] %*% dU)
  result$deriv$d_det <- result$deriv$d_det_upart + result$deriv$d_det_tpart
  result$deriv$d_neg_log_lik <-
    -result$deriv$grad_theta +
    result$deriv$d_det - result$deriv$grad_u
  result$deriv$d_log_lik <- -result$deriv$d_neg_log_lik

  result
}

#' Extract inner inverse Cholesky factor from Laplace output
#'
#' Returns \eqn{H^{-1/2}} for the inner (random-effect) Hessian from the output
#' of \code{\link{log_lik_laplace}}. Accepts the flat \code{half_H_inv} matrix
#' when \code{deriv = TRUE}, or reconstructs it from \code{hessian$chol_inner}.
#'
#' @param laplace Output of \code{log_lik_laplace(..., deriv = TRUE)} (or
#'   \code{inner_opt(..., deriv = TRUE)} with the same \code{hessian} layout).
#'
#' @return A sparse or dense matrix \eqn{H^{-1/2}} with \code{nrow = Ngamma}.
#'
#' @seealso \code{\link{log_lik_laplace}}, \code{\link{rmvnldl}}
#' @export
laplace_half_H_inv <- function(laplace) {
  if (!is.list(laplace)) {
    stop("laplace must be a list (output of log_lik_laplace)", call. = FALSE)
  }

  half_h <- laplace$hessian$half_H_inv
  if (is.null(half_h) && is.list(laplace$extra)) {
    half_h <- laplace$extra$half_H_inv
    if (is.null(half_h)) {
      half_h <- laplace$extra$extra$half_H_inv
    }
  }
  if (!is.null(half_h)) {
    return(half_h)
  }

  chol_inner <- NULL
  if (is.list(laplace$hessian)) {
    chol_inner <- laplace$hessian$chol_inner
  }
  if (is.null(chol_inner) && is.list(laplace$extra) && is.list(laplace$extra$hessian)) {
    chol_inner <- laplace$extra$hessian$chol_inner
  }
  if (is.null(chol_inner)) {
    stop(
      "laplace must contain hessian$half_H_inv (from log_lik_laplace(..., deriv = TRUE)) ",
      "or hessian$chol_inner",
      call. = FALSE
    )
  }

  ldl <- as_ldl_list(chol_inner)
  half_H_inv_from_ldl(ldl)
}
