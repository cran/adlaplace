#' @noRd
as_ldl_list <- function(chol_prec) {
  if (inherits(chol_prec, "CHMfactor") || inherits(chol_prec, "Cholesky")) {
    ex <- Matrix::expand2(chol_prec)
    return(list(
      L1 = ex$L1,
      D = as.numeric(ex$D@x),
      perm = as.integer(chol_prec@perm),
      perm_inv = as.integer(order(chol_prec@perm) - 1L)
    ))
  }
  if (is.list(chol_prec) && all(c("L1", "D", "perm") %in% names(chol_prec))) {
    perm <- as.integer(chol_prec$perm)
    perm_inv <- if ("perm_inv" %in% names(chol_prec)) {
      as.integer(chol_prec$perm_inv)
    } else {
      as.integer(order(perm) - 1L)
    }
    out <- list(
      L1 = chol_prec$L1,
      D = as.numeric(chol_prec$D),
      perm = perm,
      perm_inv = perm_inv
    )
    if ("Linv" %in% names(chol_prec)) {
      out$Linv <- chol_prec$Linv
    }
    return(out)
  }
  stop(
    "chol_prec must be a CHMfactor, Cholesky (LDL), or a list with L1, D, perm and perm_inv ",
    "(as in log_lik_laplace()$hessian$chol_inner or inner_opt()$hessian$chol_inner)"
  )
}

#' @noRd
half_H_inv_from_ldl <- function(ldl) {
  p <- length(ldl$D)
  Linv <- if (!is.null(ldl$Linv)) ldl$Linv else Matrix::solve(ldl$L1)
  halfDinv <- Matrix::Diagonal(p, ldl$D^(-0.5))

  half_H_inv_perm <- Matrix::crossprod(Linv, halfDinv)
  half_H_inv <- half_H_inv_perm[ldl$perm_inv + 1L, ]

  # H_inv = tcrossprod(half_H_inv)

  half_H_inv
}

#' Simulate from a multivariate normal using LDL of the precision matrix
#'
#' Draws \eqn{n} samples from \eqn{N(\mu, H^{-1})} where \eqn{H} is the
#' precision matrix, given its LDL factorization (unit lower \code{L1},
#' diagonal \code{D}, and permutation \code{perm}).
#'
#' @param n Number of samples.
#' @param mean Mean vector \eqn{\mu} (length \code{length(D)}).
#'   When \code{mean} is named, those names are attached to the draws.
#'   Ignored when \code{fit} is supplied.
#' @param chol_prec Either a \code{CHMfactor}, \code{Cholesky}, or \code{dCHMsimpl}
#'   from \code{Matrix::Cholesky(..., LDL = TRUE)}, or a list like
#'   \code{res$hessian$chol_inner} with components \code{L1}, \code{D},
#'   and \code{perm}. Ignored when \code{fit} is supplied.
#' @param fit Optional \code{"adlaplace_fit"} object. When supplied, overrides
#'   \code{mean} and \code{chol_prec} with \code{fit$gamma} and
#'   \code{fit$details$hessian$chol_inner}.
#'
#' @return An \code{n}-by-\code{p} matrix; each row is one draw. When
#'   \code{n = 1}, a length-\code{p} vector. Column / element names come from
#'   \code{names(mean)} when present and of length \code{p}.
#'
#' @details
#' Uses \eqn{H^{-1/2} = P^\top L^{-1} D^{-1/2}} (same as \code{reformat_chol()}
#' in \code{log_lik_deriv}), and \code{rnorm} only:
#' \eqn{x = \mu + H^{-1/2} z}, \eqn{z \sim N(0, I)}.
#'
#' @seealso \code{\link[Matrix]{Cholesky}}, \code{\link{log_lik_laplace}},
#'   \code{\link{adlaplace}}
#' @export
rmvnldl <- function(n, mean = 0, chol_prec, fit) {
  if (!missing(fit)) {
    mean <- fit$gamma
    chol_prec <- fit$details$hessian$chol_inner
    if (is.null(chol_prec) && is.list(fit$details$extra)) {
      chol_prec <- fit$details$extra$hessian$chol_inner
    }
    if (is.null(mean) || is.null(chol_prec)) {
      stop(
        "fit must provide $gamma and $details$hessian$chol_inner",
        call. = FALSE
      )
    }
  } else if (missing(chol_prec)) {
    stop("either fit or chol_prec must be supplied", call. = FALSE)
  }

  ldl <- as_ldl_list(chol_prec)
  half_H_inv <- half_H_inv_from_ldl(ldl)

  p <- length(ldl$D)
  n <- as.integer(n)
  nm <- names(mean)
  mu <- rep_len(as.numeric(mean), p)

  z <- matrix(stats::rnorm(p * n), nrow = p, ncol = n)
  draws <- as.matrix(mu + half_H_inv %*% z)

  if (n == 1L) {
    out <- as.vector(draws)
    if (length(nm) == p) {
      names(out) <- nm
    }
    out
  } else {
    out <- t(draws)
    if (length(nm) == p) {
      colnames(out) <- nm
    }
    out
  }
}

#' @noRd
log_mvn_density <- function(x, center, chol_cov) {
  delta <- x - center
  quad <- sum((backsolve(chol_cov, delta, transpose = TRUE))^2)
  log_det <- 2 * sum(log(diag(chol_cov)))
  ngamma <- length(center)
  -0.5 * (ngamma * log(2 * pi) + log_det + quad)
}

#' @noRd
full_parameters_from_gamma <- function(gamma_row, parameters, n_beta, n_theta) {
  c(
    parameters[seq_len(n_beta)],
    gamma_row,
    parameters[n_beta + seq_len(n_theta)]
  )
}

#' Posterior mean of random effects via multivariate quadrature
#'
#' Approximates \eqn{E[\gamma \mid y]} by importance-weighted quadrature over
#' \eqn{\gamma}, using a Gaussian proposal \eqn{N(\hat\gamma, H^{-1})} (Laplace
#' covariance) and correcting with \code{\link{joint_log_dens}}.
#'
#' @param parameters Outer \eqn{(\beta, \theta)} at the Laplace fit.
#' @param mode Inner mode \eqn{\hat\gamma} (e.g. \code{laplace$inner_opt$solution}).
#' @param cov \eqn{H^{-1}} for the inner block (e.g. \code{laplace$hessian$H_inv}).
#' @param ad_pack \code{ad_pack} object for \code{joint_log_dens}.
#' @param n Quadrature level passed to \pkg{mvQuad} \code{createNIGrid}.
#' @param type mvQuad rule (default \code{"nHN"}).
#' @param ndConstruction mvQuad grid construction (default \code{"sparse"}).
#' @param dec.type Cholesky decomposition flag for \code{mvQuad::rescale}.
#'
#' @return Numeric vector of length \code{n_gamma}.
#'
#' @details
#' Builds a sparse \code{nHN} grid with \pkg{mvQuad}, rescales to the Laplace
#' proposal, then forms
#' \eqn{\sum_i w_i \gamma_i \exp(\ell(\gamma_i) - \log q(\gamma_i)) /
#' \sum_i w_i \exp(\ell(\gamma_i) - \log q(\gamma_i))}
#' with log-sum-exp stabilization.
#'
#' @seealso \code{\link{log_lik_laplace}}, \code{\link{inner_opt}},
#'   \code{\link{joint_log_dens}}, \code{\link{rmvnldl}}
#' @export
mean_mvquad <- function(parameters,
                        mode,
                        cov,
                        ad_pack,
                        n = 3L,
                        type = "nHN",
                        ndConstruction = "sparse",
                        dec.type = 2L) {
  if (!requireNamespace("mvQuad", quietly = TRUE)) {
    stop("Package 'mvQuad' is required for mean_mvquad()", call. = FALSE)
  }
  if (!is(ad_pack, "ad_pack")) {
    stop("ad_pack must be an ad_pack object", call. = FALSE)
  }
  if (missing(parameters) || is.null(parameters)) {
    stop("parameters is required (outer beta and theta)", call. = FALSE)
  }
  if (missing(mode) || is.null(mode)) {
    stop("mode is required (inner gamma at Laplace mode)", call. = FALSE)
  }
  if (missing(cov) || is.null(cov)) {
    stop("cov is required (inner Laplace covariance H_inv)", call. = FALSE)
  }

  sz <- ad_pack@sizes
  n_beta <- as.integer(sz["beta"])
  n_gamma <- as.integer(sz["gamma"])
  n_theta <- as.integer(sz["theta"])

  mode <- as.numeric(mode)
  parameters <- as.numeric(parameters)

  if (length(mode) != n_gamma) {
    stop(
      "length(mode) (", length(mode), ") must equal ad_pack@sizes['gamma'] (",
      n_gamma, ")",
      call. = FALSE
    )
  }
  if (length(parameters) != n_beta + n_theta) {
    stop(
      "length(parameters) (", length(parameters),
      ") must equal n_beta + n_theta (", n_beta + n_theta, ")",
      call. = FALSE
    )
  }

  # H_inv is a symmetric CSC matrix but often stores only the lower triangle;
  # as.matrix() would zero the upper part and fail mvQuad::rescale's symmetry check.
  cov_mat <- if (inherits(cov, "Matrix")) {
    as.matrix(Matrix::forceSymmetric(cov, uplo = "L"))
  } else {
    cm <- as.matrix(cov)
    if (!isSymmetric(cm, check.attributes = FALSE)) {
      cm <- (cm + t(cm)) / 2
    }
    cm
  }
  if (!is.matrix(cov_mat) || nrow(cov_mat) != n_gamma || ncol(cov_mat) != n_gamma) {
    stop(
      "cov must be an n_gamma x n_gamma matrix (n_gamma = ", n_gamma, ")",
      call. = FALSE
    )
  }

  n <- as.integer(n)
  if (n < 1L) {
    stop("n (quadrature level) must be >= 1", call. = FALSE)
  }

  # mvQuad uses data.table::rbindlist with mismatched column names for sparse
  # Smolyak grids; harmless messages only (see data.table v1.12.2 news item 5).
  grid <- suppressMessages({
    g <- mvQuad::createNIGrid(
      dim = n_gamma,
      type = type,
      level = n,
      ndConstruction = ndConstruction
    )
    mvQuad::rescale(g, m = mode, C = cov_mat, dec.type = dec.type)
    g
  })

  nodes <- mvQuad::getNodes(grid)
  weights <- as.numeric(mvQuad::getWeights(grid))
  if (is.vector(nodes)) {
    nodes <- matrix(nodes, nrow = 1L)
  }
  n_pts <- nrow(nodes)
  if (length(weights) != n_pts) {
    stop("mvQuad grid: length(weights) must equal nrow(nodes)", call. = FALSE)
  }

  chol_cov <- chol(cov_mat)
  log_ratio <- vapply(
    seq_len(n_pts),
    function(i) {
      gamma_row <- nodes[i, , drop = TRUE]
      full <- full_parameters_from_gamma(
        gamma_row,
        parameters,
        n_beta,
        n_theta
      )
      log_p <- joint_log_dens(ad_pack, full, negative = FALSE)
      log_q <- log_mvn_density(gamma_row, mode, chol_cov)
      log_p - log_q
    },
    numeric(1)
  )

  log_ratio <- log_ratio - max(log_ratio)
  rw <- weights * exp(log_ratio)
  denom <- sum(rw)
  if (!is.finite(denom) || denom <= 0) {
    stop("mean_mvquad: non-positive or non-finite importance weights", call. = FALSE)
  }

  colSums(nodes * rw) / denom
}
