#' @include fit.R sim_fit.R
#' @importFrom stats nobs coef vcov confint fitted logLik predict qnorm
NULL

#' Methods for fitted adlaplace models
#'
#' Standard accessor methods for objects of class \code{"adlaplace_fit"}
#' returned by \code{\link{adlaplace}()}. Estimates are stored on the
#' optimization scale (log scale for parameters with
#' \code{par_info$log = TRUE}, e.g. standard deviations); methods with a
#' \code{transform} argument report the natural scale by exponentiating and, for
#' standard errors, applying the delta method.
#'
#' @param object,x An \code{adlaplace_fit} object.
#' @param transform Logical; report transformed (natural-scale) values.
#' @param level Confidence level.
#' @param parm Optional subset of parameter labels or indices.
#' @param newdata Data frame of prediction covariates.
#' @param n Number of random-effect draws for \code{predict}.
#' @param k Number of evaluation points per smooth term in \code{plot}.
#' @param draws Number of simulation draws per smooth term in \code{plot}.
#' @param digits Number of significant digits to print.
#' @param ... Passed to downstream methods.
#'
#' @return
#' \describe{
#'   \item{\code{coef}}{Named numeric vector of outer parameter estimates.
#'     With \code{transform = TRUE} (the default), parameters stored on the
#'     log scale (for example standard deviations) are exponentiated to the
#'     natural scale.}
#'   \item{\code{vcov}}{Variance-covariance matrix of the outer parameters on
#'     the optimization scale. With \code{transform = TRUE}, rows and columns
#'     for log-scale parameters are mapped to the natural scale by the delta
#'     method.}
#'   \item{\code{logLik}}{An object of class \code{"logLik"}: the
#'     Laplace-approximate marginal log-likelihood, with attributes \code{df}
#'     (number of outer parameters) and \code{nobs}.}
#'   \item{\code{nobs}}{Integer number of observations.}
#'   \item{\code{confint}}{Numeric matrix of Wald intervals, one row per outer
#'     parameter. With \code{transform = TRUE}, endpoints for log-scale
#'     parameters are exponentiated to the natural scale.}
#'   \item{\code{fitted}}{Numeric vector of length \code{nobs}: the linear
#'     predictor at the joint mode of the fixed effects and random effects.}
#'   \item{\code{predict}}{Without \code{newdata}, the same vector as
#'     \code{fitted}. With \code{newdata}, a matrix with \code{nrow(newdata)}
#'     rows and \code{n} columns of linear predictors drawn from the Laplace
#'     approximation to the random effects.}
#'   \item{\code{summary}}{A list of class \code{"summary.adlaplace_fit"} with
#'     the coefficient table (natural-scale estimates, standard errors, and
#'     interval endpoints), the log-likelihood, degrees of freedom, the number
#'     of observations and random effects, the outer optimizer convergence
#'     code, and the maximum absolute profile gradient.}
#'   \item{\code{print}, \code{print.summary}, \code{plot}}{No return value
#'     used by the caller. They print or draw and return the input object
#'     invisibly. \code{plot} draws each smooth term as curves simulated from
#'     the Laplace approximation, with the pointwise mean highlighted, or the
#'     random-effect modes when the model has no smooth term.}
#' }
#'
#' @name adlaplace_fit-methods
NULL

.fit_transform_idx <- function(object) {
  which(object$par_info$log %in% TRUE)
}

.fit_se <- function(object) {
  vc <- object$details$vcov
  npar <- nrow(object$par_info)
  if (is.null(vc)) {
    return(rep(NA_real_, npar))
  }
  sqrt(pmax(0, diag(vc)))
}

#' @rdname adlaplace_fit-methods
#' @export
coef.adlaplace_fit <- function(object, transform = TRUE, ...) {
  est <- object$par_info$mle_internal
  names(est) <- object$par_info$label
  if (transform) {
    idx <- .fit_transform_idx(object)
    est[idx] <- exp(est[idx])
  }
  est
}

#' @rdname adlaplace_fit-methods
#' @export
vcov.adlaplace_fit <- function(object, transform = FALSE, ...) {
  vc <- object$details$vcov
  if (is.null(vc)) {
    stop(
      "no outer Hessian available; refit with hessian = TRUE",
      call. = FALSE
    )
  }
  if (transform) {
    est <- object$par_info$mle_internal
    d <- rep(1, length(est))
    idx <- .fit_transform_idx(object)
    d[idx] <- exp(est[idx])
    vc <- vc * tcrossprod(d)
  }
  vc
}

#' @rdname adlaplace_fit-methods
#' @export
logLik.adlaplace_fit <- function(object, ...) {
  ll <- object$details$log_lik
  if (is.null(ll)) {
    ll <- NA_real_
  }
  structure(
    ll,
    df = nrow(object$par_info),
    nobs = object$nobs,
    class = "logLik"
  )
}

#' @rdname adlaplace_fit-methods
#' @export
nobs.adlaplace_fit <- function(object, ...) {
  object$nobs
}

#' @rdname adlaplace_fit-methods
#' @export
confint.adlaplace_fit <- function(
  object, parm, level = 0.95, transform = TRUE, ...
) {
  est <- object$par_info$mle_internal
  names(est) <- object$par_info$label
  se <- .fit_se(object)
  zq <- stats::qnorm(1 - (1 - level) / 2)
  ci <- cbind(est - zq * se, est + zq * se)
  probs <- c((1 - level) / 2, 1 - (1 - level) / 2)
  dimnames(ci) <- list(
    names(est),
    paste(format(100 * probs, trim = TRUE), "%")
  )
  if (transform) {
    idx <- .fit_transform_idx(object)
    ci[idx, ] <- exp(ci[idx, ])
  }
  if (!missing(parm)) {
    ci <- ci[parm, , drop = FALSE]
  }
  ci
}

#' @rdname adlaplace_fit-methods
#' @export
fitted.adlaplace_fit <- function(object, ...) {
  info <- object$model_data$term_data$info
  n_beta <- max(c(0L, nrow(info$beta)))
  eta <- rep(0, object$nobs)
  if (n_beta > 0L) {
    eta <- eta + as.vector(
      object$model_data$term_data$X %*%
        object$par_info$mle_internal[seq_len(n_beta)]
    )
  }
  A <- object$model_data$term_data$A
  if (!is.null(A) && ncol(A) > 0L) {
    eta <- eta + as.vector(A %*% object$gamma)
  }
  eta
}

#' @rdname adlaplace_fit-methods
#' @details \code{predict} draws random effects from the Laplace (Gaussian)
#'   approximation at the mode via \code{\link{sim_fit}} and returns the linear
#'   predictor on \code{newdata}: a \code{nrow(newdata)} by \code{n} matrix of
#'   simulated values (plus fixed-effect contributions from terms whose
#'   variables appear in \code{newdata}). Without \code{newdata} it returns
#'   \code{fitted(object)}.
#' @export
predict.adlaplace_fit <- function(object, newdata, n = 500L, ...) {
  if (missing(newdata)) {
    return(fitted(object))
  }
  sim_fit(x = newdata, data = object$model_data, fit = object$details, n = n)
}

#' @rdname adlaplace_fit-methods
#' @method print adlaplace_fit
#' @export
print.adlaplace_fit <- function(
  x, digits = max(3L, getOption("digits") - 3L), ...
) {
  cat("Laplace-approximate maximum likelihood fit (adlaplace)\n\n")
  cat("Call:\n")
  print(x$call)
  if (is.null(x$details$outer_opt)) {
    cat("\nOuter optim failed:", x$details$error, "\n")
    cat(
      "  df:", nrow(x$par_info),
      "  n:", x$nobs,
      "  random effects:", length(x$gamma),
      "\n"
    )
    return(invisible(x))
  }
  cat("\nCoefficients (natural scale):\n")
  print(round(coef(x), digits))
  cat(
    "\nLog-likelihood:", format(as.numeric(logLik(x)), digits = digits),
    "  df:", nrow(x$par_info),
    "  n:", x$nobs,
    "  random effects:", length(x$gamma),
    "\n"
  )
  if (!identical(x$details$outer_opt$convergence, 0L)) {
    cat("optim convergence code:", x$details$outer_opt$convergence, "\n")
  }
  invisible(x)
}

#' @rdname adlaplace_fit-methods
#' @method summary adlaplace_fit
#' @export
summary.adlaplace_fit <- function(object, level = 0.95, ...) {
  par <- object$par_info
  zq <- stats::qnorm(1 - (1 - level) / 2)
  log_idx <- par$log %in% TRUE

  lo <- par$mle_internal - zq * par$se_internal
  hi <- par$mle_internal + zq * par$se_internal
  lo[log_idx] <- exp(lo[log_idx])
  hi[log_idx] <- exp(hi[log_idx])

  coefficients <- cbind(
    Estimate = par$mle,
    `Std. Error` = ifelse(log_idx, NA_real_, par$se_internal),
    lower = lo,
    upper = hi
  )
  rownames(coefficients) <- par$label

  grad <- object$details$deriv$d_neg_log_lik
  out <- list(
    call = object$call,
    coefficients = coefficients,
    logLik = as.numeric(logLik(object)),
    df = nrow(par),
    nobs = object$nobs,
    n_random = length(object$gamma),
    convergence = if (is.null(object$details$outer_opt)) {
      NA_integer_
    } else {
      object$details$outer_opt$convergence
    },
    error = object$details$error,
    max_grad = if (!is.null(grad)) suppressWarnings(max(abs(grad))) else NA_real_,
    level = level
  )
  class(out) <- "summary.adlaplace_fit"
  out
}

#' @rdname adlaplace_fit-methods
#' @method print summary.adlaplace_fit
#' @export
print.summary.adlaplace_fit <- function(
  x, digits = max(3L, getOption("digits") - 3L), ...
) {
  cat("Laplace-approximate maximum likelihood fit (adlaplace)\n\n")
  cat("Call:\n")
  print(x$call)
  cat(
    "\nLog-likelihood:", format(x$logLik, digits = digits),
    "  df:", x$df,
    "  n:", x$nobs,
    "\n"
  )
  if (!is.null(x$error)) {
    cat("Outer optim failed:", x$error, "\n\n")
  } else {
    cat(
      "Convergence code:", x$convergence,
      "  max|gradient|:", format(x$max_grad, digits = 3L),
      "\n\n"
    )
  }
  cat(
    "Coefficients (natural scale, ",
    format(100 * x$level, trim = TRUE), "% CI):\n",
    sep = ""
  )
  print(round(x$coefficients, digits))
  cat("\nRandom effects:", x$n_random, "coefficients\n")
  invisible(x)
}

#' @rdname adlaplace_fit-methods
#' @details \code{plot} draws each smooth term (terms with more than one knot)
#'   on a grid spanning its knots, showing simulated curves from the Laplace
#'   approximation with the pointwise mean in red. With no smooth terms it
#'   plots the random-effect modes.
#' @export
plot.adlaplace_fit <- function(x, k = 200L, draws = 100L, ...) {
  smooth_terms <- Filter(
    function(tt) methods::is(tt, "model_term") && length(tt@knots) > 1L,
    x$model_data$terms
  )
  if (length(smooth_terms) > 0L) {
    for (tt in smooth_terms) {
      v <- tt@name
      rng <- range(tt@knots)
      newx <- data.frame(seq(rng[1], rng[2], length.out = k))
      names(newx) <- v
      sims <- sim_fit(newx, data = x$model_data, fit = x$details, n = draws)
      graphics::matplot(
        newx[[v]], sims,
        type = "l", lty = 1, col = "#00000020",
        xlab = v, ylab = "linear predictor", ...
      )
      graphics::lines(newx[[v]], rowMeans(sims), col = "red", lwd = 2)
    }
  } else {
    graphics::plot(
      x$gamma,
      xlab = "random effect index", ylab = "mode", ...
    )
    graphics::abline(h = 0, lty = 2)
  }
  invisible(x)
}
