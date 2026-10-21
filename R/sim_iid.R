#' @noRd
iid_gamma_index <- function(gamma_info) {
  if (is.null(gamma_info) || !is.data.frame(gamma_info) ||
      !"model" %in% names(gamma_info)) {
    stop("`fit` must contain model_data$term_data$info$gamma with a model column",
      call. = FALSE
    )
  }
  which(gamma_info$model == "iid")
}

#' @noRd
inner_hessian_from_fit <- function(fit) {
  h <- fit$details$hessian
  if (is.null(h)) {
    stop("`fit` must contain details$hessian from log_lik_laplace(..., deriv = TRUE)",
      call. = FALSE
    )
  }
  if (!is.null(h$inner)) {
    return(h$inner)
  }
  if (!is.null(h$H)) {
    n_gamma <- length(fit$gamma)
    if (nrow(h$H) == n_gamma && ncol(h$H) == n_gamma) {
      return(h$H)
    }
  }
  stop(
    "`fit$details$hessian` must contain inner precision matrix $inner",
    call. = FALSE
  )
}

#' Marginal covariance of IID random effects via Woodbury identity
#'
#' @param H_inner Inner (gamma) precision matrix.
#' @param which_is_iid Logical index of IID rows/columns.
#' @param H_inv Optional full inner covariance (e.g. \code{hessian$H_inv}).
#' @return Dense covariance matrix for the IID block.
#' @keywords internal
#' @noRd
var_iid_from_hessian <- function(H_inner, which_is_iid, H_inv = NULL) {
  n_iid <- sum(which_is_iid)
  if (n_iid == 0L) {
    stop("no IID components in which_is_iid", call. = FALSE)
  }
  Dinv <- H_inner[which_is_iid, which_is_iid, drop = FALSE]
  which_other <- !which_is_iid

  if (!any(which_other)) {
    return(as.matrix(Matrix::solve(Dinv)))
  }

  if (requireNamespace("WoodburyMatrix", quietly = TRUE)) {
    X <- H_inner[which_is_iid, which_other, drop = FALSE]
    B <- H_inner[which_other, which_other, drop = FALSE]
    for_var <- WoodburyMatrix::WoodburyMatrix(
      A = Matrix::solve(Dinv),
      B = B,
      X = X,
      symmetric = TRUE
    )
    return(as.matrix(WoodburyMatrix::solve(for_var)))
  }

  if (!is.null(H_inv)) {
    return(as.matrix(H_inv[which_is_iid, which_is_iid, drop = FALSE]))
  }

  X <- as.matrix(H_inner[which_is_iid, which_other, drop = FALSE])
  B <- as.matrix(H_inner[which_other, which_other, drop = FALSE])
  D <- as.matrix(Dinv)
  # Matches WoodburyMatrix path: (D + X B^{-1} X')^{-1}
  as.matrix(solve(D + X %*% solve(B, t(X))))
}

#' @noRd
rmvn_cov <- function(n, mean, cov) {
  n <- as.integer(n)
  p <- length(mean)
  cov <- as.matrix(cov)
  if (nrow(cov) != p || ncol(cov) != p) {
    stop("cov dimension does not match length(mean)", call. = FALSE)
  }
  R <- chol(cov)
  z <- matrix(stats::rnorm(p * n), nrow = p, ncol = n)
  draws <- as.matrix(mean + crossprod(R, z))
  out <- t(draws)
  if (!is.null(names(mean))) {
    colnames(out) <- names(mean)
  }
  out
}

#' Simulate IID random effects with pointwise and global intervals
#'
#' Identifies IID components in an \code{\link{adlaplace}} fit, forms their
#' marginal Laplace covariance (Woodbury identity when mixed with other random
#' effects), draws samples, and summarizes pointwise quantiles. When \pkg{GET}
#' is available, also returns a global central region across levels.
#'
#' @param fit An \code{"adlaplace_fit"} from \code{\link{adlaplace}}.
#' @param n Number of draws.
#' @param probs Quantile levels for pointwise intervals.
#' @param coverage Coverage for \code{\link[GET]{central_region}} global
#'   envelopes (used only when \pkg{GET} is installed).
#' @param ... Currently unused.
#'
#' @return A named list with one element per IID term. Each element is a list
#'   with \code{x} (basis / factor levels), \code{sim} (levels-by-draws matrix),
#'   \code{pointwise} (quantile matrix with columns named by \code{probs}), and
#'   \code{global} (matrix with columns \code{lo}, \code{central}, \code{hi},
#'   or \code{NULL} if \pkg{GET} is unavailable). Attributes
#'   \code{var_iid}, \code{iid_index}, and \code{iid_labels} store the joint
#'   marginal covariance and index metadata.
#'
#' @details
#' IID rows are those with \code{model == "iid"} in
#' \code{fit$model_data$term_data$info$gamma}. The marginal covariance uses
#' \pkg{WoodburyMatrix} when installed; otherwise the IID block of
#' \code{fit$details$hessian$H_inv} (if present) or a dense partitioned solve.
#'
#' @seealso \code{\link{sim_random}}, \code{\link{rmvnldl}}, \code{\link{iid}}
#' @export
sim_iid <- function(
  fit,
  n = 500L,
  probs = c(0.1, 0.5, 0.9),
  coverage = 0.8,
  ...
) {
  if (!inherits(fit, "adlaplace_fit")) {
    stop("`fit` must be an adlaplace_fit object", call. = FALSE)
  }
  gamma_info <- fit$model_data$term_data$info$gamma
  iid_idx <- iid_gamma_index(gamma_info)
  if (!length(iid_idx)) {
    stop("no IID random-effect components found in fit", call. = FALSE)
  }

  which_is_iid <- rep(FALSE, nrow(gamma_info))
  which_is_iid[iid_idx] <- TRUE
  iid_labels <- gamma_info$gamma_label[iid_idx]
  mode_iid <- fit$gamma[iid_labels]
  if (anyNA(mode_iid) || length(mode_iid) != length(iid_labels)) {
    stop("`fit$gamma` must contain named modes for all IID gamma labels",
      call. = FALSE
    )
  }

  H_inner <- inner_hessian_from_fit(fit)
  if (nrow(H_inner) != nrow(gamma_info) || ncol(H_inner) != nrow(gamma_info)) {
    stop(
      "inner Hessian dimension (", nrow(H_inner), " x ", ncol(H_inner),
      ") does not match number of gamma effects (", nrow(gamma_info), ")",
      call. = FALSE
    )
  }

  H_inv <- fit$details$hessian$H_inv
  var_iid <- var_iid_from_hessian(H_inner, which_is_iid, H_inv = H_inv)
  dimnames(var_iid) <- list(iid_labels, iid_labels)

  sims_all <- rmvn_cov(n = n, mean = mode_iid, cov = var_iid)

  terms <- unique(gamma_info$term[iid_idx])
  out <- vector("list", length(terms))
  names(out) <- terms

  for (tm in terms) {
    rows <- which(gamma_info$term[iid_idx] == tm)
    labels_here <- iid_labels[rows]
    x <- gamma_info$basis[iid_idx][rows]
    sim <- t(sims_all[, labels_here, drop = FALSE])
    rownames(sim) <- as.character(x)
    out[[tm]] <- list(
      x = x,
      sim = sim,
      pointwise = sim_pointwise_quantiles(sim, probs = probs),
      global = sim_global_envelope(sim, coverage = coverage)
    )
  }

  structure(
    out,
    class = c("sim_iid", "list"),
    var_iid = var_iid,
    iid_index = iid_idx,
    iid_labels = iid_labels
  )
}
