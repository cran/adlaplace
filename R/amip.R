#' @include grad_obs_units.R obs_groups.R fit.R
NULL

#' Linear functional of the inner field \eqn{\gamma}
#'
#' Builds a named numeric vector \eqn{a} aligned to
#' \code{fit$model_data$term_data$info$gamma$gamma_label} so that
#' \eqn{\phi(\gamma) = a^\top\gamma}.
#'
#' @param fit An \code{"adlaplace_fit"} from \code{\link{adlaplace}}.
#' @param term Optional character name of a covariate whose **random** terms
#'   contribute via \code{\link{design}} (e.g. \code{"pm10"}).
#' @param new_x Data frame of prediction points for \code{term} (required when
#'   \code{term} is set). Typically one row, e.g. \code{data.frame(pm10 = 80)}.
#' @param coef Optional gamma label (or labels) to set to 1 (unit / contrast
#'   vectors), e.g. a treatment coefficient name.
#'
#' @return Named numeric vector of length \code{n_gamma}.
#' @seealso \code{\link{influence_scores}}, \code{\link{amip_set}},
#'   \code{\link{design}}, \code{\link{sim_random}}
#' @export
phi_gamma <- function(fit, term = NULL, new_x = NULL, coef = NULL) {
  if (!inherits(fit, "adlaplace_fit")) {
    stop("`fit` must be an adlaplace_fit object", call. = FALSE)
  }
  md <- fit$model_data
  if (is.null(md) || is.null(md$term_data$info$gamma)) {
    stop("`fit` must contain model_data$term_data$info$gamma", call. = FALSE)
  }
  gamma_lab <- md$term_data$info$gamma$gamma_label
  a <- numeric(length(gamma_lab))
  names(a) <- gamma_lab

  if (!is.null(coef)) {
    coef <- as.character(coef)
    miss <- setdiff(coef, gamma_lab)
    if (length(miss)) {
      stop(
        "coef label(s) not found in gamma: ",
        paste(miss, collapse = ", "),
        call. = FALSE
      )
    }
    a[coef] <- 1
  }

  if (!is.null(term)) {
    if (!is.character(term) || length(term) != 1L || !nzchar(term)) {
      stop("`term` must be a non-empty character scalar", call. = FALSE)
    }
    if (is.null(new_x)) {
      stop("`new_x` is required when `term` is set", call. = FALSE)
    }
    if (!is.data.frame(new_x) || nrow(new_x) < 1L) {
      stop("`new_x` must be a data frame with at least one row", call. = FALSE)
    }
    terms <- md$terms
    pm_terms <- Filter(
      function(tt) {
        identical(tt@name, term) &&
          identical(as.character(tt@model_role), "random")
      },
      terms
    )
    if (!length(pm_terms)) {
      stop("no random terms found for variable '", term, "'", call. = FALSE)
    }
    design_list <- lapply(pm_terms, design, data = new_x)
    design_list <- Filter(Negate(is.null), design_list)
    if (!length(design_list)) {
      stop("design() returned NULL for all random terms of '", term, "'", call. = FALSE)
    }
    new_design <- as.matrix(do.call(cbind, design_list))
    miss_cols <- setdiff(colnames(new_design), gamma_lab)
    if (length(miss_cols)) {
      stop(
        "design columns not in gamma labels: ",
        paste(miss_cols, collapse = ", "),
        call. = FALSE
      )
    }
    # Use first row of new_x (typical AMIP QoI is a scalar functional).
    a[colnames(new_design)] <- a[colnames(new_design)] +
      as.numeric(new_design[1L, ])
  }

  if (is.null(coef) && is.null(term)) {
    stop("provide `term`/`new_x` and/or `coef`", call. = FALSE)
  }
  if (!any(abs(a) > 0)) {
    stop("linear functional `a` is identically zero", call. = FALSE)
  }
  a
}

#' Per-unit influence scores for an inner linear functional
#'
#' Freezes outer parameters at the MLE, re-optimizes \eqn{\gamma}, forms
#' \eqn{v = H^{-1}a} from the fit-pack inner Hessian, and streams one-unit
#' observation gradients via \code{\link{grad_obs_units}} (temporary tapes).
#' Scores are \eqn{\psi_u = -v^\top g_u}.
#'
#' @param fit An \code{"adlaplace_fit"} from \code{\link{adlaplace}}.
#' @param a Named or positional numeric vector of length \code{n_gamma}
#'   (e.g. from \code{\link{phi_gamma}}).
#' @param batch_size Passed to \code{\link{grad_obs_units}}.
#' @param control Control list for \code{\link{inner_opt}}.
#' @param verbose Logical; forwarded to \code{\link{inner_opt}}.
#'
#' @return A list with \code{phi}, \code{gamma_hat}, \code{H}, \code{v},
#'   \code{G}, \code{psi}, \code{x_full}, and \code{n_units}.
#' @seealso \code{\link{phi_gamma}}, \code{\link{amip_set}},
#'   \code{\link{grad_obs_units}}, \code{\link{ad_pack_drop}}
#' @export
influence_scores <- function(
  fit,
  a,
  batch_size = 32L,
  control = list(report.level = 0, report.freq = 0),
  verbose = FALSE
) {
  if (!inherits(fit, "adlaplace_fit")) {
    stop("`fit` must be an adlaplace_fit object", call. = FALSE)
  }
  md <- fit$model_data
  gamma_lab <- md$term_data$info$gamma$gamma_label
  a <- as.numeric(a)
  if (length(a) != length(gamma_lab)) {
    stop(
      "`a` has length ", length(a), " but n_gamma = ", length(gamma_lab),
      call. = FALSE
    )
  }
  if (!is.null(names(a))) {
    if (!identical(names(a), gamma_lab)) {
      # Reorder if names are a permutation of gamma labels.
      if (!setequal(names(a), gamma_lab)) {
        stop("`a` names must match gamma labels", call. = FALSE)
      }
      a <- a[gamma_lab]
    }
  }

  x_outer <- as.numeric(fit$par_info$mle_internal)
  gamma0 <- as.numeric(fit$gamma)
  io <- inner_opt(
    parameters = x_outer,
    gamma = gamma0,
    ad_pack = fit$ad_pack,
    control = control,
    deriv = TRUE,
    return_hessians = TRUE,
    verbose = verbose
  )
  gamma_hat <- as.numeric(io$inner_opt$solution)
  names(gamma_hat) <- names(fit$gamma)
  H <- io$hessian$inner
  x_full <- as.numeric(io$full_parameters)
  phi <- sum(a * gamma_hat)

  v <- as.numeric(Matrix::solve(H, a))
  obs <- md$observations[[1L]]
  G <- grad_obs_units(
    obs,
    x_full,
    config = fit$config,
    batch_size = batch_size,
    inner = TRUE,
    negative = TRUE
  )
  n_units <- n_obs_units(obs)
  if (ncol(G) != n_units) {
    stop(
      "grad_obs_units returned ", ncol(G), " columns but n_obs_units = ",
      n_units,
      call. = FALSE
    )
  }
  psi <- as.numeric(-Matrix::crossprod(v, G))

  list(
    phi = phi,
    gamma_hat = gamma_hat,
    H = H,
    v = v,
    G = G,
    psi = psi,
    x_full = x_full,
    n_units = n_units,
    a = a,
    x_outer = x_outer
  )
}

#' Approximate most influential removal set (AMIP)
#'
#' Greedy rule: drop units with the largest scores of the same sign as
#' \code{phi} until \eqn{\sum_{u \in S} \psi_u} matches the sign of
#' \eqn{\phi} and \eqn{|\sum| \ge |\phi|}. Leaving out \eqn{S} changes
#' \eqn{\phi} by approximately \eqn{-\sum \psi_S}.
#'
#' @param psi Numeric vector of per-unit scores (e.g. from
#'   \code{\link{influence_scores}}).
#' @param phi Scalar quantity of interest at the full-data mode.
#'
#' @return A list with \code{idx} (1-based indices to drop), \code{alpha}
#'   (fraction of units), and \code{approx_change}. If the linear
#'   approximation cannot flip the sign, \code{idx} is empty and
#'   \code{alpha} is \code{NA}.
#' @seealso \code{\link{influence_scores}}, \code{\link{ad_pack_drop}}
#' @export
amip_set <- function(psi, phi) {
  psi <- as.numeric(psi)
  phi <- as.numeric(phi)[1L]
  n <- length(psi)
  if (!is.finite(phi) || phi == 0 || !n) {
    return(list(
      idx = integer(0),
      alpha = NA_real_,
      approx_change = 0
    ))
  }
  if (phi > 0) {
    ord <- order(psi, decreasing = TRUE)
  } else {
    ord <- order(psi)
  }
  cum_psi <- cumsum(psi[ord])
  m_star <- which(cum_psi / phi >= 1)[1L]
  if (is.na(m_star)) {
    return(list(
      idx = integer(0),
      alpha = NA_real_,
      approx_change = -cum_psi[length(cum_psi)]
    ))
  }
  idx <- ord[seq_len(m_star)]
  list(
    idx = as.integer(idx),
    alpha = length(idx) / n,
    approx_change = -cum_psi[m_star]
  )
}

#' Retape an AD pack after dropping observation units
#'
#' Builds a new \code{\link{ad_pack}} on \code{fit$model_data} with
#' observation shards filtered to the remaining units (same coarse sharding
#' as \code{fit$config$obs_groups} when present). Random and parameter shards
#' are retaped; reuse of the merged fit pack is not attempted.
#'
#' @param fit An \code{"adlaplace_fit"} from \code{\link{adlaplace}}.
#' @param drop Integer vector of **1-based** unit indices to remove.
#' @param num_threads OpenMP thread count; default from \code{fit$config}.
#' @param verbose Logical; forwarded into the retape config.
#'
#' @return An \code{ad_pack} usable with \code{\link{inner_opt}} at the
#'   original outer MLE.
#' @seealso \code{\link{obs_groups_units}}, \code{\link{influence_scores}},
#'   \code{\link{amip_set}}
#' @export
ad_pack_drop <- function(fit, drop, num_threads = NULL, verbose = FALSE) {
  if (!inherits(fit, "adlaplace_fit")) {
    stop("`fit` must be an adlaplace_fit object", call. = FALSE)
  }
  md <- fit$model_data
  n_units <- n_obs_units(fit)
  drop <- as.integer(drop)
  if (!length(drop)) {
    stop("`drop` is empty; nothing to remove", call. = FALSE)
  }
  if (any(is.na(drop)) || any(drop < 1L) || any(drop > n_units)) {
    stop(
      "`drop` must be 1-based indices in [1, ", n_units, "]",
      call. = FALSE
    )
  }
  if (anyDuplicated(drop)) {
    stop("`drop` must be unique", call. = FALSE)
  }
  keep0 <- setdiff(seq.int(0L, n_units - 1L), drop - 1L)
  if (!length(keep0)) {
    stop("dropping all units leaves an empty observation domain", call. = FALSE)
  }

  cfg <- as.list(fit$config)
  og_fit <- cfg[["obs_groups"]]
  if (!is.null(og_fit) && nrow(og_fit) == n_units) {
    og_drop <- obs_groups_units(
      n_domain = n_units,
      units = keep0,
      grouping = "filter",
      obs_groups = og_fit
    )
  } else {
    og_drop <- obs_groups_units(
      n_domain = n_units,
      units = keep0,
      grouping = "together"
    )
  }
  cfg$obs_groups <- og_drop
  cfg$obs_units <- NULL
  cfg$verbose <- isTRUE(verbose)
  if (is.null(num_threads)) {
    num_threads <- cfg$num_threads
  }
  if (is.null(num_threads)) {
    num_threads <- 1L
  }
  ad_pack(
    md,
    config = cfg,
    num_threads = as.integer(num_threads)[1L],
    reorder_shards = "none"
  )
}
