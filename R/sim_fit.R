#' @noRd
chol_inner_from_fit <- function(fit) {
  chol_inner <- fit$hessian$chol_inner
  if (is.null(chol_inner) && is.list(fit$extra) && is.list(fit$extra$hessian)) {
    chol_inner <- fit$extra$hessian$chol_inner
  }
  if (is.null(chol_inner)) {
    stop(
      "fit must contain inner Cholesky factor in ",
      "$hessian$chol_inner (from log_lik_laplace()) ",
      "or $extra$hessian$chol_inner (legacy layout)",
      call. = FALSE
    )
  }
  chol_inner
}

#' @noRd
outer_par_from_fit <- function(fit) {
  if (!is.null(fit$outer_opt$par)) {
    return(fit$outer_opt$par)
  }
  if (!is.null(fit$parameters)) {
    return(fit$parameters)
  }
  stop(
    "fit must contain outer_opt$par (e.g. from adlaplace()$details) ",
    "or parameters (e.g. from log_lik_laplace())",
    call. = FALSE
  )
}

#' Simulate linear predictors on a new covariate grid
#'
#' Draws random effects from the Laplace approximation and evaluates the
#' linear predictor on \code{x} by summing fixed (\eqn{\beta}) and random
#' (\eqn{\gamma}) contributions from each model term.
#'
#' @param x Data frame of prediction covariates (same variables as the fitted model).
#'   Only terms whose variables appear in \code{x} contribute.
#' @param data Object returned by \code{\link{model_data}}.
#' @param fit Result from \code{\link{log_lik_laplace}} at the fitted outer parameters.
#' @param n Number of draws for random-effect simulation.
#'
#' @return The combined linear predictor: a \code{nrow(x)} by \code{n} matrix of
#'   simulated values (fixed-effect contributions are constant across columns).
#'
#' @export
sim_fit <- function(x, data, fit, n = 500L) {
  outer_par <- outer_par_from_fit(fit)
  gamma_sims <- rmvnldl(
    n,
    mean = fit$inner_opt$solution,
    chol_prec = chol_inner_from_fit(fit)
  )
  n_draws <- nrow(gamma_sims)
  n_grid <- nrow(x)

  if (is_model_data_bundle(data)) {
    info <- data$term_data$info
    term_list <- data$term_data$terms
  } else if (is(data, "density_data") && is.list(data@precision) && !is.null(data@precision$info)) {
    info <- data@precision$info
    term_list <- list()
  } else {
    stop("`data` must be output from model_data()", call. = FALSE)
  }

  design_list <- lapply(
    term_list,
    function(term) {
      if (!any(term@name %in% names(x))) {
        return(NULL)
      }
      beta_here <- beta_info(term, data = x)
      if (!is.null(beta_here)) {
        if (any(beta_here$term %in% names(x))) {
          design_here <- design(term, data = x)
          par_id <- which(info$parameters$label == beta_here$label)
          return(
            as.matrix(
              design_here[, beta_here$beta_label, drop = FALSE] %*%
                outer_par[par_id]
            )
          )
        }
        return(NULL)
      }
      gamma_here <- random_info(term, data = x)
      if (!is.null(gamma_here)) {
        if (any(gamma_here$term %in% names(x))) {
          gamma_here_id <- match(
            gamma_here$gamma_label,
            info$gamma$gamma_label
          )
          if (any(is.na(gamma_here_id))) {
            warning("no match for labels: ", gamma_here$label[1])
            return(NULL)
          }
          design_here <- design(term, data = x)
          return(
            tcrossprod(
              design_here[, gamma_here$gamma_label, drop = FALSE],
              gamma_sims[, gamma_here_id, drop = FALSE]
            )
          )
        }
        return(NULL)
      }
      NULL
    }
  )
  design_list <- design_list[!vapply(design_list, is.null, logical(1))]

  eta_fixed <- matrix(0, nrow = n_grid, ncol = 1L)
  eta_random <- matrix(0, nrow = n_grid, ncol = n_draws)

  if (length(design_list)) {
    ncols <- vapply(design_list, ncol, integer(1))
    if (!all(ncols %in% c(1L, n_draws))) {
      warning("some design matrices have the wrong number of columns")
    }
    is_fixed <- ncols == 1L
    is_random <- ncols == n_draws
    if (any(is_fixed)) {
      eta_fixed <- Reduce(`+`, design_list[is_fixed])
      if (!is.matrix(eta_fixed)) {
        eta_fixed <- matrix(eta_fixed, ncol = 1L)
      }
    }
    if (any(is_random)) {
      eta_random <- Reduce(`+`, design_list[is_random])
    }
  }

  eta <- as.vector(eta_fixed) + eta_random

  eta
}

#' @noRd
sim_pointwise_quantiles <- function(sim, probs) {
  pointwise <- t(apply(sim, 1L, stats::quantile, probs = probs))
  colnames(pointwise) <- as.character(probs)
  pointwise
}

#' @noRd
sim_global_envelope <- function(sim, coverage) {
  if (!requireNamespace("GET", quietly = TRUE)) {
    return(NULL)
  }
  env <- GET::central_region(
    GET::create_curve_set(list(obs = sim)),
    coverage = coverage
  )
  as.matrix(env[, c("lo", "central", "hi"), drop = FALSE])
}

#' Simulate a random-effect contribution for one covariate
#'
#' Builds a prediction grid for variable \code{x}, forms the design matrix for
#' all random terms on that variable, and multiplies by Laplace draws of the
#' random effects. Intended for plotting smooths (\code{iwp} / \code{rpoly})
#' and similar random contributions from an \code{\link{adlaplace}} fit.
#'
#' @param x Character scalar: name of the covariate / grouping variable.
#' @param fit An \code{"adlaplace_fit"} from \code{\link{adlaplace}}.
#' @param new_x Optional one-column data frame of prediction points for
#'   \code{x}. When missing, a grid is built from term knots via
#'   \code{\link[base]{pretty}} (or from unique \code{basis} levels in
#'   \code{fit$model_data$term_data$info$gamma} when there are no knots, e.g. IID).
#' @param gamma_sims Optional matrix of random-effect draws (rows = draws,
#'   columns named by gamma labels). When missing, drawn with
#'   \code{\link{rmvnldl}(n = num_sim, fit = fit)}.
#' @param num_grid Target length for the default \code{pretty} grid.
#' @param num_sim Number of draws when \code{gamma_sims} is missing.
#' @param probs Quantile levels for pointwise intervals.
#' @param coverage Coverage for \code{\link[GET]{central_region}} global
#'   envelopes (used only when \pkg{GET} is installed).
#' @param transform Function applied to each simulated linear predictor before
#'   summarizing intervals (and returned as \code{sim}); default identity.
#'   Use \code{exp} for relative-risk scale.
#'
#' @return A list with \code{x} (numeric or character grid), \code{sim}
#'   (an \code{length(x)} by draws matrix after \code{transform}),
#'   \code{pointwise} (quantile matrix), and \code{global} (GET envelope with
#'   columns \code{lo}, \code{central}, \code{hi}, or \code{NULL} if \pkg{GET}
#'   is unavailable).
#'
#' @details
#' Only terms with \code{model_role == "random"} whose \code{@name} equals \code{x}
#' are included (so companion \code{rpoly} bases for an \code{iwp} are included,
#' while fixed \code{fpoly} bases are not). Column names of the design must
#' match columns of \code{gamma_sims}.
#'
#' IID prediction grids use factor levels stored in the gamma info table; levels
#' not present at fit time are not invented here.
#'
#' @seealso \code{\link{rmvnldl}}, \code{\link{sim_fit}}, \code{\link{sim_iid}},
#'   \code{\link{design}}
#' @export
sim_random <- function(
  x,
  fit,
  new_x,
  gamma_sims,
  num_grid = 101L,
  num_sim = 100L,
  probs = c(0.1, 0.5, 0.9),
  coverage = 0.8,
  transform = identity
) {
  if (!is.character(x) || length(x) != 1L || !nzchar(x)) {
    stop("`x` must be a non-empty character scalar", call. = FALSE)
  }
  if (is.null(fit$model_data) || is.null(fit$model_data$terms)) {
    stop("`fit` must be an adlaplace_fit with $model_data$terms", call. = FALSE)
  }

  terms <- fit$model_data$terms
  term_seq <- vapply(terms, methods::slot, character(1L), "name")

  terms_here <- terms[term_seq == x]
  terms_here <- Filter(
    function(tt) identical(as.character(tt@model_role), "random"),
    terms_here
  )
  if (!length(terms_here)) {
    stop("no random terms found for variable '", x, "'", call. = FALSE)
  }

  if (missing(gamma_sims)) {
    gamma_sims <- rmvnldl(n = num_sim, fit = fit)
  }
  if (is.null(colnames(gamma_sims))) {
    stop("`gamma_sims` must have column names (gamma labels)", call. = FALSE)
  }

  if (missing(new_x)) {
    knots_list <- lapply(terms, methods::slot, "knots")
    knots_here <- unlist(knots_list[term_seq == x], use.names = FALSE)
    if (length(knots_here)) {
      grid <- pretty(knots_here, n = num_grid)
    } else {
      gamma_info <- fit$model_data$term_data$info$gamma
      grid <- sort(unique(gamma_info[gamma_info$term == x, "basis"]))
      if (!length(grid)) {
        stop(
          "no knots or gamma basis levels found for variable '", x, "'",
          call. = FALSE
        )
      }
    }
    new_x <- stats::setNames(data.frame(grid, stringsAsFactors = FALSE), x)
  } else {
    if (!is.data.frame(new_x) || !(x %in% names(new_x))) {
      stop("`new_x` must be a data frame containing column '", x, "'",
        call. = FALSE
      )
    }
  }

  design_list <- lapply(terms_here, design, data = new_x)
  design_list <- Filter(Negate(is.null), design_list)
  if (!length(design_list)) {
    stop("design() returned no columns for variable '", x, "'", call. = FALSE)
  }
  new_design <- do.call(cbind, design_list)
  new_design <- as.matrix(new_design)

  missing_cols <- setdiff(colnames(new_design), colnames(gamma_sims))
  if (length(missing_cols)) {
    stop(
      "design columns not found in gamma_sims: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }

  sim_here <- tcrossprod(
    new_design,
    gamma_sims[, colnames(new_design), drop = FALSE]
  )
  if (!is.function(transform)) {
    stop("`transform` must be a function", call. = FALSE)
  }
  sim_here <- transform(sim_here)
  list(
    x = new_x[[x]],
    sim = sim_here,
    pointwise = sim_pointwise_quantiles(sim_here, probs = probs),
    global = sim_global_envelope(sim_here, coverage = coverage)
  )
}
