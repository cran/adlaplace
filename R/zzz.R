#' @useDynLib adlaplace, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @import Matrix
#' @import methods
#' @import RCppAD
NULL

loadNamespace("Matrix")

.onLoad <- function(libname, pkgname) {
  if (isTRUE(has_openmp())) {
    warm_openmp_runtime()
  }
  invisible(NULL)
}

.my_beta_init <- 0
.my_beta_lower <- -Inf
.my_beta_upper <- Inf
.my_beta_parscale <- 1

.my_theta_init <- 0.02
.my_theta_lower <- 1e-9
.my_theta_upper <- Inf
.my_theta_parscale <- 1

#' @keywords internal
ad_shard_label <- function(shard, name = NULL) {
  if (!is.null(name) && length(name) == 1L && nzchar(name)) {
    return(name)
  }
  paste0(shard@ad_kind, "/", shard@density)
}

#' @keywords internal
init_from_info_block <- function(block) {
  if (is.null(block) || !is.data.frame(block) || nrow(block) == 0L) {
    return(numeric(0))
  }
  as.numeric(block$init)
}

#' Fill missing \code{config$beta} / \code{config$theta} / \code{config$gamma}
#' from \code{term_data$info}.
#'
#' Explicit \code{config$beta}, \code{config$theta}, and \code{config$gamma}
#' are left unchanged. \code{transform_theta} is resolved to a per-theta
#' logical vector (from \code{info$theta$log} unless the caller passed a full
#' vector or \code{FALSE}). When \code{config$theta} is missing, seeds are
#' \code{apply_theta_log(info$theta, cols = "init")$init}. When
#' \code{config$gamma} is missing, seeds are zeros of length
#' \code{nrow(info$gamma)} (\code{info$gamma} has no \code{init} column).
#'
#' @keywords internal
fill_config_from_info <- function(config, info) {
  if (is.null(config)) {
    config <- list()
  }
  if (is.null(info)) {
    return(config)
  }

  if (is.null(config[["beta"]])) {
    config$beta <- init_from_info_block(info[["beta"]])
  }

  if (is.null(config[["gamma"]])) {
    gamma_info <- info[["gamma"]]
    n_gamma <- if (is.null(gamma_info) || !is.data.frame(gamma_info)) {
      0L
    } else {
      nrow(gamma_info)
    }
    config$gamma <- rep(0, n_gamma)
  }

  theta_info <- info[["theta"]]
  if (is.null(theta_info) || !is.data.frame(theta_info)) {
    theta_info <- data.frame()
  }
  n_theta <- nrow(theta_info)

  force_no_log <- identical(config[["transform_theta"]], FALSE)
  log_flags <- if (force_no_log) {
    rep(FALSE, n_theta)
  } else if (is.logical(config[["transform_theta"]]) &&
    length(config[["transform_theta"]]) == n_theta &&
    n_theta > 0L) {
    config[["transform_theta"]]
  } else if (n_theta > 0L && "log" %in% names(theta_info)) {
    flags <- theta_info$log
    flags[is.na(flags)] <- TRUE
    as.logical(flags)
  } else {
    rep(TRUE, n_theta)
  }
  config$transform_theta <- log_flags

  if (is.null(config[["theta"]])) {
    if (n_theta > 0L) {
      theta_for_log <- theta_info
      if (!"log" %in% names(theta_for_log)) {
        theta_for_log$log <- TRUE
      }
      theta_for_log$log <- log_flags
      config$theta <- apply_theta_log(
        theta_for_log,
        cols = "init",
        active = TRUE
      )$init
    } else {
      config$theta <- numeric(0)
    }
  }

  config
}

#' @keywords internal
is_model_data_bundle <- function(x) {
  is.list(x) &&
    all(c("term_data", "observations", "random", "parameters") %in% names(x)) &&
    is.list(x$observations) &&
    is.list(x$random) &&
    is.list(x$parameters)
}
