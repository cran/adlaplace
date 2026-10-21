#' Per-unit observation gradients (streaming, grad-only tapes)
#'
#' Computes sparse gradients of each observation **unit** contribution at a
#' fixed full parameter vector \code{x}, without re-running
#' \code{\link{inner_opt}}. Each unit is taped temporarily with
#' \code{config$hessian_sparsity = FALSE} (sparse Jac only), reverse-mode
#' differentiated, then discarded.
#'
#' Units are ELGM stratum columns when \code{data@elgm_matrix} has columns;
#' otherwise units are rows of the observation vector \code{y}.
#'
#' @param data An observation \code{density_data} (e.g.
#'   \code{fit$model_data$observations[[1]]}), or a \code{model_data()} bundle
#'   (uses the first observation shard).
#' @param x Numeric full parameter vector \code{c(beta, gamma, theta)} at the
#'   evaluation point (typically \code{inner_opt()$full_parameters}).
#' @param config Config list passed to the observation tape builder. Always
#'   forced to \code{hessian_sparsity = FALSE}; \code{compact_tape} defaults to
#'   \code{TRUE} and \code{transform_theta} defaults to \code{TRUE} when
#'   omitted (same as \code{\link{adlaplace}} / \code{\link{ad_pack}}). Supply
#'   \code{beta}/\code{gamma}/\code{theta} seeds if needed for recording
#'   (taken from \code{x} when omitted). Prefer passing
#'   \code{fit$config} (or at least its \code{transform_theta}) when evaluating
#'   at an MLE from \code{\link{adlaplace}}.
#' @param units Optional integer vector of **0-based** unit indices; default
#'   all units.
#' @param batch_size Positive integer; how many one-unit tapes to hold at once.
#' @param inner Logical; if \code{TRUE} (default), return only the
#'   \eqn{\gamma} block (rows = \code{n_gamma}).
#' @param negative Logical; if \code{TRUE} (default), return the negative log
#'   density gradient (same sign as \code{\link{grad}} /
#'   \code{\link{inner_opt}}).
#'
#' @return A \code{dgCMatrix} of size \code{n_gamma x length(units)} when
#'   \code{inner = TRUE}, or \code{n_full x length(units)} when
#'   \code{inner = FALSE}. Column \code{j} is the gradient for
#'   \code{units[j]}.
#'
#' @section Config note:
#' \code{hessian_sparsity = FALSE} yields packs usable for
#' \code{\link{grad}(..., inner = TRUE)} but not for Hessian /
#' \code{\link{inner_opt}} LDL paths.
#'
#' @seealso \code{\link{grad}}, \code{\link{inner_opt}}, \code{\link{ad_pack}}
#' @export
grad_obs_units <- function(
  data,
  x,
  config = list(),
  units = NULL,
  batch_size = 32L,
  inner = TRUE,
  negative = TRUE
) {
  if (is.list(data) && !methods::is(data, "density_data")) {
    if (is.null(data$observations) || !length(data$observations)) {
      stop(
        "`data` list must be a model_data() bundle with $observations",
        call. = FALSE
      )
    }
    data <- data$observations[[1L]]
  }
  if (!methods::is(data, "density_data")) {
    stop("`data` must be a density_data observation shard", call. = FALSE)
  }
  if (!identical(as.character(data@ad_kind), "observations")) {
    stop("`data@ad_kind` must be \"observations\"", call. = FALSE)
  }
  if (length(data@density) != 1L || is.na(data@density) || !nzchar(data@density)) {
    stop("`data@density` is required", call. = FALSE)
  }

  x <- as.numeric(x)
  batch_size <- as.integer(batch_size)[1L]
  if (is.na(batch_size) || batch_size < 1L) {
    stop("`batch_size` must be a positive integer", call. = FALSE)
  }

  n_beta <- nrow(data@beta_map)
  n_gamma <- nrow(data@gamma_map)
  n_theta <- nrow(data@theta_map)
  n_full <- n_beta + n_gamma + n_theta
  if (length(x) != n_full) {
    stop(
      "`x` has length ", length(x), " but expected n_full = ", n_full,
      call. = FALSE
    )
  }

  cfg <- as.list(config)
  if (is.null(cfg$beta)) {
    cfg$beta <- if (n_beta > 0L) x[seq_len(n_beta)] else numeric(0)
  }
  if (is.null(cfg$gamma)) {
    cfg$gamma <- if (n_gamma > 0L) {
      x[seq.int(n_beta + 1L, length.out = n_gamma)]
    } else {
      numeric(0)
    }
  }
  if (is.null(cfg$theta)) {
    cfg$theta <- if (n_theta > 0L) {
      x[seq.int(n_beta + n_gamma + 1L, length.out = n_theta)]
    } else {
      numeric(0)
    }
  }
  cfg$hessian_sparsity <- FALSE
  if (is.null(cfg$compact_tape)) {
    cfg$compact_tape <- TRUE
  }
  if (is.null(cfg$transform_theta)) {
    cfg$transform_theta <- TRUE
  }

  units_arg <- if (is.null(units)) {
    integer(0)
  } else {
    as.integer(units)
  }

  .grad_obs_units_cpp(
    data,
    x,
    cfg,
    units_arg,
    batch_size,
    isTRUE(inner),
    isTRUE(negative)
  )
}
