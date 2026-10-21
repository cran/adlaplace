#' Hyperparameter prior term
#'
#' @description Places a prior on an existing theta (SD) parameter. Does not
#' create a new theta; it emits an \code{ad_kind = "parameters"} shard that
#' contributes to the joint log density.
#' @name prior-class
#' @aliases prior
#' @docType class
#' @title Hyperparameter prior term
#' @exportClass prior
NULL

setClass(
  "prior",
  slots = list(
    theta = "ANY",
    dist = "character",
    median = "numeric",
    rate = "numeric"
  ),
  contains = "model_term",
  prototype = prototype(
    name = "prior",
    label = "prior",
    knots = numeric(0),
    ref_value = numeric(0),
    p.order = as.integer(0),
    init = numeric(0),
    lower = numeric(0),
    upper = numeric(0),
    parscale = numeric(0),
    model_role = factor("fixed", levels = .model_role_levels),
    density = "exp_prior",
    ad_kind = "parameters",
    theta = 0L,
    dist = "exp",
    median = 1,
    rate = log(2)
  )
)

#' @rdname prior-class
#' @param theta Zero-based index into \code{info$theta$id}, or a character
#'   theta label (e.g. \code{"ID_iid"}).
#' @param dist Prior family. Currently only \code{"exp"} (exponential on the
#'   SD scale) is supported.
#' @param median Prior median on the SD scale. For \code{dist = "exp"}, the
#'   rate is \code{log(2) / median}.
#' @param rate Optional exponential rate (overrides \code{median} when
#'   supplied).
#' @return A \code{prior} term object.
#' @export
prior <- function(theta = 0L, dist = "exp", median = 1, rate = NULL) {
  dist <- as.character(dist)[1L]
  if (!identical(dist, "exp")) {
    stop("only dist = \"exp\" is currently supported", call. = FALSE)
  }
  if (is.character(theta)) {
    if (length(theta) != 1L || !nzchar(theta)) {
      stop("theta label must be a single non-empty string", call. = FALSE)
    }
    theta_id <- theta
  } else {
    theta_id <- as.integer(theta)[1L]
    if (is.na(theta_id) || theta_id < 0L) {
      stop("theta must be a non-negative 0-based index or a label", call. = FALSE)
    }
  }
  if (is.null(rate)) {
    if (length(median) != 1L || !is.finite(median) || median <= 0) {
      stop("median must be a single positive finite value", call. = FALSE)
    }
    rate <- log(2) / median
  } else {
    if (length(rate) != 1L || !is.finite(rate) || rate <= 0) {
      stop("rate must be a single positive finite value", call. = FALSE)
    }
    median <- log(2) / rate
  }
  label <- if (is.character(theta_id)) {
    paste0("prior_", theta_id)
  } else {
    paste0("prior_theta", theta_id)
  }
  methods::new(
    "prior",
    name = label,
    label = label,
    formula = ~0,
    density = "exp_prior",
    ad_kind = "parameters",
    theta = theta_id,
    dist = dist,
    median = median,
    rate = rate
  )
}

#' Resolve a prior term onto a 1-based theta map row
#' @noRd
prior_theta_row <- function(term, theta_info) {
  if (!is.data.frame(theta_info) || nrow(theta_info) < 1L) {
    stop("prior() requires at least one theta parameter in the model", call. = FALSE)
  }
  theta <- term@theta
  if (is.character(theta)) {
    idx <- which(theta_info$label == theta)
    if (length(idx) != 1L) {
      stop(
        "prior theta label '", theta, "' not found in info$theta$label",
        call. = FALSE
      )
    }
    return(as.integer(idx))
  }
  id <- as.integer(theta)[1L]
  if (!"id" %in% names(theta_info)) {
    stop("info$theta must include an id column", call. = FALSE)
  }
  idx <- which(theta_info$id == id)
  if (length(idx) != 1L) {
    stop(
      "prior theta id ", id, " not found in info$theta$id (0-based)",
      call. = FALSE
    )
  }
  as.integer(idx)
}

#' @keywords internal
build_prior_shard <- function(term_here, all_data, counts) {
  theta_row <- prior_theta_row(term_here, all_data$info$theta)
  density_data(
    y = term_here@rate,
    beta_map = counts$beta,
    gamma_map = counts$gamma,
    theta_map = list(theta_row, as.integer(counts$theta)),
    density = term_here@density,
    ad_kind = "parameters",
    package = term_here@package
  )
}
