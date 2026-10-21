#' Format Laplace estimates using parameter metadata
#'
#' Maps flat Laplace parameter vectors onto labeled tables from \code{info}
#' (as stored on \code{ad_pack@info} or \code{model_data()$term_data$info}).
#'
#' @param info List with \code{beta}, \code{gamma}, \code{theta}, and
#'   \code{parameters} data frames.
#' @param full_parameters Numeric vector \code{c(beta, gamma, theta)}. When
#'   supplied, \code{parameters} and \code{gamma} are ignored.
#' @param parameters Numeric outer vector \code{c(beta, theta)}; used when
#'   \code{full_parameters} is \code{NULL}.
#' @param gamma Numeric inner vector; used when \code{full_parameters} is
#'   \code{NULL}.
#'
#' @return A list with data frames \code{parameters} and \code{gamma}, or
#'   \code{list()} when \code{info} is missing or incomplete.
#' @export
format_parameters <- function(
  info,
  full_parameters = NULL,
  parameters = NULL,
  gamma = NULL
) {
  if (is.null(info) || !is.list(info)) {
    return(list())
  }
  required <- c("beta", "gamma", "theta")
  if (!all(vapply(required, function(nm) {
    is.data.frame(info[[nm]])
  }, logical(1)))) {
    return(list())
  }

  n_beta <- nrow(info$beta)
  n_gamma <- nrow(info$gamma)
  n_theta <- nrow(info$theta)

  if (!is.null(full_parameters)) {
    fp <- as.numeric(full_parameters)
  } else {
    if (is.null(parameters) || is.null(gamma)) {
      stop(
        "must supply full_parameters or both parameters and gamma",
        call. = FALSE
      )
    }
    parameters <- as.numeric(parameters)
    gamma <- as.numeric(gamma)
    if (length(parameters) != n_beta + n_theta) {
      stop(
        "length(parameters) (", length(parameters),
        ") must equal n_beta + n_theta (", n_beta + n_theta, ")",
        call. = FALSE
      )
    }
    if (length(gamma) != n_gamma) {
      stop(
        "length(gamma) (", length(gamma),
        ") must equal n_gamma (", n_gamma, ")",
        call. = FALSE
      )
    }
    fp <- c(
      parameters[seq_len(n_beta)],
      gamma,
      parameters[seq(n_beta + 1L, length.out = n_theta)]
    )
  }

  n_expected <- n_beta + n_gamma + n_theta
  if (length(fp) != n_expected) {
    stop(
      "length(full_parameters) (", length(fp),
      ") must equal n_beta + n_gamma + n_theta (", n_expected, ")",
      call. = FALSE
    )
  }

  if (!is.data.frame(info$parameters)) {
    stop("info$parameters must be a data frame", call. = FALSE)
  }
  if (nrow(info$parameters) != n_beta + n_theta) {
    stop(
      "nrow(info$parameters) (", nrow(info$parameters),
      ") must equal n_beta + n_theta (", n_beta + n_theta, ")",
      call. = FALSE
    )
  }

  out_parameters <- info$parameters
  theta_mle <- fp[seq(to = length(fp), length.out = n_theta)]
  if ("log" %in% names(info$theta)) {
    idx <- which(info$theta$log %in% TRUE)
    if (length(idx) > 0L) {
      theta_mle[idx] <- exp(theta_mle[idx])
    }
  }
  out_parameters$mle <- c(fp[seq_len(n_beta)], theta_mle)

  out_gamma <- info$gamma
  out_gamma$mode <- fp[seq(n_beta + 1L, length.out = n_gamma)]
  if (n_gamma > 0L && "gamma_label" %in% names(out_gamma)) {
    rownames(out_gamma) <- out_gamma$gamma_label
  }

  list(
    parameters = out_parameters[, 
    setdiff(names(out_parameters), c("init","lower","upper", "parscale"))],
    gamma = out_gamma[, setdiff(names(out_gamma), c("parscale", "basis", "order"))]
  )
}
