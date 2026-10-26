#' Predictive standard deviation of a random term
#'
#' Generic for the unit-scale predictive SD
#' \eqn{\sigma_x(h) = \mathrm{SD}\bigl[f(x+h)\mid f(x), f'(x),\ldots,f^{(p-1)}(x)\bigr]}
#' of a random model term. Returns `sd` times the unit PSD; the default
#' `sd = 1` gives the prior-conversion factor used by [psd_prior()].
#'
#' @param term A [model_term-class] (or a list returned by a constructor such as
#'   [iwp()] / `mgp()`).
#' @param h Numeric step size(s). Recycled against `x` when both are vectors.
#' @param x Optional evaluation location(s). Ignored by stationary-increment
#'   models such as [iwp()].
#' @param sd Positive scale for the term's SD hyperparameter (default 1).
#' @param ... Passed to methods.
#'
#' @return Numeric vector of predictive SDs, same length as the recycled
#'   `(h, x)` grid.
#'
#' @seealso [psd_prior()], [iwp()]
#' @export
setGeneric("psd", function(term, h, x = NULL, sd = 1, ...) {
  standardGeneric("psd")
})

#' @rdname psd
#' @export
setMethod("psd", "model_term", function(term, h, x = NULL, sd = 1, ...) {
  stop("no psd() method for class ", paste(class(term), collapse = "/"), call. = FALSE)
})

#' @rdname psd
#' @export
setMethod("psd", "list", function(term, h, x = NULL, sd = 1, ...) {
  target <- psd_term_from_list(term)
  psd(target, h = h, x = x, sd = sd, ...)
})

#' @rdname psd
#' @export
setMethod("psd", "iwp", function(term, h, x = NULL, sd = 1, ...) {
  if (!is.null(x)) {
    # Stationary increments: location does not enter the formula.
    x <- NULL
  }
  h <- as.numeric(h)
  if (!length(h) || any(!is.finite(h))) {
    stop("h must be finite", call. = FALSE)
  }
  sd <- as.numeric(sd)[1L]
  if (!is.finite(sd) || sd < 0) {
    stop("sd must be a non-negative finite value", call. = FALSE)
  }
  p <- as.integer(term@p.order)[1L]
  if (!is.finite(p) || p < 1L) {
    stop("iwp order p must be a positive integer", call. = FALSE)
  }
  shifted <- term@knots - term@ref_value
  s <- iwp_mean_spacing(shifted)
  if (!is.finite(s) || s <= 0) {
    s <- 1
  }
  hs <- abs(h) / s
  denom <- (2 * p - 1) * (factorial(p - 1)^2)
  abs(sd) * sqrt((hs^(2 * p - 1)) / denom)
})

#' PC prior on the predictive SD of a random term
#'
#' Builds an exponential PC prior
#' \eqn{P\bigl(\sigma_x(h) > u\bigr) = \alpha} on the predictive SD of `term`,
#' returning a [prior()] term with rate
#' \eqn{-\log(\alpha)\,\mathrm{psd}(\mathrm{term}, h, x, \mathrm{sd}=1)/u}.
#'
#' @param term A term with a [psd()] method, or a list returned by [iwp()] /
#'   `mgp()` (the element that carries a theta is selected).
#' @param u Positive threshold on the predictive-SD scale.
#' @param alpha Tail probability (default 0.5).
#' @param h Positive step size for the predictive SD.
#' @param x Optional location passed to [psd()] (required by some nonstationary
#'   terms such as `mgp`).
#'
#' @return A [prior()] term targeting the SD theta of `term`.
#' @seealso [psd()], [prior()]
#' @export
psd_prior <- function(term, u, alpha = 0.5, h, x = NULL) {
  if (is.list(term) && !is.object(term)) {
    term <- psd_term_from_list(term)
  }
  if (!is.numeric(u) || length(u) != 1L || !is.finite(u) || u <= 0) {
    stop("u must be a single positive finite value", call. = FALSE)
  }
  if (!is.numeric(alpha) || length(alpha) != 1L || !is.finite(alpha) ||
    alpha <= 0 || alpha >= 1) {
    stop("alpha must be in (0, 1)", call. = FALSE)
  }
  if (missing(h) || !is.numeric(h) || length(h) != 1L || !is.finite(h) || h <= 0) {
    stop("h must be a single positive finite value", call. = FALSE)
  }
  unit <- tryCatch(
    psd(term, h = h, x = x, sd = 1),
    error = function(e) {
      stop(conditionMessage(e), call. = FALSE)
    }
  )
  if (length(unit) != 1L || !is.finite(unit) || unit <= 0) {
    stop("psd(term, h, x, sd = 1) must return a single positive value", call. = FALSE)
  }
  theta_label <- psd_theta_label(term)
  prior(
    theta = theta_label,
    rate = -log(alpha) * as.numeric(unit) / u
  )
}

#' Pick the list element that has a theta (and a psd method).
#' @noRd
psd_term_from_list <- function(term_list) {
  if (!length(term_list)) {
    stop("empty term list", call. = FALSE)
  }
  has_theta <- vapply(term_list, function(tm) {
    inherits(tm, "model_term") && !is.null(theta_info(tm))
  }, logical(1))
  if (!any(has_theta)) {
    stop("no element of the list has a theta parameter", call. = FALSE)
  }
  term_list[[which(has_theta)[1L]]]
}

#' @noRd
psd_theta_label <- function(term) {
  info <- theta_info(term)
  if (is.null(info) || !is.data.frame(info) || nrow(info) < 1L) {
    stop("term has no theta parameter for psd_prior()", call. = FALSE)
  }
  if (nrow(info) > 1L) {
    # Prefer an explicit "_sd" label when present (e.g. rsmatern).
    sd_rows <- grep("_sd$", info$label)
    if (length(sd_rows) == 1L) {
      return(info$label[[sd_rows]])
    }
    stop(
      "psd_prior() needs a unique SD theta; found labels: ",
      paste(info$label, collapse = ", "),
      call. = FALSE
    )
  }
  info$label[[1L]]
}
