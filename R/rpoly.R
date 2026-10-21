#' Random Polynomial Model Term
#'
#' @description Creates and manages random polynomial model terms.
#' @name rpoly-class
#' @aliases rpoly
#' @docType class
#' @title Random Polynomial Model Term
#' @exportClass rpoly
#'
#' @section Methods:
#' The following methods are available for `rpoly` objects:
#' \describe{
#'   \item{\code{design(term, data)}}{Creates design matrix for rpoly term}
#'   \item{\code{precision(term, data)}}{Creates precision matrix for rpoly term}
#'   \item{\code{theta_info(term)}}{Extracts theta parameter information}
#'   \item{\code{beta_info(term, data)}}{Extracts beta parameter information}
#'   \item{\code{random_info(term, data)}}{Extracts random effects information}
#' }
NULL

setClass("rpoly",
  slots = list(
    sd = "numeric",
    basis_scale = "numeric"
  ),
  contains = "model_term",
  prototype = prototype(
    knots = numeric(0),
    sd = numeric(0),
    basis_scale = 1,
    model_role = factor("random", levels = .model_role_levels),
    density = "random_diagonal",
    ad_kind = "random"
  )
)

#' @rdname rpoly-class
#' @param x Variable name.
#' @param p Polynomial degree (default: 2).
#' @param ref_value Reference value for the polynomial.
#' @param sd Standard deviation for random effects.
#' @param basis_scale Positive scale for the covariate. Column \code{k} of the
#'   raw polynomial is divided by \code{basis_scale^k}. The default \code{1}
#'   leaves the raw polynomial unchanged. \code{\link{iwp}} passes the mean
#'   knot spacing so the polynomial matches the normalized IWP basis.
#' @param term A `rpoly` term object.
#' @param data A data frame containing the variables used in the term.
#' @return A `rpoly` term object.
#' @export
rpoly <- function(x, p = 2, ref_value = 0, sd = Inf, basis_scale = 1) {
  if (!missing(sd) && (length(sd) != 1 || sd <= 0)) {
    stop("sd must be a single positive numeric value")
  }
  if (!missing(p) && (length(p) != 1 || (round(p) != p) || p < 0)) {
    stop("p must be a single non-negative integer")
  }
  if (!missing(ref_value) && length(ref_value) != 1) {
    stop("ref_value must be a single value")
  }
  if (length(basis_scale) != 1 || !is.finite(basis_scale) || basis_scale <= 0) {
    stop("basis_scale must be a single positive finite value")
  }

  methods::new("rpoly",
    name = x,
    label = paste(x, "rpoly", sep = "_"),
    formula = stats::as.formula(paste0("~ 0 + ", x), env = new.env()),
    p.order = as.integer(p),
    ref_value = ref_value,
    sd = rep_len(sd, p),
    basis_scale = basis_scale
  )
}

scale_raw_poly_columns <- function(D, basis_scale) {
  if (length(basis_scale) != 1L || !is.finite(basis_scale) || basis_scale == 1) {
    return(D)
  }
  for (j in seq_len(ncol(D))) {
    D[, j] <- D[, j] / basis_scale^j
  }
  D
}

#' @describeIn rpoly-class Design method for rpoly objects
#' @export
#' @param term A `rpoly` term object.
#' @param data A data frame containing the variables used in the term.
#' @return A design matrix for the random polynomial term.
setMethod("design", "rpoly", function(term, data) {
  if (term@p.order == 0) {
    return(NULL)
  }
  D <- stats::poly(
    data[[term@name]] - term@ref_value,
    degree = term@p.order,
    raw = TRUE
  )
  D <- D[, 1:ncol(D), drop = FALSE]
  D <- scale_raw_poly_columns(D, term@basis_scale)

  colnames(D) <- paste0(term@name, "_rpoly_", seq.int(1, length.out = term@p.order))
  D
})

#' @describeIn rpoly-class Precision method for rpoly objects
#' @export
#' @param term A `rpoly` term object.
#' @param data A data frame containing the variables used in the term.
#' @return A precision matrix for the random polynomial term.
setMethod("precision", "rpoly", function(term, data) {
  if (term@p.order == 0) {
    return(NULL)
  }
  if (any(term@sd == Inf)) {
    return(NULL)
  }

  # Create precision matrix based on standard deviation
  p <- term@p.order
  sd_values <- rep_len(term@sd, p)

  # For random effects, we typically use identity matrix scaled by 1/sd^2
  precision_mat <- Matrix::Diagonal(n = p, x = 1 / (sd_values^2))
  dimnames(precision_mat) <- list(
    paste0(term@name, "_rpoly_", seq.int(1, length.out = term@p.order))
  )[c(1, 1)]
  return(precision_mat)
})

#' @describeIn rpoly-class Random info method for rpoly objects
#' @export
#' @param term A `rpoly` term object.
#' @param data A data frame containing the variables used in the term.
#' @return A data frame containing random effects information for the random polynomial term.
setMethod("random_info", "rpoly", function(term, data) {
  if (term@p.order == 0) {
    return(NULL)
  }

  the_label <- term@label
  the_colnames <- paste0(the_label, "_", seq.int(1, length.out = term@p.order))

  result <- data.frame(
    term = term@name,
    model = "rpoly",
    label = the_label,
    by = NA, by_labels = NA,
    basis = NA,
    order = seq.int(1, term@p.order),
    gamma_label = the_colnames
  )

  return(result)
})

