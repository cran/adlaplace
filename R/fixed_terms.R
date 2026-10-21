#' intercept Model Term
#'
#' @description Creates and manages intercept model terms for fixed effects.
#'   With a finite \code{sd}, the intercept is treated as a random effect with
#'   a known-SD Gaussian prior (\code{N(0, sd^2)}), using
#'   \code{random_diagonal} without a theta parameter.
#' @name intercept-class
#' @aliases intercept
#' @docType class
#' @title intercept Model Term
#' @exportClass intercept
#'
#' @section Methods:
#' The following methods are available for `intercept` objects:
#' \describe{
#'   \item{\code{design(term, data)}}{Creates design matrix for intercept term}
#'   \item{\code{precision(term, data)}}{Creates precision matrix for intercept term}
#'   \item{\code{theta_info(term)}}{Extracts theta parameter information}
#'   \item{\code{beta_info(term, data)}}{Extracts beta parameter information}
#'   \item{\code{random_info(term, data)}}{Extracts random effects information}
#' }
NULL


setClass("intercept",
  slots = list(sd = "numeric"),
  contains = "model_term",
  prototype = list(
    name = "intercept",
    label = "intercept",
    formula = . ~ 1,
    ref_value = numeric(0),
    p.order = integer(0),
    knots = numeric(0),
    sd = Inf,
    model_role = factor("fixed", levels = .model_role_levels),
    density = NA_character_,
    ad_kind = NA_character_
  )
)

#' @export
#' @rdname intercept-class
#' @param init Initial value for beta parameter (default: 0)
#' @param lower Lower bound for beta parameter (default: -Inf)
#' @param upper Upper bound for beta parameter (default: Inf)
#' @param parscale Parameter scale for optimization (default: 1)
#' @param sd Prior standard deviation. \code{Inf} (default) keeps a fixed
#'   effect; a finite positive value places a known-SD Gaussian prior and
#'   moves the intercept into the random-effect block.
#' @return A intercept term object
#' @examples
#' # Create a intercept term
#' intercept_term <- intercept(init=2)
#' @export
intercept <- function(init = .my_beta_init,
                      lower = .my_beta_lower,
                      upper = .my_beta_upper,
                      parscale = .my_beta_parscale,
                      sd = Inf) {
  if (length(sd) != 1L || (!is.finite(sd) && !is.infinite(sd)) || sd <= 0) {
    stop("sd must be a single positive numeric value (or Inf)", call. = FALSE)
  }
  known_sd <- is.finite(sd)
  methods::new(
    "intercept",
    label = "intercept",
    init = init,
    lower = lower,
    upper = upper,
    parscale = parscale,
    sd = sd,
    model_role = factor(
      if (known_sd) "random" else "fixed",
      levels = .model_role_levels
    ),
    density = if (known_sd) "random_diagonal" else NA_character_,
    ad_kind = if (known_sd) "random" else NA_character_
  )
}

#' @describeIn intercept-class Creates design matrix for intercept term
#' @param term A intercept term object
#' @param data A data frame containing the term variable
#' @export
setMethod("design", "intercept", function(term, data) {
  matrix(1, nrow(data), ncol=1,
    dimnames = list(rownames(data),"intercept")
    )
})

#' @describeIn intercept-class Extracts beta parameter information for intercept term
#' @param term A intercept term object
#' @param data A data frame containing the term variable
#' @export
setMethod("beta_info", "intercept", function(term, data) {
  if (identical(as.character(term@model_role), "random")) {
    return(NULL)
  }
  data.frame(
    term = "intercept",
    model = "intercept",
    label = term@label,
    order = NA,
    beta_label = "intercept",
    init = term@init,
    lower = term@lower,
    upper = term@upper,
    parscale = term@parscale
  )
})

#' @describeIn intercept-class Precision for known-SD intercept prior
#' @export
setMethod("precision", "intercept", function(term, data) {
  if (!is.finite(term@sd)) {
    return(NULL)
  }
  precision_mat <- Matrix::Diagonal(n = 1L, x = 1 / (term@sd^2))
  dimnames(precision_mat) <- list("intercept", "intercept")
  precision_mat
})

#' @describeIn intercept-class Random info for known-SD intercept prior
#' @export
setMethod("random_info", "intercept", function(term, data) {
  if (!is.finite(term@sd)) {
    return(NULL)
  }
  data.frame(
    term = "intercept",
    model = "intercept",
    label = term@label,
    by = NA, by_labels = NA,
    basis = NA,
    order = NA,
    gamma_label = "intercept"
  )
})

#' Linear Model Term
#'
#' @description Creates and manages linear model terms for fixed effects.
#'   With a finite \code{sd}, coefficients are random effects with a known-SD
#'   Gaussian prior (\code{N(0, sd^2)}).
#' @name linear-class
#' @aliases linear
#' @docType class
#' @title Linear Model Term
#' @exportClass linear
#'
#' @section Methods:
#' The following methods are available for `linear` objects:
#' \describe{
#'   \item{\code{design(term, data)}}{Creates design matrix for linear term}
#'   \item{\code{precision(term, data)}}{Creates precision matrix for linear term}
#'   \item{\code{theta_info(term)}}{Extracts theta parameter information}
#'   \item{\code{beta_info(term, data)}}{Extracts beta parameter information}
#'   \item{\code{random_info(term, data)}}{Extracts random effects information}
#' }
NULL

setClass("linear",
  slots = list(sd = "numeric"),
  contains = "model_term",
  prototype = list(
    ref_value = numeric(0),
    p.order = integer(0),
    knots = numeric(0),
    sd = Inf,
    model_role = factor("fixed", levels = .model_role_levels),
    density = NA_character_,
    ad_kind = NA_character_
  )
)

#' @export
#' @rdname linear-class
#' @param x Variable name (character) or symbol in a formula (e.g. \code{linear(x)} in
#'   \code{model_data()}).
#' @param init Initial value for beta parameter (default: 0)
#' @param lower Lower bound for beta parameter (default: -Inf)
#' @param upper Upper bound for beta parameter (default: Inf)
#' @param parscale Parameter scale for optimization (default: 1)
#' @param sd Prior standard deviation. \code{Inf} (default) keeps a fixed
#'   effect; a finite positive value places a known-SD Gaussian prior and
#'   moves the coefficients into the random-effect block.
#' @return A linear term object
#' @examples
#' # Create a linear term
#' linear_term <- linear(x = "temperature")
#' @export
linear <- function(x,
                   init = .my_beta_init,
                   lower = .my_beta_lower,
                   upper = .my_beta_upper,
                   parscale = .my_beta_parscale,
                   sd = Inf) {
  if (is.symbol(x) || is.name(x)) {
    x <- as.character(x)
  } else if (!is.character(x)) {
    x <- as.character(x)
  }
  if (length(x) != 1L) {
    stop("x must be a single variable name", call. = FALSE)
  }
  if (length(sd) != 1L || (!is.finite(sd) && !is.infinite(sd)) || sd <= 0) {
    stop("sd must be a single positive numeric value (or Inf)", call. = FALSE)
  }
  known_sd <- is.finite(sd)
  methods::new(
    "linear",
    name = x,
    label = paste(x, "linear", sep = "_"),
    formula = stats::as.formula(paste0("~ 0 + ", x)),
    init = init,
    lower = lower,
    upper = upper,
    parscale = parscale,
    sd = sd,
    model_role = factor(
      if (known_sd) "random" else "fixed",
      levels = .model_role_levels
    ),
    density = if (known_sd) "random_diagonal" else NA_character_,
    ad_kind = if (known_sd) "random" else NA_character_
  )
}

#' @describeIn linear-class Creates design matrix for linear term
#' @param term A linear term object
#' @param data A data frame containing the term variable
#' @export
setMethod("design", "linear", function(term, data) {
  res <- Matrix::sparse.model.matrix(term@formula, data, drop.unused.levels = FALSE)
  if (is.factor(data[[term@name]])) {
    res <- res[, -1, drop = FALSE]
  }
  colnames(res) <- paste0(term@name, "_linear_", colnames(res))
  res
})

#' @describeIn linear-class Extracts beta parameter information for linear term
#' @param term A linear term object
#' @param data A data frame containing the term variable
#' @export
setMethod("beta_info", "linear", function(term, data) {
  if (identical(as.character(term@model_role), "random")) {
    return(NULL)
  }
  the_colnames <- colnames(design(term, data))
  the_label <- term@label

  result <- data.frame(
    term = term@name,
    model = "linear",
    label = the_label,
    order = NA,
    beta_label = the_colnames,
    init = term@init,
    lower = term@lower,
    upper = term@upper,
    parscale = term@parscale
  )

  return(result)
})

#' @describeIn linear-class Precision for known-SD linear prior
#' @export
setMethod("precision", "linear", function(term, data) {
  if (!is.finite(term@sd)) {
    return(NULL)
  }
  D <- design(term, data)
  n <- ncol(D)
  if (is.null(n) || n < 1L) {
    return(NULL)
  }
  labels <- colnames(D)
  precision_mat <- Matrix::Diagonal(n = n, x = rep_len(1 / (term@sd^2), n))
  dimnames(precision_mat) <- list(labels, labels)
  precision_mat
})

#' @describeIn linear-class Random info for known-SD linear prior
#' @export
setMethod("random_info", "linear", function(term, data) {
  if (!is.finite(term@sd)) {
    return(NULL)
  }
  the_colnames <- colnames(design(term, data))
  data.frame(
    term = term@name,
    model = "linear",
    label = term@label,
    by = NA, by_labels = NA,
    basis = NA,
    order = NA,
    gamma_label = the_colnames
  )
})

#' Fixed Polynomial Model Term
#'
#' @description Creates and manages fixed polynomial model terms.
#' @name fpoly-class
#' @aliases fpoly
#' @docType class
#' @exportClass fpoly
#'
#' @section Methods:
#' The following methods are available for `fpoly` objects:
#' \describe{
#'   \item{\code{design(term, data)}}{Creates design matrix for fpoly term}
#'   \item{\code{precision(term, data)}}{Creates precision matrix for fpoly term}
#'   \item{\code{theta_info(term)}}{Extracts theta parameter information}
#'   \item{\code{beta_info(term, data)}}{Extracts beta parameter information}
#'   \item{\code{random_info(term, data)}}{Extracts random effects information}
#' }
NULL

setClass("fpoly",
         slots = list(
           init = "numeric",
           lower = "numeric",
           upper = "numeric",
           parscale = "numeric",
           basis_scale = "numeric"
         ),
         contains = "model_term",
         prototype = prototype(
           knots = numeric(0),
           init = numeric(0),
           lower = numeric(0),
           upper = numeric(0),
           parscale = numeric(0),
           basis_scale = 1,
           model_role = factor("fixed", levels = .model_role_levels),
           density = NA_character_,
           ad_kind = NA_character_
         )
)

#' @rdname fpoly-class
#' @param x Variable name.
#' @param x Variable name.
#' @param p Polynomial degree (default: 2).
#' @param ref_value Reference value for the polynomial.
#' @param init Initial values for beta parameters.
#' @param lower Lower bounds for beta parameters.
#' @param upper Upper bounds for beta parameters.
#' @param parscale Parameter scales for optimization.
#' @param basis_scale Positive scale for the covariate. Column \code{k} of the
#'   raw polynomial is divided by \code{basis_scale^k}. \code{\link{iwp}} passes
#'   the mean knot spacing when the polynomial is the fixed part of an IWP.
#' @param term A `fpoly` term object.
#' @param data A data frame containing the variables used in the term.
#' @return A `fpoly` term object.
#' @export
#' @examples
#' # Create a fixed polynomial term
#' fpoly_term <- fpoly(x = "temp", p = 3, ref_value = 15)
fpoly <- function(x, p = 2, ref_value = 0,
                  init = .my_beta_init,
                  lower = .my_beta_lower,
                  upper = .my_beta_upper,
                  parscale = .my_beta_parscale,
                  basis_scale = 1) {
  if (length(basis_scale) != 1 || !is.finite(basis_scale) || basis_scale <= 0) {
    stop("basis_scale must be a single positive finite value")
  }
  methods::new("fpoly",
    name = x,
    label = paste(x, "fpoly", sep = "_"),
    formula = stats::as.formula(paste0("~ 0 + ", x), env = new.env()),
    p.order = as.integer(p),
    ref_value = ref_value,
    init = rep_len(init, p),
    lower = rep_len(lower, p),
    upper = rep_len(upper, p),
    parscale = rep_len(parscale, p),
    basis_scale = basis_scale
  )
}

#' @describeIn fpoly-class Design method for fpoly objects
#' @export
#' @param term A `fpoly` term object.
#' @param data A data frame containing the variables used in the term.
#' @return A design matrix for the fixed polynomial term, or NULL if p.order is 0.
setMethod("design", "fpoly", function(term, data) {
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
  seq_order <- seq.int(1, length.out = term@p.order)

  colnames(D) <- paste0(term@name, "_fpoly_", seq_order)
  D
})

#' @describeIn fpoly-class Beta info method for fpoly objects
#' @export
#' @param term A `fpoly` term object.
#' @param data A data frame containing the variables used in the term.
#' @return A data frame containing beta parameter information for the fixed polynomial term.
setMethod("beta_info", "fpoly", function(term, data) {
  the_colnames <- colnames(design(term, data))
  the_label <- term@label

  result <- data.frame(
    term = term@name,
    model = "fpoly",
    label = the_label,
    order = NA,
    beta_label = the_colnames,
    init = term@init,
    lower = term@lower,
    upper = term@upper,
    parscale = term@parscale
  )

  return(result)
})
