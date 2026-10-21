setClass("iwp",
  slots = list(),
  contains = "model_term",
  prototype = prototype(
    # Default values for iwp-specific behavior
    model_role = factor("random", levels = .model_role_levels),
    density = "random_diagonal",
    ad_kind = "random"
  )
)

#' Integrated Wiener Process Term
#'
#' @description Creates and manages integrated Wiener process (IWP) model terms.
#' @name iwp-class
#' @docType class
#' @exportClass iwp
#'
#' @section Methods:
#' The following methods are available for `iwp` objects:
#' \describe{
#'   \item{\code{design(term, data)}}{Creates design matrix for IWP term}
#'   \item{\code{precision(term, data)}}{Creates precision matrix for IWP term}
#'   \item{\code{theta_info(term)}}{Extracts theta parameter information}
#'   \item{\code{beta_info(term, data)}}{Extracts beta parameter information}
#'   \item{\code{random_info(term, data)}}{Extracts random effects information}
#' }
#'
#' @return
#' \describe{
#'   \item{\code{design}}{Numeric design matrix for the integrated Wiener
#'     process basis, with one column per basis function and one row per
#'     observation. Entries are the basis evaluated at \code{data}.}
#'   \item{\code{precision}}{Sparse precision matrix for those basis
#'     coefficients. The penalty is the integrated Wiener process prior.}
#'   \item{\code{theta_info}}{One-row data frame describing the
#'     standard-deviation parameter: label, initial value, bounds, parameter
#'     scale, and whether optimization is on the log scale.}
#'   \item{\code{random_info}}{Data frame with one row per random coefficient,
#'     including \code{gamma_label}, the name of that coefficient in a fitted
#'     model.}
#' }
NULL

#' Integrated Wiener Process Term Constructor
#'
#' @description Creates an integrated Wiener process (IWP) model term.
#' The basis is divided by the mean knot spacing to the power \code{p}, and the
#' penalty weights are the knot spacings divided by that mean spacing. Design
#' entries and the standard-deviation parameter are then unchanged if \code{x}
#' and the knots are rescaled by the same factor.
#' @name iwp
#' @param x Variable name.
#' @param p Order of the integrated Wiener process (default: 2).
#' @param ref_value Reference value for the basis.
#' @param knots Vector of knot locations.
#' @param range Range of the data (optional).
#' @param init Initial values for theta parameters.
#' @param lower Lower bounds for theta parameters.
#' @param upper Upper bounds for theta parameters.
#' @param parscale Parameter scales for optimization.
#' @param log Whether theta is optimized on the log scale.
#' @param boundary_is_random Whether boundary should be treated as random.
#' @param include_poly Whether to include polynomial terms.
#' @return A list containing the `iwp` term object and optionally polynomial terms.
#' @examples
#' knots <- seq(0, 1, length.out = 5)
#' iwp_term <- iwp(x = "age", knots = knots)
#' iwp_term
#' @export
iwp <- function(
  x, p = 2,
  ref_value = 0,
  knots,
  range = NULL,
  init = .my_theta_init,
  lower = .my_theta_lower,
  upper = .my_theta_upper,
  parscale = .my_theta_parscale,
  log = TRUE,
  boundary_is_random = TRUE,
  include_poly = TRUE
) {
  # Check all arguments other than knots are length 1. knots must be length > 1
  if (length(x) != 1) stop("x must be a single variable name")
  if (length(p) != 1) stop("p must be a single value")
  if (length(ref_value) != 1) stop("ref_value must be a single value")
  if (!is.null(range) && length(range) != 2) stop("range must be a vector of length 2")
  if (length(init) != 1) stop("init must be a single value")
  if (length(lower) != 1) stop("lower must be a single value")
  if (length(upper) != 1) stop("upper must be a single value")
  if (length(parscale) != 1) stop("parscale must be a single value")
  if (length(log) != 1) stop("log must be a single value")
  if (length(knots) < 2) stop("knots must have length >= 2")
  if (length(boundary_is_random) != 1) stop("boundary_is_random must be a single value")
  if (length(include_poly) != 1) stop("include_poly must be a single value")

  x <- strip_term_name(as.character(x))
  the_f <- stats::as.formula(paste0("~ 0 + ", x), env = new.env())
  result <- list()
  iwp_name <- paste(c(x, "iwp"), collapse = "_")

  ref_value <- ref_align(ref_value, knots)
  knot_spacing <- iwp_mean_spacing(knots)
  if (!is.finite(knot_spacing) || knot_spacing <= 0) {
    knot_spacing <- 1
  }

  result[[iwp_name]] <- methods::new("iwp",
    name = x,
    label = iwp_name,
    formula = the_f,
    p.order = as.integer(p),
    ref_value = ref_value,
    knots = knots,
    init = init[1],
    lower = lower[1],
    upper = upper[1],
    parscale = parscale[1],
    log = log[1]
    # type is already set in prototype, no need to repeat
  )
  if (p == 1) {
    include_poly <- FALSE
  }
  if (include_poly) {
    if (boundary_is_random) {
      poly_name <- paste(c(x, "rpoly"), collapse = "_")
      result[[poly_name]] <- rpoly(
        x = x,
        p = p - 1,
        ref_value = ref_value,
        basis_scale = knot_spacing
      )
    } else {
      poly_name <- paste(c(x, "fpoly"), collapse = "_")
      result[[poly_name]] <- fpoly(
        x = x,
        p = p - 1,
        ref_value = ref_value,
        basis_scale = knot_spacing
      )
    }
  }
  result
}


local_poly_positive_axis <- function(knots, refined_x, p) {
  dif <- diff(knots)
  nn <- length(refined_x)
  n <- length(knots)
  n_cols <- n - 1L
  if (nn < 1L || n_cols < 1L) {
    return(Matrix::sparseMatrix(
      i = integer(0),
      j = integer(0),
      x = numeric(0),
      dims = c(nn, max(0L, n_cols)),
      repr = "C"
    ))
  }

  # Upper bound: every (row, col) may be nonzero past the leftmost knots.
  max_nnz <- as.integer(nn) * as.integer(n_cols)
  i_idx <- integer(max_nnz)
  j_idx <- integer(max_nnz)
  x_val <- numeric(max_nnz)
  nnz <- 0L
  fact_p <- factorial(p)
  k_seq <- seq_len(p)
  fact_k <- factorial(k_seq)
  fact_pk <- factorial(p - k_seq)

  for (j in seq_len(nn)) {
    xj <- refined_x[[j]]
    for (i in seq_len(n_cols)) {
      if (xj <= knots[[i]]) {
        next
      }
      if (xj <= knots[[i + 1L]] && xj >= knots[[i]]) {
        val <- (1 / fact_p) * (xj - knots[[i]])^p
      } else {
        val <- sum(
          (dif[[i]]^k_seq) * ((xj - knots[[i + 1L]])^(p - k_seq)) /
            (fact_k * fact_pk)
        )
      }
      if (val != 0) {
        nnz <- nnz + 1L
        i_idx[[nnz]] <- j
        j_idx[[nnz]] <- i
        x_val[[nnz]] <- val
      }
    }
  }

  if (nnz < max_nnz) {
    length(i_idx) <- nnz
    length(j_idx) <- nnz
    length(x_val) <- nnz
  }
  Matrix::sparseMatrix(
    i = i_idx,
    j = j_idx,
    x = x_val,
    dims = c(nn, n_cols),
    repr = "C"
  )
}

local_poly <- function(knots, refined_x, p) {
  if (min(knots) >= 0) {
    local_poly_positive_axis(knots, refined_x, p)
  } else if (max(knots) <= 0) {
    knots_neg <- unique(sort(ifelse(knots < 0, -knots, 0)))
    refined_x_neg <- ifelse(refined_x < 0, -refined_x, 0)
    local_poly_positive_axis(knots_neg, refined_x_neg, p)
  } else {
    knots_neg <- unique(sort(ifelse(knots < 0, -knots, 0)))
    refined_x_neg <- ifelse(refined_x < 0, -refined_x, 0)
    knots_pos <- unique(sort(ifelse(knots > 0, knots, 0)))
    refined_x_pos <- ifelse(refined_x > 0, refined_x, 0)
    cbind(
      local_poly_positive_axis(knots_neg, refined_x_neg, p),
      local_poly_positive_axis(knots_pos, refined_x_pos, p)
    )
  }
}

compute_weights_precision <- function(knots) {
  if (min(knots) >= 0) {
    methods::as(diag(diff(knots)), "matrix")
  } else if (max(knots) < 0) {
    knots_neg <- knots
    knots_neg <- unique(sort(ifelse(knots < 0, -knots, 0)))
    methods::as(diag(diff(knots_neg)), "matrix")
  } else {
    knots_neg <- knots
    knots_neg <- unique(sort(ifelse(knots < 0, -knots, 0)))
    knots_pos <- knots
    knots_pos <- unique(sort(ifelse(knots > 0, knots, 0)))
    d1 <- diff(knots_neg)
    d2 <- diff(knots_pos)
    Precweights1 <- diag(d1, nrow = length(d1))
    Precweights2 <- diag(d2, nrow = length(d2))
    methods::as(Matrix::bdiag(Precweights1, Precweights2), "matrix")
  }
}

#' Align Reference Value to Nearest Knot
#'
#' @description Aligns a reference value to the nearest knot in a set of knots.
#' This is useful for ensuring that reference values used in polynomial terms
#' are aligned with the actual knot positions.
#'
#' @param ref_value A numeric value representing the desired reference point.
#' @param knots A numeric vector of knot positions.
#'
#' @return The knot value that is closest to the specified reference value.
#'
#' @examples
#' # Align reference value to nearest knot
#' knots <- c(10, 20, 30, 40, 50)
#' adlaplace::ref_align(23, knots) # Returns 20
#'
#' @export
#' @keywords internal
ref_align <- function(ref_value, knots) {
  knots[which.min(abs(knots - ref_value))]
}

#' Mean positive gap between finite knots.
#'
#' @keywords internal
iwp_mean_spacing <- function(knots) {
  knots <- sort(unique(knots[is.finite(knots)]))
  if (length(knots) < 2L) {
    return(NA_real_)
  }
  gaps <- diff(knots)
  gaps <- gaps[gaps > 0]
  if (!length(gaps)) {
    return(NA_real_)
  }
  mean(gaps)
}

iwp_basis_divisor <- function(knots, p) {
  spacing <- iwp_mean_spacing(knots)
  if (!is.finite(spacing) || spacing <= 0 || p <= 0) {
    return(1)
  }
  spacing^p
}

# Method definitions for iwp class
#' @describeIn iwp-class Creates design matrix for IWP term
#' @param term An iwp term object
#' @param data A data frame containing the term variable
#' @export
setMethod("design", "iwp", function(term, data) {
  if (any(data[[term@name]] < min(term@knots)) || any(data[[term@name]] > max(term@knots))) {
    warning("knots don't span the range of x ", term@name)
    print(range(term@knots))
    print(range(data[[term@name]]))
  }

  shifted_knots <- term@knots - term@ref_value
  result <- local_poly(
    shifted_knots,
    data[[term@name]] - term@ref_value,
    term@p.order
  )
  result <- result / iwp_basis_divisor(shifted_knots, term@p.order)

  knots_string <- formatC(seq.int(ncol(result)),
    width = ceiling(log10(ncol(result))), flag = "0"
  )

  colnames(result) <- paste0(term@name, "_iwp_k", knots_string)
  result
})

#' @describeIn iwp-class Creates precision matrix for IWP term
#' @param term An iwp term object
#' @param data A data frame containing the term variable
#' @export
setMethod("precision", "iwp", function(term, data) {
  shifted_knots <- term@knots - term@ref_value
  weights <- compute_weights_precision(shifted_knots)
  spacing <- iwp_mean_spacing(shifted_knots)
  if (is.finite(spacing) && spacing > 0) {
    weights <- weights / spacing
  }
  result <- Matrix::Matrix(weights)

  knots_string <- formatC(seq.int(nrow(result)),
    width = ceiling(log10(nrow(result))), flag = "0"
  )

  dimnames(result) <- list(
    paste0(term@name, "_iwp_k", knots_string)
  )[c(1, 1)]
  result
})

#' @describeIn iwp-class Extracts theta parameter information for IWP term
#' @param term An iwp term object
#' @export
setMethod("theta_info", "iwp", function(term) {
  result <- data.frame(
    term = term@name, model = "iwp",
    label = term@label,
    init = term@init,
    lower = term@lower, upper = term@upper,
    parscale = term@parscale,
    model_role = term@model_role,
    log = term@log
  )
  return(result)
})

#' @describeIn iwp-class Extracts random effects information for IWP term
#' @param term An iwp term object
#' @param data A data frame containing the term variable
#' @export
setMethod("random_info", "iwp", function(term, data) {
  basis <- seq(1, length.out = length(term@knots) - 1)

  result <- expand.grid(
    term = term@name,
    model = "iwp",
    label = term@label,
    by = NA, # iwp doesn't have hierarchical structure
    by_labels = NA,
    basis = basis,
    order = term@p.order,
    stringsAsFactors = FALSE
  )

  bnumPad <- formatC(result$basis,
    width = max(ceiling(c(1, log10(max(result$basis)))), na.rm = TRUE),
    flag = "0"
  )
  result$gamma_label <- paste0(result$label, "_k", bnumPad)

  result
})
