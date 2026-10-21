#' Dirichlet-multinomial case-crossover response term
#'
#' @description Model term for the case-crossover Dirichlet-multinomial log density
#' registered as \code{dirichlet_multinomial}.
#' @name dirichlet_multinom-class
#' @aliases dirichlet_multinom-class
#' @docType class
#' @title Dirichlet-multinomial response term
#' @exportClass dirichlet_multinom
#'
#' @slot by Character vector of stratification variables for the case-crossover design.
#'
#' @section Slots (inherited from \code{model}):
#' \describe{
#'   \item{\code{ad_pack}}{Character scalar \code{"dirichlet_multinomial"}.}
#'   \item{\code{ad_kind}}{Character scalar \code{"observations"}.}
#'   \item{\code{package}}{Character scalar \code{"adlaplace"}.}
#' }
NULL

setClass(
  "dirichlet_multinom",
  slots = c(by = "character"),
  contains = "model_term",
  prototype = prototype(
    knots = numeric(0),
    ref_value = numeric(0),
    p.order = as.integer(1),
    init = 1e-3,
    lower = 0,
    upper = Inf,
    parscale = 1,
    model_role = factor("response", levels = .model_role_levels),
    density = "dirichlet_multinomial",
    ad_kind = "observations",
    package = "adlaplace",
    by = character(0)
  )
)

#' Dirichlet-multinomial case-crossover response term
#'
#' @param x Outcome variable name.
#' @param by Character vector of stratification variables (e.g. \code{c("year", "month", "dow")}).
#' @param init Initial value for overdispersion \code{sd}.
#' @param lower Lower bound for \code{sd}.
#' @param upper Upper bound for \code{sd}.
#' @param parscale Parameter scale for optimization.
#' @param log Whether theta is optimized on the log scale.
#' @return A \code{dirichlet_multinom} object.
#' @export
dirichlet_multinom <- function(x,
                               by,
                               init = 1e-3,
                               lower = 0,
                               upper = Inf,
                               parscale = 1,
                               log = TRUE) {
  x <- strip_term_name(as.character(x))
  by <- as.character(by)
  methods::new(
    "dirichlet_multinom",
    name = x,
    label = paste(x, "dirichlet_multinom", sep = "_"),
    formula = stats::as.formula(paste(x, "~."), env = new.env()),
    init = init,
    lower = lower,
    upper = upper,
    parscale = parscale,
    log = log,
    by = by
  )
}

#' Case-crossover stratum map for AD shards
#'
#' @param term A \code{dirichlet_multinom} term object.
#' @param data Analysis-ready data frame.
#' @return Sparse \code{ngCMatrix} mapping observations to strata columns.
#' @describeIn dirichlet_multinom-class Case-crossover stratum map for AD shards.
#' @export
setMethod("elgm_matrix", "dirichlet_multinom", function(term, data) {
  if (!length(term@by)) {
    stop("dirichlet_multinom(..., by = ...) is required", call. = FALSE)
  }
  missing_by <- setdiff(term@by, names(data))
  if (length(missing_by)) {
    stop(
      "by variables not in data: ", paste(missing_by, collapse = ", "),
      call. = FALSE
    )
  }

  by_vars <- data[term@by]
  by_codes <- lapply(by_vars, function(col) {
    if (is.factor(col)) as.integer(col) else col
  })
  # lex.order = TRUE: first by-variable is the slowest (like order())
  stratum <- interaction(by_codes, drop = TRUE, lex.order = TRUE)
  stratum_id <- as.integer(stratum)
  if (!any(!is.na(stratum_id))) {
    stop("by variables did not form strata", call. = FALSE)
  }
  if (isTRUE(mean(table(stratum_id) <= 1L) > 0.5)) {
    warning(
      "more than half of strata only have one observation, ",
      "might be missing the zeros in dataset?"
    )
  }

  outcome <- as.numeric(data[[term@name[[1L]]]])
  n_outcome <- as.numeric(rowsum(outcome, group = stratum_id, na.rm = TRUE))
  n_days <- as.integer(tabulate(stratum_id))
  keep <- which(n_outcome > 0 & n_days > 1L)
  stratum_sub <- match(stratum_id, keep)
  which_keep <- which(!is.na(stratum_sub))

  first <- match(seq_len(nlevels(stratum)), stratum_id)
  stratum_names <- do.call(
    paste,
    c(lapply(by_vars, function(col) as.character(col)[first]), list(sep = "."))
  )

  Matrix::sparseMatrix(
    i = which_keep,
    j = stratum_sub[which_keep],
    dims = c(nrow(data), length(keep)),
    dimnames = list(NULL, stratum_names[keep])
  )
})

#' @describeIn dirichlet_multinom-class Design method (not used).
#' @export
setMethod("design", "dirichlet_multinom", function(term, data) {
  NULL
})

#' @describeIn dirichlet_multinom-class Precision method (not used).
#' @export
setMethod("precision", "dirichlet_multinom", function(term, data) {
  NULL
})

#' @describeIn dirichlet_multinom-class Theta info for overdispersion \code{sd}.
#' @export
setMethod("theta_info", "dirichlet_multinom", function(term) {
  data.frame(
    term = term@name,
    model = "dirichlet_multinom",
    label = paste0(term@label, "_sd"),
    init = term@init,
    lower = term@lower,
    upper = term@upper,
    parscale = term@parscale,
    model_role = term@model_role,
    log = term@log,
    stringsAsFactors = FALSE
  )
})

#' @describeIn dirichlet_multinom-class Beta info method (not used).
#' @export
setMethod("beta_info", "dirichlet_multinom", function(term, data) {
  NULL
})

#' @describeIn dirichlet_multinom-class Random info method (not used).
#' @export
setMethod("random_info", "dirichlet_multinom", function(term, data) {
  NULL
})
