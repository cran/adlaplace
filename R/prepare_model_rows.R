#' @keywords internal
parse_model_terms <- function(formula, verbose = FALSE) {
  if (methods::is(formula, "formula")) {
    collect_terms(formula, verbose = verbose)
  } else {
    terms <- formula
    if (methods::is(terms, "model_term")) {
      terms <- list(terms)
    }
    terms
  }
}

#' @keywords internal
required_model_variables <- function(terms) {
  covariates <- unique(unlist(lapply(terms, function(t) t@name)))
  covariates <- unique(unlist(strsplit(covariates, ":", fixed = TRUE)))

  random_slope_terms <- unique(unlist(lapply(terms, function(t) {
    if ("mult" %in% methods::slotNames(t)) {
      t@mult
    } else {
      character(0)
    }
  })))

  size_vars <- unique(unlist(lapply(terms, function(t) {
    if (methods::is(t, "binomial") && length(t@size) == 1L && nzchar(t@size)) {
      t@size
    } else {
      character(0)
    }
  })))

  strata_vars <- unique(unlist(lapply(terms, function(t) {
    if (methods::is(t, "model_term") &&
        identical(t@ad_kind, "observations") &&
        "by" %in% methods::slotNames(t) &&
        is.character(t@by)) {
      t@by
    } else {
      character(0)
    }
  })))

  unique(c(covariates, strata_vars, random_slope_terms, size_vars))
}

#' Drop incomplete rows and sort case-crossover strata
#'
#' @param data Data frame.
#' @param terms Named list of model term objects.
#' @param verbose Print row counts and variable names.
#' @return Subset and reordered data frame.
#' @keywords internal
prepare_model_rows <- function(data, terms, verbose = FALSE) {
  required_vars <- intersect(required_model_variables(terms), names(data))

  if (verbose) {
    cat("variables:\n")
    print(required_vars)
  }

  if (length(required_vars) > 0L) {
    data <- data[stats::complete.cases(data[, required_vars, drop = FALSE]), , drop = FALSE]
  }

  if (verbose) {
    cat("data has ", nrow(data), " rows\n")
  }

  obs_terms <- Filter(function(t) {
    methods::is(t, "model_term") && identical(t@ad_kind, "observations")
  }, terms)

  for (term in obs_terms) {
    outcome_var <- term@name
    if (outcome_var %in% names(data) && anyNA(data[[outcome_var]])) {
      warning("missing values in outcome, treating as zeros", call. = FALSE)
      data[[outcome_var]][is.na(data[[outcome_var]])] <- 0
    }
  }

  strata_vars <- unique(unlist(lapply(obs_terms, function(t) {
    if ("by" %in% methods::slotNames(t) && length(t@by)) {
      t@by
    } else {
      character(0)
    }
  })))
  strata_vars <- intersect(strata_vars, names(data))
  if (length(strata_vars) > 0L) {
    ord <- do.call(order, data[, strata_vars, drop = FALSE])
    data <- data[ord, , drop = FALSE]
  }

  rownames(data) <- NULL
  data
}

#' @keywords internal
parameter_density <- function(obs_ad_fun) {
  if (grepl("_extra$", obs_ad_fun)) {
    return(obs_ad_fun)
  }
  subbed <- sub("_obs$", "_extra", obs_ad_fun, perl = TRUE)
  if (!identical(subbed, obs_ad_fun)) {
    return(subbed)
  }
  paste0(obs_ad_fun, "_extra")
}
