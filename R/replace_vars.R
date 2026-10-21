#' Replace data-column names in model terms
#'
#' Rewrites the data-column references on model terms (and formulas / term lists)
#' according to a named character map. Used by modular-inference workflows that
#' share one model structure across exposure sample columns
#' (e.g. \code{sqrt_pm} -> \code{sqrt_pm_s3}).
#'
#' @param x A \code{model_term}, \code{dirichlet_multinom}, formula, or list of
#'   such objects.
#' @param map Named character vector: names are current column names, values are
#'   replacement column names. Names absent from \code{map} are left unchanged.
#'
#' @return An object of the same type as \code{x}, with column references updated
#'   and labels / formulas regenerated where applicable.
#'
#' @examples
#' terms <- collect_terms(nbinom(y) ~ fpoly(sqrt_pm, p = 1))
#' replace_vars(terms, c(sqrt_pm = "sqrt_pm_s1"))
#'
#' @export
setGeneric("replace_vars", function(x, map) standardGeneric("replace_vars"))

#' @noRd
.replace_map_lookup <- function(name, map) {
  if (!length(name) || is.na(name) || !nzchar(name)) {
    return(name)
  }
  if (name %in% names(map)) {
    as.character(map[[name]])
  } else {
    as.character(name)
  }
}

#' @noRd
.replace_formula_symbols <- function(f, map) {
  if (!inherits(f, "formula")) {
    return(f)
  }
  env <- environment(f)
  replace_lang <- function(expr) {
    if (is.symbol(expr)) {
      nm <- as.character(expr)
      if (nm %in% names(map)) {
        return(as.symbol(map[[nm]]))
      }
      return(expr)
    }
    if (is.call(expr)) {
      parts <- lapply(as.list(expr), replace_lang)
      return(as.call(parts))
    }
    expr
  }
  parts <- lapply(as.list(f), replace_lang)
  out <- as.call(parts)
  class(out) <- "formula"
  environment(out) <- if (is.null(env)) emptyenv() else env
  out
}

#' @noRd
.regenerate_term_label <- function(term, new_name) {
  cls <- class(term)[1L]
  # Keep trailing role suffix from the previous label when possible
  old <- term@label
  suffix <- sub(paste0("^", gsub("([.|()\\^{}+$*?]|\\[|\\])", "\\\\\\1", term@name)), "", old)
  if (!nzchar(suffix) || identical(suffix, old)) {
    paste(new_name, cls, sep = "_")
  } else if (startsWith(suffix, "_")) {
    paste0(new_name, suffix)
  } else {
    paste(new_name, cls, sep = "_")
  }
}

#' @rdname replace_vars
#' @export
setMethod("replace_vars", "model_term", function(x, map) {
  if (is.null(map) || !length(map)) {
    return(x)
  }
  map <- stats::setNames(as.character(map), names(map))
  new_name <- .replace_map_lookup(x@name, map)
  if (identical(new_name, x@name)) {
    # Still rewrite formula symbols (e.g. companions that reference other vars)
    x@formula <- .replace_formula_symbols(x@formula, map)
    return(x)
  }
  # Regenerate label while @name is still the old prefix (suffix strip depends on it)
  new_label <- .regenerate_term_label(x, new_name)
  x@name <- new_name
  x@label <- new_label
  x@formula <- .replace_formula_symbols(x@formula, map)
  x
})

#' @rdname replace_vars
#' @export
setMethod("replace_vars", "dirichlet_multinom", function(x, map) {
  if (is.null(map) || !length(map)) {
    return(x)
  }
  map <- stats::setNames(as.character(map), names(map))
  x <- methods::callNextMethod(x, map)
  if (length(x@by)) {
    x@by <- vapply(x@by, .replace_map_lookup, character(1), map = map)
  }
  x
})

#' @rdname replace_vars
#' @export
setMethod("replace_vars", "formula", function(x, map) {
  if (is.null(map) || !length(map)) {
    return(x)
  }
  map <- stats::setNames(as.character(map), names(map))
  .replace_formula_symbols(x, map)
})

#' @rdname replace_vars
#' @export
setMethod("replace_vars", "list", function(x, map) {
  if (is.null(map) || !length(map)) {
    return(x)
  }
  out <- lapply(x, replace_vars, map = map)
  # Preserve names; update names that were themselves variable names when possible
  if (!is.null(names(out))) {
    nm <- names(out)
    for (i in seq_along(nm)) {
      if (methods::is(out[[i]], "model_term") && nzchar(out[[i]]@label)) {
        nm[i] <- out[[i]]@label
      }
    }
    names(out) <- nm
  }
  out
})
