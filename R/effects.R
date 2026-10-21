#' Per-shard layout for AD density evaluation
#'
#' @param x Character vector of term names.
#' @return Character vector with surrounding quotes removed.
#' @keywords internal
#' @export
strip_term_name <- function(x) {
  x <- as.character(x)
  gsub("^\"|\"$", "", x)
}

#' Parse Model Terms from Formula
#'
#' @description
#' Parses a formula and creates model terms by evaluating constructor calls in
#' the formula environment (so \code{pkg::term(...)} and terms from attached
#' packages on the search path work). Falls back to the \pkg{adlaplace}
#' namespace for built-in constructors.
#'
#' @param formula Model formula with constructor terms (e.g. \code{iwp(x, ...)},
#'   \code{iid(fac)}, \code{nbinom(y, ...)} on the LHS). Bare symbols such as
#'   \code{x} are coerced to \code{linear(x)}. Unnamed first positional symbols
#'   are treated as column names in \code{data}; named arguments (e.g.
#'   \code{x = sites} for spatial terms) are evaluated as objects in the
#'   formula environment. The named argument \code{size} is also treated as a
#'   column name when passed as a symbol (\code{binom(y, size = N)}).
#' @param verbose print extra information
#' @return List of model term objects
#'
#' @export
collect_terms <- function(formula, verbose = FALSE) {
  if (!methods::is(formula, "formula")) {
    warning("formula must be of class formula")
  }

  formula_env <- environment(formula)
  if (is.null(formula_env)) {
    formula_env <- parent.frame()
  }

  formula_terms <- stats::terms(formula)
  term_labels <- rownames(attr(formula_terms, "factors"))
  response_idx <- attr(formula_terms, "response")
  response_label <- if (response_idx > 0L) term_labels[response_idx] else NULL

  terms_1 <- lapply(term_labels, function(lab) {
    # a bare response symbol (e.g. y ~ x) defaults to a Gaussian
    # observation term; bare covariate symbols default to linear()
    if (identical(lab, response_label) && !grepl("[(]", lab)) {
      if (verbose) {
        message("Response '", lab, "' defaulting to normal(", lab, ")")
      }
      term_obj <- normal(lab)
      return(name_single_term(term_obj))
    }

    parsed <- eval_term_label(lab, formula_env)
    if (isTRUE(parsed$ok)) {
      term_obj <- parsed$value
      if (verbose) {
        message(
          "Model term ", substr(lab, 1, 20),
          "... parsed from formula"
        )
      }
    } else if (grepl("[(]", lab)) {
      stop(format_term_parse_error(lab, parsed), call. = FALSE)
    } else {
      term_obj <- linear(lab)
    }
    name_single_term(term_obj)
  })

  terms_1 <- do.call(c, terms_1)

  # add intercept
  if (
    attr(stats::terms(formula), "intercept") &&
      !any(vapply(terms_1, inherits, logical(1), what = "intercept"))
  ) {
    terms_1$intercept <- intercept()
  }

  terms_1
}

#' List name for one model term: class, data column, and \code{@mult}.
#' A constructor that already returned several terms keeps its own names.
#' @noRd
name_single_term <- function(term_obj) {
  if (!is.list(term_obj)) {
    term_obj <- list(term_obj)
  }
  if (length(term_obj) != 1L) {
    return(term_obj)
  }
  names(term_obj) <- term_element_name(term_obj[[1L]])
  term_obj
}

#' @noRd
term_element_name <- function(term) {
  if (inherits(term, "intercept")) {
    return("intercept")
  }
  pieces <- c(class(term)[1L], term@name)
  if (methods::.hasSlot(term, "mult")) {
    mult <- term@mult
    if (length(mult) == 1L && !is.na(mult) && nzchar(mult)) {
      pieces <- c(pieces, mult)
    }
  }
  paste(pieces, collapse = "_")
}

#' Column-name formals coerced even when named (symbols -> character).
#' @noRd
.column_name_args <- c("size")

#' @noRd
coerce_term_call_symbols <- function(expr) {
  if (is.symbol(expr) || is.name(expr)) {
    return(call("linear", as.character(expr)))
  }
  if (!is.call(expr)) {
    return(expr)
  }
  args <- as.list(expr)
  nm <- names(args)
  for (i in seq_along(args)) {
    if (i == 1L) {
      next
    }
    a <- args[[i]]
    arg_name <- if (is.null(nm)) {
      ""
    } else {
      nm[[i]]
    }
    if (is.null(arg_name) || is.na(arg_name)) {
      arg_name <- ""
    }
    if (is.symbol(a) || is.name(a)) {
      is_unnamed <- !nzchar(arg_name)
      # Only the first positional (unnamed) argument is treated as a column
      # name, so named object args like matern(x = sites) stay objects.
      # Named size=N is also a column name.
      if ((is_unnamed && i == 2L) || arg_name %in% .column_name_args) {
        args[[i]] <- as.character(a)
      }
    } else if (is.call(a)) {
      args[[i]] <- coerce_term_call_symbols(a)
    }
  }
  as.call(args)
}

#' @noRd
try_error_message <- function(x) {
  cond <- attr(x, "condition")
  if (!is.null(cond)) {
    msg <- conditionMessage(cond)
    if (length(msg) && nzchar(msg)) {
      return(trimws(msg))
    }
  }
  msg <- paste(as.character(x), collapse = "\n")
  msg <- sub("^Error in [^\n]*:\\s*", "", msg)
  msg <- sub("^Error:\\s*", "", msg)
  trimws(msg)
}

#' @noRd
term_constructor_label <- function(expr) {
  if (!is.call(expr)) {
    return(NA_character_)
  }
  head <- expr[[1L]]
  if (is.call(head) && identical(head[[1L]], quote(`::`))) {
    return(paste0(as.character(head[[2L]]), "::", as.character(head[[3L]])))
  }
  if (is.call(head) && identical(head[[1L]], quote(`:::`))) {
    return(paste0(as.character(head[[2L]]), ":::", as.character(head[[3L]])))
  }
  as.character(head)
}

#' @noRd
term_constructor_available <- function(expr, formula_env) {
  if (!is.call(expr)) {
    return(FALSE)
  }
  head <- expr[[1L]]
  if (is.call(head) &&
      (identical(head[[1L]], quote(`::`)) || identical(head[[1L]], quote(`:::`)))) {
    pkg <- as.character(head[[2L]])
    fun <- as.character(head[[3L]])
    if (!requireNamespace(pkg, quietly = TRUE)) {
      return(FALSE)
    }
    return(exists(fun, envir = asNamespace(pkg), inherits = FALSE, mode = "function"))
  }
  fun <- as.character(head)
  exists(fun, envir = formula_env, inherits = TRUE, mode = "function") ||
    exists(fun, envir = asNamespace("adlaplace"), inherits = FALSE, mode = "function")
}

#' @noRd
looks_like_missing_function <- function(msg) {
  grepl(
    "could not find function|is not an exported object|there is no package called|object '[^']+' of mode 'function' was not found",
    msg,
    ignore.case = TRUE
  )
}

#' @noRd
format_term_parse_error <- function(lab, parsed) {
  ctor <- parsed$constructor
  msg <- parsed$error
  missing_ctor <- isFALSE(parsed$constructor_ok) ||
    (is.na(parsed$constructor_ok) && looks_like_missing_function(msg))

  if (missing_ctor) {
    ctor_txt <- if (length(ctor) && !is.na(ctor) && nzchar(ctor)) {
      paste0(" '", ctor, "'")
    } else {
      ""
    }
    paste0(
      "Failed to parse term '", lab, "': constructor", ctor_txt,
      " is not available. Use pkg::term(...) or library(pkg) so constructors ",
      "are visible.",
      if (length(msg) && nzchar(msg)) paste0(" (", msg, ")") else ""
    )
  } else {
    paste0(
      "Failed to evaluate term '", lab, "': ", msg,
      " First unnamed arguments in formula calls are treated as data column ",
      "names (character); pass objects with named arguments ",
      "(e.g. x = sites) or compute nested values outside the formula."
    )
  }
}

#' @noRd
eval_term_label <- function(lab, formula_env) {
  expr <- parse(text = lab, keep.source = FALSE)[[1]]
  expr <- coerce_term_call_symbols(expr)
  ctor <- term_constructor_label(expr)
  ctor_ok <- term_constructor_available(expr, formula_env)

  try_result <- try(
    eval(expr, envir = formula_env),
    silent = TRUE
  )
  if (!inherits(try_result, "try-error")) {
    return(list(ok = TRUE, value = try_result))
  }
  err_msg <- try_error_message(try_result)

  try_ns <- try(
    eval(expr, envir = asNamespace("adlaplace")),
    silent = TRUE
  )
  if (!inherits(try_ns, "try-error")) {
    return(list(ok = TRUE, value = try_ns))
  }

  list(
    ok = FALSE,
    error = err_msg,
    constructor = ctor,
    constructor_ok = ctor_ok
  )
}
