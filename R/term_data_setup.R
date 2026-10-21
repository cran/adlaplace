#' Apply log transform to selected theta columns
#'
#' When \code{active} is \code{TRUE}, logs values in \code{cols} for rows where
#' \code{theta_info$log} is \code{TRUE}. Missing or \code{NA} \code{log}
#' entries default to \code{TRUE}.
#'
#' @param theta_info Data frame from \code{info$theta} (see \code{\link{term_data_setup}}).
#' @param cols Character vector of column names to transform (e.g. \code{"init"}).
#' @param active Logical; when \code{FALSE}, returns \code{theta_info} unchanged.
#' @return A copy of \code{theta_info} with selected entries logged.
#' @export
apply_theta_log <- function(theta_info, cols = "init", active = TRUE) {
  if (!active || nrow(theta_info) == 0L) {
    return(theta_info)
  }
  out <- theta_info
  log <- if ("log" %in% names(out)) {
    out$log
  } else {
    rep(TRUE, nrow(out))
  }
  log[is.na(log)] <- TRUE
  log_transform_columns(out, cols, which(log))
}

#' @keywords internal
ensure_theta_log <- function(theta_setup) {
  if (nrow(theta_setup) == 0L) {
    theta_setup$log <- logical(0)
    return(theta_setup)
  }
  if (!"log" %in% names(theta_setup)) {
    theta_setup$log <- TRUE
  } else {
    theta_setup$log[is.na(theta_setup$log)] <- TRUE
  }
  theta_setup
}

#' @keywords internal
log_transform_columns <- function(df, cols, idx) {
  if (length(idx) == 0L) {
    return(df)
  }
  out <- df
  for (col in cols) {
    if (col %in% names(out)) {
      out[idx, col] <- log(out[idx, col])
    }
  }
  out
}

#' @keywords internal
parse_term_data_setup_terms <- function(formula, verbose = FALSE) {
  if (methods::is(formula, "formula")) {
    return(collect_terms(formula, verbose = verbose))
  }

  terms <- formula
  if (inherits(terms, "model_term")) {
    terms <- list(terms)
  }
  inherits_seq <- unlist(lapply(terms, inherits, what = "model_term"))
  if (!all(inherits_seq)) {
    warning(
      "formula must be of class formula or a list of objects which inherit class model_term"
    )
  }
  terms
}

#' Column-bind design blocks; coerce to sparse when any block is sparse.
#'
#' Drops \code{NULL} pieces. When any remaining piece inherits
#' \code{sparseMatrix}, every piece is coerced to \code{dgCMatrix} before
#' \code{cbind} so a leading dense block cannot strip the \code{Matrix} class.
#' When none are sparse, returns a base dense matrix via \code{cbind}.
#'
#' @param pieces List of design matrices (may include \code{NULL}).
#' @return A design matrix (base \code{matrix} or \code{dgCMatrix}).
#' @keywords internal
cbind_design_blocks <- function(pieces) {
  pieces <- Filter(Negate(is.null), pieces)
  if (!length(pieces)) {
    return(NULL)
  }
  any_sparse <- any(vapply(
    pieces,
    function(m) inherits(m, "sparseMatrix"),
    logical(1L)
  ))
  if (any_sparse) {
    pieces <- lapply(pieces, function(m) {
      if (inherits(m, "dgCMatrix")) {
        return(m)
      }
      as_dgC(m)
    })
  }
  do.call(cbind, pieces)
}

#' @keywords internal
bind_design_matrices <- function(terms, data) {
  terms <- lapply(terms, add_by_levels, data)
  design_list <- lapply(terms, design, data = data)

  terms_with_beta <- vapply(terms, function(t) methods::slot(t, "model_role") == "fixed", logical(1L))
  terms_with_gamma <- vapply(terms, function(t) methods::slot(t, "model_role") == "random", logical(1L))

  if (any(terms_with_beta)) {
    x_matrix <- cbind_design_blocks(design_list[terms_with_beta])
    if (is.null(x_matrix)) {
      x_matrix <- matrix(nrow = nrow(data), ncol = 0)
    }
  } else {
    x_matrix <- matrix(nrow = nrow(data), ncol = 0)
  }

  if (any(terms_with_gamma)) {
    a_matrix <- cbind_design_blocks(design_list[terms_with_gamma])
    if (is.null(a_matrix)) {
      a_matrix <- matrix(nrow = nrow(data), ncol = 0)
    }
  } else {
    a_matrix <- matrix(nrow = nrow(data), ncol = 0)
  }

  list(terms = terms, X = x_matrix, A = a_matrix)
}

#' @keywords internal
empty_theta_setup <- function() {
  data.frame(
    term = character(),
    model = character(),
    label = character(),
    init = numeric(),
    lower = numeric(),
    upper = numeric(),
    parscale = numeric(),
    model_role = factor(levels = .model_role_levels),
    log = logical(0),
    stringsAsFactors = FALSE
  )
}

#' @keywords internal
empty_beta_setup <- function() {
  data.frame(
    term = character(),
    model = character(),
    label = character(),
    order = numeric(),
    beta_label = character(),
    init = numeric(),
    lower = numeric(),
    upper = numeric(),
    parscale = numeric(),
    stringsAsFactors = FALSE
  )
}

#' @keywords internal
empty_gamma_setup <- function() {
  data.frame(
    term = character(),
    model = character(),
    label = character(),
    gamma_label = character(),
    stringsAsFactors = FALSE
  )
}

#' @keywords internal
build_theta_setup <- function(terms) {
  theta_info_list <- Filter(Negate(is.null), lapply(terms, theta_info))
  # Normalize log on each frame before rbind so mixed term types
  # (different column sets) share a common log column.
  theta_info_list <- lapply(theta_info_list, ensure_theta_log)
  if (length(theta_info_list) == 0L) {
    theta_setup <- empty_theta_setup()
  } else {
    theta_setup <- do.call(rbind, theta_info_list)
  }
  theta_setup <- ensure_theta_log(theta_setup)
  theta_setup$id <- seq.int(0, length.out = nrow(theta_setup))
  theta_setup
}

#' @keywords internal
reorder_setup_by_columns <- function(setup, matrix_cols, label_col, empty_warning) {
  if (ncol(matrix_cols) == 0L || nrow(setup) == 0L) {
    return(setup)
  }
  reorder <- match(colnames(matrix_cols), setup[[label_col]])
  if (any(is.na(reorder))) {
    warning(empty_warning)
    reorder <- reorder[!is.na(reorder)]
    if (length(reorder) == 0L) {
      return(data.frame())
    }
  }
  setup[reorder, , drop = FALSE]
}

#' @keywords internal
build_beta_setup <- function(terms, data, x_matrix) {
  beta_setup <- do.call(rbind, lapply(terms, beta_info, data = data))
  if (is.null(beta_setup)) {
    beta_setup <- empty_beta_setup()
  }
  reorder_setup_by_columns(
    beta_setup,
    x_matrix,
    "beta_label",
    "some beta labels not found in X matrix columns"
  )
}

#' @keywords internal
build_gamma_setup <- function(terms, data, a_matrix, theta_setup) {
  random_info_list <- Filter(Negate(is.null), lapply(terms, random_info, data = data))
  if (length(random_info_list) == 0L) {
    gamma_setup <- empty_gamma_setup()
  } else {
    gamma_setup <- do.call(rbind, random_info_list)
  }
  gamma_setup$id <- seq.int(0, length.out = nrow(gamma_setup))
  if (nrow(gamma_setup) > 0L && nrow(theta_setup) > 0L) {
    gamma_setup$theta_id <- theta_setup[match(
      gamma_setup$label, theta_setup$label
    ), "id"]
  }
  gamma_setup <- reorder_setup_by_columns(
    gamma_setup,
    a_matrix,
    "gamma_label",
    paste(
      "some random-effect column names were not found among gamma labels;",
      "dropping unmatched random effects"
    )
  )
  if (nrow(gamma_setup) > 0L && "gamma_label" %in% names(gamma_setup)) {
    rownames(gamma_setup) <- gamma_setup$gamma_label
  }
  gamma_setup
}

#' @keywords internal
extract_response <- function(terms, formula_in, data) {
  obs_idx <- which(vapply(terms, function(t) {
    methods::is(t, "model_term") && !is.na(t@ad_kind) &&
      identical(t@ad_kind, "observations")
  }, logical(1L)))

  term_idx <- integer(0)
  if (length(obs_idx) == 1L) {
    term_idx <- obs_idx[1L]
  } else if (length(obs_idx) > 1L) {
    warning("multiple observation-density terms; using the first")
    term_idx <- obs_idx[1L]
  }

  if (length(term_idx) == 1L) {
    return(as.numeric(data[[terms[[term_idx]]@name]]))
  }

  if (methods::is(formula_in, "formula")) {
    lhs <- formula_in[[2L]]
    if (is.symbol(lhs)) {
      lhs_name <- as.character(lhs)
      if (lhs_name %in% names(data)) {
        return(as.numeric(data[[lhs_name]]))
      }
      warning("response variable '", lhs_name, "' not found in data")
      return(numeric(0))
    }
    warning("no response variable")
    return(numeric(0))
  }

  warning("no response variable")
  numeric(0)
}

#' @keywords internal
empty_parameters_info <- function() {
  data.frame(
    term = character(),
    model = character(),
    label = character(),
    init = numeric(),
    lower = numeric(),
    upper = numeric(),
    parscale = numeric(),
    model_role = factor(levels = .model_role_levels),
    log = logical(0),
    stringsAsFactors = FALSE
  )
}

#' @keywords internal
build_parameters_info <- function(beta_df, theta_df, names_common) {
  beta_rows <- if (!is.null(beta_df) && nrow(beta_df) > 0L) {
    out <- beta_df[, names_common, drop = FALSE]
    out$log <- FALSE
    out
  } else {
    NULL
  }
  theta_rows <- if (nrow(theta_df) > 0L) {
    cols <- intersect(c(names_common, "log"), names(theta_df))
    theta_df[, cols, drop = FALSE]
  } else {
    NULL
  }
  out <- do.call(rbind, Filter(Negate(is.null), list(beta_rows, theta_rows)))
  if (is.null(out) || nrow(out) == 0L) {
    return(empty_parameters_info())
  }
  rownames(out) <- NULL
  log <- out$log
  log[is.na(log)] <- FALSE
  log_transform_columns(out, c("init", "lower", "upper"), which(log))
}

#' Build design matrices and parameter metadata from a formula
#'
#' @param formula Model formula or named list of term objects.
#' @param data Data frame.
#' @param verbose Print parsing messages.
#' @return List with \code{terms}, \code{y}, \code{A}, \code{X}, \code{data}, \code{info}.
#'   The \code{info$theta} data frame includes a logical \code{log} column:
#'   \code{TRUE} means the parameter is stored on the log scale when
#'   \code{config$transform_theta} is \code{TRUE} (default for all rows unless a
#'   model term sets otherwise). \code{info$parameters} stacks \code{beta} and
#'   \code{theta} rows with the same \code{log} column (\code{FALSE} for
#'   fixed effects); \code{init}, \code{lower}, and \code{upper} are already on
#'   the optimization (log) scale where \code{log} is \code{TRUE}.
#' @export
term_data_setup <- function(formula, data, verbose = FALSE) {
  formula_in <- formula
  terms <- parse_term_data_setup_terms(formula, verbose = verbose)
  design <- bind_design_matrices(terms, data)
  terms <- design$terms
  x_matrix <- design$X
  a_matrix <- design$A

  theta_setup <- build_theta_setup(terms)
  beta_setup <- build_beta_setup(terms, data, x_matrix)
  gamma_setup <- build_gamma_setup(terms, data, a_matrix, theta_setup)
  y <- extract_response(terms, formula_in, data)

  if (is.null(beta_setup)) {
    beta_theta_names <- names(theta_setup)
  } else {
    beta_theta_names <- intersect(colnames(beta_setup), colnames(theta_setup))
  }
  beta_theta_names <- setdiff(beta_theta_names, "order")

  list(
    y = y,
    A = a_matrix,
    X = x_matrix,
    data = data,
    info = list(
      beta = beta_setup,
      gamma = gamma_setup,
      theta = theta_setup,
      parameters = build_parameters_info(
        beta_setup,
        theta_setup,
        beta_theta_names
      )
    ),
    terms = terms
  )
}
