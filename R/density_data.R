#' @include classes.R
NULL

#' Parameter-block row counts for an \code{density_data}
#'
#' @param x An \code{density_data} object.
#' @param ... Unused; for S4 generic compatibility.
#' @return Named list with \code{n_beta}, \code{n_gamma}, and \code{n_theta}.
#' @export
setGeneric("sizes", function(x, ...) standardGeneric("sizes"))

#' @rdname sizes
#' @export
setMethod("sizes", "density_data", function(x) {
  list(
    n_beta = nrow(x@beta_map),
    n_gamma = nrow(x@gamma_map),
    n_theta = nrow(x@theta_map)
  )
})

#' @keywords internal
validate_density_data_dims <- function(y, A, X) {
  n_obs <- length(y)
  if (n_obs == 0L) {
    if (!is.null(A) && (nrow(A) != 0L || ncol(A) != 0L)) {
      stop("y is empty but A is non-empty")
    }
    if (!is.null(X) && (nrow(X) != 0L || ncol(X) != 0L)) {
      stop("y is empty but X is non-empty")
    }
    return(invisible(NULL))
  }
  if (!is.null(A) && nrow(A) != n_obs) {
    stop("nrow(A) (", nrow(A), ") must match length(y) (", n_obs, ")")
  }
  if (!is.null(X) && nrow(X) != n_obs) {
    stop("nrow(X) (", nrow(X), ") must match length(y) (", n_obs, ")")
  }
  invisible(NULL)
}

#' @keywords internal
validate_density_data_maps <- function(data, kind) {
  layout <- sizes(data)
  if (identical(kind, "parameters")) {
    if (layout$n_theta == 0L || ncol(data@theta_map) == 0L) {
      stop("parameters shards require a non-empty theta_map")
    }
  } else if (identical(kind, "random")) {
    if (ncol(data@beta_map) != 0L) {
      stop("random shards require beta_map with zero columns")
    }
    if (layout$n_gamma == 0L) {
      stop("random shards require a non-empty gamma_map")
    }
  } else if (identical(kind, "observations")) {
    if (length(data@y) == 0L) {
      stop("observation shards require non-empty y on density_data")
    }
  }
  invisible(layout)
}

#' Resolve inner \code{gamma} starting values for optimization.
#' @keywords internal
resolve_gamma_start <- function(config, cache = NULL, num_gamma, gamma = missing()) {
  num_gamma <- as.integer(num_gamma)[1L]
  if (!missing(gamma)) {
    if (length(gamma) != num_gamma) {
      stop(
        "length(gamma) (", length(gamma), ") must equal num_gamma (", num_gamma, ")",
        call. = FALSE
      )
    }
    return(gamma)
  }
  if (!is.null(cache[["gamma"]]) && length(cache[["gamma"]]) == num_gamma) {
    return(cache[["gamma"]])
  }
  if (!is.null(config[["gamma"]]) && length(config[["gamma"]]) == num_gamma) {
    return(config[["gamma"]])
  }
  rep(0, num_gamma)
}

#' Fill missing \code{config$gamma} before \code{ad_pack_ptr()} tape build.
#' @keywords internal
normalize_config_for_ptr <- function(config, data, kind = NULL) {
  layout <- sizes(data)
  if (identical(kind, "random")) {
    return(config)
  }
  if (identical(kind, "observations") && is.null(config[["obs_groups"]])) {
    # ELGM (e.g. dirichlet_multinom): dens indexes obs_groups as strata columns
    # of elgm_matrix, not rows of y. Prefer ensure_config_obs_groups() upstream
    # (ad_pack/fit) so num_shards builds a proper map; here fall back to one
    # shard over all strata so default_obs_groups(length(y)) cannot OOB.
    elgm <- data@elgm_matrix
    n_elgm <- if (methods::is(elgm, "Matrix")) ncol(elgm) else 0L
    if (n_elgm > 0L) {
      config$obs_groups <- default_obs_groups(n_elgm)
    } else {
      n_obs <- length(data@y)
      if (n_obs > 0L) {
        config$obs_groups <- default_obs_groups(n_obs)
      }
    }
  }
  if (is.null(config[["beta"]])) {
    config$beta <- numeric(0)
  }
  n_gamma <- layout$n_gamma
  gamma_len <- if (is.null(config[["gamma"]])) {
    0L
  } else {
    length(config[["gamma"]])
  }
  if (n_gamma > 0L && gamma_len != n_gamma) {
    config$gamma <- rep(0, n_gamma)
  }
  config
}

#' @keywords internal
validate_config_layout <- function(model, config, kind = NULL) {
  layout <- sizes(model)
  if (length(config$beta) != layout$n_beta) {
    stop(
      "length(config$beta) (", length(config$beta),
      ") must match nrow(beta_map) (", layout$n_beta, ")"
    )
  }
  if (is.null(kind) || !identical(kind, "random")) {
    gamma_len <- if (is.null(config[["gamma"]])) {
      layout$n_gamma
    } else {
      length(config[["gamma"]])
    }
    if (gamma_len != layout$n_gamma) {
      stop(
        "length(config$gamma) (", gamma_len,
        ") must match nrow(gamma_map) (", layout$n_gamma, ")"
      )
    }
  }
  n_theta_config <- if (!is.null(config$theta)) length(config$theta) else NA_integer_
  if (is.na(n_theta_config) || n_theta_config == 0L) {
    n_theta_config <- layout$n_theta
  }
  if (n_theta_config != layout$n_theta) {
    stop(
      "length(config$theta) (", n_theta_config,
      ") must match nrow(theta_map) (", layout$n_theta, ")"
    )
  }
  invisible(layout)
}

#' Coerce a sparse matrix to \code{dgCMatrix}.
#' @keywords internal
as_dgC <- function(m) {
  m <- methods::as(m, "CsparseMatrix")
  m <- methods::as(m, "dMatrix")
  m <- methods::as(m, "generalMatrix")
  if (!inherits(m, "dgCMatrix")) {
    warning(
      "as_dgC expected class 'dgCMatrix' but got class: ",
      paste(class(m), collapse = ", "),
      call. = FALSE
    )
    m <- methods::as(m, "dgCMatrix")
  }
  m
}

#' Coerce a sparse matrix to upper-triangle \code{dsCMatrix} (\code{uplo = "U"}).
#'
#' Used for \code{random_mult} precision \code{Q}. The C++ backend rejects
#' \code{dgCMatrix}; coerce here before building the tape.
#' @keywords internal
as_dsC_upper <- function(m) {
  if (inherits(m, "dsCMatrix") && identical(m@uplo, "U")) {
    return(Matrix::drop0(m))
  }
  Matrix::forceSymmetric(as_dgC(m), uplo = "U")
}

#' Normalize \code{random_mult} precision list so \code{Q} is \code{dsCMatrix}.
#' @keywords internal
normalize_random_mult_precision <- function(precision) {
  if (is.null(precision) || !is.list(precision) || is.null(precision$Q)) {
    return(precision)
  }
  precision$Q <- as_dsC_upper(precision$Q)
  precision
}

#' Coerce a sparse matrix to \code{ngCMatrix} (Matrix >= 1.6).
#' @keywords internal
as_ngC <- function(m) {
  if (inherits(m, "ngCMatrix")) {
    return(m)
  }
  m <- methods::as(m, "CsparseMatrix")
  m <- methods::as(m, "nMatrix")
  m <- methods::as(m, "generalMatrix")
  if (!inherits(m, "ngCMatrix")) {
    warning(
      "as_ngC expected class 'ngCMatrix' but got class: ",
      paste(class(m), collapse = ", "),
      call. = FALSE
    )
    m <- methods::as(m, "ngCMatrix")
  }
  m
}

#' Coerce map shorthand: length-1 \code{nrow}, length-2 \code{c(row, nrow)},
#' or \code{list(indices, nrow)}.
#' @keywords internal
coerce_map_shorthand <- function(x, what) {
  if (is.null(x)) {
    return(NULL)
  }
  if (is.list(x)) {
    if (length(x) != 2L) {
      return(x)
    }
    idx_raw <- x[[1]]
    n_raw <- x[[2]]
    idx <- as.integer(idx_raw)
    n <- as.integer(n_raw)
    if (length(n_raw) != 1L || is.na(n) || !isTRUE(all.equal(as.numeric(n_raw), as.numeric(n)))) {
      stop(what, ": list shorthand second element must be scalar nrow", call. = FALSE)
    }
    if (length(idx_raw) != length(idx) ||
      any(is.na(idx)) ||
      !isTRUE(all.equal(as.numeric(idx_raw), as.numeric(idx)))) {
      stop(what, ": list shorthand indices must be finite integers", call. = FALSE)
    }
    if (n < 0L) {
      stop(what, ": nrow must be non-negative", call. = FALSE)
    }
    if (length(idx) > 0L && (any(idx < 1L) || any(idx > n))) {
      stop(
        what, ": list shorthand indices must be between 1 and nrow (", n, ")",
        call. = FALSE
      )
    }
    return(Matrix::sparseMatrix(
      i = idx,
      j = seq.int(1L, length.out = length(idx)),
      x = 1,
      dims = c(n, length(idx)),
      repr = "C"
    ))
  }
  if (!is.atomic(x)) {
    return(x)
  }
  if (length(x) == 1L) {
    n <- as.integer(x)
    if (is.na(n)) {
      return(x)
    }
    if (n < 0L) {
      stop(what, ": nrow must be non-negative", call. = FALSE)
    }
    return(Matrix::Matrix(nrow = n, ncol = 0L))
  }
  if (length(x) == 2L) {
    row <- as.integer(x[1])
    n <- as.integer(x[2])
    if (is.na(row) || is.na(n)) {
      stop(what, ": shorthand c(row, nrow) must be finite integers", call. = FALSE)
    }
    if (n < 0L) {
      stop(what, ": nrow must be non-negative", call. = FALSE)
    }
    if (n == 0L) {
      return(Matrix::sparseMatrix(
        i = integer(0),
        j = integer(0),
        dims = c(0L, 1L),
        repr = "C"
      ))
    }
    if (row < 1L || row > n) {
      stop(
        what, ": row index (", row, ") must be between 1 and nrow (", n, ")",
        call. = FALSE
      )
    }
    return(Matrix::sparseMatrix(
      i = row,
      j = 1L,
      x = 1,
      dims = c(n, 1L),
      repr = "C"
    ))
  }
  x
}

#' @keywords internal
validate_map_at_most_one_per_row_col <- function(m, what) {
  if (nrow(m) == 0L || ncol(m) == 0L) {
    return(invisible(NULL))
  }
  rs <- Matrix::rowSums(m)
  cs <- Matrix::colSums(m)
  if (any(rs > 1L)) {
    stop(what, ": more than one nonzero per row", call. = FALSE)
  }
  if (any(cs > 1L)) {
    stop(what, ": more than one nonzero per column", call. = FALSE)
  }
  invisible(NULL)
}

#' @keywords internal
validate_map_exactly_one_per_column <- function(m, what) {
  if (ncol(m) == 0L) {
    return(invisible(NULL))
  }
  cs <- Matrix::colSums(m)
  if (any(cs != 1L)) {
    stop(
      what,
      ": each column must have exactly one nonzero when the design is non-empty",
      call. = FALSE
    )
  }
  invisible(NULL)
}

#' @keywords internal
validate_design_parameter_maps <- function(beta_map,
                                           gamma_map,
                                           n_beta_design,
                                           n_gamma_design) {
  if (n_beta_design > 0L) {
    if (ncol(beta_map) != n_beta_design) {
      stop(
        "beta_map ncol (", ncol(beta_map),
        ") must match ncol(X) (", n_beta_design, ")",
        call. = FALSE
      )
    }
    validate_map_exactly_one_per_column(beta_map, "beta_map")
  }
  if (n_gamma_design > 0L) {
    if (ncol(gamma_map) != n_gamma_design) {
      stop(
        "gamma_map ncol (", ncol(gamma_map),
        ") must match ncol(A) (", n_gamma_design, ")",
        call. = FALSE
      )
    }
    validate_map_exactly_one_per_column(gamma_map, "gamma_map")
  }
  invisible(NULL)
}

#' Transpose of a design matrix for \code{ATp} / \code{XTp} slots.
#' @keywords internal
design_Tp <- function(M, n_obs, what = "M") {
  if (is.null(M)) {
    return(as_dgC(Matrix::sparseMatrix(
      i = integer(0),
      j = integer(0),
      dims = c(0L, n_obs),
      repr = "C"
    )))
  }
  if (nrow(M) != n_obs) {
    stop(
      "nrow(", what, ") (", nrow(M), ") must match length(y) (", n_obs, ")",
      call. = FALSE
    )
  }
  as_dgC(Matrix::t(M))
}

#' Build \code{density_data} from designs and \code{config} layout sizes.
#'
#' Used by examples with a plain data list (\code{y}, \code{A}, \code{X}).
#'
#' @keywords internal
density_data_from_config_matrices <- function(y, A, X, config, theta_local_row = 0L) {
  n_gamma <- if (!is.null(config[["gamma"]])) {
    length(config[["gamma"]])
  } else if (!is.null(A) && ncol(A) > 0L) {
    ncol(A)
  } else {
    0L
  }
  n_theta <- length(config$theta)
  gamma_map <- if (n_gamma > 0L) {
    if (!is.null(A) && ncol(A) > 0L) {
      Matrix::Diagonal(ncol(A))
    } else {
      Matrix::Matrix(nrow = n_gamma, ncol = 0L)
    }
  } else {
    NULL
  }
  theta_map <- if (n_theta > 0L) {
    Matrix::sparseMatrix(
      i = as.integer(theta_local_row),
      j = 0L,
      x = 1,
      dims = c(n_theta, 1L),
      index1 = FALSE,
      repr = "C"
    )
  } else {
    NULL
  }
  density_data(
    y = y,
    A = A,
    X = X,
    gamma_map = gamma_map,
    theta_map = theta_map
  )
}

#' @keywords internal
density_data_from_mats <- function(y = numeric(0),
                              A = NULL,
                              X = NULL,
                              beta_map = NULL,
                              gamma_map = NULL,
                              theta_map = NULL,
                              elgm_matrix = NULL,
                              density = NA_character_,
                              ad_kind = NA_character_,
                              package = NA_character_,
                              precision = NULL,
                              weights = numeric(0)) {
  n_obs <- length(y)
  validate_density_data_dims(y, A, X)
  ATp <- design_Tp(A, n_obs, "A")
  XTp <- design_Tp(X, n_obs, "X")
  if (length(density) != 1L || is.na(density) || !nzchar(density)) {
    density <- NA_character_
  } else {
    density <- as.character(density)
  }
  if (length(ad_kind) != 1L || is.na(ad_kind) || !nzchar(ad_kind)) {
    ad_kind <- NA_character_
  } else {
    ad_kind <- as.character(ad_kind)
  }
  if (length(package) != 1L || is.na(package) || !nzchar(package)) {
    if (!is.na(ad_kind) && nzchar(ad_kind) && !is.na(density) && nzchar(density)) {
      package <- "adlaplace"
    } else {
      package <- NA_character_
    }
  } else {
    package <- as.character(package)
  }
  beta_map <- coerce_map_shorthand(beta_map, "beta_map")
  gamma_map <- coerce_map_shorthand(gamma_map, "gamma_map")
  theta_map <- coerce_map_shorthand(theta_map, "theta_map")
  if (is.null(beta_map)) {
    n_beta <- if (!is.null(X) && ncol(X) > 0L) ncol(X) else 0L
    beta_map <- if (n_beta > 0L) {
      Matrix::Diagonal(n_beta)
    } else {
      Matrix::Matrix(nrow = 0L, ncol = 0L)
    }
  }
  if (is.null(gamma_map)) {
    n_gamma <- if (!is.null(A) && ncol(A) > 0L) ncol(A) else 0L
    gamma_map <- if (n_gamma > 0L) {
      Matrix::Diagonal(n_gamma)
    } else {
      Matrix::Matrix(nrow = 0L, ncol = 0L)
    }
  }
  if (is.null(theta_map)) {
    theta_map <- Matrix::Matrix(nrow = 0L, ncol = 0L)
  }
  if (is.null(elgm_matrix)) {
    elgm_matrix <- Matrix::Matrix(nrow = n_obs, ncol = 0L)
  }
  beta_map <- as_ngC(beta_map)
  gamma_map <- as_ngC(gamma_map)
  theta_map <- as_ngC(theta_map)
  elgm_matrix <- as_ngC(elgm_matrix)
  validate_map_at_most_one_per_row_col(beta_map, "beta_map")
  validate_map_at_most_one_per_row_col(gamma_map, "gamma_map")
  n_beta_design <- if (is.null(X)) 0L else ncol(X)
  n_gamma_design <- if (is.null(A)) 0L else ncol(A)
  validate_design_parameter_maps(
    beta_map,
    gamma_map,
    n_beta_design,
    n_gamma_design
  )
  weights <- as.numeric(weights)
  if (length(weights) > 0L && length(weights) != n_obs) {
    stop(
      "length(weights) (", length(weights), ") must match length(y) (", n_obs, ")",
      call. = FALSE
    )
  }
  if (identical(density, "random_mult")) {
    precision <- normalize_random_mult_precision(precision)
  }
  methods::new(
    "density_data",
    y = as.numeric(y),
    ATp = ATp,
    XTp = XTp,
    beta_map = beta_map,
    gamma_map = gamma_map,
    theta_map = theta_map,
    elgm_matrix = elgm_matrix,
    density = density,
    ad_kind = ad_kind,
    package = package,
    precision = precision,
    weights = weights
  )
}

#' @include classes.R
#' @describeIn density_data Construct a density_data object
#'
#' @param y Response vector (default empty), or an existing \code{density_data} object
#'   to recast as another shard.
#' @param A Random-effects design matrix (\code{nrow(A) = length(y)}; any
#'   \code{Matrix} class).
#' @param X Fixed-effects design matrix (\code{nrow(X) = length(y)}; any
#'   \code{Matrix} class).
#' @param beta_map Beta parameter map (any \code{Matrix}; coerced to
#'   \code{ngCMatrix}), or a shorthand: length-1 integer \code{n} gives
#'   \code{Matrix::Matrix(nrow = n, ncol = 0)}; length-2 \code{c(row, n)} gives
#'   one column with a structural \code{1} at row \code{row} (1-based);
#'   and \code{list(indices, n)} gives one column per entry in \code{indices}
#'   with structural \code{1}s at those rows. \code{nrow} is the global beta
#'   block size; column \code{j} maps local \code{X} column \code{j} to a global
#'   beta row. When \code{ncol(X) > 0}, \code{ncol(beta_map)} must equal
#'   \code{ncol(X)} with exactly one nonzero per column. If
#'   \code{NULL}, defaults to \code{Matrix::Diagonal(ncol(X))} when
#'   \code{ncol(X) > 0}, otherwise an empty matrix.
#' @param gamma_map Gamma parameter map (any \code{Matrix}; coerced to
#'   \code{ngCMatrix}; at most one nonzero per row and column), or the same
#'   shorthand forms as \code{beta_map}. Column \code{j} maps local \code{A}
#'   column \code{j} into the global gamma block. When \code{ncol(A) > 0},
#'   \code{ncol(gamma_map)} must equal \code{ncol(A)} with exactly one nonzero
#'   per column. If \code{NULL}, defaults to
#'   \code{Matrix::Diagonal(ncol(A))} when \code{ncol(A) > 0},
#'   otherwise an empty matrix.
#' @param theta_map Theta parameter map (any \code{Matrix}; coerced to
#'   \code{ngCMatrix}), or the same shorthand forms as
#'   \code{beta_map}. Column \code{j} selects global theta row for component
#'   \code{j}. If \code{NULL}, defaults to an empty matrix.
#' @param elgm_matrix Optional ELGM map (any \code{Matrix}; coerced to
#'   \code{ngCMatrix}). If \code{NULL}, defaults to \code{Matrix::Matrix(nrow = length(y), ncol = 0)}.
#' @param density Registered AD density name for this shard (optional).
#' @param ad_kind Shard kind (\code{"observations"}, \code{"parameters"}, \code{"random"}; optional).
#' @param package Package recording AD tapes for this shard (optional).
  #' @param precision Optional precision payload (any R object). For
  #'   \code{density = "random_mult"}, a list with \code{Q} as
  #'   \code{dsCMatrix} (\code{uplo = "U"}); \code{dgCMatrix} is coerced via
  #'   \code{as_dsC_upper()}.
#' @param weights Optional per-observation weights (e.g. binomial trial counts).
#'   Empty means all ones.
#' @return An S4 object of class \code{"density_data"}: one
#'   automatic-differentiation shard. Slots hold the response, the transposed
#'   fixed- and random-effect design blocks, the maps from local design columns
#'   into the global parameter vectors, the registered density name, and the
#'   shard kind (\code{"observations"}, \code{"parameters"}, or \code{"random"}).
#' @export
density_data <- function(y = missing(),
                    A = NULL,
                    X = NULL,
                    beta_map = NULL,
                    gamma_map = NULL,
                    theta_map = NULL,
                    elgm_matrix = NULL,
                    density = NA_character_,
                    ad_kind = NA_character_,
                    package = NA_character_,
                    precision = NULL,
                    weights = numeric(0)) {
  if (missing(y)) {
    return(density_data_from_mats(
      y = numeric(0),
      A = A,
      X = X,
      beta_map = beta_map,
      gamma_map = gamma_map,
      theta_map = theta_map,
      elgm_matrix = elgm_matrix,
      density = density,
      ad_kind = ad_kind,
      package = package,
      precision = precision,
      weights = weights
    ))
  }
  if (is(y, "density_data")) {
    if (missing(ad_kind) && missing(density) && missing(package)) {
      stop(
        "recasting density_data requires ad_kind, density, and/or package",
        call. = FALSE
      )
    }
    kind <- if (missing(ad_kind)) y@ad_kind else as.character(ad_kind)
    fun <- if (missing(density)) y@density else as.character(density)
    if (length(kind) != 1L || is.na(kind) || !nzchar(kind)) {
      stop("ad_kind must be a non-empty string", call. = FALSE)
    }
    if (length(fun) != 1L || is.na(fun) || !nzchar(fun)) {
      stop("density must be a non-empty string", call. = FALSE)
    }
    pkg <- if (missing(package)) y@package else as.character(package)
    prec <- if (missing(precision)) y@precision else precision
    if (identical(fun, "random_mult")) {
      prec <- normalize_random_mult_precision(prec)
    }
    wts <- if (missing(weights)) y@weights else as.numeric(weights)
    out <- methods::new(
      "density_data",
      y = y@y,
      ATp = y@ATp,
      XTp = y@XTp,
      beta_map = y@beta_map,
      gamma_map = y@gamma_map,
      theta_map = y@theta_map,
      elgm_matrix = y@elgm_matrix,
      density = fun,
      ad_kind = kind,
      package = pkg,
      precision = prec,
      weights = wts
    )
    validate_density_data_maps(out, kind)
    return(out)
  }
  density_data_from_mats(
    y = y,
    A = A,
    X = X,
    beta_map = beta_map,
    gamma_map = gamma_map,
    theta_map = theta_map,
    elgm_matrix = elgm_matrix,
    density = density,
    ad_kind = ad_kind,
    package = package,
    precision = precision,
    weights = weights
  )
}
