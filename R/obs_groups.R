#' Default observation shard map (one column, all observations)
#'
#' @param n_obs Number of observations (\code{length(y)}).
#' @return A \code{ngCMatrix} with one shard column and a structural nonzero
#'   mapping each observation row to that shard (0-based row indices).
#' @keywords internal
default_obs_groups <- function(n_obs) {
  n_obs <- as.integer(n_obs)[1L]
  if (is.na(n_obs) || n_obs < 1L) {
    return(as_ngC(Matrix::sparseMatrix(
      i = integer(0),
      j = integer(0),
      dims = c(0L, 1L),
      index1 = FALSE
    )))
  }
  as_ngC(Matrix::sparseMatrix(
    i = seq.int(0L, n_obs - 1L),
    j = rep(0L, n_obs),
    dims = c(n_obs, 1L),
    index1 = FALSE
  ))
}

#' Number of removable observation units
#'
#' Units are ELGM stratum columns when \code{elgm_matrix} has columns;
#' otherwise rows of the observation vector \code{y}. Same domain as
#' \code{\link{grad_obs_units}}.
#'
#' @param data An observation \code{density_data}, a \code{model_data()}
#'   bundle, or an \code{"adlaplace_fit"}.
#' @return Integer length-1 count of units.
#' @export
n_obs_units <- function(data) {
  obs <- .resolve_obs_density(data)
  elgm <- obs@elgm_matrix
  if (methods::is(elgm, "Matrix") && ncol(elgm) > 0L) {
    return(as.integer(ncol(elgm)))
  }
  as.integer(length(obs@y))
}

#' Build \code{obs_groups} for a subset of observation units
#'
#' Restricts which observation units are recorded on observation tapes.
#' Domain indices are **0-based** (ELGM stratum columns, or rows of \code{y}).
#'
#' @param n_domain Number of units in the full domain.
#' @param units Integer vector of 0-based unit indices to keep. Default all.
#' @param grouping One of:
#'   \describe{
#'     \item{\code{"identity"}}{One shard column per selected unit (scores).}
#'     \item{\code{"together"}}{One shard containing all selected units.}
#'     \item{\code{"filter"}}{Start from \code{obs_groups}, drop unselected
#'       rows, drop empty columns (keeps coarse fit sharding).}
#'   }
#' @param obs_groups Existing shard map; required when
#'   \code{grouping = "filter"}.
#' @return A \code{ngCMatrix} suitable for \code{config$obs_groups}.
#' @seealso \code{\link{n_obs_units}}, \code{\link{grad_obs_units}},
#'   \code{\link{ad_pack_drop}}
#' @export
obs_groups_units <- function(
  n_domain,
  units = NULL,
  grouping = c("identity", "together", "filter"),
  obs_groups = NULL
) {
  grouping <- match.arg(grouping)
  n_domain <- as.integer(n_domain)[1L]
  if (is.na(n_domain) || n_domain < 1L) {
    stop("`n_domain` must be a positive integer", call. = FALSE)
  }
  if (is.null(units)) {
    units <- seq.int(0L, n_domain - 1L)
  } else {
    units <- as.integer(units)
  }
  if (!length(units)) {
    stop("`units` is empty", call. = FALSE)
  }
  if (any(is.na(units)) || any(units < 0L) || any(units >= n_domain)) {
    stop(
      "`units` must be 0-based indices in [0, ", n_domain, ")",
      call. = FALSE
    )
  }
  if (anyDuplicated(units)) {
    stop("`units` must be unique", call. = FALSE)
  }

  if (identical(grouping, "identity")) {
    n_u <- length(units)
    return(as_ngC(Matrix::sparseMatrix(
      i = units,
      j = seq.int(0L, n_u - 1L),
      dims = c(n_domain, n_u),
      index1 = FALSE
    )))
  }

  if (identical(grouping, "together")) {
    return(as_ngC(Matrix::sparseMatrix(
      i = units,
      j = rep(0L, length(units)),
      dims = c(n_domain, 1L),
      index1 = FALSE
    )))
  }

  # grouping == "filter"
  if (is.null(obs_groups)) {
    stop(
      "`obs_groups` is required when grouping = \"filter\"",
      call. = FALSE
    )
  }
  og <- as_ngC(obs_groups)
  if (nrow(og) != n_domain) {
    stop(
      "`obs_groups` has ", nrow(og), " rows but n_domain = ", n_domain,
      call. = FALSE
    )
  }
  keep <- logical(n_domain)
  keep[units + 1L] <- TRUE
  og_t <- methods::as(og, "TsparseMatrix")
  sel <- keep[og_t@i + 1L]
  if (!any(sel)) {
    stop("filtering obs_groups left no selected units", call. = FALSE)
  }
  i_keep <- og_t@i[sel]
  j_keep <- og_t@j[sel]
  # Remap shard columns to drop empties, preserving relative order.
  j_levels <- sort(unique(j_keep))
  j_new <- match(j_keep, j_levels) - 1L
  as_ngC(Matrix::sparseMatrix(
    i = i_keep,
    j = j_new,
    dims = c(n_domain, length(j_levels)),
    index1 = FALSE
  ))
}

#' @keywords internal
.resolve_obs_density <- function(data) {
  if (inherits(data, "adlaplace_fit")) {
    data <- data$model_data
  }
  if (is.list(data) && !methods::is(data, "density_data")) {
    if (is.null(data$observations) || !length(data$observations)) {
      stop(
        "`data` list must be a model_data() bundle with $observations",
        call. = FALSE
      )
    }
    data <- data$observations[[1L]]
  }
  if (!methods::is(data, "density_data")) {
    stop("`data` must be a density_data observation shard", call. = FALSE)
  }
  if (!identical(as.character(data@ad_kind), "observations")) {
    stop("`data@ad_kind` must be \"observations\"", call. = FALSE)
  }
  data
}

#' Build \code{config$obs_groups} from design / ELGM when missing
#'
#' Used by \code{\link{adlaplace}} and \code{\link{ad_pack}} so that
#' \code{num_shards} alone is enough for case-crossover (ELGM) models.
#' When \code{elgm_matrix} has columns, shards index strata, not raw rows of
#' \code{y}.
#'
#' When \code{config$obs_groups} is missing and \code{config$obs_units} is set
#' (0-based unit indices), builds an identity shard map over those units via
#' \code{\link{obs_groups_units}}. Optional \code{config$obs_units_grouping}
#' selects \code{"identity"} (default) or \code{"together"}.
#'
#' @param config Config list (may already contain \code{obs_groups}).
#' @param A Random-effects design (\code{term_data$A}).
#' @param elgm_matrix Optional ELGM stratum map.
#' @param num_shards Maximum shards (from \code{config$num_shards} if missing).
#' @param num_threads Used for \code{min_shards} when ELGM is present.
#' @param n_domain Optional unit-domain size for \code{obs_units}; inferred from
#'   \code{elgm_matrix} or \code{nrow(A)} when omitted.
#' @return \code{config} with \code{obs_groups} filled when possible.
#' @keywords internal
ensure_config_obs_groups <- function(
  config,
  A,
  elgm_matrix = NULL,
  num_shards = NULL,
  num_threads = 1L,
  n_domain = NULL
) {
  if (!is.null(config[["obs_groups"]])) {
    return(config)
  }

  obs_units <- config[["obs_units"]]
  if (!is.null(obs_units)) {
    if (is.null(n_domain)) {
      has_elgm <- !is.null(elgm_matrix) &&
        methods::is(elgm_matrix, "Matrix") &&
        ncol(elgm_matrix) > 0L
      if (has_elgm) {
        n_domain <- ncol(elgm_matrix)
      } else if (!is.null(A) && nrow(A) > 0L) {
        n_domain <- nrow(A)
      } else {
        stop(
          "config$obs_units set but cannot infer n_domain from elgm_matrix / A",
          call. = FALSE
        )
      }
    }
    grouping <- config[["obs_units_grouping"]]
    if (is.null(grouping)) {
      grouping <- "identity"
    }
    grouping <- as.character(grouping)[1L]
    if (!grouping %in% c("identity", "together")) {
      stop(
        "config$obs_units_grouping must be \"identity\" or \"together\" ",
        "(use obs_groups_units(..., grouping = \"filter\") with an existing ",
        "obs_groups for filter)",
        call. = FALSE
      )
    }
    config$obs_groups <- obs_groups_units(
      n_domain = n_domain,
      units = obs_units,
      grouping = grouping
    )
    return(config)
  }

  if (is.null(A) || ncol(A) < 1L) {
    return(config)
  }
  A <- as_dgC(A)
  if (is.null(num_shards)) {
    num_shards <- config[["num_shards"]]
  }
  if (is.null(num_shards)) {
    num_shards <- 100L
  }
  num_shards <- as.integer(num_shards)[1L]
  num_threads <- as.integer(num_threads)[1L]
  if (is.na(num_threads) || num_threads < 1L) {
    num_threads <- 1L
  }
  if (is.na(num_shards) || num_shards < 1L) {
    num_shards <- 100L
  }
  has_elgm <- !is.null(elgm_matrix) &&
    methods::is(elgm_matrix, "Matrix") &&
    ncol(elgm_matrix) > 0L
  if (has_elgm) {
    config$obs_groups <- obs_groups(
      A,
      elgm_matrix = elgm_matrix,
      num_shards = num_shards,
      min_shards = min(num_shards, num_threads * 4L)
    )
  } else {
    config$obs_groups <- obs_groups(A, num_shards = num_shards)
  }
  config
}

#' Partition observations into AD shards by sparsity pattern
#'
#' The function partitions columns into \code{num_shards} shards using
#' quantiles of the first right singular vector, optionally using \pkg{RSpectra}
#' for efficiency when available.
#'
#' The resulting grouping is returned as a pattern sparse matrix whose columns
#' correspond to shards and whose nonzeros mark membership. Loadings are used
#' only to form the partition. Shards are ordered from most heterogeneous to
#' most homogeneous.
#'
#' @param A Random-effects design matrix (\code{nrow(A)} = number of observations).
#' @param elgm_matrix A numeric matrix (or matrix-like object) for extended latent gaussian models.
#' @param num_shards Integer giving the maximum number of shards to construct.
#'   The actual number of shards may be smaller if fewer distinct loadings
#'   are present.
#' @param min_shards Integer giving the minimum number of shards. When there
#'   are fewer distinct singular-vector loadings than \code{min_shards}, only
#'   the largest exact-loading group is split until \code{min_shards} is reached
#'   (or that group cannot be split further).
#'
#' @details
#' If the \pkg{RSpectra} package is available, the leading singular vector
#' is computed using \code{RSpectra::svds}; otherwise, a full singular value
#' decomposition via \code{\link[base]{svd}} is used.
#'
#' Shard boundaries are defined by empirical quantiles of the loadings.
#' Shards are subsequently reordered so that shards with larger within-shard
#' variability appear first.
#'
#' @return
#' A sparse matrix of class \code{"ngCMatrix"} (from \pkg{Matrix}), with
#' one column per shard and one row per observation (or ELGM stratum).
#' Nonzero entries mark shard membership; there is no \code{@x} slot.
#'
#' @examples
#' set.seed(1)
#' A <- matrix(rnorm(100), 20, 5)
#' G <- obs_groups(A, num_shards = 3)
#' G
#' @export
obs_groups <- function(A, elgm_matrix, num_shards, min_shards = 0) {
  ATp <- Matrix::t(A)
  if (missing(num_shards)) {
    num_shards <- ncol(ATp)
  }

  if (!missing(elgm_matrix)) {
    ATp_t <- methods::as(Matrix::Matrix(ATp), "TsparseMatrix")
    ATp_t <- data.frame(row = ATp_t@j, gamma = ATp_t@i)
    elgm_matrix_t <- methods::as(elgm_matrix, "TsparseMatrix")
    elgm_matrix_t <- data.frame(row = elgm_matrix_t@i, strata = elgm_matrix_t@j)

    data.table::setDT(ATp_t)
    data.table::setDT(elgm_matrix_t)

    A_elgm_merge <- merge(ATp_t, elgm_matrix_t)
    A_elgm_merge <- A_elgm_merge[, c("gamma", "strata")]
    A_elgm_merge <- A_elgm_merge[!duplicated(A_elgm_merge), ]
    A_elgm_merge <- A_elgm_merge[order(A_elgm_merge$strata, A_elgm_merge$gamma), ]

    ATp <- Matrix::sparseMatrix(
      i = A_elgm_merge$gamma,
      j = A_elgm_merge$strata,
      x = rep(1.0, nrow(A_elgm_merge)),
      index1 = FALSE, dims = c(nrow(ATp), ncol(elgm_matrix))
    )
  }

  if (inherits(ATp, "ngCMatrix")) {
    ATp <- methods::as(ATp, "dMatrix")
  }
  # RSpectra::svds expects numeric Matrix input in a compressed sparse form.
  ATp <- methods::as(methods::as(ATp, "dMatrix"), "CsparseMatrix")
  if (requireNamespace("RSpectra", quietly = TRUE) && (
    max(dim(ATp)) > 100
  )) {
    loadings <- RSpectra::svds(ATp, k = 1)$v[, 1]
  } else {
    if (max(dim(ATp)) > 1e5) {
      warning("ATp matrix is very large, consider installing the RSpectra package")
    }
    sv <- svd(ATp)
    loadings <- sv$v[, 1]
  }

  uniqueLoadings <- sort(unique(loadings[is.finite(loadings)]))
  if (!length(uniqueLoadings)) {
    stop("obs_groups: no finite singular-vector loadings", call. = FALSE)
  }

  # Discrete groups from exact loadings. If fewer than min_shards, split only
  # the largest group into enough pieces to reach min_shards.
  group_id <- match(loadings, uniqueLoadings)
  if (length(uniqueLoadings) < min_shards) {
    counts <- tabulate(group_id, nbins = length(uniqueLoadings))
    big <- which.max(counts)
    idx <- which(group_id == big)
    n_pieces <- min(
      length(idx),
      as.integer(min_shards) - length(uniqueLoadings) + 1L
    )
    if (n_pieces > 1L) {
      piece <- floor((seq_along(idx) - 1L) * n_pieces / length(idx))
      new_ids <- c(big, seq.int(length(uniqueLoadings) + 1L, length.out = n_pieces - 1L))
      group_id[idx] <- new_ids[piece + 1L]
    }
  }

  n_groups <- length(unique(group_id[!is.na(group_id)]))
  if (n_groups > num_shards) {
    # Too many distinct loadings: quantile-merge on the continuous loadings.
    groupCut <- unique(as.numeric(stats::quantile(
      uniqueLoadings, seq(0, 1, len = num_shards + 1)
    )))
    groupCut[1] <- groupCut[1] - 1
    groupCut[length(groupCut)] <- groupCut[length(groupCut)] + 1
    groupCut <- unique(groupCut)
    if (length(groupCut) < 2L) {
      stop("obs_groups: could not form unique cut breaks", call. = FALSE)
    }
    loadingsCut <- cut(loadings, groupCut)
    shard_id <- as.integer(factor(
      loadingsCut, names(sort(table(loadingsCut), decreasing = TRUE))
    )) - 1L
  } else {
    # One shard per discrete group (after any largest-group split).
    shard_id <- as.integer(factor(
      group_id, names(sort(table(group_id), decreasing = TRUE))
    )) - 1L
  }

  shard_sd <- pmax(tapply(loadings, shard_id, stats::sd), 0, na.rm = TRUE)
  shard_order <- order(shard_sd, decreasing = TRUE) - 1L
  shard_id_ordered <- match(shard_id, shard_order) - 1L

  as_ngC(Matrix::sparseMatrix(
    i = seq(0, len = length(loadings)),
    j = shard_id_ordered,
    dims = c(length(loadings), max(shard_id_ordered) + 1L),
    index1 = FALSE
  ))
}
