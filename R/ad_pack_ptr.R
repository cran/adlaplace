#' @include density_data.R
NULL

#' Build raw AD handle for one density shard
#'
#' Constructs CppAD tapes for a single shard. The density kind and name come
#' from \code{data@ad_kind} and \code{data@density}. For \code{ad_kind = "random"},
#' \code{data@precision} is passed straight to the backend. For
#' \code{random_diagonal} it must be a numeric vector of diagonal precision
#' weights with \code{length == ncol(gamma_map)}. For \code{random_mult} it must
#' be \code{list(Q, log_det, rank)} where \code{Q} is a square
#' \code{dsCMatrix} with \code{uplo = "U"} (upper triangle only); use
#' \code{adlaplace:::as_dsC_upper(Q)} to coerce a \code{dgCMatrix}. Missing
#' precision means the shard is not built (e.g. diffuse \code{rpoly} with
#' \code{sd = Inf} via \code{model_data()}); calling \code{ad_pack_ptr()}
#' without it is an error.
#'
#' Merge handles for multiple shards with \code{c()} before calling
#' \code{\link{ad_pack}}.
#'
#' @param data An \code{density_data} object with \code{ad_kind} and \code{ad_pack}
#'   slots set.
#' @param config Model configuration list (\code{beta}, \code{theta}, etc.).
#'   \code{gamma} is optional; when missing, zeros of length
#'   \code{nrow(gamma_map)} are used as AD tape seeds.
#'   \code{config$num_threads} does not assign OpenMP threads; use
#'   \code{num_threads} on \code{\link{ad_pack}} after merging shards.
#'   When \code{info} is supplied, missing \code{beta} / \code{theta} /
#'   \code{gamma} / \code{transform_theta} are filled from it; explicit
#'   \code{config} entries win. Missing \code{gamma} becomes zeros of length
#'   \code{nrow(info$gamma)}.
#' @param info Optional \code{model_data()$term_data$info} list. When provided,
#'   tape seeds use \code{info$beta$init} and \code{info$theta$init} (the latter
#'   on the log scale where \code{info$theta$log} is \code{TRUE}); \code{gamma}
#'   defaults to zeros (\code{info$gamma} has no \code{init} column).
#' @return External pointer of class \code{ad_pack_ptr}.
#' @export
ad_pack_ptr <- function(data, config = NULL, info = NULL) {
  if (missing(data) || !is(data, "density_data")) {
    stop("`data` must be an density_data object", call. = FALSE)
  }
  kind <- data@ad_kind
  name <- data@density
  if (length(name) != 1L || is.na(name) || !nzchar(name)) {
    stop("data@density is required (density name)", call. = FALSE)
  }
  if (length(kind) != 1L || is.na(kind) || !nzchar(kind)) {
    stop("data@ad_kind is required (observations/parameters/random)", call. = FALSE)
  }

  validate_density_data_maps(data, kind)
  if (is.null(config)) {
    config <- list()
  }
  config <- fill_config_from_info(config, info)
  config <- normalize_config_for_ptr(config, data, kind)
  validate_config_layout(data, config, kind)

  pkg <- data@package
  if (length(pkg) != 1L || is.na(pkg) || !nzchar(pkg)) {
    pkg <- "adlaplace"
  }

  # ad_kind "random": each density has its own Rcpp entry point named
  # create_ad_shard_<ad_pack>, looked up in data@package (default "adlaplace").
  # e.g. random_diagonal -> adlaplace:::create_ad_shard_random_diagonal(data, config)
  if (identical(kind, "random")) {
    fn <- utils::getFromNamespace(paste0("create_ad_shard_", name), pkg)
    return(fn(data, config))
  }

  # ad_kind "observations"/"parameters": dispatch to the backend package's
  # get_ad_pack_raw_obs / get_ad_pack_raw_parameters (tapes must be recorded
  # in that package's DSO); the density name is resolved inside that package.
  builder <- switch(kind,
    observations = "get_ad_pack_raw_obs",
    parameters = "get_ad_pack_raw_parameters",
    stop(
      "unknown ad_kind `", kind,
      "`; expected observations, parameters, or random",
      call. = FALSE
    )
  )
  fn <- utils::getFromNamespace(builder, pkg)
  fn(data, config, name)
}

#' Combine \code{ad_pack_ptr} handles
#'
#' Moves AD shards from one or more \code{ad_pack_ptr} objects into a single
#' handle. Inputs are cleared after the merge (ownership moves).
#'
#' @param x An \code{ad_pack_ptr}.
#' @param ... Additional \code{ad_pack_ptr} objects.
#' @return A combined \code{ad_pack_ptr}.
#' @seealso \code{\link{clone_ad_pack_ptr}}, \code{\link{ad_pack}}
#' @export
#' @method c ad_pack_ptr
c.ad_pack_ptr <- function(x, ...) {
  dots <- list(...)
  if (length(dots) == 0L) {
    return(x)
  }
  shards <- c(list(x), dots)
  if (!all(vapply(shards, function(s) is(s, "ad_pack_ptr"), logical(1L)))) {
    stop("all arguments must be ad_pack_ptr")
  }
  if (length(shards) == 1L) {
    return(shards[[1L]])
  }
  c_ad_pack_ptr(shards)
}

#' Deep copy of an \code{ad_pack_ptr} handle
#'
#' Returns a new \code{ad_pack_ptr} with independent C++ state (CppAD tapes and
#' sparsity patterns). Hessian templates are not copied; call \code{\link{ad_pack}}
#' on the clone to attach maps. The source handle is left unchanged, unlike
#' \code{\link{c.ad_pack_ptr}} which moves shards and clears the inputs.
#' OpenMP thread affinity from a prior \code{ad_pack(..., num_threads > 1)} is
#' cleared on the clone so \code{\link{joint_log_dens}} / \code{\link{grad}} /
#' \code{\link{hessian}} remain available as serial debug APIs.
#'
#' @param x An \code{ad_pack_ptr} object.
#' @return A new \code{ad_pack_ptr}.
#' @export
clone_ad_pack_ptr <- function(x) {
  if (!inherits(x, "ad_pack_ptr")) {
    stop("`x` must be an ad_pack_ptr object")
  }
  clone_ad_pack_ptr_(x)
}
