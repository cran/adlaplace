#' Build a minimal density_data for shard tests
#'
#' Optional \code{ad_kind} / \code{density} / \code{precision} are stored on
#' the returned object so it can be passed directly to \code{ad_pack_ptr()}.
#' @keywords internal
test_ad_data <- function(y, A, X, config,
                         theta_local_row = 0L,
                         ad_kind = NA_character_,
                         density = NA_character_,
                         precision = NULL) {
  data <- adlaplace:::density_data_from_config_matrices(
    y = y,
    A = A,
    X = X,
    config = config,
    theta_local_row = theta_local_row
  )
  data@ad_kind <- ad_kind
  data@density <- density
  data@precision <- precision
  data
}

#' Set \code{ad_kind}, \code{density} and \code{precision} on a \code{density_data}
#' so it can be passed to \code{ad_pack_ptr()}.
#' @keywords internal
as_shard <- function(data, ad_kind, density, precision = NULL) {
  data@ad_kind <- ad_kind
  data@density <- density
  data@precision <- precision
  data
}

#' Random-shard density_data for tests with a single Amat block
#' @keywords internal
test_random_shard <- function(data, config, gamma_ids, theta_id, Q,
                              density = "random_diagonal") {
  n_gamma_full <- length(config$gamma)
  n_term <- length(gamma_ids)
  gamma_map <- Matrix::sparseMatrix(
    i = as.integer(gamma_ids),
    j = seq.int(0L, length.out = n_term),
    dims = c(n_gamma_full, n_term),
    index1 = FALSE,
    repr = "C"
  )
  adlaplace:::density_data(
    beta_map = Matrix::Matrix(nrow = length(config$beta), ncol = 0L),
    gamma_map = gamma_map,
    theta_map = data@theta_map,
    ad_kind = "random",
    density = density,
    precision = Q
  )
}

#' \code{mgcv::gamSim(6)} always \code{cat()}s from a nested \code{gamSim(1)}
#' call, even when \code{verbose = FALSE}. Capture that progress text.
#' @keywords internal
quiet_gamsim <- function(...) {
  sim <- NULL
  utils::capture.output(sim <- mgcv::gamSim(...))
  sim
}
