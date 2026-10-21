#' Midpoint finite differences along a coordinate grid
#'
#' @param y Numeric vector or matrix of values along \code{grid}.
#' @param grid Numeric coordinate values (length \code{nrow(y)} or \code{length(y)}).
#' @return List with \code{mid} (midpoints) and \code{fd} (slopes).
#' @keywords internal
fd_midpoint <- function(y, grid) {
  grid <- as.numeric(grid)
  dx <- diff(grid)
  mid <- grid[-1L] - dx / 2
  if (is.null(dim(y))) {
    list(mid = mid, fd = as.numeric(diff(y) / dx))
  } else {
    list(mid = mid, fd = sweep(diff(y), 1L, dx, "/"))
  }
}

#' Scan joint log-density AD derivatives vs finite differences
#'
#' Evaluates the joint log density and its AD gradient/Hessian while varying
#' one coordinate of the full parameter vector (beta, gamma, theta). Returns
#' plain matrices shaped for \code{matplot()}/\code{matlines()}.
#'
#' @param ad_pack An \code{ad_pack} object (typically with \code{num_threads = 1}).
#' @param x Numeric full parameter vector.
#' @param index Integer (1-based) coordinate of \code{x} to vary.
#' @param width Half-width of the scan interval around \code{x[index]}.
#' @param n Integer number of grid points (default 11).
#' @param ad_shards Optional 0-based shard indices passed to
#'   \code{\link{joint_log_dens}} / \code{\link{grad}} / \code{\link{hessian}}.
#' @param negative Logical; passed through to the joint-density evaluators.
#' @return A list with:
#'   \item{grid}{Coordinate values (length \code{n}).}
#'   \item{mid}{Midpoints for finite differences (length \code{n - 1}).}
#'   \item{value}{Joint density along the grid.}
#'   \item{fd_value}{Midpoint differences of \code{value}.}
#'   \item{grad}{\code{n x length(x)} matrix of AD gradients (rows = grid).}
#'   \item{fd_grad}{\code{(n-1) x length(x)} midpoint slopes of \code{grad}.}
#'   \item{hess_row}{\code{n x length(x)} AD Hessian rows for \code{index}.}
#' @export
deriv_scan_joint <- function(ad_pack,
                             x,
                             index,
                             width = 0.1,
                             n = 11L,
                             ad_shards = NULL,
                             negative = FALSE) {
  x <- as.numeric(x)
  index <- as.integer(index)[1L]
  n <- as.integer(n)[1L]
  if (index < 1L || index > length(x)) {
    stop("index must be in 1:length(x)", call. = FALSE)
  }
  if (n < 3L) {
    stop("n must be at least 3", call. = FALSE)
  }

  grid <- x[index] + seq(-width, width, length.out = n)
  par_grid <- matrix(x, nrow = n, ncol = length(x), byrow = TRUE)
  par_grid[, index] <- grid
  x_list <- split(par_grid, row(par_grid))

  value <- vapply(
    x_list,
    joint_log_dens,
    numeric(1),
    ad_pack = ad_pack,
    ad_shards = ad_shards,
    negative = negative
  )

  grad <- do.call(
    rbind,
    lapply(
      x_list,
      grad,
      ad_pack = ad_pack,
      inner = FALSE,
      ad_shards = ad_shards,
      negative = negative
    )
  )

  hess_row <- matrix(NA_real_, nrow = n, ncol = length(x))
  for (i in seq_len(n)) {
    H <- as.matrix(hessian(
      ad_pack,
      x_list[[i]],
      inner = FALSE,
      ad_shards = ad_shards,
      negative = negative
    ))
    hess_row[i, ] <- H[index, ]
  }

  fd_v <- fd_midpoint(value, grid)
  fd_g <- fd_midpoint(grad, grid)

  list(
    grid = grid,
    mid = fd_v$mid,
    value = value,
    fd_value = fd_v$fd,
    grad = grad,
    fd_grad = fd_g$fd,
    hess_row = hess_row,
    index = index
  )
}

#' Scan Laplace profile AD derivatives vs finite differences
#'
#' Evaluates \code{\link{log_lik_laplace}(deriv = TRUE)} while varying one
#' coordinate of the outer parameter vector (beta, theta only; gammas are
#' profiled out). Returns plain matrices shaped for \code{matplot()}/\code{matlines()}.
#'
#' @param ad_pack An \code{ad_pack} object.
#' @param x Numeric outer parameter vector (length beta + theta).
#' @param index Integer (1-based) coordinate of \code{x} to vary.
#' @param width Half-width of the scan interval around \code{x[index]}.
#' @param n Integer number of grid points (default 7).
#' @param config Config list passed to \code{log_lik_laplace}.
#' @param gamma Optional gamma start for the inner optimization.
#' @param control Inner-optimization control list.
#' @return A list with:
#'   \item{grid}{Coordinate values (length \code{n}).}
#'   \item{mid}{Midpoints for finite differences.}
#'   \item{value}{Negative log likelihood along the grid.}
#'   \item{fd_value}{Midpoint differences of \code{value}.}
#'   \item{grad}{\code{n x length(x)} AD profile gradients.}
#'   \item{fd_grad}{Midpoint slopes of \code{grad}.}
#'   \item{half_log_det}{Half log-determinant along the grid.}
#'   \item{d_det}{AD derivative of the half log-determinant wrt outer params.}
#'   \item{fd_det}{Midpoint differences of \code{half_log_det}.}
#'   \item{u_hat}{\code{n x n_gamma} inner modes.}
#'   \item{dU_index}{\code{n x n_gamma} AD \code{dU} slices for \code{index}.}
#'   \item{fd_u}{Midpoint slopes of \code{u_hat}.}
#'   \item{mid_result}{Full \code{log_lik_laplace} result at the middle grid point.}
#' @export
deriv_scan_laplace <- function(ad_pack,
                               x,
                               index,
                               width = 0.5,
                               n = 7L,
                               config = list(),
                               gamma = NULL,
                               control = list(
                                 maxit = 100L,
                                 report.level = 0,
                                 report.freq = 0
                               )) {
  x <- as.numeric(x)
  index <- as.integer(index)[1L]
  n <- as.integer(n)[1L]
  if (index < 1L || index > length(x)) {
    stop("index must be in 1:length(x) (outer beta/theta only)", call. = FALSE)
  }
  if (n < 3L) {
    stop("n must be at least 3", call. = FALSE)
  }

  grid <- x[index] + seq(-width, width, length.out = n)
  par_grid <- matrix(x, nrow = n, ncol = length(x), byrow = TRUE)
  par_grid[, index] <- grid
  x_list <- split(par_grid, row(par_grid))

  scan_args <- list(
    ad_pack = ad_pack,
    config = config,
    deriv = TRUE,
    control = control
  )
  if (!is.null(gamma)) {
    scan_args$gamma <- gamma
  }

  res_scan <- lapply(x_list, function(xi) {
    do.call(log_lik_laplace, c(list(x = xi), scan_args))
  })

  value <- vapply(res_scan, `[[`, numeric(1), "neg_log_lik")
  half_log_det <- vapply(
    res_scan,
    function(r) as.numeric(r$hessian$half_log_det)[1L],
    numeric(1)
  )
  grad <- do.call(
    rbind,
    lapply(res_scan, function(r) as.numeric(r$deriv$d_neg_log_lik))
  )
  d_det <- do.call(
    rbind,
    lapply(res_scan, function(r) {
      d <- r$deriv$d_det
      if (is.null(d)) {
        rep(NA_real_, length(x))
      } else {
        as.numeric(d)
      }
    })
  )
  u_hat <- do.call(
    rbind,
    lapply(res_scan, function(r) as.numeric(r$inner_opt$solution))
  )
  dU_index <- do.call(
    rbind,
    lapply(res_scan, function(r) {
      dU <- r$gradient$outer$dU
      if (is.null(dU)) {
        rep(NA_real_, ncol(u_hat))
      } else {
        as.numeric(as.matrix(dU)[, index])
      }
    })
  )

  fd_v <- fd_midpoint(value, grid)
  fd_g <- fd_midpoint(grad, grid)
  fd_d <- fd_midpoint(half_log_det, grid)
  fd_u <- fd_midpoint(u_hat, grid)

  list(
    grid = grid,
    mid = fd_v$mid,
    value = value,
    fd_value = fd_v$fd,
    grad = grad,
    fd_grad = fd_g$fd,
    half_log_det = half_log_det,
    d_det = d_det,
    fd_det = fd_d$fd,
    u_hat = u_hat,
    dU_index = dU_index,
    fd_u = fd_u$fd,
    index = index,
    mid_result = res_scan[[(n + 1L) %/% 2L]]
  )
}
