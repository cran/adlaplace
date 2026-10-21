test_that("permuted design columns match identity maps via beta_map/gamma_map", {
  set.seed(31)
  n <- 25L
  n_re <- 4L
  X <- Matrix::Matrix(cbind(1, rnorm(n), rnorm(n)))
  A <- Matrix::sparseMatrix(
    i = seq_len(n),
    j = sample(n_re, n, replace = TRUE),
    x = 1,
    dims = c(n, n_re)
  )
  beta <- c(1.5, -0.7, 0.3)
  gamma <- rnorm(n_re, sd = 0.4)
  sigma <- 0.6
  eta <- as.vector(X %*% beta + A %*% gamma)
  y <- rnorm(n, eta, sigma)

  config <- list(
    beta = beta,
    gamma = gamma,
    theta = log(sigma),
    transform_theta = TRUE
  )
  theta_map <- Matrix::sparseMatrix(i = 1L, j = 1L, dims = c(1L, 1L))

  obs_ref <- adlaplace:::density_data(
    y = y,
    A = A,
    X = X,
    theta_map = theta_map,
    ad_kind = "observations",
    density = "gaussian_obs"
  )
  extra_ref <- adlaplace::density_data(
    obs_ref,
    ad_kind = "parameters",
    density = "gaussian_extra"
  )

  beta_perm <- c(3L, 1L, 2L)
  gamma_perm <- c(4L, 2L, 1L, 3L)
  X_perm <- X[, beta_perm, drop = FALSE]
  A_perm <- A[, gamma_perm, drop = FALSE]
  beta_map_perm <- Matrix::sparseMatrix(
    i = beta_perm,
    j = seq_len(length(beta_perm)),
    dims = c(length(beta), length(beta_perm))
  )
  gamma_map_perm <- Matrix::sparseMatrix(
    i = gamma_perm,
    j = seq_len(length(gamma_perm)),
    dims = c(length(gamma), length(gamma_perm))
  )

  obs_perm <- adlaplace:::density_data(
    y = y,
    A = A_perm,
    X = X_perm,
    beta_map = beta_map_perm,
    gamma_map = gamma_map_perm,
    theta_map = theta_map,
    ad_kind = "observations",
    density = "gaussian_obs"
  )
  extra_perm <- adlaplace::density_data(
    obs_perm,
    ad_kind = "parameters",
    density = "gaussian_extra"
  )

  x <- c(beta, gamma, log(sigma))
  manual <- sum(stats::dnorm(y, eta, sigma, log = TRUE))

  ptr_ref <- adlaplace::ad_pack_ptr(obs_ref, config)
  ptr_extra_ref <- adlaplace::ad_pack_ptr(extra_ref, config)
  ad_ref <- adlaplace::joint_log_dens(ptr_ref, x, negative = FALSE) +
    adlaplace::joint_log_dens(ptr_extra_ref, x, negative = FALSE)

  ptr_perm <- adlaplace::ad_pack_ptr(obs_perm, config)
  ptr_extra_perm <- adlaplace::ad_pack_ptr(extra_perm, config)
  ad_perm <- adlaplace::joint_log_dens(ptr_perm, x, negative = FALSE) +
    adlaplace::joint_log_dens(ptr_extra_perm, x, negative = FALSE)

  expect_equal(ad_ref, manual, tolerance = 1e-10)
  expect_equal(ad_perm, manual, tolerance = 1e-10)
  expect_equal(ad_perm, ad_ref, tolerance = 1e-10)
})

test_that("design maps require ncol match and one nonzero per column", {
  X <- matrix(c(1, 1, 2, 2), nrow = 2L, ncol = 2L)
  expect_error(
    adlaplace:::density_data(
      y = c(0, 0),
      X = X,
      beta_map = Matrix::Diagonal(1L)
    ),
    "beta_map ncol"
  )
  expect_error(
    adlaplace:::density_data(
      y = c(0, 0),
      X = X,
      beta_map = Matrix::sparseMatrix(
        i = 1L,
        j = 2L,
        dims = c(2L, 2L)
      )
    ),
    "exactly one nonzero"
  )
})
