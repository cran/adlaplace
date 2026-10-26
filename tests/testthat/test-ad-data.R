test_that("density_data recasts shard from existing density_data", {
  obs <- adlaplace::density_data(
    y = 1:3,
    beta_map = 2L,
    gamma_map = 3L,
    theta_map = c(3L, 3L),
    ad_kind = "observations",
    density = "nbinom_obs"
  )
  extra <- adlaplace::density_data(
    obs,
    ad_kind = "parameters",
    density = "nbinom_extra"
  )
  expect_identical(extra@y, obs@y)
  expect_identical(extra@beta_map, obs@beta_map)
  expect_identical(extra@gamma_map, obs@gamma_map)
  expect_identical(extra@theta_map, obs@theta_map)
  expect_equal(extra@ad_kind, "parameters")
  expect_equal(extra@density, "nbinom_extra")
  expect_false(identical(extra, obs))
})

test_that("length-1 integer map args become zero-column matrices", {
  model <- adlaplace::density_data(
    beta_map = 2L,
    gamma_map = 3L,
    theta_map = 1L
  )
  expect_equal(sizes(model), list(n_beta = 2L, n_gamma = 3L, n_theta = 1L))
  expect_equal(ncol(model@beta_map), 0L)
  expect_equal(ncol(model@gamma_map), 0L)
  expect_equal(ncol(model@theta_map), 0L)
  expect_true("elgm_matrix" %in% methods::slotNames(model))
})

test_that("length-2 map shorthand builds one-column maps", {
  model <- adlaplace::density_data(
    beta_map = c(1L, 2L),
    theta_map = c(2L, 3L)
  )
  expect_equal(sizes(model), list(n_beta = 2L, n_gamma = 0L, n_theta = 3L))
  expect_equal(nrow(model@beta_map), 2L)
  expect_equal(ncol(model@beta_map), 1L)
  expect_equal(model@beta_map@i + 1L, 1L)
  expect_equal(nrow(model@theta_map), 3L)
  expect_equal(ncol(model@theta_map), 1L)
  expect_equal(model@theta_map@i + 1L, 2L)
})

test_that("length-2 map shorthand validates row index", {
  expect_error(
    adlaplace::density_data(theta_map = c(0L, 1L)),
    "row index"
  )
  expect_error(
    adlaplace::density_data(theta_map = c(4L, 3L)),
    "row index"
  )
})

test_that("list(indices, nrow) shorthand builds one-to-one columns", {
  model <- adlaplace::density_data(
    gamma_map = list(c(1L, 3L), 4L)
  )
  expect_equal(nrow(model@gamma_map), 4L)
  expect_equal(ncol(model@gamma_map), 2L)
  expect_equal(model@gamma_map@i + 1L, c(1L, 3L))
})

test_that("list(indices, nrow) shorthand validates shape and values", {
  expect_error(
    adlaplace::density_data(theta_map = list(c(1L, 2L), c(3L, 4L))),
    "second element must be scalar nrow"
  )
  expect_error(
    adlaplace::density_data(theta_map = list(c(1.5), 3L)),
    "indices must be finite integers"
  )
  expect_error(
    adlaplace::density_data(theta_map = list(c(4L), 3L)),
    "indices must be between 1 and nrow"
  )
})

test_that("random_diagonal ad_pack_ptr accepts numeric precision vector", {
  nr <- 4L
  model <- adlaplace::density_data(
    gamma_map = Matrix::sparseMatrix(
      i = seq_len(nr),
      j = seq_len(nr),
      x = 1,
      dims = c(6L, nr)
    ),
    theta_map = c(1L, 1L),
    ad_kind = "random",
    density = "random_diagonal",
    precision = rep(2, nr)
  )
  config <- list(
    beta = numeric(0),
    gamma = rep(0, 6),
    theta = 0.1,
    transform_theta = FALSE
  )
  ptr <- adlaplace::ad_pack_ptr(model, config)
  expect_true(is(ptr, "ad_pack_ptr"))
})

test_that("random_diagonal ad_pack_ptr errors when precision is NULL", {
  nr <- 4L
  model <- adlaplace::density_data(
    gamma_map = Matrix::sparseMatrix(
      i = seq_len(nr),
      j = seq_len(nr),
      x = 1,
      dims = c(6L, nr)
    ),
    theta_map = c(1L, 1L),
    ad_kind = "random",
    density = "random_diagonal",
    precision = NULL
  )
  config <- list(
    beta = numeric(0),
    gamma = rep(0, 6),
    theta = 0.1,
    transform_theta = FALSE
  )
  expect_error(
    adlaplace::ad_pack_ptr(model, config),
    "precision is required"
  )
})

test_that("beta_map and gamma_map default to diagonal maps from X and A", {
  A <- Matrix::sparseMatrix(
    i = c(0L, 1L),
    j = c(0L, 1L),
    x = 1,
    dims = c(2L, 3L),
    index1 = FALSE,
    repr = "C"
  )
  X <- matrix(1, 2, 1)
  model <- adlaplace::density_data(y = c(1, 2), A = A, X = X, theta_map = Matrix::Diagonal(1L))
  expect_equal(sizes(model)$n_beta, 1L)
  expect_equal(sizes(model)$n_gamma, 3L)
  expect_equal(ncol(model@beta_map), 1L)
  expect_equal(ncol(model@gamma_map), 3L)
})

test_that("theta_map defaults to empty map when omitted", {
  model <- adlaplace::density_data(
    y = 1:4,
    A = matrix(0, 4, 4),
    X = matrix(0, 4, 2)
  )
  expect_equal(sizes(model)$n_theta, 0L)
  expect_equal(nrow(model@theta_map), 0L)
  expect_equal(ncol(model@theta_map), 0L)
})

test_that("density_data does not require theta inference", {
  A <- Matrix::sparseMatrix(
    i = 0L,
    j = 0L,
    x = 1,
    dims = c(1L, 1L),
    index1 = FALSE,
    repr = "C"
  )
  expect_silent(suppressWarnings(adlaplace::density_data(y = 1, A = A, X = matrix(1, 1, 1))))
})

test_that("elgm columns set the design row count", {
  n_y <- 4L
  n_cell <- 7L
  y <- rpois(n_y, 1)
  A <- Matrix::sparseMatrix(
    i = seq_len(n_cell),
    j = rep(1L, n_cell),
    x = 1,
    dims = c(n_cell, 1L)
  )
  X <- Matrix::Matrix(1, n_cell, 1L)
  elgm <- Matrix::sparseMatrix(
    i = c(1L, 1L, 2L, 3L, 4L, 4L),
    j = c(1L, 2L, 3L, 4L, 5L, 7L),
    x = c(0.4, 0.6, 1, 1, 0.2, 0.8),
    dims = c(n_y, n_cell)
  )
  model <- adlaplace::density_data(
    y = y,
    A = A,
    X = X,
    elgm_matrix = elgm,
    density = "poisson_aggregate",
    ad_kind = "observations"
  )
  expect_equal(length(model@y), n_y)
  expect_equal(dim(model@elgm_matrix), c(n_y, n_cell))
  expect_equal(ncol(model@ATp), n_cell)
  expect_equal(ncol(model@XTp), n_cell)

  # Observation-level ELGM (strata) still aligns the design with y.
  A_obs <- Matrix::Diagonal(n_y)
  X_obs <- Matrix::Matrix(1, n_y, 1L)
  elgm_strata <- Matrix::sparseMatrix(
    i = seq_len(n_y),
    j = c(1L, 1L, 2L, 2L),
    dims = c(n_y, 2L)
  )
  expect_silent(adlaplace::density_data(
    y = y,
    A = A_obs,
    X = X_obs,
    elgm_matrix = elgm_strata
  ))

  expect_error(
    adlaplace::density_data(y = y, A = A, X = X),
    "nrow\\(A\\) \\(7\\) must match length\\(y\\) \\(4\\)"
  )
  expect_error(
    adlaplace::density_data(
      y = y[-1L],
      A = A,
      X = X,
      elgm_matrix = elgm
    ),
    "length\\(y\\) \\(3\\) must match nrow\\(elgm_matrix\\) \\(4\\) or ncol\\(elgm_matrix\\) \\(7\\)"
  )
  expect_error(
    adlaplace::density_data(
      y = y,
      A = A[1:3, , drop = FALSE],
      X = X[1:3, , drop = FALSE],
      elgm_matrix = elgm
    ),
    "must match ncol\\(elgm_matrix\\) \\(7\\) or length\\(y\\) \\(4\\)"
  )
})

test_that("aggregated ELGM indexes districts by column", {
  n_cell <- 5L
  n_dist <- 2L
  elgm <- Matrix::sparseMatrix(
    i = c(1L, 2L, 3L, 3L, 4L, 5L),
    j = c(1L, 1L, 1L, 2L, 2L, 2L),
    x = log(c(10, 20, 15, 15, 40, 50)),
    dims = c(n_cell, n_dist)
  )
  model <- adlaplace::density_data(
    y = c(4, 7),
    A = Matrix::Diagonal(n_cell),
    X = Matrix::Matrix(0, n_cell, 0L),
    elgm_matrix = elgm,
    density = "poisson_aggregate",
    ad_kind = "observations"
  )
  expect_equal(length(model@y), n_dist)
  expect_equal(dim(model@elgm_matrix), c(n_cell, n_dist))
  expect_equal(ncol(model@ATp), n_cell)

  expect_error(
    adlaplace::density_data(
      y = c(4, 7, 1),
      A = Matrix::Diagonal(n_cell),
      X = Matrix::Matrix(0, n_cell, 0L),
      elgm_matrix = elgm
    ),
    "nrow\\(elgm_matrix\\) \\(5\\) or ncol\\(elgm_matrix\\) \\(2\\)"
  )
  expect_error(
    adlaplace::density_data(
      y = c(4, 7),
      A = Matrix::Diagonal(n_dist),
      X = Matrix::Matrix(0, n_dist, 0L),
      elgm_matrix = elgm
    ),
    "must match nrow\\(elgm_matrix\\) \\(5\\)"
  )
})

test_that("validate_config_layout allows missing theta when theta_map empty", {
  A <- Matrix::sparseMatrix(
    i = c(0L, 1L),
    j = c(0L, 1L),
    x = 1,
    dims = c(2L, 2L),
    index1 = FALSE,
    repr = "C"
  )
  model <- adlaplace::density_data(y = 1:2, A = A, X = matrix(1, 2, 2))
  config <- list(
    beta = rep(0, 2),
    gamma = rep(0, 2)
  )
  expect_silent(adlaplace:::validate_config_layout(model, config))
})
