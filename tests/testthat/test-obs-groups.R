test_that("obs_groups ELGM product covers districts without a cartesian join", {
  set.seed(1)
  n_cell <- 80L
  n_basis <- 12L
  n_dist <- 30L
  A <- Matrix::sparseMatrix(
    i = rep(seq_len(n_cell), each = 4L),
    j = sample.int(n_basis, n_cell * 4L, replace = TRUE),
    x = 1,
    dims = c(n_cell, n_basis)
  )
  district <- sample.int(n_dist, n_cell, replace = TRUE)
  extra <- sample.int(n_cell, 15L)
  E <- Matrix::sparseMatrix(
    i = c(seq_len(n_cell), extra),
    j = c(district, (district[extra] %% n_dist) + 1L),
    x = 1,
    dims = c(n_cell, n_dist)
  )
  G <- adlaplace::obs_groups(A, elgm_matrix = E, num_shards = 8L)
  expect_s4_class(G, "ngCMatrix")
  expect_equal(nrow(G), n_dist)
  expect_lte(ncol(G), 8L)
  expect_true(all(Matrix::rowSums(G) == 1))

  A5 <- Matrix::sparseMatrix(
    i = c(1L, 2L, 3L, 4L, 5L),
    j = c(1L, 1L, 2L, 3L, 3L),
    dims = c(5L, 3L)
  )
  E5 <- Matrix::sparseMatrix(
    i = c(1L, 2L, 3L, 3L, 4L, 5L),
    j = c(1L, 1L, 1L, 2L, 2L, 2L),
    dims = c(5L, 2L)
  )
  G5 <- adlaplace::obs_groups(A5, elgm_matrix = E5, num_shards = 2L)
  expect_equal(nrow(G5), 2L)
  expect_lte(ncol(G5), 2L)
  expect_true(all(Matrix::rowSums(G5) == 1))
  expect_error(
    adlaplace::obs_groups(A5, elgm_matrix = Matrix::t(E5), num_shards = 2L),
    "nrow\\(elgm_matrix\\)"
  )
})

test_that("config$obs_groups accepts diag, Diagonal, and dense 0/1", {
  set.seed(2)
  n <- 6L
  X <- Matrix::Matrix(1, n, 1L)
  A <- Matrix::Diagonal(n)
  beta <- 0.2
  gamma <- rnorm(n, sd = 0.1)
  eta <- as.numeric(X %*% beta + as.numeric(gamma))
  y <- rpois(n, exp(eta))
  model <- adlaplace::density_data(
    y = y,
    A = A,
    X = X,
    density = "poisson_obs",
    ad_kind = "observations"
  )
  x <- c(beta, gamma)
  forms <- list(
    diag = diag(n),
    Diagonal = Matrix::Diagonal(n),
    dense = matrix(as.numeric(diag(n)), n, n),
    dgC = methods::as(
      methods::as(Matrix::Diagonal(n), "generalMatrix"),
      "CsparseMatrix"
    ),
    together = matrix(1, n, 1L)
  )
  lls <- vapply(forms, function(og) {
    ptr <- adlaplace::ad_pack_ptr(
      model,
      list(
        beta = beta,
        gamma = gamma,
        theta = numeric(0),
        obs_groups = og,
        verbose = FALSE
      )
    )
    adlaplace::joint_log_dens(ptr, x, negative = FALSE)
  }, numeric(1))
  expect_equal(max(lls) - min(lls), 0, tolerance = 1e-8)
  expect_equal(adlaplace::n_groups(
    adlaplace::ad_pack_ptr(
      model,
      list(
        beta = beta, gamma = gamma, theta = numeric(0),
        obs_groups = Matrix::Diagonal(n), verbose = FALSE
      )
    )
  ), n)

  bad <- diag(n)
  bad[1L, 1L] <- 2
  expect_error(
    adlaplace::ad_pack_ptr(
      model,
      list(beta = beta, gamma = gamma, obs_groups = bad, verbose = FALSE)
    ),
    "0 or 1"
  )
  bad[1L, 1L] <- -1
  expect_error(
    adlaplace::ad_pack_ptr(
      model,
      list(beta = beta, gamma = gamma, obs_groups = bad, verbose = FALSE)
    ),
    "0 or 1"
  )
})
