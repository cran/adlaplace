test_that("rmvnldl sample mean matches MVN with LDL precision", {
  set.seed(2026)
  H <- matrix(c(4, 0.5, 0.5, 3), 2L, 2L)
  mu <- c(0.2, -0.4)
  chol_prec <- Matrix::Cholesky(
    Matrix::forceSymmetric(H, uplo = "L"),
    LDL = TRUE
  )
  draws <- rmvnldl(8000L, mean = mu, chol_prec = chol_prec)
  expect_equal(dim(draws), c(8000L, 2L))
  expect_equal(colMeans(draws), mu, tolerance = 0.05)
  target_cov <- solve(H)
  expect_equal(cov(draws), target_cov, tolerance = 0.08)
})

test_that("rmvnldl accepts list LDL from laplace output", {
  H <- diag(2, 3)
  chol_list <- list(
    L1 = Matrix::Diagonal(3, 1),
    D = diag(H),
    perm = 0:2,
    perm_inv = 0:2
  )
  set.seed(1)
  x <- rmvnldl(5L, mean = rep(0, 3), chol_prec = chol_list)
  expect_equal(dim(x), c(5L, 3L))
  expect_true(all(is.finite(x)))
})

test_that("rmvnldl with n = 1 returns a length-p vector", {
  H <- matrix(c(2, 0, 0, 3), 2L, 2L)
  chol_prec <- Matrix::Cholesky(
    Matrix::forceSymmetric(H, uplo = "L"),
    LDL = TRUE
  )
  set.seed(3)
  x <- rmvnldl(1L, mean = c(1, 2), chol_prec = chol_prec)
  expect_equal(length(x), 2L)
  expect_true(all(is.finite(x)))
})

test_that("rmvnldl propagates names(mean) to draws", {
  H <- diag(2, 3)
  chol_list <- list(
    L1 = Matrix::Diagonal(3, 1),
    D = diag(H),
    perm = 0:2,
    perm_inv = 0:2
  )
  mu <- c(a = 0.1, b = -0.2, c = 0.3)
  set.seed(4)
  mat <- rmvnldl(5L, mean = mu, chol_prec = chol_list)
  expect_equal(colnames(mat), names(mu))
  set.seed(5)
  vec <- rmvnldl(1L, mean = mu, chol_prec = chol_list)
  expect_equal(names(vec), names(mu))
})

test_that("rmvnldl fit= overrides mean and chol_prec", {
  H <- diag(2, 3)
  chol_list <- list(
    L1 = Matrix::Diagonal(3, 1),
    D = diag(H),
    perm = 0:2,
    perm_inv = 0:2
  )
  mu <- c(a = 0.1, b = -0.2, c = 0.3)
  fit <- list(
    gamma = mu,
    details = list(extra = list(hessian = list(chol_inner = chol_list)))
  )
  set.seed(6)
  from_fit <- rmvnldl(4L, fit = fit)
  set.seed(6)
  from_args <- rmvnldl(4L, mean = mu, chol_prec = chol_list)
  expect_equal(from_fit, from_args)
  expect_equal(colnames(from_fit), names(mu))
})
