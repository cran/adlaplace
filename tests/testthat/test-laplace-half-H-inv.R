test_that("laplace_half_H_inv finds half_H_inv in hessian", {
  p <- 4L
  half <- Matrix::Diagonal(p, 0.5)
  laplace <- list(
    hessian = list(half_H_inv = half)
  )
  expect_equal(adlaplace::laplace_half_H_inv(laplace), half)
})

test_that("laplace_half_H_inv finds half_H_inv in nested extra (legacy)", {
  p <- 4L
  half <- Matrix::Diagonal(p, 0.5)
  laplace <- list(
    extra = list(
      gradient = list(inner = rep(0, p)),
      hessian = list(chol_inner = list(L1 = Matrix::Diagonal(p), D = rep(1, p))),
      extra = list(half_H_inv = half)
    )
  )
  expect_equal(adlaplace::laplace_half_H_inv(laplace), half)
})

test_that("laplace_half_H_inv finds half_H_inv at top of extra", {
  p <- 3L
  half <- Matrix::Diagonal(p, 0.25)
  laplace <- list(extra = list(half_H_inv = half))
  expect_equal(adlaplace::laplace_half_H_inv(laplace), half)
})

test_that("laplace_half_H_inv rebuilds from hessian chol_inner", {
  skip_if_not_installed("trustOptim")
  set.seed(0)
  p <- 3L
  H <- Matrix::forceSymmetric(
    matrix(c(2, 0.5, 0, 0.5, 3, 0.1, 0, 0.1, 4), p, p)
  )
  chol <- Matrix::Cholesky(H, LDL = TRUE, perm = TRUE)
  laplace <- list(
    hessian = list(chol_inner = chol)
  )
  half <- adlaplace::laplace_half_H_inv(laplace)
  expect_equal(nrow(half), p)
  expect_equal(ncol(half), p)
  expect_true(max(abs(Matrix::tcrossprod(half) - Matrix::solve(H))) < 1e-8)
})

test_that("laplace_half_H_inv works on log_lik_laplace deriv output", {
  set.seed(1)
  Nobs <- 40L
  Nrandom <- 3L
  X <- Matrix::Matrix(cbind(1, rbinom(Nobs, 1, 0.5)))
  Amat <- Matrix::sparseMatrix(
    i = seq_len(Nobs),
    j = sample(Nrandom, Nobs, replace = TRUE),
    x = 1
  )
  config <- list(
    beta = rep(0, ncol(X)),
    theta = c(-1, -1),
    transform_theta = TRUE,
    gamma = rep(0, Nrandom),
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 8L),
    verbose = FALSE
  )
  model <- test_ad_data(
    y = rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config,
    theta_local_row = length(config$theta) - 1L
  )
  random_shard <- test_random_shard(
    data = model,
    config = config,
    gamma_ids = seq.int(0L, length.out = Nrandom),
    theta_id = 0L,
    Q = rep(1, Nrandom)
  )
  ad_ptr <- do.call(c, list(
    adlaplace::ad_pack_ptr(as_shard(model, "observations", "nbinom_obs"), config),
    adlaplace::ad_pack_ptr(random_shard, config),
    adlaplace::ad_pack_ptr(as_shard(model, "parameters", "nbinom_extra"), config)
  ))
  ad_pack <- adlaplace::ad_pack(ad_ptr)
  laplace <- adlaplace::log_lik_laplace(
    x = c(config$beta, config$theta),
    config = list(verbose = FALSE),
    gamma = config$gamma,
    ad_pack = ad_pack,
    control = list(maxit = 5L, report.level = 0, report.freq = 0),
    deriv = TRUE
  )
  half <- adlaplace::laplace_half_H_inv(laplace)
  expect_equal(nrow(half), Nrandom)
  expect_true(!is.null(laplace$hessian$half_H_inv))
})
