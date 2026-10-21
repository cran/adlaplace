test_that("hessian_map chol_inner_list includes half_H_inv and H_inv patterns", {
  set.seed(2)
  Nobs <- 20L
  X <- Matrix::Matrix(cbind(1, rbinom(Nobs, 1, prob = 0.5)))
  Amat <- Matrix::sparseMatrix(
    i = seq_len(Nobs),
    j = sample(4L, Nobs, replace = TRUE),
    x = 1
  )
  config <- list(
    beta = rep(0, 2),
    theta = -1,
    transform_theta = TRUE,
    gamma = rep(0, ncol(Amat)),
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 4L),
    num_threads = 1L,
    verbose = FALSE
  )
  model <- test_ad_data(
    y = rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config
  )
  random_shard <- test_random_shard(
    data = model,
    config = config,
    gamma_ids = seq.int(0L, length.out = ncol(Amat)),
    theta_id = 0L,
    Q = rep(1, ncol(Amat))
  )
  ad_ptr <- do.call(c, list(
    adlaplace::ad_pack_ptr(as_shard(model, "observations", "nbinom_obs"), config),
    adlaplace::ad_pack_ptr(random_shard, config),
    adlaplace::ad_pack_ptr(as_shard(model, "parameters", "nbinom_extra"), config)
  ))
  af <- adlaplace::ad_pack(ad_ptr, num_threads = 1L)
  cil <- af@chol_inner_list
  expect_true(!is.null(cil$half_H_inv))
  expect_true(!is.null(cil$H_inv))
  expect_equal(nrow(cil$half_H_inv), ncol(Amat))
  expect_equal(nrow(cil$H_inv), ncol(Amat))
  H_pat <- cil$H_inv
  col_idx <- rep.int(seq_len(ncol(H_pat)) - 1L, diff(H_pat@p))
  expect_true(all(H_pat@i <= col_idx))
})

test_that("inner_opt deriv half_H_inv and H_inv match R reference", {
  set.seed(41)
  Nobs <- 30L
  X <- Matrix::Matrix(cbind(1, rbinom(Nobs, 1, prob = 0.5)))
  Amat <- Matrix::sparseMatrix(
    i = seq_len(Nobs),
    j = sample(4L, Nobs, replace = TRUE),
    x = 1
  )
  config <- list(
    beta = rep(0, 2),
    theta = -1,
    transform_theta = TRUE,
    gamma = rep(0, ncol(Amat)),
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 8L),
    num_threads = 1L,
    verbose = FALSE
  )
  model <- test_ad_data(
    y = rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config
  )
  random_shard <- test_random_shard(
    data = model,
    config = config,
    gamma_ids = seq.int(0L, length.out = ncol(Amat)),
    theta_id = 0L,
    Q = rep(1, ncol(Amat))
  )
  ad_ptr <- do.call(c, list(
    adlaplace::ad_pack_ptr(as_shard(model, "observations", "nbinom_obs"), config),
    adlaplace::ad_pack_ptr(random_shard, config),
    adlaplace::ad_pack_ptr(as_shard(model, "parameters", "nbinom_extra"), config)
  ))
  af <- adlaplace::ad_pack(ad_ptr, num_threads = 1L)
  inner_deriv <- adlaplace::inner_opt(
    c(config$beta, config$theta), config$gamma,
    ad_pack = af,
    control = list(maxit = 3L, report.level = 0, report.freq = 0),
    deriv = TRUE
  )
  ci <- inner_deriv$hessian$chol_inner
  expect_true(!is.null(inner_deriv$hessian$half_H_inv))
  expect_true(!is.null(inner_deriv$hessian$H_inv))
  H_inv <- inner_deriv$hessian$H_inv
  expect_true(inherits(H_inv, "dsCMatrix"))
  expect_equal(H_inv@uplo, "U")

  half_ref <- adlaplace:::half_H_inv_from_ldl(adlaplace:::as_ldl_list(ci))
  expect_true(max(abs(inner_deriv$hessian$half_H_inv - half_ref)) < 1e-8)

  H_inv_ref <- Matrix::tcrossprod(half_ref)
  expect_true(max(abs(H_inv - H_inv_ref)) < 1e-8)

  H_inner <- inner_deriv$hessian$inner
  expect_true(max(abs(as.matrix(H_inv) - as.matrix(Matrix::solve(H_inner)))) < 1e-6)

  v <- stats::rnorm(ncol(Amat))
  expect_true(max(abs(as.vector(H_inv %*% v) - as.vector(Matrix::solve(H_inner, v)))) < 1e-6)
})

test_that("H_inv dsCMatrix with coupled random-effect groups (vignette-like)", {
  set.seed(0)
  Nobs <- 200L
  Nrandom1 <- 10L
  Nrandom2 <- 25L
  X <- Matrix::Matrix(cbind(1, stats::rbinom(Nobs, 1, 0.5)))
  Amat <- cbind(
    Matrix::sparseMatrix(i = seq_len(Nobs), j = sample(Nrandom1, Nobs, replace = TRUE)),
    Matrix::sparseMatrix(i = seq_len(Nobs), j = sample(Nrandom2, Nobs, replace = TRUE))
  )
  config <- list(
    beta = rep(0, ncol(X)),
    theta = c(-1, -1, -1),
    transform_theta = TRUE,
    gamma = rep(0, ncol(Amat)),
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 40L),
    num_threads = 1L,
    verbose = FALSE,
    package = "adlaplace"
  )
  n_beta <- length(config$beta)
  n_gamma <- length(config$gamma)
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
    gamma_ids = seq.int(0L, length.out = ncol(Amat)),
    theta_id = 0L,
    Q = rep(1, ncol(Amat))
  )
  ad_ptr <- do.call(c, list(
    adlaplace::ad_pack_ptr(as_shard(model, "observations", "nbinom_obs"), config),
    adlaplace::ad_pack_ptr(random_shard, config),
    adlaplace::ad_pack_ptr(as_shard(model, "parameters", "nbinom_extra"), config)
  ))
  af <- adlaplace::ad_pack(ad_ptr, num_threads = 1L)
  ll <- adlaplace::log_lik_laplace(
    x = c(config$beta, config$theta),
    config = list(verbose = FALSE),
    gamma = config$gamma,
    ad_pack = af,
    control = list(maxit = 5L, report.level = 0, report.freq = 0),
    deriv = TRUE
  )
  H_inv <- ll$hessian$H_inv
  H_inner <- ll$hessian$inner
  expect_true(inherits(H_inv, "dsCMatrix"))
  expect_equal(H_inv@uplo, "U")
  expect_true(max(abs(as.matrix(H_inv) - as.matrix(Matrix::solve(H_inner)))) < 1e-5)

  v <- stats::rnorm(n_gamma)
  expect_true(max(abs(as.vector(H_inv %*% v) - as.vector(Matrix::solve(H_inner, v)))) < 1e-5)

  seq_gamma1 <- seq.int(n_beta + 1L, length.out = n_gamma)
  dU_check <- -Matrix::solve(
    H_inner,
    ll$hessian$outer[seq_gamma1, -seq_gamma1, drop = FALSE]
  )
  expect_true(max(abs(as.matrix(dU_check - ll$gradient$outer$dU))) < 1e-5)
})
