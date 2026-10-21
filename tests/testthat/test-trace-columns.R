test_that("ad_pack chol_inner_list includes trace_columns", {
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
  n_gamma <- af@sizes["gamma"]
  n_groups <- adlaplace:::n_groups(af@ptr)

  expect_true(!is.null(cil$trace_columns))
  expect_equal(unname(dim(cil$trace_columns)), c(unname(n_gamma), n_groups))
  expect_s4_class(cil$trace_columns, "ngCMatrix")
})

test_that("trace_columns matches legacy log_lik_deriv column builder", {
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
  n_beta <- af@sizes["beta"]
  n_gamma <- af@sizes["gamma"]

  legacy <- adlaplace:::trace_columns_from_pattern(
    group_sparsity = af@group_sparsity,
    n_beta = n_beta,
    n_gamma = n_gamma,
    half_H_inv_pat = af@chol_inner_list$half_H_inv
  )
  expect_equal(af@chol_inner_list$trace_columns, legacy)
})

test_that("inner_opt trace3 matches standalone trace_hinv_t", {
  set.seed(7)
  Nobs <- 40L
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

  io <- adlaplace::inner_opt(
    parameters = c(config$beta, config$theta),
    gamma = config$gamma,
    ad_pack = af,
    control = list(maxit = 5L, report.level = 0, report.freq = 0),
    deriv = TRUE,
    verbose = FALSE
  )
  expect_true(!is.null(io$hessian$trace3))
  expect_true(all(is.finite(io$hessian$trace3)))

  legacy <- adlaplace::trace_hinv_t(
    ad_pack = af,
    x = io$full_parameters,
    LinvPt = io$hessian$half_H_inv,
    LinvPtColumns = af@chol_inner_list$trace_columns,
    verbose = FALSE
  )
  expect_equal(io$hessian$trace3, legacy, tolerance = 1e-10)
})

test_that("log_lik_laplace deriv uses trace from inner_opt", {
  set.seed(0)
  Nobs <- 80L
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
  x <- c(config$beta, config$gamma, config$theta)

  ll <- adlaplace::log_lik_laplace(
    x = c(config$beta, config$theta),
    config = config,
    gamma = config$gamma,
    ad_pack = af,
    control = list(maxit = 5L, report.level = 0, report.freq = 0),
    deriv = TRUE
  )
  expect_true(is.finite(ll$log_lik))
  expect_equal(length(ll$deriv$d_neg_log_lik), length(config$beta) + length(config$theta))
  expect_true(is.finite(sum(ll$deriv$d_log_lik)))
})
