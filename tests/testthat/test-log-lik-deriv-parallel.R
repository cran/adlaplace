test_that("ad_pack assigns threads for multi-shard model", {
  skip_if_not(adlaplace:::has_openmp(), "OpenMP not available in this build")
  set.seed(0)
  Nobs <- 120L
  Nrandom1 <- 4L
  Nrandom2 <- 5L
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
    verbose = FALSE,
    package = "adlaplace"
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
    gamma_ids = seq.int(0L, length.out = ncol(Amat)),
    theta_id = 0L,
    Q = rep(1, ncol(Amat))
  )
  ad_ptr <- do.call(c, list(
    adlaplace::ad_pack_ptr(as_shard(model, "observations", "nbinom_obs"), config),
    adlaplace::ad_pack_ptr(random_shard, config),
    adlaplace::ad_pack_ptr(as_shard(model, "parameters", "nbinom_extra"), config)
  ))
  x <- c(config$beta, config$gamma, config$theta)
  dens_ptr <- adlaplace::clone_ad_pack_ptr(ad_ptr)
  expect_true(is.finite(adlaplace::joint_log_dens(dens_ptr, x, negative = FALSE)))
  ad_pack <- adlaplace::ad_pack(ad_ptr, num_threads = 2L, reorder_shards = "none")
  n <- adlaplace:::n_groups(ad_pack@ptr)
  owners <- vapply(seq_len(n) - 1L, function(g) {
    adlaplace:::get_thread_owner(ad_pack@ptr, g)
  }, integer(1))
  expect_equal(owners, (seq_len(n) - 1L) %% 2L)
  expect_error(
    adlaplace::joint_log_dens(ad_pack, x, negative = FALSE),
    "serial debug APIs|num_threads"
  )
})

test_that("log_lik_laplace deriv=TRUE with multi-thread ad_pack", {
  skip_if_not(adlaplace:::has_openmp(), "OpenMP not available in this build")
  set.seed(0)
  Nobs <- 120L
  Nrandom1 <- 4L
  Nrandom2 <- 5L
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
    verbose = FALSE,
    package = "adlaplace"
  )
  n_beta <- length(config$beta)
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
  ad_pack <- adlaplace::ad_pack(ad_ptr, num_threads = 2L)

  ll_deriv <- adlaplace::log_lik_laplace(
    x = c(config$beta, config$theta),
    config = list(verbose = FALSE),
    gamma = config$gamma,
    ad_pack = ad_pack,
    control = list(maxit = 6L, report.level = 0, report.freq = 0),
    deriv = TRUE
  )
  expect_true(is.finite(ll_deriv$log_lik))
  expect_equal(length(ll_deriv$deriv$d_neg_log_lik), n_beta + length(config$theta))
  expect_true(all(is.finite(ll_deriv$deriv$d_neg_log_lik)))
  expect_true(all(is.finite(ll_deriv$extra$trace3)))
})

test_that("log_lik_laplace deriv=TRUE with serial threads", {
  set.seed(0)
  Nobs <- 80L
  X <- Matrix::Matrix(cbind(1, stats::rbinom(Nobs, 1, 0.5)))
  Amat <- Matrix::sparseMatrix(i = seq_len(Nobs), j = sample(4L, Nobs, replace = TRUE))
  config <- list(
    beta = rep(0, ncol(X)),
    theta = c(-1, -1),
    transform_theta = TRUE,
    gamma = rep(0, ncol(Amat)),
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 16L),
    verbose = FALSE,
    package = "adlaplace"
  )
  n_beta <- length(config$beta)
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
  ad_pack <- adlaplace::ad_pack(ad_ptr, num_threads = 1L)

  ll_deriv <- adlaplace::log_lik_laplace(
    x = c(config$beta, config$theta),
    config = list(verbose = FALSE),
    gamma = config$gamma,
    ad_pack = ad_pack,
    control = list(maxit = 6L, report.level = 0, report.freq = 0),
    deriv = TRUE
  )
  expect_true(is.finite(ll_deriv$log_lik))
  expect_equal(length(ll_deriv$deriv$d_neg_log_lik), n_beta + length(config$theta))
  expect_true(all(is.finite(ll_deriv$deriv$d_neg_log_lik)))
})

test_that("ad_pack_ptr has no thread assignment until ad_pack", {
  skip_if_not(adlaplace:::has_openmp(), "OpenMP not available in this build")
  set.seed(1)
  Nobs <- 60L
  X <- Matrix::Matrix(cbind(1, stats::rbinom(Nobs, 1, 0.5)))
  Amat <- Matrix::sparseMatrix(i = seq_len(Nobs), j = sample(4L, Nobs, replace = TRUE))
  config10 <- list(
    beta = rep(0, ncol(X)),
    theta = -1,
    transform_theta = TRUE,
    gamma = rep(0, ncol(Amat)),
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 20L),
    verbose = FALSE
  )
  model <- test_ad_data(
    y = rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config10,
    theta_local_row = 0L
  )
  ptr_obs <- adlaplace::ad_pack_ptr(as_shard(model, "observations", "nbinom_obs"), config10)
  expect_false(adlaplace:::get_owner_thread_assigned(ptr_obs, 0L))

  ptr <- do.call(c, list(
    ptr_obs,
    adlaplace::ad_pack_ptr(as_shard(model, "parameters", "nbinom_extra"), config10)
  ))
  expect_false(adlaplace:::get_owner_thread_assigned(ptr, 0L))

  af <- adlaplace::ad_pack(ptr, num_threads = 2L, reorder_shards = "none")
  n <- adlaplace:::n_groups(af@ptr)
  owners <- vapply(seq_len(n) - 1L, function(g) {
    adlaplace:::get_thread_owner(af@ptr, g)
  }, integer(1))
  expect_equal(owners, (seq_len(n) - 1L) %% 2L)
  expect_true(all(vapply(seq_len(n) - 1L, function(g) {
    adlaplace:::get_owner_thread_assigned(af@ptr, g)
  }, logical(1))))
})

test_that("ad_pack(ptr, num_threads = 1L) assigns all shards to thread 0", {
  set.seed(2)
  Nobs <- 40L
  X <- Matrix::Matrix(cbind(1, stats::rbinom(Nobs, 1, 0.5)))
  Amat <- Matrix::sparseMatrix(i = seq_len(Nobs), j = sample(4L, Nobs, replace = TRUE))
  config1 <- list(
    beta = rep(0, ncol(X)),
    theta = -1,
    transform_theta = TRUE,
    gamma = rep(0, ncol(Amat)),
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 12L),
    verbose = FALSE
  )
  model <- test_ad_data(
    y = rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config1,
    theta_local_row = 0L
  )
  ptr <- adlaplace::ad_pack_ptr(as_shard(model, "observations", "nbinom_obs"), config1)
  af <- adlaplace::ad_pack(ptr, num_threads = 1L)
  owners <- vapply(seq_len(adlaplace:::n_groups(af@ptr)) - 1L, function(g) {
    adlaplace:::get_thread_owner(af@ptr, g)
  }, integer(1))
  expect_true(all(owners == 0L))

  ll <- adlaplace::log_lik_laplace(
    x = c(config1$beta, config1$theta),
    config = list(verbose = FALSE),
    gamma = config1$gamma,
    ad_pack = af,
    control = list(maxit = 5L, report.level = 0, report.freq = 0),
    deriv = TRUE
  )
  expect_true(is.finite(ll$log_lik))
})
