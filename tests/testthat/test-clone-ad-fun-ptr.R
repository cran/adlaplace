test_that("clone_ad_pack_ptr matches joint_log_dens on source", {
  set.seed(11)
  Nobs <- 20L
  X <- Matrix::Matrix(cbind(1, rbinom(Nobs, 1, prob = 0.5)))
  Amat <- Matrix::sparseMatrix(
    i = seq_len(Nobs),
    j = sample(3L, Nobs, replace = TRUE),
    x = 1
  )
  config <- list(
    beta = rep(0, 2),
    theta = -1,
    transform_theta = TRUE,
    gamma = rep(0, ncol(Amat)),
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 4L),
    verbose = FALSE
  )
  model <- test_ad_data(
    y = rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config
  )
  ptr <- adlaplace::ad_pack_ptr(as_shard(model, "observations", "nbinom_obs"), config)
  copy <- adlaplace::clone_ad_pack_ptr(ptr)
  x <- c(config$beta, config$gamma, config$theta)
  expect_true(is(copy, "ad_pack_ptr"))
  expect_false(identical(ptr, copy))
  expect_equal(
    adlaplace::joint_log_dens(ptr, x, negative = FALSE),
    adlaplace::joint_log_dens(copy, x, negative = FALSE)
  )
})

test_that("clone survives after c() invalidates source shard handle", {
  set.seed(12)
  Nobs <- 15L
  X <- Matrix::Matrix(cbind(1, rbinom(Nobs, 1, prob = 0.5)))
  Amat <- Matrix::sparseMatrix(
    i = seq_len(Nobs),
    j = sample(2L, Nobs, replace = TRUE),
    x = 1
  )
  config <- list(
    beta = rep(0, 2),
    theta = -0.5,
    transform_theta = TRUE,
    gamma = rep(0, ncol(Amat)),
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 3L),
    verbose = FALSE
  )
  model <- test_ad_data(
    y = rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config
  )
  obs <- adlaplace::ad_pack_ptr(as_shard(model, "observations", "nbinom_obs"), config)
  extra <- adlaplace::ad_pack_ptr(as_shard(model, "parameters", "nbinom_extra"), config)
  obs_copy <- adlaplace::clone_ad_pack_ptr(obs)
  x <- c(config$beta, config$gamma, config$theta)
  combined <- c(obs, extra)
  expect_error(
    adlaplace::joint_log_dens(obs, x, negative = FALSE),
    "NULL|cleared|invalid",
    ignore.case = TRUE
  )
  expect_true(is.finite(
    adlaplace::joint_log_dens(obs_copy, x, negative = FALSE)
  ))
})

test_that("clone after ad_pack() preserves laplace eval", {
  set.seed(13)
  Nobs <- 25L
  X <- Matrix::Matrix(cbind(1, rbinom(Nobs, 1, prob = 0.5)))
  Amat <- Matrix::sparseMatrix(
    i = seq_len(Nobs),
    j = sample(3L, Nobs, replace = TRUE),
    x = 1
  )
  config <- list(
    beta = rep(0, 2),
    theta = -1,
    transform_theta = TRUE,
    gamma = rep(0, ncol(Amat)),
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 6L),
    verbose = FALSE
  )
  model <- test_ad_data(
    y = rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config
  )
  ad_ptr <- do.call(c, list(
    adlaplace::ad_pack_ptr(as_shard(model, "observations", "nbinom_obs"), config),
    adlaplace::ad_pack_ptr(as_shard(model, "parameters", "nbinom_extra"), config)
  ))
  af <- adlaplace::ad_pack(ad_ptr)
  copy <- adlaplace::clone_ad_pack_ptr(ad_ptr)
  x <- c(config$beta, config$gamma, config$theta)
  expect_equal(
    adlaplace::joint_log_dens(ad_ptr, x, negative = FALSE),
    adlaplace::joint_log_dens(copy, x, negative = FALSE)
  )
  inner_src <- adlaplace::inner_opt(
    c(config$beta, config$theta), config$gamma,
    ad_pack = af,
    control = list(maxit = 2L, report.level = 0, report.freq = 0),
    deriv = FALSE
  )
  af_copy <- adlaplace::ad_pack(copy)
  inner_copy <- adlaplace::inner_opt(
    c(config$beta, config$theta), config$gamma,
    ad_pack = af_copy,
    control = list(maxit = 2L, report.level = 0, report.freq = 0),
    deriv = FALSE
  )
  expect_equal(inner_copy$log_lik, inner_src$log_lik)
})
