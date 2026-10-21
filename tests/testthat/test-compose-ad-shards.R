test_that("c() combines shards for small GLMM", {
  set.seed(2)
  Nobs <- 40L
  Nrandom1 <- 3L
  X <- Matrix::Matrix(cbind(1, rbinom(Nobs, 1, prob = 0.5)))
  Amat <- Matrix::sparseMatrix(
    i = seq_len(Nobs),
    j = sample(Nrandom1, Nobs, replace = TRUE),
    x = 1
  )
  config <- list(
    beta = rep(0, 2),
    theta = -1,
    transform_theta = TRUE,
    gamma = rep(0, ncol(Amat)),
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 10L),
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
  expect_true(is(ad_ptr, "ad_pack_ptr"))
  expect_equal(
    adlaplace:::n_groups(ad_ptr),
    ncol(config$obs_groups) + 2L
  )
  x <- c(config$beta, config$gamma, config$theta)
  expect_true(is.finite(adlaplace::joint_log_dens(ad_ptr, x)))
})

test_that("ad_pack_ptr obs-only builds observation groups only", {
  set.seed(3)
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
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 5L),
    verbose = FALSE
  )
  model <- test_ad_data(
    y = rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config
  )
  ad_obs <- adlaplace::ad_pack_ptr(
    as_shard(model, "observations", "nbinom_obs"),
    config
  )

  expect_error(
    adlaplace::ad_pack_ptr(as_shard(model, "observations", ""), config),
    "data@density is required",
    fixed = TRUE
  )
  expect_equal(adlaplace:::n_groups(ad_obs), ncol(config$obs_groups))
  expect_true(is(ad_obs, "ad_pack_ptr"))
})
