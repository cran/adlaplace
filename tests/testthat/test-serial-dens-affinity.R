test_that("joint_log_dens/grad/hessian reject multi-thread ad_pack handles", {
  skip_if_not(adlaplace:::has_openmp(), "OpenMP not available in this build")
  set.seed(21)
  Nobs <- 30L
  X <- Matrix::Matrix(cbind(1, stats::rbinom(Nobs, 1, 0.5)))
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
    verbose = FALSE
  )
  model <- test_ad_data(
    y = stats::rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config
  )
  ad_ptr <- do.call(c, list(
    adlaplace::ad_pack_ptr(as_shard(model, "observations", "nbinom_obs"), config),
    adlaplace::ad_pack_ptr(as_shard(model, "parameters", "nbinom_extra"), config)
  ))
  plain <- adlaplace::clone_ad_pack_ptr(ad_ptr)
  af <- adlaplace::ad_pack(ad_ptr, num_threads = 2L)
  x <- c(config$beta, config$gamma, config$theta)

  expect_error(
    adlaplace::joint_log_dens(af, x, negative = FALSE),
    "serial debug APIs|num_threads"
  )
  expect_error(
    adlaplace::grad(af, x, negative = FALSE),
    "serial debug APIs|num_threads"
  )
  expect_error(
    adlaplace::hessian(af, x, negative = FALSE),
    "serial debug APIs|num_threads"
  )

  expect_true(is.finite(adlaplace::joint_log_dens(plain, x, negative = FALSE)))
  expect_true(all(is.finite(adlaplace::grad(plain, x, negative = FALSE))))

  dens_clone <- adlaplace::clone_ad_pack_ptr(af@ptr)
  expect_false(adlaplace:::get_owner_thread_assigned(dens_clone, 0L))
  expect_true(is.na(adlaplace:::get_configured_num_threads(dens_clone)))
  expect_true(is.finite(
    adlaplace::joint_log_dens(dens_clone, x, negative = FALSE)
  ))
})

test_that("joint_log_dens still works on ad_pack with num_threads = 1", {
  set.seed(22)
  Nobs <- 20L
  X <- Matrix::Matrix(cbind(1, stats::rbinom(Nobs, 1, 0.5)))
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
    y = stats::rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config
  )
  ad_ptr <- do.call(c, list(
    adlaplace::ad_pack_ptr(as_shard(model, "observations", "nbinom_obs"), config),
    adlaplace::ad_pack_ptr(as_shard(model, "parameters", "nbinom_extra"), config)
  ))
  af <- adlaplace::ad_pack(ad_ptr, num_threads = 1L)
  x <- c(config$beta, config$gamma, config$theta)
  expect_true(is.finite(adlaplace::joint_log_dens(af, x, negative = FALSE)))
})
