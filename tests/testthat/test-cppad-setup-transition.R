# Regression: tape build (serial) -> ad_pack(N>1) -> inner_opt(deriv=FALSE)
# must not hit CppAD parallel_setup thread_num() >= num_threads on macOS.

build_transition_fixture <- function(num_threads = 2L,
                                       num_shards = 20L,
                                       include_random = TRUE) {
  set.seed(42)
  Nobs <- 80L
  X <- Matrix::Matrix(cbind(1, stats::rbinom(Nobs, 1, 0.5)))
  Amat <- Matrix::sparseMatrix(
    i = seq_len(Nobs),
    j = sample(4L, Nobs, replace = TRUE)
  )
  config <- list(
    beta = rep(0, ncol(X)),
    theta = c(-1, -1),
    transform_theta = TRUE,
    gamma = rep(0, ncol(Amat)),
    obs_groups = adlaplace::obs_groups(Amat, num_shards = num_shards),
    verbose = FALSE,
    package = "adlaplace"
  )
  model <- test_ad_data(
    y = rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config,
    theta_local_row = 1L
  )
  ptrs <- list(
    adlaplace::ad_pack_ptr(as_shard(model, "observations", "nbinom_obs"), config),
    adlaplace::ad_pack_ptr(as_shard(model, "parameters", "nbinom_extra"), config)
  )
  if (include_random) {
    random_shard <- test_random_shard(
      data = model,
      config = config,
      gamma_ids = seq.int(0L, length.out = ncol(Amat)),
      theta_id = 0L,
      Q = rep(1, ncol(Amat))
    )
    ptrs <- c(
      list(adlaplace::ad_pack_ptr(random_shard, config)),
      ptrs
    )
  }
  ad_pack <- adlaplace::ad_pack(do.call(c, ptrs), num_threads = num_threads)
  list(
    config = config,
    ad_pack = ad_pack,
    parameters = c(config$beta, config$theta),
    gamma = config$gamma
  )
}

expect_inner_opt_finite <- function(fx) {
  result <- adlaplace::inner_opt(
    parameters = fx$parameters,
    gamma = fx$gamma,
    ad_pack = fx$ad_pack,
    control = list(maxit = 5L, report.level = 0, report.freq = 0),
    deriv = FALSE
  )
  expect_true(is.finite(result$neg_log_lik))
  invisible(result)
}

test_that("tape build -> ad_pack(2) -> inner_opt deriv=FALSE (mixed shards)", {
  skip_if_not(adlaplace:::has_openmp(), "OpenMP not available in this build")
  fx <- build_transition_fixture(num_threads = 2L, include_random = TRUE)
  expect_inner_opt_finite(fx)
})

test_that("tape build -> ad_pack(2) -> inner_opt deriv=FALSE (obs + extra only)", {
  skip_if_not(adlaplace:::has_openmp(), "OpenMP not available in this build")
  fx <- build_transition_fixture(num_threads = 2L, include_random = FALSE)
  expect_inner_opt_finite(fx)
})

test_that("back-to-back inner_opt after tape build does not crash", {
  skip_if_not(adlaplace:::has_openmp(), "OpenMP not available in this build")
  fx <- build_transition_fixture(num_threads = 2L, include_random = TRUE)
  expect_inner_opt_finite(fx)
  expect_inner_opt_finite(fx)
})
