# CppAD team teardown: deriv=TRUE uses one CppadParallelScope in inner_opt (opt + trace).

cppad_teardown_fixture <- function(num_threads = 2L) {
  set.seed(42)
  Nobs <- 80L
  X <- Matrix::Matrix(cbind(1, stats::rbinom(Nobs, 1, 0.5)))
  Amat <- Matrix::sparseMatrix(i = seq_len(Nobs), j = sample(4L, Nobs, replace = TRUE))
  config <- list(
    beta = rep(0, ncol(X)),
    theta = c(-1, -1),
    transform_theta = TRUE,
    gamma = rep(0, ncol(Amat)),
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 20L),
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
  list(
    config = config,
    ad_pack = adlaplace::ad_pack(ad_ptr, num_threads = num_threads)
  )
}

test_that("log_lik_laplace deriv=TRUE runs trace inside inner_opt (team teardown)", {
  skip_if_not(adlaplace:::has_openmp(), "OpenMP not available in this build")
  fx <- cppad_teardown_fixture(num_threads = 2L)
  ll <- adlaplace::log_lik_laplace(
    x = c(fx$config$beta, fx$config$theta),
    config = list(verbose = FALSE),
    gamma = fx$config$gamma,
    ad_pack = fx$ad_pack,
    control = list(maxit = 5L, report.level = 0, report.freq = 0),
    deriv = TRUE
  )
  expect_true(is.finite(ll$log_lik))
  expect_true(all(is.finite(ll$deriv$d_neg_log_lik)))
  expect_true(all(is.finite(ll$extra$trace3)))
})

test_that("back-to-back log_lik_laplace deriv=TRUE reuses CppAD team teardown", {
  skip_if_not(adlaplace:::has_openmp(), "OpenMP not available in this build")
  fx <- cppad_teardown_fixture(num_threads = 2L)
  args <- list(
    x = c(fx$config$beta, fx$config$theta),
    config = list(verbose = FALSE),
    gamma = fx$config$gamma,
    ad_pack = fx$ad_pack,
    control = list(maxit = 4L, report.level = 0, report.freq = 0),
    deriv = TRUE
  )
  ll1 <- do.call(adlaplace::log_lik_laplace, args)
  ll2 <- do.call(adlaplace::log_lik_laplace, args)
  expect_true(is.finite(ll1$log_lik))
  expect_true(is.finite(ll2$log_lik))
  expect_true(all(is.finite(ll2$extra$trace3)))
})

test_that("serial log_lik_laplace deriv=TRUE still completes", {
  fx <- cppad_teardown_fixture(num_threads = 1L)
  ll <- adlaplace::log_lik_laplace(
    x = c(fx$config$beta, fx$config$theta),
    config = list(verbose = FALSE),
    gamma = fx$config$gamma,
    ad_pack = fx$ad_pack,
    control = list(maxit = 4L, report.level = 0, report.freq = 0),
    deriv = TRUE
  )
  expect_true(is.finite(ll$log_lik))
  expect_true(all(is.finite(ll$extra$trace3)))
})

test_that("repeated parallel log_lik_laplace deriv=TRUE survives thread_alloc teardown", {
  # Four OpenMP threads exceed CRAN's two-core cap (CPU/elapsed NOTE).
  skip_on_cran()
  skip_if_not(adlaplace:::has_openmp(), "OpenMP not available in this build")
  fx <- cppad_teardown_fixture(num_threads = 4L)
  args <- list(
    x = c(fx$config$beta, fx$config$theta),
    config = list(verbose = FALSE),
    gamma = fx$config$gamma,
    ad_pack = fx$ad_pack,
    control = list(maxit = 3L, report.level = 0, report.freq = 0),
    deriv = TRUE
  )
  for (i in seq_len(20L)) {
    ll <- suppressWarnings(
      do.call(adlaplace::log_lik_laplace, args),
      classes = "adlaplace_inner_status"
    )
    expect_true(is.finite(ll$log_lik), info = paste("iter", i))
    expect_true(all(is.finite(ll$deriv$d_neg_log_lik)), info = paste("iter", i))
    expect_true(all(is.finite(ll$extra$trace3)), info = paste("iter", i))
  }
})
