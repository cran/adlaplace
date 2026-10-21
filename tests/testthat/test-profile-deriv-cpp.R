test_that("C++ profile deriv matches R log_lik_deriv oracle", {
  set.seed(17)
  Nobs <- 40L
  X <- Matrix::Matrix(cbind(1, rbinom(Nobs, 1, prob = 0.5)))
  Amat <- Matrix::sparseMatrix(
    i = seq_len(Nobs),
    j = sample(5L, Nobs, replace = TRUE),
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
  x <- c(config$beta, config$theta)
  ctrl <- list(maxit = 8L, report.level = 0, report.freq = 0)

  io <- adlaplace::inner_opt(
    x, config$gamma,
    ad_pack = af, control = ctrl, deriv = TRUE, return_hessians = TRUE
  )
  expect_true(!is.null(io$deriv))
  expect_true(!is.null(io$dU))
  expect_equal(length(io$deriv$d_neg_log_lik), length(x))

  # Rebuild hessian$trace3 for the R oracle (log_lik_laplace moves it to gradient).
  hess_pack <- io$hessian
  expect_true(!is.null(hess_pack$trace3))

  r_ref <- adlaplace:::log_lik_deriv(
    full_parameters = io$full_parameters,
    hessian_pack = hess_pack,
    grad = io$gradient$outer,
    ad_pack = af
  )

  expect_equal(
    as.numeric(io$deriv$d_neg_log_lik),
    as.numeric(r_ref$deriv$d_neg_log_lik),
    tolerance = 1e-8
  )
  expect_equal(
    as.numeric(io$deriv$d_det),
    as.numeric(r_ref$deriv$d_det),
    tolerance = 1e-8
  )
  expect_equal(
    as.numeric(io$deriv$grad_theta),
    as.numeric(r_ref$deriv$grad_theta),
    tolerance = 1e-8
  )
  expect_equal(
    as.numeric(io$deriv$grad_u),
    as.numeric(r_ref$deriv$grad_u),
    tolerance = 1e-8
  )
  expect_true(max(abs(as.matrix(io$dU) - as.matrix(r_ref$extra$dU))) < 1e-8)

  ll <- adlaplace::log_lik_laplace(
    x = x,
    config = list(verbose = FALSE),
    gamma = config$gamma,
    ad_pack = af,
    control = ctrl,
    deriv = TRUE
  )
  expect_equal(
    as.numeric(ll$deriv$d_neg_log_lik),
    as.numeric(r_ref$deriv$d_neg_log_lik),
    tolerance = 1e-8
  )
  expect_true(max(abs(as.matrix(ll$gradient$outer$dU) - as.matrix(r_ref$extra$dU))) < 1e-8)
  expect_null(ll$hessian$trace3)
  expect_equal(as.numeric(ll$gradient$outer$trace3), as.numeric(hess_pack$trace3))
})

test_that("slim outer_fg agrees with full log_lik_laplace and skips Hessians", {
  set.seed(19)
  Nobs <- 35L
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
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 6L),
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
  x <- c(config$beta, config$theta)
  ctrl <- list(maxit = 8L, report.level = 0, report.freq = 0)

  slim <- adlaplace::inner_opt(
    x, config$gamma,
    ad_pack = af, control = ctrl, deriv = TRUE, return_hessians = FALSE
  )
  expect_true(!is.null(slim$deriv$d_neg_log_lik))
  expect_null(slim$hessian$inner)
  expect_null(slim$hessian$outer)
  expect_null(slim$hessian$half_H_inv)
  expect_null(slim$hessian$H_inv)
  expect_null(slim$dU)

  full <- adlaplace::log_lik_laplace(
    x = x,
    config = list(verbose = FALSE),
    gamma = config$gamma,
    ad_pack = af,
    control = ctrl,
    deriv = TRUE,
    return_hessians = TRUE
  )
  expect_equal(slim$neg_log_lik, full$neg_log_lik, tolerance = 1e-10)
  expect_equal(
    as.numeric(slim$deriv$d_neg_log_lik),
    as.numeric(full$deriv$d_neg_log_lik),
    tolerance = 1e-10
  )
  expect_equal(
    as.numeric(slim$inner_opt$solution),
    as.numeric(full$inner_opt$solution),
    tolerance = 1e-10
  )

  cache <- new.env(parent = emptyenv())
  cache$gamma <- config$gamma
  fn_val <- adlaplace::outer_fn(
    x = x, config = config, cache = cache, ad_pack = af, control_inner = ctrl
  )
  gr_val <- adlaplace::outer_gr(
    x = x, config = config, cache = cache, ad_pack = af, control_inner = ctrl
  )
  expect_equal(fn_val, full$neg_log_lik, tolerance = 1e-10)
  expect_equal(as.numeric(gr_val), as.numeric(full$deriv$d_neg_log_lik), tolerance = 1e-10)
  expect_equal(cache$fg_evals, 1L)
  expect_false(exists("fg_result", envir = cache, inherits = FALSE))
  expect_true(is.null(cache$fg_result) || !inherits(cache$fg_result$hessian$inner, "Matrix"))
})
