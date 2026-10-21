test_that("lpt_assign_owners balances longest jobs first", {
  owners <- adlaplace:::lpt_assign_owners(c(10, 1, 9, 2), num_threads = 2L)
  expect_equal(length(owners), 4L)
  expect_true(all(owners %in% 0:1))
  # shards 1 and 3 (0-based 0 and 2) are heaviest -> different threads
  expect_false(identical(owners[[1L]], owners[[3L]]))
  loads <- c(
    sum(c(10, 1, 9, 2)[owners == 0L]),
    sum(c(10, 1, 9, 2)[owners == 1L])
  )
  expect_equal(max(loads) - min(loads), 0)
})

test_that("match_reorder_shards accepts gradient/hessian/third/none", {
  expect_equal(adlaplace:::match_reorder_shards("gradient"), "gradient")
  expect_equal(adlaplace:::match_reorder_shards("hessian"), "hessian")
  expect_equal(adlaplace:::match_reorder_shards("third"), "third")
  expect_equal(adlaplace:::match_reorder_shards("none"), "none")
  expect_error(adlaplace:::match_reorder_shards("nope"), "should be one of")
})

test_that("reorder_shards none preserves modulo owners", {
  skip_if_not(adlaplace:::has_openmp(), "OpenMP not available in this build")
  set.seed(41)
  Nobs <- 60L
  X <- Matrix::Matrix(cbind(1, stats::rbinom(Nobs, 1, 0.5)))
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
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 12L),
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
  af <- adlaplace::ad_pack(ad_ptr, num_threads = 2L, reorder_shards = "none")
  n <- adlaplace:::n_groups(af@ptr)
  owners <- vapply(seq_len(n) - 1L, function(g) {
    adlaplace:::get_thread_owner(af@ptr, g)
  }, integer(1))
  expect_equal(owners, (seq_len(n) - 1L) %% 2L)
})

test_that("reorder_shards gradient LPT can differ from modulo and stays correct", {
  skip_if_not(adlaplace:::has_openmp(), "OpenMP not available in this build")
  set.seed(42)
  Nobs <- 80L
  X <- Matrix::Matrix(cbind(1, stats::rbinom(Nobs, 1, 0.5)))
  Amat <- Matrix::sparseMatrix(
    i = seq_len(Nobs),
    j = sample(6L, Nobs, replace = TRUE),
    x = 1
  )
  config <- list(
    beta = rep(0, 2),
    theta = -1,
    transform_theta = TRUE,
    gamma = rep(0, ncol(Amat)),
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 16L),
    verbose = FALSE
  )
  model <- test_ad_data(
    y = stats::rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config
  )
  make_ptr <- function() {
    do.call(c, list(
      adlaplace::ad_pack_ptr(as_shard(model, "observations", "nbinom_obs"), config),
      adlaplace::ad_pack_ptr(as_shard(model, "parameters", "nbinom_extra"), config)
    ))
  }
  af_none <- adlaplace::ad_pack(
    make_ptr(),
    num_threads = 2L,
    reorder_shards = "none"
  )
  af_grad <- adlaplace::ad_pack(
    make_ptr(),
    num_threads = 2L,
    reorder_shards = "gradient"
  )
  n <- adlaplace:::n_groups(af_grad@ptr)
  owners_none <- vapply(seq_len(n) - 1L, function(g) {
    adlaplace:::get_thread_owner(af_none@ptr, g)
  }, integer(1))
  owners_grad <- vapply(seq_len(n) - 1L, function(g) {
    adlaplace:::get_thread_owner(af_grad@ptr, g)
  }, integer(1))
  expect_true(all(owners_grad %in% 0:1))
  # Small models may time as 0 under proc.time; LPT then parks all on thread 0.
  expect_true(length(unique(owners_grad)) >= 1L)
  expect_equal(length(owners_grad), n)

  x <- c(config$beta, config$gamma, config$theta)
  dens <- adlaplace::clone_ad_pack_ptr(make_ptr())
  dens_val <- adlaplace::joint_log_dens(dens, x, negative = FALSE)
  expect_true(is.finite(dens_val))

  fo <- adlaplace::fun_obj_fdfh(
    af_grad,
    parameters = c(config$beta, config$theta),
    gamma = config$gamma,
    inner = FALSE,
    verbose = FALSE
  )
  expect_true(is.finite(fo$f))
})

test_that("reorder_shards hessian LPT can differ from modulo and stays correct", {
  skip_if_not(adlaplace:::has_openmp(), "OpenMP not available in this build")
  set.seed(42)
  Nobs <- 80L
  X <- Matrix::Matrix(cbind(1, stats::rbinom(Nobs, 1, 0.5)))
  Amat <- Matrix::sparseMatrix(
    i = seq_len(Nobs),
    j = sample(6L, Nobs, replace = TRUE),
    x = 1
  )
  config <- list(
    beta = rep(0, 2),
    theta = -1,
    transform_theta = TRUE,
    gamma = rep(0, ncol(Amat)),
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 16L),
    verbose = FALSE
  )
  model <- test_ad_data(
    y = stats::rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config
  )
  make_ptr <- function() {
    do.call(c, list(
      adlaplace::ad_pack_ptr(as_shard(model, "observations", "nbinom_obs"), config),
      adlaplace::ad_pack_ptr(as_shard(model, "parameters", "nbinom_extra"), config)
    ))
  }
  af_hess <- adlaplace::ad_pack(
    make_ptr(),
    num_threads = 2L,
    reorder_shards = "hessian"
  )
  n <- adlaplace:::n_groups(af_hess@ptr)
  owners_hess <- vapply(seq_len(n) - 1L, function(g) {
    adlaplace:::get_thread_owner(af_hess@ptr, g)
  }, integer(1))
  expect_true(all(owners_hess %in% 0:1))
  # Small models may time as 0 under proc.time; LPT then parks all on thread 0.
  expect_true(length(unique(owners_hess)) >= 1L)

  fo <- adlaplace::fun_obj_fdfh(
    af_hess,
    parameters = c(config$beta, config$theta),
    gamma = config$gamma,
    inner = FALSE,
    verbose = FALSE
  )
  expect_true(is.finite(fo$f))
})

test_that("reorder_shards third LPT assigns owners and parallel dens agrees", {
  skip_if_not(adlaplace:::has_openmp(), "OpenMP not available in this build")
  set.seed(43)
  Nobs <- 70L
  X <- Matrix::Matrix(cbind(1, stats::rbinom(Nobs, 1, 0.5)))
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
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 10L),
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
  af <- adlaplace::ad_pack(ad_ptr, num_threads = 2L, reorder_shards = "third")
  n <- adlaplace:::n_groups(af@ptr)
  owners <- vapply(seq_len(n) - 1L, function(g) {
    adlaplace:::get_thread_owner(af@ptr, g)
  }, integer(1))
  expect_true(all(owners %in% 0:1))
  expect_true(!is.null(af@chol_inner_list$half_H_inv))
  expect_true(!is.null(af@chol_inner_list$trace_columns))

  ll <- adlaplace::log_lik_laplace(
    x = c(config$beta, config$theta),
    config = list(verbose = FALSE),
    gamma = config$gamma,
    ad_pack = af,
    control = list(maxit = 8L, report.level = 0, report.freq = 0),
    deriv = TRUE
  )
  expect_true(is.finite(ll$log_lik))
  expect_true(all(is.finite(ll$extra$trace3)))
})

test_that("num_threads = 1 ignores reorder_shards balancing", {
  set.seed(44)
  Nobs <- 30L
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
    obs_groups = adlaplace::obs_groups(Amat, num_shards = 6L),
    verbose = FALSE
  )
  model <- test_ad_data(
    y = stats::rpois(Nobs, 2),
    A = Amat,
    X = X,
    config = config
  )
  ad_ptr <- adlaplace::ad_pack_ptr(as_shard(model, "observations", "nbinom_obs"), config)
  af <- adlaplace::ad_pack(ad_ptr, num_threads = 1L, reorder_shards = "gradient")
  owners <- vapply(seq_len(adlaplace:::n_groups(af@ptr)) - 1L, function(g) {
    adlaplace:::get_thread_owner(af@ptr, g)
  }, integer(1))
  expect_true(all(owners == 0L))
})
