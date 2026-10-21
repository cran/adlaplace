build_fun_obj_test_model <- function(num_threads) {
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
  # Plain dens handle before multi-thread ad_pack() mutates affinity on ad_ptr.
  dens_ptr <- adlaplace::clone_ad_pack_ptr(ad_ptr)
  ad_pack <- adlaplace::ad_pack(ad_ptr, num_threads = num_threads)
  gamma <- stats::rnorm(n_gamma, sd = 0.1)
  list(
    ad_pack = ad_pack,
    dens_ptr = dens_ptr,
    config = config,
    n_beta = n_beta,
    n_gamma = n_gamma,
    parameters = c(config$beta, config$theta),
    gamma = gamma,
    x_full = c(config$beta, gamma, config$theta)
  )
}

expect_fun_obj_parity <- function(m, inner) {
  fo <- adlaplace::fun_obj_fdfh(
    m$ad_pack,
    m$parameters,
    m$gamma,
    inner = inner
  )
  expect_true(is.finite(fo$f))
  expect_true(all(is.finite(fo$grad)))

  expect_equal(
    fo$f,
    adlaplace::joint_log_dens(m$dens_ptr, m$x_full, negative = TRUE),
    tolerance = 1e-8
  )

  grad_ref <- adlaplace::grad(
    m$dens_ptr,
    m$x_full,
    inner = inner,
    negative = TRUE
  )
  if (inner) {
    gamma_idx <- (m$n_beta + 1L):(m$n_beta + m$n_gamma)
    grad_ref <- grad_ref[gamma_idx]
  }
  expect_equal(fo$grad, grad_ref, tolerance = 1e-8)

  hess_ref <- adlaplace::hessian(
    m$dens_ptr,
    m$x_full,
    inner = inner,
    negative = TRUE
  )
  if (inner) {
    gamma_idx <- (m$n_beta + 1L):(m$n_beta + m$n_gamma)
    hess_ref <- hess_ref[gamma_idx, gamma_idx]
  }
  expect_equal(as.matrix(fo$hessian), as.matrix(hess_ref), tolerance = 1e-8)
}

test_that("fun_obj_fdfh matches direct eval (serial, inner=TRUE)", {
  m <- build_fun_obj_test_model(num_threads = 1L)
  expect_fun_obj_parity(m, inner = TRUE)
})

test_that("fun_obj_fdfh matches direct eval (serial, inner=FALSE)", {
  m <- build_fun_obj_test_model(num_threads = 1L)
  expect_fun_obj_parity(m, inner = FALSE)
})

test_that("fun_obj_fdfh OpenMP region uses the requested team", {
  skip_if_not(adlaplace:::has_openmp(), "OpenMP not available in this build")
  m <- build_fun_obj_test_model(num_threads = 2L)
  lines <- capture.output(adlaplace::fun_obj_fdfh(
    m$ad_pack,
    m$parameters,
    m$gamma,
    inner = TRUE,
    verbose = TRUE
  ))
  # Windows keeps the process high-water mark, so an earlier test that used
  # more than 2 threads raises this region above the ad_pack request.
  threads_line <- lines[grepl("threads = ", lines, fixed = TRUE)]
  expect_true(length(threads_line) >= 1L)
  expected <- sub(".*threads = ([0-9]+).*", "\\1", threads_line[[1L]])
  expect_true(as.integer(expected) >= 2L)
  team <- lines[grepl("parallel_team=", lines, fixed = TRUE)]
  expect_true(length(team) >= 1L)
  expect_true(any(grepl(paste0("parallel_team=", expected), team, fixed = TRUE)))
})

test_that("fun_obj_fdfh matches direct eval (multi-thread, inner=TRUE)", {
  skip_if_not(adlaplace:::has_openmp(), "OpenMP not available in this build")
  m <- build_fun_obj_test_model(num_threads = 2L)
  expect_fun_obj_parity(m, inner = TRUE)
})

test_that("fun_obj_fdfh matches direct eval (multi-thread, inner=FALSE)", {
  skip_if_not(adlaplace:::has_openmp(), "OpenMP not available in this build")
  m <- build_fun_obj_test_model(num_threads = 2L)
  expect_fun_obj_parity(m, inner = FALSE)
})
