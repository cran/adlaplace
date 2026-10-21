test_that("compact_tape dens/grad/hess/log_lik match full-domain tapes", {
  set.seed(42)
  Nobs <- 120L
  X <- Matrix::Matrix(cbind(1, rbinom(Nobs, 1, 0.4)))
  Amat <- Matrix::sparseMatrix(
    i = seq_len(Nobs),
    j = sample(12L, Nobs, replace = TRUE),
    x = 1
  )
  y <- rnbinom(Nobs, size = 5, mu = 2)
  og <- adlaplace::obs_groups(Amat, num_shards = 6L)

  build_ptr <- function(compact) {
    config <- list(
      beta = c(0.1, -0.2),
      theta = c(log(5), -1, -1),
      transform_theta = TRUE,
      gamma = rnorm(ncol(Amat), sd = 0.1),
      obs_groups = og,
      compact_tape = compact,
      num_threads = 1L,
      verbose = FALSE
    )
    model <- test_ad_data(
      y = y,
      A = Amat,
      X = X,
      config = config,
      theta_local_row = 0L
    )
    list(
      ptr = adlaplace::ad_pack_ptr(
        as_shard(model, "observations", "nbinom_obs"),
        config
      ),
      config = config
    )
  }

  compact <- build_ptr(TRUE)
  full <- build_ptr(FALSE)
  n_g <- adlaplace:::n_groups(compact$ptr)
  expect_equal(n_g, adlaplace:::n_groups(full$ptr))

  domains_c <- vapply(
    seq_len(n_g) - 1L,
    function(g) adlaplace:::get_tape_sizes(compact$ptr, g)$domain,
    integer(1)
  )
  domains_f <- vapply(
    seq_len(n_g) - 1L,
    function(g) adlaplace:::get_tape_sizes(full$ptr, g)$domain,
    integer(1)
  )
  expect_true(all(domains_f == domains_f[1L]))
  expect_true(max(domains_c) <= domains_f[1L])
  expect_true(mean(domains_c) < domains_f[1L])

  x <- c(compact$config$beta, compact$config$gamma, compact$config$theta)
  expect_equal(
    joint_log_dens(compact$ptr, x, negative = FALSE),
    joint_log_dens(full$ptr, x, negative = FALSE),
    tolerance = 1e-10
  )
  expect_equal(
    as.numeric(grad(compact$ptr, x, negative = FALSE)),
    as.numeric(grad(full$ptr, x, negative = FALSE)),
    tolerance = 1e-8
  )
  Hc <- as.matrix(hessian(compact$ptr, x, negative = FALSE))
  Hf <- as.matrix(hessian(full$ptr, x, negative = FALSE))
  expect_equal(Hc, Hf, tolerance = 1e-7)

  pack_c <- ad_pack(compact$ptr, num_threads = 1L)
  pack_f <- ad_pack(full$ptr, num_threads = 1L)
  params <- c(compact$config$beta, compact$config$theta)
  ll_c <- log_lik_laplace(
    params,
    gamma = compact$config$gamma,
    ad_pack = pack_c,
    control = list(report.level = -1L)
  )
  ll_f <- log_lik_laplace(
    params,
    gamma = full$config$gamma,
    ad_pack = pack_f,
    control = list(report.level = -1L)
  )
  expect_equal(ll_c$log_lik, ll_f$log_lik, tolerance = 1e-6)
})
