test_that("fill_config_from_info seeds beta/theta from info", {
  skip_if_not_installed("mgcv")
  dat <- quiet_gamsim(6, n = 80, scale = 0.2, dist = "poisson", verbose = FALSE)
  md <- adlaplace::model_data(
    adlaplace::nbinom(y, lower = 1e-9) ~ x1 + adlaplace::iid(fac, init = 0.25),
    data = dat,
    verbose = FALSE
  )
  info <- md$term_data$info
  filled <- adlaplace:::fill_config_from_info(NULL, info)

  expect_equal(filled$beta, as.numeric(info$beta$init))
  expect_equal(
    filled$theta,
    adlaplace::apply_theta_log(info$theta, cols = "init")$init
  )
  expect_equal(filled$gamma, rep(0, nrow(info$gamma)))
  expect_equal(as.logical(filled$transform_theta), as.logical(info$theta$log))
})

test_that("fill_config_from_info keeps explicit config beta/theta/gamma", {
  skip_if_not_installed("mgcv")
  dat <- quiet_gamsim(6, n = 40, scale = 0.2, dist = "poisson", verbose = FALSE)
  md <- adlaplace::model_data(
    adlaplace::nbinom(y, lower = 1e-9) ~ x1 + adlaplace::iid(fac, init = 0.25),
    data = dat,
    verbose = FALSE
  )
  info <- md$term_data$info
  beta_override <- info$beta$init + 1
  theta_override <- rep(-1, nrow(info$theta))
  gamma_override <- rep(0.3, nrow(info$gamma))
  filled <- adlaplace:::fill_config_from_info(
    list(
      beta = beta_override,
      theta = theta_override,
      gamma = gamma_override,
      transform_theta = FALSE
    ),
    info
  )

  expect_equal(filled$beta, beta_override)
  expect_equal(filled$theta, theta_override)
  expect_equal(filled$gamma, gamma_override)
  expect_equal(filled$transform_theta, rep(FALSE, nrow(info$theta)))
})

test_that("ad_pack_ptr with info matches explicit config", {
  skip_if_not_installed("mgcv")
  dat <- quiet_gamsim(6, n = 80, scale = 0.2, dist = "poisson", verbose = FALSE)
  md <- adlaplace::model_data(
    adlaplace::nbinom(y, lower = 1e-9) ~ x1 + adlaplace::iid(fac, init = 0.25),
    data = dat,
    verbose = FALSE
  )
  info <- md$term_data$info
  config <- list(
    beta = as.numeric(info$beta$init),
    theta = adlaplace::apply_theta_log(info$theta, cols = "init")$init,
    gamma = rep(0, nrow(info$gamma)),
    transform_theta = TRUE,
    verbose = FALSE
  )

  obs <- md$observations[[1L]]
  ptr_info <- adlaplace::ad_pack_ptr(data = obs, info = info)
  ptr_cfg <- adlaplace::ad_pack_ptr(data = obs, config = config)

  af_info <- adlaplace::ad_pack(ptr_info, num_threads = 1L)
  af_cfg <- adlaplace::ad_pack(ptr_cfg, num_threads = 1L)

  n_beta <- nrow(info$beta)
  n_gamma <- nrow(obs@gamma_map)
  n_theta <- nrow(info$theta)
  x <- c(
    config$beta,
    rep(0, n_gamma),
    config$theta
  )
  expect_equal(length(x), n_beta + n_gamma + n_theta)

  ld_info <- adlaplace::joint_log_dens(af_info, x, negative = FALSE)
  ld_cfg <- adlaplace::joint_log_dens(af_cfg, x, negative = FALSE)
  expect_equal(ld_info, ld_cfg)
  expect_true(is.finite(ld_info))
})
