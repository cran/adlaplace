test_that("format_parameters returns list() for missing info", {
  expect_identical(adlaplace::format_parameters(NULL), list())
  expect_identical(adlaplace::format_parameters(list()), list())
  expect_identical(
    adlaplace::format_parameters(list(beta = data.frame(x = 1))),
    list()
  )
})

test_that("format_parameters maps full_parameters onto info tables", {
  skip_if_not_installed("mgcv")
  dat <- quiet_gamsim(6, n = 80, scale = 0.2, dist = "poisson", verbose = FALSE)
  md <- adlaplace::model_data(
    adlaplace::nbinom(y, lower = 1e-9) ~
      x1 +
      adlaplace::iid(fac, init = 0.25),
    data = dat,
    verbose = FALSE
  )
  info <- md$term_data$info
  n_beta <- nrow(info$beta)
  n_gamma <- nrow(info$gamma)
  n_theta <- nrow(info$theta)
  fp <- c(
    info$beta$init,
    rep(0.1, n_gamma),
    adlaplace::apply_theta_log(info$theta, cols = "init")$init
  )

  out <- adlaplace::format_parameters(info, full_parameters = fp)
  expect_named(out, c("parameters", "gamma"))
  expect_equal(nrow(out$parameters), n_beta + n_theta)
  expect_equal(nrow(out$gamma), n_gamma)
  expect_equal(out$parameters$mle[seq_len(n_beta)], fp[seq_len(n_beta)])
  expect_equal(out$gamma$mode, fp[seq(n_beta + 1L, length.out = n_gamma)])

  theta_idx <- which(info$theta$log)
  if (length(theta_idx) > 0L) {
    expect_equal(
      out$parameters$mle[n_beta + theta_idx],
      exp(fp[seq(to = length(fp), length.out = n_theta)][theta_idx])
    )
  }
})

test_that("format_parameters matches parameters + gamma input", {
  skip_if_not_installed("mgcv")
  dat <- quiet_gamsim(6, n = 80, scale = 0.2, dist = "poisson", verbose = FALSE)
  md <- adlaplace::model_data(
    adlaplace::nbinom(y, lower = 1e-9) ~ x1 + adlaplace::iid(fac, init = 0.25),
    data = dat,
    verbose = FALSE
  )
  info <- md$term_data$info
  n_beta <- nrow(info$beta)
  n_gamma <- nrow(info$gamma)
  n_theta <- nrow(info$theta)
  fp <- c(
    info$beta$init,
    rep(0.2, n_gamma),
    adlaplace::apply_theta_log(info$theta, cols = "init")$init
  )
  outer <- c(fp[seq_len(n_beta)], fp[seq(to = length(fp), length.out = n_theta)])
  gamma <- fp[seq(n_beta + 1L, length.out = n_gamma)]

  out_full <- adlaplace::format_parameters(info, full_parameters = fp)
  out_split <- adlaplace::format_parameters(
    info,
    parameters = outer,
    gamma = gamma
  )
  expect_equal(out_split$parameters$mle, out_full$parameters$mle)
  expect_equal(out_split$gamma$mode, out_full$gamma$mode)
})

test_that("ad_pack from model_data stores info slot", {
  skip_if_not_installed("mgcv")
  dat <- quiet_gamsim(6, n = 80, scale = 0.2, dist = "poisson", verbose = FALSE)
  md <- adlaplace::model_data(
    adlaplace::nbinom(y, lower = 1e-9) ~ x1 + adlaplace::iid(fac, init = 0.25),
    data = dat,
    verbose = FALSE
  )
  af <- adlaplace::ad_pack(
    md,
    config = list(
      transform_theta = TRUE,
      num_shards = 4L,
      num_threads = 1L,
      verbose = FALSE
    )
  )
  expect_true("info" %in% methods::slotNames(af))
  expect_equal(af@info, md$term_data$info)
})

test_that("ad_pack from ptr has empty info", {
  skip_if_not_installed("mgcv")
  dat <- quiet_gamsim(6, n = 80, scale = 0.2, dist = "poisson", verbose = FALSE)
  md <- adlaplace::model_data(
    adlaplace::nbinom(y, lower = 1e-9) ~ x1 + adlaplace::iid(fac, init = 0.25),
    data = dat,
    verbose = FALSE
  )
  config <- list(
    beta = md$term_data$info$beta$init,
    theta = md$term_data$info$theta$init,
    gamma = rep(0, nrow(md$term_data$info$gamma)),
    transform_theta = TRUE,
    verbose = FALSE
  )
  config$theta <- adlaplace::apply_theta_log(md$term_data$info$theta, cols = "init")$init
  shards <- unname(c(md$observations, md$random, md$parameters))
  ptr <- do.call(c, lapply(shards, adlaplace::ad_pack_ptr, config = config))
  af <- adlaplace::ad_pack(ptr)
  expect_identical(af@info, list())
})
