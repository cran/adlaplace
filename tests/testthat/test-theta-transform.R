test_that("term_data_setup defaults log to TRUE for all theta rows", {
  skip_if_not_installed("mgcv")
  dat <- quiet_gamsim(6, n = 80, scale = 0.2, dist = "poisson", verbose = FALSE)
  md <- adlaplace::model_data(
    adlaplace::nbinom(y, lower = 1e-9) ~
      x1 +
      adlaplace::iid(fac, init = 0.25),
    data = dat,
    verbose = FALSE
  )
  theta <- md$term_data$info$theta
  expect_true("log" %in% names(theta))
  expect_false(any(is.na(theta$log)))
  expect_true(all(theta$log))
})

test_that("apply_theta_log logs only rows with log TRUE", {
  theta_info <- data.frame(
    label = c("sd", "alpha"),
    init = c(0.5, 0.8),
    lower = c(1e-9, -Inf),
    upper = c(Inf, Inf),
    log = c(TRUE, FALSE),
    stringsAsFactors = FALSE
  )
  out <- adlaplace::apply_theta_log(
    theta_info,
    cols = c("init", "lower", "upper")
  )
  expect_equal(out$init, c(log(0.5), 0.8))
  expect_equal(out$lower[1], log(1e-9))
  expect_equal(out$lower[2], -Inf)
  expect_equal(out$upper, c(Inf, Inf))
})

test_that("apply_theta_log is a no-op when active is FALSE", {
  theta_info <- data.frame(
    label = "sd",
    init = 0.5,
    log = TRUE,
    stringsAsFactors = FALSE
  )
  out <- adlaplace::apply_theta_log(theta_info, active = FALSE)
  expect_equal(out$init, 0.5)
})

test_that("iid log=FALSE is selective in info$theta", {
  skip_if_not_installed("mgcv")
  dat <- quiet_gamsim(6, n = 60, scale = 0.2, dist = "poisson", verbose = FALSE)
  md <- adlaplace::model_data(
    adlaplace::nbinom(y, lower = 1e-9) ~
      x1 +
      adlaplace::iid(fac, init = 0.25, log = FALSE),
    data = dat,
    verbose = FALSE
  )
  theta <- md$term_data$info$theta
  iid_row <- grepl("iid", theta$label)
  expect_false(any(theta$log[iid_row]))
  expect_true(all(theta$log[!iid_row]))
})

test_that("fit$par_info has enriched columns and log-aware se", {
  skip_if_not_installed("mgcv")
  set.seed(11L)
  n <- 120L
  g <- sample(6L, n, replace = TRUE)
  dat <- data.frame(
    y = rpois(n, exp(0.2 + rnorm(6L)[g] * 0.2)),
    x = rnorm(n),
    g = factor(g)
  )
  fit <- suppressWarnings(
    adlaplace::adlaplace(
      adlaplace::nbinom(y, lower = 1e-9, init = 0.2) ~
        x + adlaplace::iid(g, init = 0.25),
      data = dat,
      config = list(num_shards = 4L),
      control = list(maxit = 60L),
      verbose = FALSE
    ),
    classes = "adlaplace_inner_status"
  )
  expect_equal(
    names(fit$par_info),
    c(
      "label", "mle", "se", "mle_internal", "se_internal",
      "init", "lower", "upper", "parscale", "log"
    )
  )
  expect_false("term" %in% names(fit$par_info))
  expect_false("model" %in% names(fit$par_info))
  idx_log <- which(fit$par_info$log %in% TRUE)
  idx_raw <- which(!fit$par_info$log %in% TRUE)
  expect_true(all(is.na(fit$par_info$se[idx_log])))
  expect_equal(
    fit$par_info$mle[idx_log],
    exp(fit$par_info$mle_internal[idx_log]),
    tolerance = 1e-10
  )
  if (length(idx_raw)) {
    expect_equal(
      fit$par_info$mle[idx_raw],
      fit$par_info$mle_internal[idx_raw],
      tolerance = 1e-10
    )
    expect_equal(
      fit$par_info$se[idx_raw],
      fit$par_info$se_internal[idx_raw],
      tolerance = 1e-10
    )
  }
  if (!is.null(fit$details$vcov)) {
    expect_equal(
      unname(fit$par_info$se_internal),
      unname(sqrt(pmax(0, diag(fit$details$vcov)))),
      tolerance = 1e-8
    )
  }
  expect_false("par" %in% names(fit))
  expect_false("logLik" %in% names(fit))
  expect_false("vcov" %in% names(fit))
  expect_false("optim" %in% names(fit))
  expect_true(!is.null(fit$details$outer_opt))
  expect_true(!is.null(fit$details$inner_opt))
  expect_true(!is.null(fit$details$gradient$outer$trace3))
  expect_null(fit$details$hessian$trace3)
})
