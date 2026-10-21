simulate_fit_data <- function(n = 400L, n_re = 8L, seed = 1L) {
  set.seed(seed)
  g <- sample(n_re, n, replace = TRUE)
  re <- rnorm(n_re, sd = 0.3)
  x <- rbinom(n, 1, 0.5)
  eta <- 1 + 0.5 * x + re[g]
  nb_sd <- 0.3
  z <- rgamma(n, nb_sd^(-2), nb_sd^(-2))
  data.frame(
    y = rpois(n, exp(eta) * z),
    x = x,
    g = factor(g)
  )
}

test_that("adlaplace() fits a nbinom GLMM and methods work", {
  dat <- simulate_fit_data()
  fit <- adlaplace(
    nbinom(y, lower = 1e-9, init = 0.15) ~ x + iid(g, init = 0.3),
    data = dat,
    config = list(num_shards = 10L),
    control = list(maxit = 100L),
    verbose = FALSE
  )

  expect_s3_class(fit, "adlaplace_fit")
  expect_true(is.finite(as.numeric(logLik(fit))))
  expect_identical(fit$details$outer_opt$convergence, 0L)

  co <- coef(fit)
  expect_true(all(is.finite(co)))
  expect_equal(length(co), nrow(fit$par_info))
  # transformed parameters reported on the natural (positive) scale
  expect_true(all(co[fit$par_info$log %in% TRUE] > 0))
  # rough recovery of the fixed effects (matched by label, order-free)
  beta_labels <- fit$par_info$label[
    seq_len(nrow(fit$model_data$term_data$info$beta))
  ]
  int_label <- grep("intercept", beta_labels, value = TRUE)[1]
  x_label <- setdiff(beta_labels, int_label)[1]
  expect_equal(unname(co[int_label]), 1, tolerance = 0.35)
  expect_equal(unname(co[x_label]), 0.5, tolerance = 0.35)

  vc <- vcov(fit)
  expect_true(isSymmetric(vc, tol = 1e-6))
  expect_true(all(diag(vc) > 0))

  ci <- confint(fit)
  expect_equal(dim(ci), c(length(co), 2L))
  expect_true(all(ci[, 1] < ci[, 2]))
  # estimates inside their own intervals
  expect_true(all(co > ci[, 1] & co < ci[, 2]))

  ll <- logLik(fit)
  expect_s3_class(ll, "logLik")
  expect_equal(attr(ll, "nobs"), nrow(dat))
  expect_equal(nobs(fit), nrow(dat))

  expect_equal(length(fitted(fit)), nrow(dat))
  expect_equal(length(fit$gamma), nlevels(dat$g))

  sm <- summary(fit)
  expect_s3_class(sm, "summary.adlaplace_fit")
  log_idx <- fit$par_info$log %in% TRUE
  se <- sm$coefficients[, "Std. Error"]
  expect_true(all(is.finite(se[!log_idx])))
  if (any(log_idx)) {
    expect_true(all(is.na(se[log_idx])))
  }
  expect_output(print(sm), "Coefficients")
  expect_output(print(fit), "Coefficients")
})

test_that("adlaplace() with hessian = FALSE has no vcov", {
  dat <- simulate_fit_data(n = 200L, seed = 2L)
  fit <- adlaplace(
    nbinom(y, lower = 1e-9, init = 0.15) ~ x + iid(g, init = 0.3),
    data = dat,
    config = list(num_shards = 5L),
    hessian = FALSE,
    control = list(maxit = 50L)
  )
  expect_null(fit$details$vcov)
  expect_error(vcov(fit), "hessian = TRUE")
  expect_true(all(is.na(confint(fit))))
})

test_that("adlaplace() default hessian is TRUE for <=5 outer params", {
  dat <- simulate_fit_data(n = 150L, seed = 3L)
  fit <- adlaplace(
    nbinom(y, lower = 1e-9, init = 0.15) ~ x + iid(g, init = 0.3),
    data = dat,
    config = list(num_shards = 5L),
    control = list(maxit = 40L)
  )
  expect_lte(nrow(fit$par_info), 5L)
  expect_false(is.null(fit$details$vcov))
})

test_that("adlaplace() returns a fit object when outer fn is non-finite", {
  dat <- simulate_fit_data(n = 80L, n_re = 4L, seed = 9L)
  local_mocked_bindings(
    outer_fn = function(x, config, cache, ad_pack, control_inner = list(), ...) {
      assign("last_par_fn", x, cache)
      Inf
    },
    .package = "adlaplace"
  )
  msgs <- character()
  withCallingHandlers(
    fit <- adlaplace(
      nbinom(y, lower = 1e-9, init = 0.15) ~ x + iid(g, init = 0.3),
      data = dat,
      config = list(num_shards = 4L),
      hessian = FALSE,
      control = list(maxit = 1L)
    ),
    message = function(m) {
      msgs <<- c(msgs, conditionMessage(m))
      invokeRestart("muffleMessage")
    }
  )
  expect_s3_class(fit, "adlaplace_fit")
  expect_null(fit$details$outer_opt)
  expect_true(nzchar(fit$details$error))
  expect_true(methods::is(fit$ad_pack, "ad_pack"))
  expect_true(is.environment(fit$cache))
  expect_true(all(is.na(fit$par_info$mle)))
  expect_true(is.na(as.numeric(logLik(fit))))
  expect_output(print(fit), "Outer optim failed")
  expect_true(any(grepl("outer optim failed", msgs, fixed = TRUE)))
  expect_true(any(grepl("last outer parameters", msgs, fixed = TRUE)))
})
