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

fit_small_glmm <- function(seed = 4L) {
  dat <- simulate_fit_data(n = 200L, seed = seed)
  fit <- adlaplace(
    nbinom(y, lower = 1e-9, init = 0.15) ~ x + iid(g, init = 0.3),
    data = dat,
    config = list(num_shards = 5L),
    control = list(maxit = 80L),
    verbose = FALSE
  )
  fit$.__dat__ <- dat
  fit
}

test_that("predict without newdata returns fitted values", {
  fit <- fit_small_glmm()
  expect_equal(predict(fit), fitted(fit))
  expect_equal(length(predict(fit)), fit$nobs)
})

test_that("predict with newdata returns simulation matrix", {
  fit <- fit_small_glmm()
  dat <- fit$model_data$term_data$y
  newx <- data.frame(
    x = c(0, 1),
    g = factor(c(1L, 2L), levels = levels(fit$.__dat__$g))
  )
  set.seed(1)
  pred <- predict(fit, newdata = newx, n = 8L)
  expect_equal(dim(as.matrix(pred)), c(2L, 8L))
  expect_true(all(is.finite(pred)))
})

test_that("coef transform argument controls scale", {
  fit <- fit_small_glmm()
  nat <- coef(fit, transform = TRUE)
  raw <- coef(fit, transform = FALSE)
  idx <- which(fit$par_info$log %in% TRUE)
  expect_equal(nat[idx], exp(raw[idx]), tolerance = 1e-10)
  expect_equal(raw[-idx], nat[-idx], tolerance = 1e-10)
})

test_that("summary and confint handle missing vcov", {
  dat <- simulate_fit_data(n = 150L, seed = 5L)
  fit <- adlaplace(
    nbinom(y, lower = 1e-9, init = 0.15) ~ x + iid(g, init = 0.3),
    data = dat,
    config = list(num_shards = 4L),
    hessian = FALSE,
    control = list(maxit = 50L),
    verbose = FALSE
  )
  sm <- summary(fit)
  expect_s3_class(sm, "summary.adlaplace_fit")
  expect_true(all(is.na(sm$coefficients[, "Std. Error"])))
  expect_output(print(sm), "Coefficients")
  expect_true(all(is.na(confint(fit)[, 1])))
})

test_that("vcov transform applies delta method on log-scale parameters", {
  fit <- fit_small_glmm()
  raw <- vcov(fit, transform = FALSE)
  nat <- vcov(fit, transform = TRUE)
  idx <- which(fit$par_info$log %in% TRUE)
  d <- rep(1, length(fit$par_info$mle_internal))
  d[idx] <- exp(fit$par_info$mle_internal[idx])
  expect_equal(nat, raw * tcrossprod(d), tolerance = 1e-8)
})
