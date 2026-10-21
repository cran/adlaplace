test_that("model_data + OpenMP log_lik_laplace works", {
  # Runs under R CMD check (NOT_CRAN=true); skip interactive CRAN-like runs.
  skip_on_cran()
  skip_if_not(adlaplace:::has_openmp(), "OpenMP not available in this build")
  set.seed(1L)
  n <- 80L
  dat <- data.frame(
    y = rpois(n, 2),
    x = runif(n),
    g = factor(rep(seq_len(8L), length.out = n))
  )
  md <- adlaplace::model_data(
    data = dat,
    formula = adlaplace::nbinom(y, init = 0.2) ~ x + adlaplace::iid(g, init = 0.2)
  )
  config <- list(
    transform_theta = TRUE,
    obs_groups = adlaplace::obs_groups(md$term_data$A, num_shards = 8L),
    verbose = FALSE
  )
  af <- adlaplace::ad_pack(md, config, num_threads = 2L)
  res <- adlaplace::log_lik_laplace(
    x = md$term_data$info$parameters$init,
    ad_pack = af,
    config = config,
    control = list(report.level = 0, report.freq = 0),
    deriv = TRUE
  )
  expect_true(is.finite(res$log_lik))
  expect_equal(length(res$deriv$d_neg_log_lik), length(md$term_data$info$parameters$init))
  expect_true(all(is.finite(res$deriv$d_neg_log_lik)))
})
