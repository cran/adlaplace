test_that("psd(iwp) matches BayesGP PSD_compute_iwp when spacing is 1", {
  for (p in c(1L, 2L, 3L)) {
    knots <- seq(0, 4, by = 1) # mean spacing 1
    term <- iwp("x", p = p, knots = knots, ref_value = 0, include_poly = FALSE)$x_iwp
    h <- c(0.5, 1, 2)
    got <- psd(term, h = h)
    # BayesGP: sqrt(h^(2p-1) / ((2p-1) ((p-1)!)^2))
    expect_equal(
      got,
      sqrt((h^(2 * p - 1)) / ((2 * p - 1) * (factorial(p - 1)^2))),
      tolerance = 1e-12
    )
  }
})

test_that("psd(iwp) is invariant to joint rescaling of x, knots, and h", {
  knots <- seq(0, 1, by = 0.25)
  term <- iwp("x", p = 2, knots = knots, ref_value = 0, include_poly = FALSE)$x_iwp
  h <- c(0.1, 0.25, 0.5)
  base <- psd(term, h = h)

  knots10 <- 10 * knots
  term10 <- iwp("x", p = 2, knots = knots10, ref_value = 0, include_poly = FALSE)$x_iwp
  expect_equal(psd(term10, h = 10 * h), base, tolerance = 1e-12)
})

test_that("psd(iwp) matches design/precision prior SD at knot offsets", {
  # O-spline prior SD at a knot approaches the continuous PSD; with the
  # knot-spacing scaling both sides use the same units.
  knots <- seq(0, 1, by = 0.05)
  ref <- 0
  term <- iwp("x", p = 2, knots = knots, ref_value = ref, include_poly = FALSE)$x_iwp
  x_eval <- 0.5
  h <- x_eval - ref
  B <- as.numeric(design(term, data.frame(x = x_eval)))
  W <- diag(as.matrix(precision(term, data.frame(x = x_eval))))
  fem_sd <- sqrt(sum((B^2) / W))
  expect_equal(fem_sd, psd(term, h = h), tolerance = 5e-3)
})

test_that("psd_prior resolves iwp list labels and errors without a method", {
  terms <- iwp("x", p = 2, knots = seq(0, 1, by = 0.25), ref_value = 0)
  pr <- psd_prior(terms, u = 0.1, alpha = 0.5, h = 0.25)
  expect_identical(pr@theta, "x_iwp")
  expect_true(pr@rate > 0)

  # model_term default has no psd method
  expect_error(psd_prior(intercept(), u = 0.1, alpha = 0.5, h = 1), "no psd")
})
