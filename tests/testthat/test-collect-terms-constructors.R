test_that("constructor formulas parse via collect_terms", {
  skip_if_not_installed("mgcv")
  dat <- quiet_gamsim(6, n = 80, scale = 0.2, dist = "poisson", verbose = FALSE)

  with(dat, {
    ctor_style <-
      adlaplace::nbinom(y, lower = 1e-9) ~
      x1 +
      adlaplace::iwp(
        x2,
        p = 2,
        knots = seq(0, 1, len = 11),
        ref_value = 0.5
      ) +
      adlaplace::iid(fac)

    terms <- adlaplace::collect_terms(ctor_style, verbose = FALSE)
    expect_false("response" %in% names(terms))

    nbinom_idx <- which(vapply(terms, function(x) {
      methods::is(x, "nbinom")
    }, logical(1L)))
    expect_length(nbinom_idx, 1L)
    expect_equal(terms[[nbinom_idx]]@density, "nbinom_obs")
    expect_equal(terms[[nbinom_idx]]@lower, 1e-9)
  })
})
