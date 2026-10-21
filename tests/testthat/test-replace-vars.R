test_that("replace_vars rewrites model_term name/label/formula", {
  term <- adlaplace::fpoly("sqrt_pm", p = 1, ref_value = 4)
  out <- adlaplace::replace_vars(term, c(sqrt_pm = "sqrt_pm_s3"))
  expect_equal(out@name, "sqrt_pm_s3")
  expect_true(grepl("sqrt_pm_s3", out@label))
  expect_true(grepl("sqrt_pm_s3", deparse(out@formula)))
})

test_that("replace_vars on list of terms preserves structure", {
  terms <- list(
    adlaplace::fpoly("sqrt_pm", p = 1, ref_value = 4),
    adlaplace::dirichlet_multinom("mort_all", by = c("geo_id", "yearMonthDow"))
  )
  out <- adlaplace::replace_vars(
    terms,
    c(sqrt_pm = "sqrt_pm_s1", geo_id = "region")
  )
  expect_equal(out[[1]]@name, "sqrt_pm_s1")
  expect_equal(unname(out[[2]]@by), c("region", "yearMonthDow"))
})

test_that("replace_vars preserves a trailing label suffix", {
  term <- adlaplace::rpoly("sqrt_pm", p = 1, ref_value = 4)
  term@label <- "sqrt_pm_hrpoly_1"
  terms <- list(sqrt_pm_hrpoly_1 = term)
  out <- adlaplace::replace_vars(terms, c(sqrt_pm = "sqrt_pm_s1"))
  expect_true("sqrt_pm_s1_hrpoly_1" %in% names(out))
  expect_equal(out[["sqrt_pm_s1_hrpoly_1"]]@label, "sqrt_pm_s1_hrpoly_1")
  expect_equal(out[["sqrt_pm_s1_hrpoly_1"]]@name, "sqrt_pm_s1")
})
