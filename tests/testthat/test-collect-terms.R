test_that("collect_terms parses linear(x) without x in caller env", {
  f <- y ~ intercept() + linear(x)
  terms <- adlaplace::collect_terms(f)
  expect_true(inherits(terms$linear_x, "linear"))
  expect_equal(terms$linear_x@name, "x")
  expect_equal(terms$linear_x@label, "x_linear")
})

test_that("collect_terms parses bare covariate as linear", {
  f <- y ~ x1
  terms <- adlaplace::collect_terms(f)
  expect_true(inherits(terms$linear_x1, "linear"))
  expect_equal(terms$linear_x1@name, "x1")
})

test_that("collect_terms keeps bare outcome LHS out of model terms", {
  terms <- adlaplace::collect_terms(y ~ x1)
  expect_false("response" %in% names(terms))
  expect_false(any(vapply(terms, function(x) {
    methods::is(x, "nbinom")
  }, logical(1L))))
})

test_that("random_info reuses precomputed term label", {
  term <- adlaplace::iid("grp")[[1L]]
  expect_equal(term@label, "grp_iid")
  info <- adlaplace::random_info(term, data.frame(grp = factor(c("a", "b", "a"))))
  expect_true(all(info$label == term@label))
})

test_that("collect_terms reports missing constructor vs evaluation failure", {
  f_missing <- y ~ not_a_real_term(x)
  expect_error(
    adlaplace::collect_terms(f_missing),
    "constructor 'not_a_real_term' is not available"
  )

  # Constructor exists, but an argument expression fails while evaluating.
  make_bad_term <- function(x = NULL, knots = NULL) {
    list(x = x, knots = knots)
  }
  boom <- function() {
    stop("nested boom", call. = FALSE)
  }
  sites <- cbind(1:3, 4:6)
  f_nested <- y ~ make_bad_term(x = sites, knots = boom())
  environment(f_nested) <- environment()
  expect_error(
    adlaplace::collect_terms(f_nested),
    "Failed to evaluate term"
  )
  expect_error(
    adlaplace::collect_terms(f_nested),
    "nested boom"
  )
  expect_error(
    adlaplace::collect_terms(f_nested),
    "named arguments"
  )
})

if (!methods::isClass("toy_mult")) {
  methods::setClass(
    "toy_mult",
    slots = c(mult = "character"),
    contains = "model_term"
  )
}

test_that("collect_terms names drop package prefixes and keep columns", {
  toy_mult <- function(x, mult) {
    methods::new(
      "toy_mult",
      name = x,
      label = paste(x, mult, "toy_mult", sep = "_"),
      formula = stats::as.formula(paste0("~ 0 + ", x), env = new.env()),
      mult = mult
    )
  }
  f <- adlaplace::nbinom(y) ~ adlaplace::iid(group) + toy_mult(group, mult = "x")
  environment(f) <- environment()
  terms <- adlaplace::collect_terms(f)
  expect_false(any(grepl("::", names(terms), fixed = TRUE)))
  expect_true(all(c("nbinom_y", "iid_group", "toy_mult_group_x") %in% names(terms)))
})
