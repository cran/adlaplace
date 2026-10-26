test_that("observation_response supplies counts that are not rows of data", {
  methods::setClass(
    "toy_agg",
    slots = c(elgm = "Matrix", y = "numeric"),
    contains = "model_term",
    prototype = methods::prototype(
      model_role = factor("response", levels = adlaplace::.model_role_levels),
      density = "poisson_obs",
      ad_kind = "observations",
      y = numeric(0),
      elgm = methods::new("dgCMatrix")
    ),
    where = topenv()
  )
  methods::setMethod(
    "observation_response", "toy_agg",
    function(term, data) term@y,
    where = topenv()
  )
  methods::setMethod(
    "elgm_matrix", "toy_agg",
    function(term, data) term@elgm,
    where = topenv()
  )
  on.exit({
    methods::removeMethod("observation_response", "toy_agg", where = topenv())
    methods::removeMethod("elgm_matrix", "toy_agg", where = topenv())
    methods::removeClass("toy_agg", where = topenv())
  }, add = TRUE)

  n_cell <- 5L
  elgm <- Matrix::sparseMatrix(
    i = c(1L, 2L, 3L, 3L, 4L, 5L),
    j = c(1L, 1L, 1L, 2L, 2L, 2L),
    x = log(c(10, 20, 15, 15, 40, 50)),
    dims = c(n_cell, 2L)
  )
  toy_agg <- function(x, elgm, y) {
    methods::new(
      "toy_agg",
      name = x,
      label = paste0(x, "_toy"),
      formula = stats::as.formula(paste(x, "~ ."), env = new.env()),
      y = as.numeric(y),
      elgm = elgm
    )
  }
  dat <- data.frame(x = stats::rnorm(n_cell))
  md <- adlaplace::model_data(
    toy_agg("Y", elgm = elgm, y = c(4, 7)) ~ x,
    data = dat
  )
  expect_equal(md$term_data$y, c(4, 7))
  expect_equal(nrow(md$term_data$X), n_cell)
  expect_equal(md$observations$Y@y, c(4, 7))
  expect_equal(dim(md$observations$Y@elgm_matrix), c(n_cell, 2L))
  expect_equal(ncol(md$term_data$A), 0L)
})
