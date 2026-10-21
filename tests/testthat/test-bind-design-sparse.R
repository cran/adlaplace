# Dense reference matching the pre-sparse local_poly_positive_axis.
dense_local_poly_positive_axis <- function(knots, refined_x, p) {
  dif <- diff(knots)
  nn <- length(refined_x)
  n <- length(knots)
  D <- matrix(0, nrow = nn, ncol = n - 1)
  for (j in seq_len(nn)) {
    for (i in seq_len(n - 1L)) {
      if (refined_x[[j]] <= knots[[i]]) {
        D[j, i] <- 0
      } else if (
        refined_x[[j]] <= knots[[i + 1L]] && refined_x[[j]] >= knots[[i]]
      ) {
        D[j, i] <- (1 / factorial(p)) * (refined_x[[j]] - knots[[i]])^p
      } else {
        k <- seq_len(p)
        D[j, i] <- sum(
          (dif[[i]]^k) * ((refined_x[[j]] - knots[[i + 1L]])^(p - k)) /
            (factorial(k) * factorial(p - k))
        )
      }
    }
  }
  D
}

dense_local_poly <- function(knots, refined_x, p) {
  if (min(knots) >= 0) {
    dense_local_poly_positive_axis(knots, refined_x, p)
  } else if (max(knots) <= 0) {
    knots_neg <- unique(sort(ifelse(knots < 0, -knots, 0)))
    refined_x_neg <- ifelse(refined_x < 0, -refined_x, 0)
    dense_local_poly_positive_axis(knots_neg, refined_x_neg, p)
  } else {
    knots_neg <- unique(sort(ifelse(knots < 0, -knots, 0)))
    refined_x_neg <- ifelse(refined_x < 0, -refined_x, 0)
    knots_pos <- unique(sort(ifelse(knots > 0, knots, 0)))
    refined_x_pos <- ifelse(refined_x > 0, refined_x, 0)
    cbind(
      dense_local_poly_positive_axis(knots_neg, refined_x_neg, p),
      dense_local_poly_positive_axis(knots_pos, refined_x_pos, p)
    )
  }
}

test_that("local_poly returns dgCMatrix matching dense basis", {
  knots <- seq(0, 1, by = 0.25)
  x <- seq(0, 1, length.out = 17)
  sp <- adlaplace:::local_poly(knots, x, 2)
  expect_s4_class(sp, "dgCMatrix")
  expect_equal(as.matrix(sp), dense_local_poly(knots, x, 2), tolerance = 1e-12)
  # Zeros to the right of x are structural.
  expect_lt(Matrix::nnzero(sp), length(sp))

  knots_split <- seq(-1, 1, by = 0.5)
  x_split <- seq(-1, 1, length.out = 21)
  sp_split <- adlaplace:::local_poly(knots_split, x_split, 3)
  expect_s4_class(sp_split, "dgCMatrix")
  expect_equal(
    as.matrix(sp_split),
    dense_local_poly(knots_split, x_split, 3),
    tolerance = 1e-12
  )
})

test_that("design(iwp) is dgCMatrix with colnames", {
  knots <- seq(0, 1, length.out = 6)
  dat <- data.frame(z = seq(0, 1, length.out = 12))
  term <- adlaplace::iwp("z", p = 2, knots = knots, ref_value = 0.5)[[1]]
  A <- design(term, dat)
  expect_s4_class(A, "dgCMatrix")
  expect_true(all(grepl("^z_iwp_k", colnames(A))))
  expect_equal(ncol(A), length(knots) - 1L)
})

test_that("cbind_design_blocks coerces when any block is sparse", {
  dense <- matrix(1:6, 3, 2, dimnames = list(NULL, c("d1", "d2")))
  sparse <- Matrix::sparseMatrix(
    i = 1:3, j = c(1L, 1L, 2L), x = 1:3,
    dims = c(3L, 2L),
    dimnames = list(NULL, c("s1", "s2"))
  )

  ds <- adlaplace:::cbind_design_blocks(list(dense, sparse))
  sd <- adlaplace:::cbind_design_blocks(list(sparse, dense))
  dd <- adlaplace:::cbind_design_blocks(list(dense, dense))

  expect_s4_class(ds, "dgCMatrix")
  expect_s4_class(sd, "dgCMatrix")
  expect_equal(ncol(ds), 4L)
  expect_equal(ncol(sd), 4L)
  expect_true(is.matrix(dd))
  expect_false(inherits(dd, "Matrix"))
  expect_null(adlaplace:::cbind_design_blocks(list(NULL, NULL)))
})

test_that("model_data A is sparse for iwp + rpoly expansion", {
  set.seed(2)
  dat <- data.frame(y = rnorm(30), x = runif(30))
  md <- adlaplace::model_data(
    y ~ adlaplace::iwp(x, p = 2, knots = seq(0, 1, len = 6), ref_value = 0.5),
    data = dat,
    verbose = FALSE
  )
  expect_true(inherits(md$term_data$A, "sparseMatrix"))
  expect_equal(ncol(md$term_data$A), nrow(md$term_data$info$gamma))
})

test_that("ensure_config_obs_groups fills obs_groups for base-matrix A", {
  set.seed(3)
  A <- matrix(rnorm(50), 10, 5)
  cfg <- adlaplace:::ensure_config_obs_groups(
    list(num_shards = 3L),
    A = A
  )
  expect_false(is.null(cfg$obs_groups))
  expect_equal(ncol(cfg$obs_groups), 3L)
})

test_that("dirichlet_multinom + iwp gets multiple observation shards", {
  set.seed(4)
  n_strata <- 12L
  n_per <- 5L
  n <- n_strata * n_per
  data <- data.frame(
    count = c(rep(0L, n_per - 1L), 2L)[rep(seq_len(n_per), n_strata)],
    x = runif(n),
    region = rep(1:3, length.out = n),
    date = rep(seq_len(n_strata), each = n_per),
    year = 2002L
  )
  # Ensure each stratum has multiple days and at least one event.
  data$count <- as.integer(data$count)
  formula <- dirichlet_multinom(
    count,
    by = c("year", "region", "date"),
    init = 0.1
  ) ~ adlaplace::iwp(
    x,
    p = 2,
    knots = seq(0, 1, len = 6),
    ref_value = 0.5,
    init = 0.1
  )

  md <- model_data(formula = formula, data = data, verbose = FALSE)
  pack <- ad_pack(
    md,
    config = list(
      transform_theta = TRUE,
      num_shards = 4L,
      verbose = FALSE
    ),
    num_threads = 1L,
    reorder_shards = "none"
  )
  expect_gt(adlaplace:::n_groups(pack@ptr), 1L)
  expect_true(inherits(md$term_data$A, "sparseMatrix"))
})
