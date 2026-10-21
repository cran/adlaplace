test_that("obs_groups partitions a modest design matrix", {
  set.seed(11)
  A <- matrix(rnorm(120), nrow = 30, ncol = 4)
  G <- obs_groups(A, num_shards = 3L)
  expect_s4_class(G, "ngCMatrix")
  expect_false("x" %in% methods::slotNames(G))
  expect_equal(nrow(G), 30L)
  expect_gte(ncol(G), 1L)
  expect_lte(ncol(G), 3L)
})

test_that("obs_groups uses max(dim(ATp)) for large-matrix warning", {
  # deparse() canonicalizes 1e5 to 1e+05; the installed body has no R/ source.
  src <- paste(deparse(body(obs_groups)), collapse = "\n")
  expect_match(src, "max\\(dim\\(ATp\\)\\)\\s*>\\s*1e\\+05")
  expect_match(src, "max\\(dim\\(ATp\\)\\)\\s*>\\s*100")
})

