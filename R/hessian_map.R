#' Build grouped Hessian sparsity structures
#'
#' Constructs Hessian sparsity templates (global/outer and inner-gamma)
#' and per-group maps from shard sparsity patterns.
#'
#' @param sparsity_list List of per-group sparsity shards from \code{ad_pack()}.
#' @param n_beta Number of fixed-effect parameters.
#' @param n_gamma Number of random-effect parameters.
#' @param n_theta Number of variance parameters.
#'
#' @return A list with \code{outer}, \code{inner}, optional \code{chol_inner_list}
#'   (symbolic LDL pattern for C++: \code{L1}, \code{Linv}, \code{perm},
#'   \code{perm} (0-based; inner Hessian uses \code{index1 = FALSE}),
#'   \code{perm_inv} (0-based), \code{half_H_inv}, \code{H_inv}), optional
#'   \code{trace_columns} (added in \code{ad_pack()}, not by this function),
#'   \code{chol_inner} (\code{dCHMsimpl}
#'   for R), \code{map_outer}, \code{map_inner}, and \code{sizes} (named
#'   \code{beta}/\code{gamma}/\code{theta}; consumed internally by
#'   \code{ad_pack()}).
#'
#' @keywords internal
hessian_map <- function(sparsity_list, n_beta, n_gamma, n_theta) {
  if (n_beta < 0L || n_gamma < 0L || n_theta < 0L) {
    stop("n_beta, n_gamma, and n_theta must be non-negative")
  }

  Ngroups <- length(sparsity_list)
  Nparams <- n_beta + n_gamma + n_theta
  Sgamma0 <- seq.int(n_beta, length.out = n_gamma)

  shard_frames <- hessian_map_shard_frames(sparsity_list, Ngroups)
  sparsityOuter <- hessian_map_assign_outer_cells(
    shard_frames$outer,
    Nparams
  )
  sparsityInner <- hessian_map_assign_inner_cells(
    shard_frames$inner,
    Sgamma0,
    n_gamma
  )

  outerMat <- sparsityOuter[!duplicated(sparsityOuter$cell), , drop = FALSE]
  innerMat <- sparsityInner[!duplicated(sparsityInner$cell), , drop = FALSE]

  hessian_outer <- hessian_map_build_outer_template(outerMat, Nparams)
  hessian_inner <- hessian_map_build_inner_template(innerMat, n_gamma)
  hessian_outer_map <- hessian_map_build_shard_map(sparsityOuter, Ngroups)
  hessian_inner_map <- hessian_map_build_shard_map(sparsityInner, Ngroups)

  result_map <- hessian_map_extract_maps(
    hessian_outer_map,
    hessian_inner_map,
    hessian_inner
  )
  chol_result <- hessian_map_build_chol_inner(innerMat, n_gamma)

  list(
    outer = as_dgC(hessian_outer),
    inner = as_dgC(hessian_inner),
    chol_inner = chol_result$chol_inner,
    chol_inner_list = chol_result$chol_inner_list,
    map_outer = result_map$outer,
    map_inner = result_map$inner,
    sizes = c(beta = n_beta, gamma = n_gamma, theta = n_theta)
  )
}

hessian_map_make_sparse_df <- function(shard_id, row, col) {
  n <- length(col)
  data.frame(
    shard = rep.int(as.integer(shard_id), n),
    row = as.integer(row),
    col = as.integer(col),
    local = seq.int(0L, length.out = n),
    stringsAsFactors = FALSE
  )
}

hessian_map_bind_sparse_df <- function(lst) {
  lst <- lst[lengths(lst) > 0L]
  if (length(lst) == 0L) {
    return(data.frame(
      shard = integer(0),
      row = integer(0),
      col = integer(0),
      local = integer(0),
      stringsAsFactors = FALSE
    ))
  }
  do.call(rbind, lst)
}

hessian_map_shard_frames <- function(sparsity_list, Ngroups) {
  list_outer <- vector("list", Ngroups)
  list_inner <- vector("list", Ngroups)

  for (D in seq_len(Ngroups)) {
    shard <- sparsity_list[[D]]
    list_outer[[D]] <- hessian_map_make_sparse_df(
      shard_id = D - 1L,
      row = shard$row_hess,
      col = shard$col_hess
    )
    list_inner[[D]] <- hessian_map_make_sparse_df(
      shard_id = D - 1L,
      row = shard$row_hess_inner,
      col = shard$col_hess_inner
    )
  }

  list(
    outer = hessian_map_bind_sparse_df(list_outer),
    inner = hessian_map_bind_sparse_df(list_inner)
  )
}

hessian_map_assign_outer_cells <- function(sparsityOuter, Nparams) {
  sparsityOuter$cell <- sparsityOuter$row + sparsityOuter$col * Nparams
  sparsityOuter$cellSparse <- as.integer(factor(sparsityOuter$cell)) - 1L
  sparsityOuter
}

hessian_map_assign_inner_cells <- function(sparsityInner, Sgamma0, n_gamma) {
  sparsityInner$rowInner <- match(sparsityInner$row, Sgamma0) - 1L
  sparsityInner$colInner <- match(sparsityInner$col, Sgamma0) - 1L
  if (anyNA(sparsityInner$rowInner) || anyNA(sparsityInner$colInner)) {
    stop("inner hessian index out of gamma range")
  }
  sparsityInner$cell <- sparsityInner$rowInner + n_gamma * sparsityInner$colInner
  sparsityInner$cellSparse <- as.integer(factor(sparsityInner$cell)) - 1L
  sparsityInner
}

hessian_map_build_outer_template <- function(outerMat, Nparams) {
  Matrix::sparseMatrix(
    i = outerMat$row,
    j = outerMat$col,
    x = outerMat$cellSparse,
    repr = "C",
    symmetric = TRUE,
    index1 = FALSE,
    dims = c(Nparams, Nparams)
  )
}

hessian_map_build_inner_template <- function(innerMat, n_gamma) {
  Matrix::sparseMatrix(
    i = innerMat$rowInner,
    j = innerMat$colInner,
    x = innerMat$cellSparse,
    repr = "C",
    symmetric = TRUE,
    index1 = FALSE,
    dims = c(n_gamma, n_gamma)
  )
}

hessian_map_build_shard_map <- function(sparsity_df, Ngroups) {
  Matrix::sparseMatrix(
    i = sparsity_df$cellSparse,
    j = sparsity_df$shard,
    x = sparsity_df$local,
    index1 = FALSE,
    repr = "C",
    dims = c(length(unique(sparsity_df$cellSparse)), Ngroups)
  )
}

hessian_map_extract_maps <- function(
  hessian_outer_map,
  hessian_inner_map,
  hessian_inner
) {
  list(
    outer = list(
      p = as.integer(hessian_outer_map@p),
      local = as.integer(hessian_outer_map@x),
      global = as.integer(hessian_outer_map@i),
      dims = dim(hessian_outer_map)
    ),
    inner = list(
      p = as.integer(hessian_inner_map@p),
      local = as.integer(hessian_inner_map@x),
      global = match(hessian_inner_map@i, as.integer(hessian_inner@x)) - 1L,
      dims = c(length(hessian_inner@x), ncol(hessian_inner_map))
    )
  )
}

hessian_map_build_chol_inner <- function(innerMat, n_gamma) {
  chol_inner <- NULL
  if (n_gamma > 0L && nrow(innerMat) > 0L) {
    chol_inner <- tryCatch(
      {
        # Pattern-only dummy: off-diagonals 1, diagonal = 1 + degree so the
        # matrix is strictly diagonally dominant (hence SPD) for any graph.
        # A fixed diagonal of 10 fails on hub-and-spoke patterns (e.g.
        # hierarchical IWP with a global component coupling all groups).
        #
        # AD sparsity lists both triangles as separate cells; Matrix
        # sparseMatrix(..., symmetric = TRUE) requires a single triangle
        # (uplo="U" by default) and errors if both are supplied.
        upper <- innerMat$rowInner <= innerMat$colInner
        hi_pat <- Matrix::sparseMatrix(
          i = innerMat$rowInner[upper],
          j = innerMat$colInner[upper],
          x = 1,
          symmetric = TRUE,
          index1 = FALSE,
          dims = c(n_gamma, n_gamma)
        )
        hi_dummy <- methods::as(hi_pat, "generalMatrix")
        off_deg <- as.numeric(Matrix::rowSums(hi_dummy)) -
          as.numeric(Matrix::diag(hi_dummy))
        Matrix::diag(hi_dummy) <- off_deg + 1
        hi_dummy <- Matrix::forceSymmetric(hi_dummy)
        Matrix::Cholesky(hi_dummy, perm = TRUE, LDL = TRUE)
      },
      error = function(e) NULL
    )
  }

  chol_inner_list <- list()
  if (!is.null(chol_inner)) {
    L <- Matrix::expand2(chol_inner)$L1
    perm0 <- as.integer(chol_inner@perm)
    perm_inv <- as.integer(order(chol_inner@perm) - 1L)
    Linv <- methods::as(Matrix::solve(L), "nMatrix")
    half_H_inv_pat <- Matrix::crossprod(Linv, Matrix::Diagonal(n_gamma, 1))
    half_H_inv_pat <- half_H_inv_pat[perm_inv + 1L, ]
    H_inv_pat <- Matrix::drop0(Matrix::triu(Matrix::tcrossprod(half_H_inv_pat), k = 0L))
    chol_inner_list <- list(
      L1 = L,
      Linv = Linv,
      perm = perm0,
      perm_inv = perm_inv,
      half_H_inv = methods::as(half_H_inv_pat, "CsparseMatrix"),
      H_inv = methods::as(H_inv_pat, "CsparseMatrix")
    )
  }

  if (is.null(chol_inner)) {
    chol_inner <- Matrix::sparseMatrix(
      i = integer(0),
      j = integer(0),
      dims = c(0L, 0L)
    )
  }

  list(chol_inner = chol_inner, chol_inner_list = chol_inner_list)
}

#' Per-shard column indices into half_H_inv for trace_hinv_t
#'
#' Built from \code{group_sparsity} and the symbolic \code{half_H_inv} pattern
#' in \code{chol_inner_list}. Added to that list in \code{new_ad_pack_from_ptr()}.
#'
#' @param group_sparsity List of per-shard \code{grad_inner} index vectors.
#' @param n_beta Number of fixed-effect parameters.
#' @param n_gamma Number of random-effect parameters.
#' @param half_H_inv_pat Symbolic \code{half_H_inv} sparse matrix pattern.
#' @return Sparse \code{ngCMatrix} (\code{index1 = FALSE}) with
#'   \code{dims = c(n_gamma, length(group_sparsity))}.
#' @keywords internal
trace_columns_from_pattern <- function(
  group_sparsity,
  n_beta,
  n_gamma,
  half_H_inv_pat
) {
  n_groups <- length(group_sparsity)
  if (n_gamma < 1L || n_groups < 1L || is.null(half_H_inv_pat)) {
    return(Matrix::sparseMatrix(
      i = integer(0),
      j = integer(0),
      dims = c(n_gamma, n_groups),
      index1 = FALSE
    ))
  }

  seq_gamma0 <- seq.int(n_beta, length.out = n_gamma)
  which_columns_by_group1 <- lapply(
    group_sparsity,
    function(grad_inner, refmat) {
      grad_inner_gamma <- match(grad_inner, seq_gamma0)
      linv_here <- refmat[grad_inner_gamma, , drop = FALSE]
      which(diff(linv_here@p) > 0L) - 1L
    },
    refmat = half_H_inv_pat
  )

  Matrix::sparseMatrix(
    i = unlist(which_columns_by_group1),
    j = rep(
      seq(0L, length.out = length(which_columns_by_group1)),
      unlist(lapply(which_columns_by_group1, length))
    ),
    index1 = FALSE,
    dims = c(n_gamma, length(which_columns_by_group1))
  )
}

#' Positions of a shard's quadratic form inside the joint inverse Hessian
#'
#' Called from \code{ad_pack()} after the symbolic \code{H_inv} pattern exists.
#' The default method returns \code{NULL}, which leaves that shard on the
#' \code{trace_hinv_t} column walk. A backend method may return an integer
#' vector of 0-based slots into \code{H_inv@x}, one per stored entry of the
#' shard's precision, with \code{-1L} where the pair is absent from the pattern.
#'
#' @param shard A \code{density_data} shard.
#' @param H_inv Symbolic upper triangle of the joint inverse Hessian.
#' @return An integer vector, or \code{NULL}.
#' @export
setGeneric(
  "hinv_trace_index",
  function(shard, H_inv) standardGeneric("hinv_trace_index")
)

#' @rdname hinv_trace_index
#' @export
setMethod("hinv_trace_index", "density_data", function(shard, H_inv) {
  NULL
})
