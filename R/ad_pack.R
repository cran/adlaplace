#' @include RcppExports.R classes.R density_data.R ad_pack_ptr.R
#' @keywords internal
effective_num_threads <- function(num_threads) {
  num_threads <- as.integer(num_threads)[1]
  if (is.na(num_threads) || num_threads < 1L) {
    stop("num_threads must be a positive integer", call. = FALSE)
  }
  # Without OpenMP at compile time, ignore requested thread count.
  if (!isTRUE(has_openmp())) {
    return(1L)
  }
  # Owner maps / parallel_map follow the requested count. Windows OpenMP/CppAD
  # team clamp-up (anti-shrink) happens at eval via padded thread groups, not
  # by rewriting owners here.
  num_threads
}

#' @keywords internal
match_reorder_shards <- function(reorder_shards) {
  reorder_shards <- as.character(reorder_shards)[1L]
  match.arg(reorder_shards, c("none", "gradient", "hessian", "third"))
}

#' Longest-processing-time thread assignment
#'
#' @param costs Numeric vector of per-shard costs (length n_shards).
#' @param num_threads Positive integer thread count.
#' @return Integer vector of length n_shards with 0-based owner thread ids.
#' @keywords internal
lpt_assign_owners <- function(costs, num_threads) {
  costs <- as.numeric(costs)
  num_threads <- as.integer(num_threads)[1L]
  n <- length(costs)
  if (n < 1L) {
    return(integer(0))
  }
  if (is.na(num_threads) || num_threads < 1L) {
    stop("num_threads must be a positive integer", call. = FALSE)
  }
  owners <- integer(n)
  loads <- numeric(num_threads)
  ord <- order(costs, decreasing = TRUE, na.last = TRUE)
  for (idx in ord) {
    t <- which.min(loads) - 1L
    owners[[idx]] <- t
    c_i <- costs[[idx]]
    if (is.na(c_i)) {
      c_i <- 0
    }
    loads[[t + 1L]] <- loads[[t + 1L]] + c_i
  }
  owners
}

#' @keywords internal
profile_shard_eval_times <- function(
  ptr,
  x,
  eval_fun,
  n_rep = 2L,
  n_warmup = 1L
) {
  n_rep <- as.integer(n_rep)[1L]
  n_warmup <- as.integer(n_warmup)[1L]
  if (is.na(n_rep) || n_rep < 1L) {
    stop("n_rep must be a positive integer", call. = FALSE)
  }
  if (is.na(n_warmup) || n_warmup < 0L) {
    stop("n_warmup must be non-negative", call. = FALSE)
  }
  # One GC before the timing loop (not per shard; avoids system.time gcFirst).
  invisible(gc())
  n_shards <- n_groups(ptr)
  times <- numeric(n_shards)
  for (s in seq_len(n_shards) - 1L) {
    if (n_warmup > 0L) {
      for (w in seq_len(n_warmup)) {
        eval_fun(ptr, x, s)
      }
    }
    elapsed <- numeric(n_rep)
    for (r in seq_len(n_rep)) {
      t0 <- proc.time()[["elapsed"]]
      eval_fun(ptr, x, s)
      elapsed[[r]] <- proc.time()[["elapsed"]] - t0
    }
    times[[s + 1L]] <- mean(elapsed)
  }
  times
}

#' @keywords internal
profile_shard_gradient_times <- function(ptr, x, n_rep = 2L, n_warmup = 0L) {
  profile_shard_eval_times(
    ptr,
    x,
    eval_fun = function(ptr, x, s) {
      grad(ptr, x, ad_shards = s, inner = FALSE)
    },
    n_rep = n_rep,
    n_warmup = n_warmup
  )
}

#' @keywords internal
profile_shard_hessian_times <- function(ptr, x, n_rep = 2L, n_warmup = 1L) {
  profile_shard_eval_times(
    ptr,
    x,
    eval_fun = function(ptr, x, s) {
      hessian(ptr, x, ad_shards = s, inner = FALSE)
    },
    n_rep = n_rep,
    n_warmup = n_warmup
  )
}

# Overrides RcppExports binding (Collate: ad_pack.R after RcppExports.R):
# one GC before C++ chrono timing, matching gradient/hessian helpers.
profile_shard_trace3_times <- function(
  handle,
  x,
  half_H_inv,
  trace_columns,
  n_rep = 4L,
  n_warmup = 1L
) {
  invisible(gc())
  .Call(
    `_adlaplace_profile_shard_trace3_times`,
    handle,
    x,
    half_H_inv,
    trace_columns,
    n_rep,
    n_warmup
  )
}

#' @keywords internal
build_parallel_map <- function(n_shards, num_threads, owner_threads = NULL) {
  num_threads <- effective_num_threads(num_threads)
  if (n_shards < 1L) {
    stop("ad_pack pointer has no shards", call. = FALSE)
  }

  shard_ids <- seq_len(n_shards)
  if (is.null(owner_threads)) {
    thread_ids <- ((shard_ids - 1L) %% num_threads) + 1L
  } else {
    owner_threads <- as.integer(owner_threads)
    if (length(owner_threads) != n_shards) {
      stop("owner_threads must have length n_shards", call. = FALSE)
    }
    if (any(is.na(owner_threads) | owner_threads < 0L)) {
      stop("owner_threads must be non-negative integers", call. = FALSE)
    }
    max_owner <- max(owner_threads)
    if (max_owner + 1L > num_threads) {
      num_threads <- max_owner + 1L
    }
    thread_ids <- owner_threads + 1L
  }
  map <- Matrix::sparseMatrix(
    i = shard_ids,
    j = thread_ids,
    dims = c(n_shards, num_threads),
    index1 = TRUE,
    repr = "C"
  )
  map
}

#' Place each shard's \code{hinv_trace_index()} result on its AD group.
#'
#' Observation densities split into several groups and must not define an
#' index. A quadratic-form density is one group; its integer vector is stored
#' at that group and every other group stays \code{NULL}.
#' @keywords internal
hinv_q_index_list <- function(density_shards, group_counts, H_inv) {
  group_counts <- as.integer(group_counts)
  n_groups <- sum(group_counts)
  out <- vector("list", n_groups)
  offset <- 0L
  for (i in seq_along(density_shards)) {
    ng <- group_counts[[i]]
    idx <- hinv_trace_index(density_shards[[i]], H_inv)
    if (!is.null(idx)) {
      if (ng != 1L) {
        stop(
          "hinv_trace_index requires a density with one AD group",
          call. = FALSE
        )
      }
      out[[offset + 1L]] <- as.integer(idx)
    }
    offset <- offset + ng
  }
  if (offset != n_groups) {
    stop(
      "group_counts do not sum to the number of AD groups",
      call. = FALSE
    )
  }
  out
}

#' @keywords internal
new_ad_pack_from_ptr <- function(
  ptr,
  num_threads = 1L,
  info = list(),
  verbose = FALSE,
  reorder_shards = c("none", "gradient", "hessian", "third"),
  templates = NULL,
  density_shards = NULL,
  group_counts = NULL
) {
  if (!is(ptr, "ad_pack_ptr")) {
    stop("ptr must be an ad_pack_ptr external pointer")
  }
  num_threads <- effective_num_threads(num_threads)
  reorder_shards <- match_reorder_shards(reorder_shards)
  n_shards <- n_groups(ptr)
  if (verbose) {
    cat(
      "ad_pack: attaching Hessian templates (",
      n_shards, " AD group(s), ",
      num_threads, " thread(s)",
      ", reorder_shards=", reorder_shards, ")...\n",
      sep = ""
    )
    utils::flush.console()
  }

  sz <- get_sizes(ptr, 0L)
  n_beta <- sz$n_beta
  n_theta <- sz$n_theta
  n_gamma <- sz$n_outer - n_beta - n_theta

  reuse <- NULL
  if (!is.null(templates)) {
    if (!methods::is(templates, "ad_pack")) {
      stop("templates must be an ad_pack object", call. = FALSE)
    }
    tsz <- templates@sizes
    if (!identical(as.integer(tsz["beta"]), as.integer(n_beta)) ||
      !identical(as.integer(tsz["gamma"]), as.integer(n_gamma)) ||
      !identical(as.integer(tsz["theta"]), as.integer(n_theta))) {
      stop(
        "templates sizes do not match ptr (beta/gamma/theta)",
        call. = FALSE
      )
    }
    if (length(templates@group_sparsity) != n_shards) {
      stop(
        "templates have ", length(templates@group_sparsity),
        " shards but ptr has ", n_shards,
        call. = FALSE
      )
    }
    reuse <- templates
    if (verbose) {
      cat("  reusing Hessian templates from supplied ad_pack...\n")
      utils::flush.console()
    }
  }

  if (is.null(reuse)) {
    if (verbose) {
      cat(
        "  collecting sparsity patterns (beta=", n_beta,
        ", gamma=", n_gamma, ", theta=", n_theta, ")...\n",
        sep = ""
      )
      utils::flush.console()
    }
    shard_ids <- seq_len(n_shards) - 1L
    sparsity <- vector("list", n_shards)
    for (i in seq_along(shard_ids)) {
      g <- shard_ids[[i]]
      if (verbose && n_shards > 1L) {
        cat("    sparsity group ", i, "/", n_shards, "\n", sep = "")
        utils::flush.console()
      }
      sparsity[[i]] <- c(get_sizes(ptr, g), get_sparse_pattern(ptr, g))
    }
    if (verbose) {
      cat("  building Hessian map...\n")
      utils::flush.console()
    }
    hessian_pack <- hessian_map(
      sparsity_list = sparsity,
      n_beta = n_beta,
      n_gamma = n_gamma,
      n_theta = n_theta
    )
    chol_inner_list <- hessian_pack$chol_inner_list
    if (length(chol_inner_list) > 0L && !is.null(chol_inner_list$half_H_inv)) {
      if (verbose) {
        cat("  building trace column map...\n")
        utils::flush.console()
      }
      hessian_pack$chol_inner_list$trace_columns <- trace_columns_from_pattern(
        group_sparsity = lapply(sparsity, function(shard) shard$grad_inner),
        n_beta = n_beta,
        n_gamma = n_gamma,
        half_H_inv_pat = chol_inner_list$half_H_inv
      )
      if (!is.null(density_shards) && !is.null(chol_inner_list$H_inv)) {
        if (is.null(group_counts)) {
          stop("group_counts is required when density_shards is supplied", call. = FALSE)
        }
        hessian_pack$chol_inner_list$hinv_q_index <- hinv_q_index_list(
          density_shards,
          group_counts,
          chol_inner_list$H_inv
        )
      }
      chol_inner_list <- hessian_pack$chol_inner_list
    }
    group_sparsity <- lapply(sparsity, function(shard) shard$grad_inner)
  } else {
    hessian_pack <- list(
      outer = reuse@outer,
      inner = reuse@inner,
      map_outer = reuse@map_outer,
      map_inner = reuse@map_inner,
      chol_inner = reuse@chol_inner,
      chol_inner_list = reuse@chol_inner_list,
      sizes = reuse@sizes
    )
    chol_inner_list <- hessian_pack$chol_inner_list
    group_sparsity <- reuse@group_sparsity
  }

  if (verbose) {
    cat("  attaching Hessian templates to C++ handle...\n")
    utils::flush.console()
  }
  adlaplace_attach_hessian(ptr, hessian_pack)

  if (verbose) {
    cat("ad_pack: Hessian attach complete.\n")
    utils::flush.console()
  }

  do_balance <- reorder_shards %in% c("gradient", "hessian", "third")
  if (num_threads <= 1L || n_shards <= 1L) {
    do_balance <- FALSE
  }

  owner_threads <- NULL
  if (do_balance) {
    n_params <- as.integer(sz$n_outer)
    x_profile <- rep(0, n_params)
    costs <- NULL
    if (identical(reorder_shards, "gradient")) {
      if (verbose) {
        cat("  profiling outer gradient times per shard...\n")
        utils::flush.console()
      }
      costs <- profile_shard_gradient_times(ptr, x_profile)
    } else if (identical(reorder_shards, "hessian")) {
      if (verbose) {
        cat("  profiling outer Hessian times per shard...\n")
        utils::flush.console()
      }
      costs <- profile_shard_hessian_times(ptr, x_profile)
    } else if (identical(reorder_shards, "third")) {
      if (
        !is.null(chol_inner_list$half_H_inv) &&
          !is.null(chol_inner_list$trace_columns)
      ) {
        if (verbose) {
          cat("  profiling Reverse3 (dummy direction) times per shard...\n")
          utils::flush.console()
        }
        half_pat <- methods::as(chol_inner_list$half_H_inv, "dgCMatrix")
        costs <- profile_shard_trace3_times(
          ptr,
          x_profile,
          half_pat,
          chol_inner_list$trace_columns,
          n_rep = 2L,
          n_warmup = 1L
        )
      } else if (verbose) {
        cat(
          "  reorder_shards='third' but no half_H_inv/trace_columns; ",
          "falling back to modulo owners.\n",
          sep = ""
        )
        utils::flush.console()
      }
    }
    if (!is.null(costs)) {
      owner_threads <- lpt_assign_owners(costs, num_threads)
      if (verbose) {
        cat(
          "  LPT owners: ",
          paste(owner_threads, collapse = ","),
          "\n",
          sep = ""
        )
        cat(
          "  shard times: ",
          paste(format(costs, digits = 3L), collapse = ","),
          "\n",
          sep = ""
        )
        utils::flush.console()
      }
    }
  }

  if (verbose) {
    cat("  assigning owner threads...\n")
    utils::flush.console()
  }
  if (is.null(owner_threads)) {
    assign_owner_threads(ptr, num_threads)
  } else {
    assign_owner_threads(ptr, num_threads, owners = owner_threads)
  }
  owner_threads <- vapply(
    seq_len(n_shards) - 1L,
    function(g) get_thread_owner(ptr, g),
    integer(1)
  )
  parallel_map <- build_parallel_map(n_shards, num_threads, owner_threads)

  methods::new(
    "ad_pack",
    ptr = ptr,
    group_sparsity = group_sparsity,
    outer = hessian_pack$outer,
    inner = hessian_pack$inner,
    map_outer = hessian_pack$map_outer,
    map_inner = hessian_pack$map_inner,
    parallel_map = parallel_map,
    chol_inner = hessian_pack$chol_inner,
    chol_inner_list = chol_inner_list,
    sizes = hessian_pack$sizes,
    info = info
  )
}

#' Build AD function with Hessian templates
#'
#' @param x One of: an \code{ad_pack_ptr}, an \code{density_data} shard with
#'   \code{ad_kind}/\code{ad_pack} set, or a \code{model_data()} bundle (list
#'   with \code{observations}, \code{random}, \code{parameters}).
#' @param config Model configuration list. Required for \code{density_data} and
#'   \code{model_data}; unused for \code{ad_pack_ptr}. For \code{model_data},
#'   \code{beta}, \code{gamma}, and \code{theta} are filled from \code{x$term_data$info}
#'   when omitted (only \code{shards}, \code{transform_theta}, etc. are needed).
#'   When \code{config$num_threads} is set, it overrides the \code{num_threads}
#'   argument for \code{density_data} and \code{model_data} methods.
#'   When \code{config$reorder_shards} is set, it overrides \code{reorder_shards}.
#'   \code{config$hessian_sparsity} (default \code{TRUE}) controls whether
#'   observation / parameter / random tapes discover and calibrate Hessian
#'   sparsity. Set \code{FALSE} for grad-only packs (usable with
#'   \code{\link{grad}}, not with \code{\link{inner_opt}} LDL / Hessian paths);
#'   see \code{\link{grad_obs_units}}.
#'   \code{config$obs_hessian} is \code{"analytic"} (default) or \code{"discover"}.
#'   Analytic builds the observation Hessian pattern from design co-occurrence
#'   (and ELGM strata for Dirichlet-multinomial) instead of CppAD
#'   \code{for_hes_sparsity}. Use \code{"discover"} as an escape hatch.
#'   When \code{config$obs_groups} is missing and \code{config$obs_units} is set
#'   (0-based unit indices), observation shards are built via
#'   \code{\link{obs_groups_units}} (see also \code{\link{ad_pack_drop}}).
#' @param num_threads Positive integer; OpenMP thread count for \code{inner_opt}
#'   and parallel \code{trace_hinv_t}. Default \code{1L} (serial).
#' @param reorder_shards Character; how to assign OpenMP owner threads when
#'   \code{num_threads > 1} and there are multiple shards:
#'   \code{"none"} (default) uses \code{owner_thread = shard_index \% num_threads};
#'   \code{"gradient"} times each shard's outer
#'   \code{\link{grad}(..., inner = FALSE)} (mean of 2; no warmup), then
#'   longest-processing-time (LPT) assignment;
#'   \code{"hessian"} times each shard's outer
#'   \code{\link{hessian}(..., inner = FALSE)} (1 warmup + mean of 2), then LPT;
#'   \code{"third"} times a Reverse3 sweep with dummy directions matching the
#'   symbolic \code{half_H_inv} / \code{trace_columns} sparsity (no numeric
#'   Cholesky), then LPT.
#'   Physical shard order is unchanged; only \code{owner_thread} is rewritten.
#' @param templates Optional existing \code{ad_pack} whose Hessian templates
#'   (and group sparsity) are reused when the new pack has identical
#'   \code{beta}/\code{gamma}/\code{theta} sizes and the same number of shards.
#'   Useful when rebuilding packs for modular inference over exposure samples.
#' @param ... For \code{ad_pack_ptr} input, optional additional
#'   \code{ad_pack_ptr} shards to combine before attaching templates.
#' @return Object of class \code{ad_pack}.
#' @export
setGeneric(
  "ad_pack",
  function(x, config = NULL, num_threads = 1L,
           reorder_shards = c("none", "gradient", "hessian", "third"),
           templates = NULL, ...) {
    standardGeneric("ad_pack")
  }
)

#' @rdname ad_pack
#' @export
setMethod(
  "ad_pack",
  signature = c(x = "ad_pack_ptr"),
  function(x, config = NULL, num_threads = 1L,
           reorder_shards = c("none", "gradient", "hessian", "third"),
           templates = NULL, ...) {
    # Formals before `...` are filled positionally. Extra ad_pack_ptr shards
    # therefore bind to config, num_threads, reorder_shards, or templates
    # whenever a later argument is supplied by name. Collect those shards in
    # call order and compose them with c(), the same path as
    # ad_pack(c(ptr1, ptr2, ...)). Never forward a pointer as templates.
    parts <- list()
    verbose <- FALSE
    config_reorder <- NULL

    if (is(config, "ad_pack_ptr")) {
      parts <- c(parts, list(config))
    } else if (is.list(config)) {
      # a genuine config list: only verbose / reorder_shards used at attach
      verbose <- isTRUE(config[["verbose"]])
      config_reorder <- config[["reorder_shards"]]
    } else if (!is.null(config)) {
      stop("additional arguments must be ad_pack_ptr", call. = FALSE)
    }

    if (is(num_threads, "ad_pack_ptr")) {
      parts <- c(parts, list(num_threads))
      num_threads <- 1L
    }
    if (is(reorder_shards, "ad_pack_ptr")) {
      parts <- c(parts, list(reorder_shards))
      reorder_shards <- "none"
    }
    if (!is.null(config_reorder)) {
      reorder_shards <- config_reorder
    }
    if (is(templates, "ad_pack_ptr")) {
      parts <- c(parts, list(templates))
      templates <- NULL
    } else if (!is.null(templates) && !methods::is(templates, "ad_pack")) {
      stop("templates must be an ad_pack object", call. = FALSE)
    }

    dots <- list(...)
    if (length(dots) > 0L) {
      if (!all(vapply(dots, function(ptr) is(ptr, "ad_pack_ptr"), logical(1L)))) {
        stop("additional arguments must be ad_pack_ptr", call. = FALSE)
      }
      parts <- c(parts, dots)
    }
    if (length(parts) > 0L) {
      if (verbose) {
        cat("ad_pack: merging ", 1L + length(parts), " raw handle(s)...\n", sep = "")
        utils::flush.console()
      }
      x <- do.call(c, c(list(x), parts))
    }
    new_ad_pack_from_ptr(
      x,
      num_threads = num_threads,
      verbose = verbose,
      reorder_shards = reorder_shards,
      templates = templates
    )
  }
)

#' @rdname ad_pack
#' @export
setMethod(
  "ad_pack",
  signature = c(x = "density_data"),
  function(x, config, num_threads = 1L,
           reorder_shards = c("none", "gradient", "hessian", "third"),
           templates = NULL, ...) {
    if (missing(config) || is.null(config)) {
      stop("config is required for ad_pack(density_data, config)", call. = FALSE)
    }
    if (!is.null(config$num_threads)) {
      num_threads <- config$num_threads
    }
    if (!is.null(config$reorder_shards)) {
      reorder_shards <- config$reorder_shards
    }
    if (identical(as.character(x@ad_kind), "observations") &&
      is.null(config[["obs_groups"]]) &&
      !is.null(config[["obs_units"]])) {
      elgm <- x@elgm_matrix
      n_domain <- if (methods::is(elgm, "Matrix") && ncol(elgm) > 0L) {
        ncol(elgm)
      } else {
        length(x@y)
      }
      config <- ensure_config_obs_groups(
        config,
        A = NULL,
        elgm_matrix = if (methods::is(elgm, "Matrix") && ncol(elgm) > 0L) {
          elgm
        } else {
          NULL
        },
        num_shards = config$num_shards,
        num_threads = num_threads,
        n_domain = n_domain
      )
    }
    ptr <- ad_pack_ptr(x, config)
    verbose <- isTRUE(config[["verbose"]])
    new_ad_pack_from_ptr(
      ptr,
      num_threads = num_threads,
      verbose = verbose,
      reorder_shards = reorder_shards,
      templates = templates,
      density_shards = list(x),
      group_counts = n_groups(ptr)
    )
  }
)

#' @rdname ad_pack
#' @export
setMethod(
  "ad_pack",
  signature = c(x = "list"),
  function(x, config, num_threads = 1L,
           reorder_shards = c("none", "gradient", "hessian", "third"),
           templates = NULL, ...) {
    if (!is_model_data_bundle(x)) {
      stop(
        "list `x` must be a bundle from model_data() with ",
        "`observations`, `random`, and `parameters`",
        call. = FALSE
      )
    }
    if (missing(config) || is.null(config)) {
      stop("config is required for ad_pack(model_data, config)", call. = FALSE)
    }
    if (!is.null(config$num_threads)) {
      num_threads <- config$num_threads
    }
    if (!is.null(config$reorder_shards)) {
      reorder_shards <- config$reorder_shards
    }
    config_build <- fill_config_from_info(config, x$term_data$info)
    config_build <- ensure_config_obs_groups(
      config_build,
      A = x$term_data$A,
      elgm_matrix = x$term_data$elgm_matrix,
      num_shards = config_build$num_shards,
      num_threads = num_threads
    )
    verbose <- isTRUE(config_build[["verbose"]])
    shard_list <- c(x$observations, x$random, x$parameters)
    shard_names <- names(shard_list)
    n_shards <- length(shard_list)
    if (verbose) {
      obs_groups <- if (!is.null(config_build$obs_groups)) {
        ncol(config_build$obs_groups)
      } else {
        NA_integer_
      }
      cat(
        "ad_pack: building ", n_shards, " density shard(s)",
        if (!is.na(obs_groups)) paste0(" (", obs_groups, " observation group(s) per obs shard)") else "",
        "...\n",
        sep = ""
      )
      utils::flush.console()
    }
    ptrs <- vector("list", n_shards)
    # c() clears each source pointer, so count groups before the merge.
    group_counts <- integer(n_shards)
    for (i in seq_len(n_shards)) {
      shard <- shard_list[[i]]
      shard_name <- if (length(shard_names) >= i) shard_names[[i]] else NULL
      if (verbose) {
        cat(
          "  [", i, "/", n_shards, "] ",
          ad_shard_label(shard, shard_name),
          " (CppAD tape",
          if (identical(shard@ad_kind, "observations") && !is.null(config_build$obs_groups)) {
            paste0(", ", ncol(config_build$obs_groups), " groups")
          } else {
            ""
          },
          ")...\n",
          sep = ""
        )
        utils::flush.console()
      }
      ptrs[[i]] <- ad_pack_ptr(shard, config = config_build)
      group_counts[[i]] <- n_groups(ptrs[[i]])
      if (verbose) {
        cat("  [", i, "/", n_shards, "] done (", group_counts[[i]], " AD group(s)).\n", sep = "")
        utils::flush.console()
      }
    }
    if (verbose) {
      cat("ad_pack: merging density handles...\n")
      utils::flush.console()
    }
    new_ad_pack_from_ptr(
      do.call(c, ptrs),
      num_threads = num_threads,
      info = x$term_data$info,
      verbose = verbose,
      reorder_shards = reorder_shards,
      templates = templates,
      density_shards = shard_list,
      group_counts = group_counts
    )
  }
)

#' Rebuild an \code{ad_pack} for a new model-data design
#'
#' Covariate values in observation \code{XTp} are baked into CppAD tapes at
#' build time, so they cannot be mutated in place. This helper retapes a new
#' \code{model_data} bundle and reuses Hessian templates from \code{ad_pack}
#' whenever sizes and shard counts match (identical sparsity across exposure
#' samples). Prefer building a list of packs once for modular inference rather
#' than calling this inside every likelihood evaluation.
#'
#' @param ad_pack Existing \code{ad_pack} whose templates may be reused.
#' @param model_data A \code{model_data()} bundle with the same random-effect
#'   structure and (typically) a different fixed-effect design matrix.
#' @param config Configuration list passed to \code{\link{ad_pack}}.
#' @param num_threads,reorder_shards Forwarded to \code{\link{ad_pack}}.
#'
#' @return A new \code{ad_pack}.
#' @seealso \code{\link{ad_pack}}, \code{\link{replace_vars}}
#' @export
ad_pack_set_X <- function(
  ad_pack,
  model_data,
  config = list(),
  num_threads = 1L,
  reorder_shards = c("none", "gradient", "hessian", "third")
) {
  if (!methods::is(ad_pack, "ad_pack")) {
    stop("ad_pack must be an ad_pack object", call. = FALSE)
  }
  if (!is.list(model_data) ||
    !all(c("observations", "random", "parameters") %in% names(model_data))) {
    stop("model_data must be a model_data() bundle", call. = FALSE)
  }
  ad_pack(
    model_data,
    config = config,
    num_threads = num_threads,
    reorder_shards = reorder_shards,
    templates = ad_pack
  )
}

#' C++ backend entry points
#'
#' Low-level entry points exposed to R. Accept an \code{ad_pack} S4 object
#' (via \code{@ptr}) or a raw \code{ad_pack_ptr}.
#'
#' @param ad_pack An \code{ad_pack} S4 object or \code{ad_pack_ptr} handle.
#' @param x Numeric parameter vector of length \code{Nparams}.
#' @param ad_shards Optional integer vector of 0-based shard indices; \code{NULL} or
#'   \code{integer(0)} evaluates all shards.
#' @param negative Logical (default \code{TRUE}). If \code{TRUE}, return the
#'   **negative** log density \eqn{-\ell(x)} and its derivatives (minimization /
#'   \pkg{trustOptim} sign, consistent with \code{inner_opt()}). If \code{FALSE},
#'   return \eqn{\ell(x)}, \eqn{\nabla\ell}, and \eqn{\nabla^2\ell}.
#' @param inner Logical; if \code{TRUE}, evaluate inner-\eqn{\gamma}
#'   derivatives; if \code{FALSE}, evaluate outer derivatives at the full
#'   parameter vector.
#' @param verbose Logical; if \code{TRUE}, print threads, shards, and CppAD
#'   session phases.
#' @param parameters Numeric vector of length \code{Nbeta + Ntheta}.
#' @param gamma Numeric vector of length \code{Ngamma}.
#' @param LinvPt,LinvPtColumns Sparse factors for \code{trace_hinv_t()}.
#' @param H_inv Optional numeric values of the joint inverse Hessian, in the
#'   nonzero order of \code{chol_inner_list$H_inv}. A sparse matrix contributes
#'   its \code{@x} slot. \code{NULL} keeps every shard on the column walk.
#'
#' @section Sign convention:
#' With default \code{negative = TRUE}, \code{joint_log_dens()}, \code{grad()},
#' and \code{hessian()} match \code{inner_opt()} (negative log-density). Set
#' \code{negative = FALSE} for the joint log density and its derivatives at the
#' same \code{x}.
#'
#' @section Thread affinity:
#' These are **serial debug** APIs. They error if the handle was built with
#' \code{ad_pack(..., num_threads > 1)}. Use a plain \code{ad_pack_ptr} (or
#' \code{ad_pack(..., num_threads = 1)}), or \code{\link{clone_ad_pack_ptr}()}
#' before assigning multi-thread affinity. Parallel evaluation uses
#' \code{\link{fun_obj_fdfh}} / \code{\link{inner_opt}} /
#' \code{\link{log_lik_laplace}} on the affined handle.
#'
#' @name adlaplace_cpp
#' @return See individual functions.
NULL

#' @rdname adlaplace_cpp
#' @export
joint_log_dens <- function(ad_pack, x, ad_shards = NULL, negative = TRUE) {
  ptr <- if (isS4(ad_pack) && methods::.hasSlot(ad_pack, "ptr")) ad_pack@ptr else ad_pack
  .Call(`_adlaplace_joint_log_dens`, ptr, x, ad_shards, negative)
}

#' @rdname adlaplace_cpp
#' @export
grad <- function(ad_pack, x, ad_shards = NULL, inner = FALSE, negative = TRUE) {
  ptr <- if (isS4(ad_pack) && methods::.hasSlot(ad_pack, "ptr")) ad_pack@ptr else ad_pack
  .Call(`_adlaplace_grad`, ptr, x, ad_shards, inner, negative)
}

#' @rdname adlaplace_cpp
#' @export
hessian <- function(ad_pack, x, ad_shards = NULL, inner = FALSE, verbose = FALSE, negative = TRUE) {
  ptr <- if (isS4(ad_pack) && methods::.hasSlot(ad_pack, "ptr")) ad_pack@ptr else ad_pack
  .Call(`_adlaplace_hessian`, ptr, x, ad_shards, inner, verbose, negative)
}

#' @rdname adlaplace_cpp
#' @export
trace_hinv_t <- function(ad_pack, x, LinvPt, LinvPtColumns, verbose = FALSE,
                         H_inv = NULL) {
  ptr <- if (isS4(ad_pack) && methods::.hasSlot(ad_pack, "ptr")) ad_pack@ptr else ad_pack
  H_inv_x <- NULL
  if (!is.null(H_inv)) {
    H_inv_x <- if (is.numeric(H_inv) && is.null(dim(H_inv))) {
      as.numeric(H_inv)
    } else {
      as.numeric(methods::as(methods::as(H_inv, "generalMatrix"), "CsparseMatrix")@x)
    }
  }
  .Call("_adlaplace_trace_hinv_t", ptr, x, LinvPt, LinvPtColumns, verbose, H_inv_x)
}

#' @rdname adlaplace_cpp
#' @export
fun_obj_fdfh <- function(ad_pack, parameters, gamma, inner = TRUE, verbose = FALSE) {
  .Call(`_adlaplace_fun_obj_fdfh`, parameters, gamma, ad_pack, inner, verbose)
}
