#' Assemble model terms and per-shard \code{density_data} objects
#'
#' Parses a formula, builds design matrices, and returns formula terms plus
#' \code{density_data} shards for the observation density, parameter densities, and
#' each random-effect term.
#'
#' @param formula Model formula with constructor terms (e.g. \code{iwp()}, \code{iid()}).
#' @param data Data frame containing variables referenced in \code{formula}.
#' @param verbose Print extra information while parsing terms.
#' @param na_omit When \code{TRUE} (default), drop rows with \code{NA} in
#'   covariates, outcome, stratification (\code{by}), or random-slope
#'   \code{mult} variables; impute remaining outcome \code{NA}s as zero; and
#'   sort by observation-term \code{by} columns when present.
#' @return A list with:
#' \describe{
#'   \item{\code{terms}}{Named list of term objects from \code{\link{collect_terms}}.}
#'   \item{\code{data}}{Output of \code{\link{term_data_setup}}, including \code{elgm_matrix}
#'     when an observation term defines an \code{elgm_matrix} method.}
#'   \item{\code{observations}}{Named list of observation \code{density_data} shards.}
#'   \item{\code{parameters}}{Named list of parameter \code{density_data} shards.}
#'   \item{\code{random}}{Named list of random-effect \code{density_data} shards.}
#' }
#' @export
#' @examples
#' set.seed(1)
#' dat <- data.frame(
#'   y = rpois(30, 2),
#'   x1 = rnorm(30),
#'   x2 = runif(30)
#' )
#' md <- model_data(
#'   y ~ x1 + iwp(x2, p = 2, knots = seq(0, 1, length.out = 5)),
#'   data = dat
#' )
#' names(md$terms)
model_data <- function(formula, data, verbose = FALSE, na_omit = TRUE) {
  formula_in <- formula
  the_terms <- parse_model_terms(formula, verbose = verbose)

  if (na_omit) {
    data <- prepare_model_rows(data, the_terms, verbose = verbose)
  }

  elgm_mats <- collect_elgm_matrices(the_terms, data)
  all_data <- term_data_setup(formula = formula_in, data = data, verbose = verbose)
  the_terms <- all_data$terms

  if (length(elgm_mats) == 1L) {
    all_data$elgm_matrix <- elgm_mats[[1L]]
  }

  shard_counts <- count_model_shards(all_data)
  observations <- list()
  random <- list()
  random_parameters <- list()
  prior_parameters <- list()

  for (term_here in the_terms) {
    if (!methods::is(term_here, "model_term")) {
      warning(
        "term does not inherit from class 'model_term'; skipping",
        call. = FALSE
      )
      next
    }

    elgm_here <- elgm_mats[[term_here@name]]
    kind <- term_here@ad_kind

    if (identical(kind, "observations")) {
      observations[[term_here@name]] <- build_observation_shard(
        term_here,
        all_data,
        shard_counts,
        elgm_here
      )
    }

    if (identical(kind, "random")) {
      random_shards <- build_random_shards(
        term_here,
        all_data,
        shard_counts,
        random
      )
      if (!is.null(random_shards$random)) {
        random[[random_shards$name]] <- random_shards$random
      }
      if (!is.null(random_shards$parameter)) {
        random_parameters[[random_shards$parameter_name]] <- random_shards$parameter
      }
    }

    if (identical(kind, "parameters")) {
      prior_parameters[[term_here@label]] <- build_prior_shard(
        term_here,
        all_data,
        shard_counts
      )
    }
  }

  parameters <- build_parameter_shards(observations, random_parameters)
  if (length(prior_parameters) > 0L) {
    parameters <- c(parameters, prior_parameters)
  }

  list(
    term_data = all_data,
    observations = observations,
    parameters = parameters,
    random = random,
    terms = the_terms
  )
}

#' @keywords internal
collect_elgm_matrices <- function(terms, data) {
  obs_terms <- Filter(function(t) {
    methods::is(t, "model_term") && identical(t@ad_kind, "observations")
  }, terms)

  elgm_mats <- list()
  for (term in obs_terms) {
    term_class <- class(term)[1L]
    if (methods::hasMethod("elgm_matrix", term_class)) {
      elgm_mats[[term@name]] <- elgm_matrix(term, data)
    }
  }

  if (length(elgm_mats) > 1L) {
    warning(
      "multiple observation terms with elgm_matrix methods; ",
      "using the first for data$elgm_matrix",
      call. = FALSE
    )
  }
  elgm_mats
}

#' @keywords internal
count_model_shards <- function(all_data) {
  list(
    beta = max(c(0, nrow(all_data$info$beta))),
    gamma = max(c(0, nrow(all_data$info$gamma))),
    theta = max(c(0, nrow(all_data$info$theta)))
  )
}

#' @keywords internal
beta_gamma_maps <- function(all_data, counts) {
  beta_map <- if (counts$beta > 0L && ncol(all_data$X) > 0L) {
    list(
      match(colnames(all_data$X), all_data$info$beta$beta_label),
      as.integer(counts$beta)
    )
  } else {
    counts$beta
  }
  gamma_map <- if (counts$gamma > 0L && ncol(all_data$A) > 0L) {
    list(
      match(colnames(all_data$A), all_data$info$gamma$gamma_label),
      as.integer(counts$gamma)
    )
  } else {
    counts$gamma
  }
  list(beta_map = beta_map, gamma_map = gamma_map)
}

#' @keywords internal
observation_weights <- function(term_here, data) {
  if (methods::is(term_here, "binomial") &&
      length(term_here@size) == 1L &&
      nzchar(term_here@size)) {
    size_col <- term_here@size
    if (!size_col %in% names(data)) {
      stop(
        "binomial size column '", size_col, "' not found in data",
        call. = FALSE
      )
    }
    return(as.numeric(data[[size_col]]))
  }
  numeric(0)
}

#' @keywords internal
theta_map_for_term <- function(term_label, theta_info, n_theta) {
  list(
    grep(term_label, theta_info$label),
    as.integer(n_theta)
  )
}

#' @keywords internal
build_observation_shard <- function(term_here, all_data, counts, elgm_here) {
  maps <- beta_gamma_maps(all_data, counts)
  density_data(
    y = all_data$y,
    A = all_data$A,
    X = all_data$X,
    beta_map = maps$beta_map,
    gamma_map = maps$gamma_map,
    theta_map = theta_map_for_term(
      term_here@label,
      all_data$info$theta,
      counts$theta
    ),
    elgm_matrix = elgm_here,
    density = term_here@density,
    ad_kind = "observations",
    package = term_here@package,
    weights = observation_weights(term_here, all_data$data)
  )
}

#' @keywords internal
build_random_shards <- function(term_here, all_data, counts, random) {
  prec_mat <- precision(term_here, all_data$data)
  if (is.null(prec_mat)) {
    return(list(random = NULL, parameter = NULL))
  }

  random_name <- term_here@label
  if (random_name %in% names(random)) {
    stop(
      "duplicate random-term label '", random_name, "'; ",
      "each random term needs a unique @label",
      call. = FALSE
    )
  }

  prec_payload <- if (is.list(prec_mat)) {
    prec_mat
  } else {
    as.numeric(Matrix::diag(prec_mat))
  }
  theta_map_here <- theta_map_for_term(
    term_here@label,
    all_data$info$theta,
    counts$theta
  )

  random_shard <- density_data(
    beta_map = counts$beta,
    gamma_map = list(
      which(all_data$info$gamma$label == term_here@label),
      counts$gamma
    ),
    theta_map = theta_map_here,
    density = term_here@density,
    ad_kind = "random",
    package = term_here@package,
    precision = prec_payload
  )

  extra_name <- extra_density(term_here)
  parameter_shard <- NULL
  parameter_name <- NULL
  if (is.character(extra_name) && length(extra_name) == 1L && nzchar(extra_name)) {
    parameter_name <- paste0(random_name, "_det")
    parameter_shard <- density_data(
      beta_map = counts$beta,
      gamma_map = counts$gamma,
      theta_map = theta_map_here,
      density = extra_name,
      ad_kind = "parameters",
      package = term_here@package,
      precision = prec_payload
    )
  }

  list(
    name = random_name,
    random = random_shard,
    parameter_name = parameter_name,
    parameter = parameter_shard
  )
}

#' @keywords internal
build_parameter_shards <- function(observations, random_parameters) {
  parameters <- Filter(
    Negate(is.null),
    lapply(
      observations,
      function(obs_shard) {
        if (ncol(obs_shard@theta_map) == 0L) {
          return(NULL)
        }
        density_data(
          y = obs_shard@y,
          theta_map = obs_shard@theta_map,
          beta_map = nrow(obs_shard@beta_map),
          gamma_map = nrow(obs_shard@gamma_map),
          elgm_matrix = obs_shard@elgm_matrix,
          density = parameter_density(obs_shard@density),
          ad_kind = "parameters",
          package = obs_shard@package,
          weights = obs_shard@weights
        )
      }
    )
  )
  if (length(parameters) > 0L) {
    nm <- names(parameters)
    if (is.null(nm)) {
      nm <- rep("", length(parameters))
    }
    names(parameters) <- paste0(nm, "_extra")
  }
  if (length(random_parameters) > 0L) {
    parameters <- c(parameters, random_parameters)
  }
  parameters
}
