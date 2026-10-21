#' Valid model type factor levels
#'
#' Character vector of allowed levels for the \code{model_role} slot on
#' \code{model_term} objects.
#'
#' @return A character vector of length 3, with elements \code{"fixed"},
#'   \code{"random"}, and \code{"response"} in that order. These are the
#'   allowed levels of the \code{model_role} factor on \code{model_term}
#'   objects, indicating a fixed effect, a random effect, or the response.
#' @export
.model_role_levels <- c("fixed", "random", "response")

NULL

#' Per-shard layout for AD density evaluation
#'
#' @slot y Response vector.
#' @slot ATp Transpose of random-effects design matrix.
#' @slot XTp Transpose of fixed-effects design matrix.
#' @slot beta_map Beta parameter map (\code{ngCMatrix}). \code{nrow} is the
#'   global beta block size; column \code{j} has one structural nonzero at row
#'   \code{r} meaning local design column \code{j} of \code{X} uses global beta
#'   \code{r}. When \code{ncol(X) > 0}, require \code{ncol(beta_map) == ncol(X)}.
#' @slot gamma_map Gamma parameter map (\code{ngCMatrix}); at most one structural
#'   nonzero per row and column. Same column convention as \code{beta_map} for
#'   local \code{A} columns into the global gamma block.
#' @slot theta_map Theta parameter map (\code{ngCMatrix}). Column \code{j}
#'   selects global theta row \code{r} for this shard's theta component \code{j}.
#' @slot elgm_matrix Optional exposure-lag map (\code{ngCMatrix}; empty by default).
#' @slot density Registered AD density name for this shard.
#' @slot ad_kind Shard kind (\code{"observations"}, \code{"parameters"}, \code{"random"}).
#' @slot package Package name whose shared library records tapes for this shard
#'   (defaults to \code{"adlaplace"} when missing).
#' @slot precision Optional precision payload (any R object).
#' @slot weights Optional per-observation weights (e.g. binomial trial counts).
#'   Empty means all ones.
#' @importClassesFrom Matrix Matrix ngCMatrix
#' @exportClass density_data
setClass(
  "density_data",
  slots = c(
    y = "numeric",
    ATp = "Matrix",
    XTp = "Matrix",
    beta_map = "ngCMatrix",
    gamma_map = "ngCMatrix",
    theta_map = "ngCMatrix",
    elgm_matrix = "ngCMatrix",
    density = "character",
    ad_kind = "character",
    package = "character",
    precision = "ANY",
    weights = "numeric"
  ),
  prototype = prototype(
    weights = numeric(0)
  )
)

#' Raw AD handle (external pointer)
#'
#' S4 class for a C++ AD backend handle. Objects are created by
#' \code{\link{ad_pack_ptr}}, combined with \code{\link{c.ad_pack_ptr}}, and
#' typically passed to \code{\link{ad_pack}} to attach Hessian templates.
#'
#' @name ad_pack_ptr-class
#' @aliases ad_pack_ptr-class
#' @exportClass ad_pack_ptr
#' @seealso \code{\link{ad_pack_ptr}}, \code{\link{ad_pack}},
#'   \code{\link{clone_ad_pack_ptr}}
setClass("ad_pack_ptr", contains = "externalptr")

#' AD function with Hessian templates attached
#'
#' @slot ptr Raw combined handle (\code{ad_pack_ptr}).
#' @slot group_sparsity Per-group inner gradient sparsity patterns.
#' @slot outer Outer Hessian template (\code{dgCMatrix}).
#' @slot inner Inner-gamma Hessian template (\code{dgCMatrix}).
#' @slot map_outer Outer Hessian shard map.
#' @slot map_inner Inner Hessian shard map.
#' @slot parallel_map Thread-affinity map (\code{ngCMatrix}); rows are shards,
#'   columns are threads, with one nonzero per shard row.
#' @slot chol_inner Symbolic LDL factor or empty sparse matrix.
#' @slot chol_inner_list Numeric LDL list for C++ (\code{L1}, \code{Linv},
#'   \code{perm}, \code{perm_inv}, \code{half_H_inv}, \code{H_inv}), plus
#'   \code{trace_columns} (per-shard column indices for \code{trace_hinv_t}) and
#'   \code{hinv_q_index} (per-shard slots of \eqn{Q} into \code{H_inv}, when a
#'   shard defines \code{\link{hinv_trace_index}}).
#' @slot sizes Named numeric vector \code{beta}/\code{gamma}/\code{theta}.
#' @slot info List of parameter metadata (\code{beta}, \code{gamma},
#'   \code{theta}, \code{parameters}); populated from \code{model_data()$term_data$info}
#'   when the handle is built from a model-data bundle, otherwise empty.
#' @importClassesFrom Matrix dgCMatrix ngCMatrix
#' @exportClass ad_pack
setClass(
  "ad_pack",
  slots = c(
    ptr = "ad_pack_ptr",
    group_sparsity = "list",
    outer = "dgCMatrix",
    inner = "dgCMatrix",
    map_outer = "list",
    map_inner = "list",
    parallel_map = "ngCMatrix",
    chol_inner = "ANY",
    chol_inner_list = "list",
    sizes = "numeric",
    info = "list"
  ),
  prototype = list(
    info = list()
  )
)

#' By-Group Class
#'
#' @description A class for grouping information in hierarchical model terms.
#' Contains the grouping variable name and its levels/labels.
#' @slot term Character vector of length 1, the grouping variable name
#' @slot levels Character vector of unique levels for the grouping variable
#' @slot labels Character vector of formatted labels for the levels
#' @exportClass by_group
setClass(
  "by_group",
  slots = list(
    term = "character",
    levels = "character",
    labels = "character"
  ),
  prototype = prototype(
    term = character(0),
    levels = character(0),
    labels = character(0)
  )
)

#' Base Model Class
#'
#' @description The base S4 class that all specific model term classes inherit from.
#' Subclasses that need hierarchical grouping should include a \code{by} slot
#' (either \code{character} or \code{by_group} type) and optionally \code{by_levels}
#' and \code{by_labels} slots.
#' @slot name Character vector of length 1, the term name
#' @slot label Character scalar term label reused across metadata builders.
#' @slot formula Formula object for the term
#' @slot knots Numeric vector of knot locations (for spline terms)
#' @slot ref_value Numeric reference value for centering
#' @slot p.order Integer polynomial order
#' @slot init Numeric vector of initial values
#' @slot lower Numeric vector of lower bounds
#' @slot upper Numeric vector of upper bounds
#' @slot parscale Numeric vector of parameter scales for optimization
#' @slot log Logical; optimize/tape each theta on the log scale (recycled to
#'   the number of thetas for multi-parameter terms).
#' @slot model_role Factor indicating term role ("fixed", "random", or "response")
#' @slot density Registered AD density name for \code{ad_pack_ptr()}, or \code{NA}
#'   when the term does not map to a density shard.
#' @slot ad_kind Shard kind passed to \code{ad_pack_ptr()} (\code{"observations"},
#'   \code{"parameters"}, \code{"random"}, or \code{NA}).
#' @slot package Package whose \code{.so} records AD tapes for this term
#'   (default \code{"adlaplace"} for built-in densities).
#' @exportClass model_term
setClass("model_term",
  slots = list(
    name = "character",
    label = "character",
    formula = "formula",
    knots = "numeric",
    ref_value = "numeric",
    p.order = "integer",
    init = "numeric",
    lower = "numeric",
    upper = "numeric",
    parscale = "numeric",
    log = "logical",
    model_role = "factor",
    density = "character",
    ad_kind = "character",
    package = "character"
  ),
  prototype = prototype(
    name = character(0),
    label = character(0),
    formula = formula(),
    knots = numeric(0),
    ref_value = numeric(0),
    p.order = as.integer(1),
    init = numeric(0),
    lower = numeric(0),
    upper = numeric(0),
    parscale = numeric(0),
    log = TRUE,
    model_role = factor("fixed", levels = .model_role_levels),
    density = NA_character_,
    ad_kind = NA_character_,
    package = "adlaplace"
  )
)

#' Model Term Generics
#'
#' @description Generic functions for extracting model matrices and parameter information from model terms.
#' @param term A model term object.
#' @param data A data frame containing the variables.
#' @return Method-specific return values.
#' @name model-generics
NULL

#' @rdname model-generics
#' @export
setGeneric("design", function(term, data) standardGeneric("design"))
#' @rdname model-generics
#' @export
setGeneric("precision", function(term, data) standardGeneric("precision"))
#' @rdname model-generics
#' @export
setGeneric("theta_info", function(term) standardGeneric("theta_info"))
#' @rdname model-generics
#' @export
setGeneric("beta_info", function(term, data) standardGeneric("beta_info"))
#' @rdname model-generics
#' @export
setGeneric("random_info", function(term, data) standardGeneric("random_info"))
#' @rdname model-generics
#' @export
setGeneric("elgm_matrix", function(term, data) standardGeneric("elgm_matrix"))

#' @rdname model-generics
#' @export
setMethod("design", "model_term", function(term, data) {
  NULL
})

#' @rdname model-generics
#' @export
setMethod("precision", "model_term", function(term, data) {
  NULL
})

#' @rdname model-generics
#' @export
setMethod("theta_info", "model_term", function(term) {
  NULL
})

#' @rdname model-generics
#' @export
setMethod("beta_info", "model_term", function(term, data) {
  NULL
})

#' @rdname model-generics
#' @export
setMethod("random_info", "model_term", function(term, data) {
  NULL
})

#' Optional companion parameters density name for a random term
#'
#' When non-\code{NULL}, \code{\link{model_data}} emits an extra
#' \code{ad_kind = "parameters"} shard (e.g. FEM log-determinant) alongside the
#' random shard. Default is \code{NULL} (no companion).
#'
#' @param term A model term object.
#' @return Character density name, or \code{NULL}.
#' @rdname model-generics
#' @export
setGeneric("extra_density", function(term) standardGeneric("extra_density"))

#' @rdname model-generics
#' @export
setMethod("extra_density", "model_term", function(term) {
  NULL
})

#' By-Group Classes and Functions
#'
#' @description Functions for creating and managing by_group objects for hierarchical
#' model grouping information.
#'
#' @name by_group
#' @rdname by_group
#' @export
NULL

#' @rdname by_group
#' @param term Character, the grouping variable name
#' @param data A data frame. If provided and levels is missing, unique values are
#'   extracted from data[[term]]
#' @param levels Character vector of unique levels for the grouping variable.
#'   If missing and data is provided, unique values are extracted from data[[term]]
#' @return A by_group object
#' @examples
#' # Create with explicit levels
#' bg <- by_group(term = "group", levels = c("A", "B", "C"))
#'
#' # Create from data
#' dat <- data.frame(group = c("A", "B", "A", "C"))
#' bg <- by_group(term = "group", data = dat)
by_group <- function(term, data = NULL, levels = NULL) {
  if (missing(term)) {
    # Return empty by_group object for use in prototypes
    return(methods::new("by_group", term = character(0), levels = character(0), labels = character(0)))
  }

  if (is.null(levels)) {
    if (is.null(data)) {
      return(methods::new("by_group",
        term = term,
        levels = character(0),
        labels = character(0)
      ))
    }
    if (!(term %in% names(data))) {
      stop("Grouping variable '", term, "' not found in data")
    }
    unique_values <- unique(data[[term]])
    is_numeric <- is.numeric(unique_values)
    levels <- as.character(unique_values)
  } else {
    is_numeric <- is.numeric(levels)
  }

  # Construct labels from levels with formatC for numeric
  if (is_numeric) {
    labels <- formatC(
      as.numeric(levels),
      width = ceiling(log10(max(abs(as.numeric(levels))))),
      flag = "0"
    )
  } else {
    labels <- as.character(levels)
  }

  methods::new("by_group",
    term = term,
    levels = levels,
    labels = labels
  )
}

#' Updates the by_group levels in a model term from data.
#'
#' If the term has no by slot, returns it unchanged.
#' If levels are empty, they are populated from data.
#' If levels exist, new values in data are added with a warning.
#'
#' @param term A model term object (inherited from model class)
#' @param data A data frame containing the grouping variable
#' @return The term with updated by_group levels
#' @keywords internal
#' @noRd
add_by_levels <- function(term, data) {
  if (!"by" %in% methods::slotNames(term)) {
    return(term)
  }

  if (!inherits(term@by, "by_group")) {
    return(term)
  }

  by_var <- term@by@term
  if (length(by_var) == 0 || !is.character(by_var)) {
    return(term)
  }

  if (!(by_var %in% names(data))) {
    stop("Grouping variable '", by_var, "' not found in data")
  }

  data_levels <- as.character(unique(data[[by_var]]))

  if (length(term@by@levels) == 0) {
    # Create new by_group with levels from data
    unique_values <- unique(data[[by_var]])
    is_numeric <- is.numeric(unique_values)
    levels <- as.character(unique_values)
    if (is_numeric) {
      labels <- formatC(
        as.numeric(levels),
        width = ceiling(log10(max(abs(as.numeric(levels))))),
        flag = "0"
      )
    } else {
      labels <- as.character(levels)
    }
    term@by <- methods::new("by_group",
      term = term@by@term,
      levels = levels,
      labels = labels
    )
  } else {
    # Check if all data levels are in existing levels
    missing_levels <- setdiff(data_levels, term@by@levels)
    if (length(missing_levels) > 0) {
      warning("Adding new levels to by_group: ", paste(missing_levels, collapse = ", "))
      # Determine if existing levels are numeric
      is_numeric <- is.numeric(term@by@levels)
      new_levels <- c(term@by@levels, missing_levels)
      if (is_numeric) {
        new_labels <- formatC(
          as.numeric(new_levels),
          width = ceiling(log10(max(abs(as.numeric(new_levels))))),
          flag = "0"
        )
      } else {
        new_labels <- as.character(new_levels)
      }
      term@by <- methods::new("by_group",
        term = term@by@term,
        levels = new_levels,
        labels = new_labels
      )
    }
  }

  term
}
