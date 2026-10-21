#include <Rcpp.h>

#include <Rinternals.h>
#include "adlaplace/register.hpp"
#include "adlaplace/runtime.hpp"

//' Build raw AD handle for observation shards only
//'
//' @param model An \code{density_data} S4 object.
//' @param config Model configuration list.
//' @param name Registered observation density name (e.g. \code{"nbinom_obs"}).
//' @return External pointer of class \code{ad_pack_ptr}.
//' @keywords internal
// [[Rcpp::export]]
SEXP get_ad_pack_raw_obs(SEXP model, Rcpp::List config, std::string name) {
  ad_pack* groups = get_ad_pack_raw_obs_h(model, config, name);
  return make_ad_pack_ptr(groups);
}

//' Build raw AD handle for a parameters shard
//'
//' @param model An \code{density_data} S4 object.
//' @param config Model configuration list.
//' @param name Registered parameters density name (e.g. \code{"nbinom_extra"}).
//' @return External pointer of class \code{ad_pack_ptr}.
//' @keywords internal
// [[Rcpp::export]]
SEXP get_ad_pack_raw_parameters(SEXP model, Rcpp::List config, std::string name) {
  ad_pack* groups = get_ad_pack_raw_parameters_h(model, config, name);
  return make_ad_pack_ptr(groups);
}

//' Merge partial AD handles into one raw handle
//'
//' Concatenates shards from \code{\link{ad_pack_ptr}} (or other partial
//' builders) in list order. Does not attach \code{hessian_map}; use
//' \code{\link{ad_pack}} when templates are needed.
//'
//' @param handles List of external pointers (\code{ad_pack_ptr}).
//' @return Combined external pointer of class \code{ad_pack_ptr}.
//' @seealso \code{\link{ad_pack_ptr}}, \code{\link{ad_pack}}
//' @keywords internal
// [[Rcpp::export(name = c_ad_pack_ptr)]]
SEXP c_ad_pack_ptr(Rcpp::List handles) {
  std::vector<ad_pack*> parts;
  parts.reserve(handles.size());
  for (int i = 0; i < handles.size(); ++i) {
    SEXP h = handles[i];
    if (TYPEOF(h) != EXTPTRSXP) {
      Rcpp::stop("handles[[%d]] must be an external pointer", i + 1);
    }
    ad_pack* g = static_cast<ad_pack*>(R_ExternalPtrAddr(h));
    if (!g) {
      Rcpp::stop("handles[[%d]] external pointer is NULL", i + 1);
    }
    parts.push_back(g);
    R_ClearExternalPtr(h);
  }
  ad_pack* merged = combine_ad_fun(parts);
  return make_ad_pack_ptr(merged);
}

//' Attach hessian_map() result to an ad_pack handle
//'
//' Copies outer/inner templates and maps into the C++ \code{ad_pack} handle.
//'
//' @param handle External pointer of class \code{ad_pack_ptr}.
//' @param hessian_pack List returned by \code{hessian_map()}.
//' @keywords internal
// [[Rcpp::export]]
void adlaplace_attach_hessian(SEXP handle, Rcpp::List hessian_pack) {
  ad_pack* groups = ad_fun_from_handle(handle);
  ad_fun_attach_hessians_from_list(*groups, hessian_pack);
}

//' Number of AD shards in an \code{ad_pack_ptr} handle
//'
//' @param handle External pointer of class \code{ad_pack_ptr}.
//' @return Integer count of groups (shards).
//' @export
// [[Rcpp::export]]
int n_groups(SEXP handle) {
  ad_pack* groups = ad_fun_from_handle(handle);
  return static_cast<int>(groups->fun.size());
}

//' Sparse structure sizes for one AD shard
//'
//' Layout and sparsity sizes for one AD shard
//'
//' @param handle External pointer of class \code{ad_pack_ptr}.
//' @param group 0-based group index.
//' @return List with \code{n_inner}, \code{n_outer}, \code{n_beta}, \code{n_theta},
//'   and \code{nnz_grad_*}, \code{nnz_hes_*}.
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List get_sizes(SEXP handle, int group) {
  ad_pack* groups = ad_fun_from_handle(handle);
  ad_shard* shard = shard_handle(groups, static_cast<size_t>(group));

  int n_inner = 0;
  int n_outer = 0;
  int n_beta = 0;
  int n_theta = 0;
  int nnz_grad_inner = 0;
  int nnz_grad_outer = 0;
  int nnz_hes_inner = 0;
  int nnz_hes_outer = 0;
  if (shard->get_sizes(
      &n_inner, &n_outer, &n_beta, &n_theta,
      &nnz_grad_inner, &nnz_grad_outer,
      &nnz_hes_inner, &nnz_hes_outer) != 0) {
    Rcpp::stop("backend api->get_sizes failed for group %d", group);
  }

  return Rcpp::List::create(
    Rcpp::Named("n_inner") = n_inner,
    Rcpp::Named("n_outer") = n_outer,
    Rcpp::Named("n_beta") = n_beta,
    Rcpp::Named("n_theta") = n_theta,
    Rcpp::Named("nnz_grad_inner") = nnz_grad_inner,
    Rcpp::Named("nnz_grad_outer") = nnz_grad_outer,
    Rcpp::Named("nnz_hes_inner") = nnz_hes_inner,
    Rcpp::Named("nnz_hes_outer") = nnz_hes_outer
  );
}

//' CppAD tape sizes for one AD shard
//'
//' @param handle External pointer of class \code{ad_pack_ptr}.
//' @param group 0-based group index.
//' @return List with \code{domain}, \code{n_global}, \code{size_op},
//'   \code{size_var}, and pattern nnz counts. \code{domain} /
//'   \code{size_op} / \code{size_var} are zero when the shard never
//'   recorded a tape (e.g. analytic FEM ssq with \code{fem_analytic = TRUE});
//'   \code{n_global} and the nnz fields remain valid.
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List get_tape_sizes(SEXP handle, int group) {
  ad_pack* groups = ad_fun_from_handle(handle);
  ad_shard* shard = shard_handle(groups, static_cast<size_t>(group));
  AdTape& gp = shard->pack;
  return Rcpp::List::create(
    Rcpp::Named("domain") = static_cast<int>(gp.fun.Domain()),
    Rcpp::Named("n_global") = static_cast<int>(
      gp.n_global > 0 ? gp.n_global : gp.fun.Domain()),
    Rcpp::Named("size_op") = static_cast<int>(gp.fun.size_op()),
    Rcpp::Named("size_var") = static_cast<int>(gp.fun.size_var()),
    Rcpp::Named("nnz_grad") = static_cast<int>(gp.pattern_grad.nnz()),
    Rcpp::Named("nnz_grad_inner") = static_cast<int>(gp.pattern_grad_inner.nnz()),
    Rcpp::Named("nnz_hes") = static_cast<int>(gp.pattern_hessian.nnz()),
    Rcpp::Named("nnz_hes_inner") = static_cast<int>(gp.pattern_hessian_inner.nnz())
  );
}

//' Sparse index patterns for one AD shard
//'
//' @param handle External pointer of class \code{ad_pack_ptr}.
//' @param group 0-based group index.
//' @return List with \code{grad}, \code{grad_inner}, \code{row_hess}, \code{col_hess}, etc.
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List get_sparse_pattern(SEXP handle, int group) {
  ad_pack* groups = ad_fun_from_handle(handle);
  return sparsity_shard_from_handle(shard_handle(groups, static_cast<size_t>(group)));
}

//' Build-time owner thread for one AD shard
//'
//' @param handle External pointer of class \code{ad_pack_ptr}.
//' @param group 0-based group index.
//' @return Integer owner thread id recorded when the shard was prewarmed.
//' @keywords internal
// [[Rcpp::export]]
int get_thread_owner(SEXP handle, int group) {
  ad_pack* groups = ad_fun_from_handle(handle);
  ad_shard* shard = shard_handle(groups, static_cast<size_t>(group));
  return static_cast<int>(shard->pack.owner_thread);
}

//' Configured \code{num_threads} from \code{ad_pack()} thread assignment
//'
//' @param handle External pointer of class \code{ad_pack_ptr}.
//' @return Integer \code{num_threads} if \code{ad_pack()} assigned threads; otherwise \code{NA}.
//' @keywords internal
// [[Rcpp::export]]
Rcpp::IntegerVector get_configured_num_threads(SEXP handle) {
  ad_pack* groups = ad_fun_from_handle(handle);
  if (!groups->num_threads_configured) {
    return Rcpp::IntegerVector::create(NA_INTEGER);
  }
  return Rcpp::IntegerVector::create(
    static_cast<int>(groups->configured_num_threads)
  );
}

//' Whether a shard has an assigned OpenMP owner thread
//'
//' @param handle External pointer of class \code{ad_pack_ptr}.
//' @param group 0-based group index.
//' @return Logical scalar.
//' @keywords internal
// [[Rcpp::export]]
bool get_owner_thread_assigned(SEXP handle, int group) {
  ad_pack* groups = ad_fun_from_handle(handle);
  ad_shard* shard = shard_handle(groups, static_cast<size_t>(group));
  return shard->pack.owner_thread_assigned;
}

//' Assign OpenMP owner threads to all shards
//'
//' By default sets \code{owner_thread = shard_index \% num_threads}. When
//' \code{owners} is supplied (length \code{n_shards}, 0-based thread ids in
//' \code{[0, num_threads)}), those values are used instead (e.g. LPT balancing
//' from \code{ad_pack(..., reorder_shards=)}).
//' Called from \code{ad_pack()} after templates are attached. Required before
//' parallel \code{inner_opt} / \code{trace_hinv_t}.
//'
//' @param handle External pointer of class \code{ad_pack_ptr}.
//' @param num_threads Positive integer thread count.
//' @param owners Optional integer vector of length \code{n_shards} with
//'   0-based owner thread ids; \code{NULL} uses modulo assignment.
//' @keywords internal
// [[Rcpp::export]]
void assign_owner_threads(
  SEXP handle,
  int num_threads,
  Rcpp::Nullable<Rcpp::IntegerVector> owners = R_NilValue
) {
  if (num_threads < 1) {
    Rcpp::stop("num_threads must be a positive integer");
  }
  // Owners follow the requested num_threads. Windows anti-shrink of the
  // OpenMP/CppAD team is handled at eval by padding thread groups up to the
  // process high-water mark (see thread_groups_for_parallel_eval).
  ad_pack* groups = ad_fun_from_handle(handle);
  const std::size_t n_threads = static_cast<std::size_t>(num_threads);
  const std::size_t n_shards = groups->fun.size();

  if (owners.isNotNull()) {
    Rcpp::IntegerVector own = owners.get();
    if (static_cast<std::size_t>(own.size()) != n_shards) {
      Rcpp::stop(
        "owners must have length n_shards (%d), got %d",
        static_cast<int>(n_shards),
        own.size()
      );
    }
    for (std::size_t s = 0; s < n_shards; ++s) {
      const int t = own[static_cast<R_xlen_t>(s)];
      if (t == NA_INTEGER || t < 0 || static_cast<std::size_t>(t) >= n_threads) {
        Rcpp::stop(
          "owners[%d] = %d is outside [0, num_threads)",
          static_cast<int>(s),
          t
        );
      }
      ad_shard* shard = shard_handle(groups, s);
      shard->pack.owner_thread = static_cast<std::size_t>(t);
      shard->pack.owner_thread_assigned = true;
    }
  } else {
    for (std::size_t s = 0; s < n_shards; ++s) {
      ad_shard* shard = shard_handle(groups, s);
      shard->pack.owner_thread = shard->pack.shard_index % n_threads;
      shard->pack.owner_thread_assigned = true;
    }
  }
  groups->configured_num_threads = n_threads;
  groups->num_threads_configured = true;
}

//' Deep copy of an \code{ad_pack_ptr} handle
//'
//' Clones CppAD tapes and sparsity patterns into a new external pointer.
//' The source handle is unchanged (unlike \code{c()} on \code{ad_pack_ptr}, which moves
//' shards and clears sources).
//'
//' @param handle External pointer of class \code{ad_pack_ptr}.
//' @return New \code{ad_pack_ptr} with independent C++ state.
//' @keywords internal
// [[Rcpp::export(name = "clone_ad_pack_ptr_")]]
SEXP clone_ad_pack_ptr_impl(SEXP handle) {
  ad_pack* src = ad_fun_from_handle(handle);
  ad_pack* copy = clone_ad_pack(src);
  return make_ad_pack_ptr(copy);
}
