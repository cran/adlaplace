#include "adlaplace/runtime.hpp"
#include "adlaplace/register.hpp"
#include "adlaplace/chol_update.hpp"
#include "adlaplace/omp_compat.hpp"
#include "adlaplace/ompad.hpp"

void adlaplace_release_shard_eval_buffers(ad_pack& backend) {
  if (omp_in_parallel() != 0) {
    Rcpp::stop(
        "adlaplace_release_shard_eval_buffers: must not run inside an OpenMP "
        "parallel region");
  }
  for (std::size_t s = 0; s < backend.fun.size(); ++s) {
    shard_handle(&backend, s)->release_eval_buffers();
  }
}

ad_pack* clone_ad_pack(const ad_pack* src) {
  if (!src) {
    Rcpp::stop("clone_ad_pack: source handle is NULL");
  }
  if (src->fun.empty()) {
    Rcpp::stop("clone_ad_pack: source has no AD shards");
  }

  ad_pack* copy = new ad_pack();
  copy->abi_version = src->abi_version;
  copy->fun.reserve(src->fun.size());
  for (ad_shard* shard : src->fun) {
    if (!shard) {
      continue;
    }
    // Virtual clone keeps the dynamic type (e.g. FemSsqShard). Reconstructing
    // from AdTape via ShardFactory cannot carry subclass payload.
    ad_shard* cloned = shard->clone();
    cloned->pack.owner_thread = 0;
    cloned->pack.owner_thread_assigned = false;
    copy->fun.push_back(cloned);
  }
  if (copy->fun.empty()) {
    delete copy;
    Rcpp::stop("clone_ad_pack: source has no valid shards");
  }
  copy->configured_num_threads = 1;
  copy->num_threads_configured = false;
  return copy;
}

hessian_template hessian_template_from_dgc(
  const DgCView& tpl,
  const char* label) {

  if (tpl.p.size() == 0) {
    Rcpp::stop("%s Hessian template missing; attach hessian_map result first", label);
  }

  const int n = tpl.nrow();
  const R_xlen_t p_len = tpl.p.size();
  const R_xlen_t i_len = tpl.i.size();
  if (static_cast<R_xlen_t>(n + 1) != p_len) {
    Rcpp::stop(
      "%s Hessian p_len=%d but expected nrow+1=%d",
      label,
      static_cast<int>(p_len),
      n + 1
    );
  }

  std::vector<int> p(static_cast<size_t>(p_len));
  std::vector<int> i(static_cast<size_t>(i_len));
  std::vector<int> x(static_cast<size_t>(i_len));
  for (R_xlen_t k = 0; k < p_len; ++k) {
    p[static_cast<size_t>(k)] = tpl.p[k];
  }
  for (R_xlen_t k = 0; k < i_len; ++k) {
    i[static_cast<size_t>(k)] = tpl.i[k];
    x[static_cast<size_t>(k)] = static_cast<int>(tpl.value<double>(k));
  }

  Eigen::Map<const hessian_template> upper(
    n, n, static_cast<int>(i_len),
    p.data(), i.data(), x.data()
  );
  hessian_template out = upper.selfadjointView<Eigen::Upper>();
  out.makeCompressed();
  return out;
}

void ad_fun_attach_hessians_from_list(
  ad_pack& shards,
  const Rcpp::List& ad_pack) {

  if (!ad_pack.containsElementNamed("outer") || !ad_pack.containsElementNamed("inner")) {
    Rcpp::stop("ad_pack list must contain 'outer' and 'inner' (from hessian_map())");
  }
  if (!ad_pack.containsElementNamed("map_outer") || !ad_pack.containsElementNamed("map_inner")) {
    Rcpp::stop("ad_pack list must contain 'map_outer' and 'map_inner'");
  }
  if (!ad_pack.containsElementNamed("sizes")) {
    Rcpp::stop("ad_pack list must contain 'sizes'");
  }

  shards.sizes = IntVecView(static_cast<SEXP>(ad_pack["sizes"]));
  if (!shards.sizes.has_name("beta") ||
      !shards.sizes.has_name("gamma") ||
      !shards.sizes.has_name("theta")) {
    Rcpp::stop("sizes must be a named vector: beta, gamma, theta");
  }

  if (shards.hessians_attached) {
    return;
  }

  shards.hessian_outer = hessian_template_from_dgc(
    DgCView(Rcpp::as<Rcpp::S4>(ad_pack["outer"])), "outer");

  shards.hessian_inner = hessian_template_from_dgc(
    DgCView(Rcpp::as<Rcpp::S4>(ad_pack["inner"])), "inner");

  shards.map_outer = hessian_map_view(Rcpp::as<Rcpp::List>(ad_pack["map_outer"]));
  shards.map_inner = hessian_map_view(Rcpp::as<Rcpp::List>(ad_pack["map_inner"]));

  shards.hessians_attached = true;
  ad_fun_attach_chol_pattern_from_list(shards, ad_pack);
}

namespace {

static void check_ad_fun_abi(const ad_pack& backend) {
  if (backend.abi_version != ADLAPLACE_ABI_VERSION) {
    Rcpp::stop(
      "ad_pack ABI mismatch (got %d, need %d): rebuild the extension package against this adlaplace",
      backend.abi_version,
      ADLAPLACE_ABI_VERSION
    );
  }
}

static void validate_ad_fun_eval(const ad_pack& backend) {
  check_ad_fun_abi(backend);
  if (backend.fun.empty()) {
    Rcpp::stop("ad_pack has no AD shards");
  }
}

static void validate_ad_fun_laplace(const ad_pack& backend) {
  validate_ad_fun_eval(backend);
  if (backend.sizes.sexp == R_NilValue ||
      !backend.sizes.has_name("beta") ||
      !backend.sizes.has_name("gamma") ||
      !backend.sizes.has_name("theta")) {
    Rcpp::stop("ad_pack missing sizes; call ad_pack() before inner_opt()");
  }
  if (!backend.hessians_attached) {
    Rcpp::stop("ad_pack missing Hessian templates; call ad_pack() before inner_opt()");
  }
}

static Rcpp::List ad_fun_list_from_s4(const Rcpp::S4& obj) {
  return Rcpp::List::create(
    Rcpp::Named("ad_pack") = obj.slot("ptr"),
    Rcpp::Named("group_sparsity") = obj.slot("group_sparsity"),
    Rcpp::Named("outer") = obj.slot("outer"),
    Rcpp::Named("inner") = obj.slot("inner"),
    Rcpp::Named("map_outer") = obj.slot("map_outer"),
    Rcpp::Named("map_inner") = obj.slot("map_inner"),
    Rcpp::Named("sizes") = obj.slot("sizes"),
    Rcpp::Named("chol_inner_list") = obj.slot("chol_inner_list")
  );
}

}  // namespace

ad_shard* shard_handle(ad_pack* backend, size_t shard) {
  if (!backend) Rcpp::stop("ad_pack handle is NULL");
  if (shard >= backend->fun.size()) {
    Rcpp::stop("shard index %d out of range [0, %d]", (int)shard, (int)backend->fun.size() - 1);
  }
  ad_shard* s = backend->fun[shard];
  if (!s) Rcpp::stop("ad_pack.fun[%d] is NULL", (int)shard);
  return s;
}

SEXP ad_fun_handle_sexp(const Rcpp::List& ad_fun_list) {
  if (ad_fun_list.containsElementNamed("ad_pack")) {
    return ad_fun_list["ad_pack"];
  }
  Rcpp::stop("ad_pack list must contain component 'ad_pack'");
}

ad_pack* ad_fun_from_list(const Rcpp::List& ad_fun_list) {
  SEXP handle_ptr = ad_fun_handle_sexp(ad_fun_list);
  auto* backend = static_cast<::ad_pack*>(R_ExternalPtrAddr(handle_ptr));
  if (!backend) {
    Rcpp::stop("ad_pack external pointer is NULL (cleared?)");
  }

  ad_fun_attach_hessians_from_list(*backend, ad_fun_list);
  return backend;
}

ad_pack* ad_fun_from_handle(SEXP handle) {
  auto* backend = static_cast<::ad_pack*>(R_ExternalPtrAddr(handle));
  if (!backend) {
    Rcpp::stop("ad_pack external pointer is NULL (cleared?)");
  }
  check_ad_fun_abi(*backend);
  return backend;
}

ad_pack* resolve_ad_pack_eval(SEXP ad_pack_ptr) {
  if (Rf_isNull(ad_pack_ptr)) {
    Rcpp::stop("ad_pack_ptr must not be NULL");
  }
  if (!Rf_inherits(ad_pack_ptr, "ad_pack_ptr")) {
    Rcpp::stop("expected ad_pack_ptr");
  }
  ad_pack* backend = ad_fun_from_handle(ad_pack_ptr);
  validate_ad_fun_eval(*backend);
  return backend;
}

ad_pack* resolve_ad_pack_laplace(const Rcpp::S4& ad_fun_s4) {
  if (!ad_fun_s4.inherits("ad_pack")) {
    Rcpp::stop("expected an ad_pack object; call ad_pack(ad_pack_ptr) first");
  }
  ::ad_pack* backend = ad_fun_from_list(ad_fun_list_from_s4(ad_fun_s4));
  validate_ad_fun_laplace(*backend);
  return backend;
}

std::vector<size_t> resolve_shard_indices(
  size_t n_shards,
  const Rcpp::IntegerVector& shards) {
  std::vector<size_t> shard_idx;
  if (shards.size() == 0) {
    shard_idx.resize(n_shards);
    for (size_t s = 0; s < n_shards; ++s) shard_idx[s] = s;
    return shard_idx;
  }

  shard_idx.reserve(shards.size());
  for (R_xlen_t k = 0; k < shards.size(); ++k) {
    if (shards[k] == NA_INTEGER) {
      Rcpp::stop("shards contains NA at position %d", (int)k + 1);
    }
    if (shards[k] < 0 || static_cast<size_t>(shards[k]) >= n_shards) {
      Rcpp::stop("shards index %d out of range [0, %d]", shards[k], (int)n_shards - 1);
    }
    shard_idx.push_back(static_cast<size_t>(shards[k]));
  }
  return shard_idx;
}

Rcpp::List sparsity_shard_from_handle(ad_shard* shard) {
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
    Rcpp::stop("shard get_sizes failed");
  }

  std::vector<int> grad_inner(static_cast<size_t>(nnz_grad_inner));
  std::vector<int> grad_outer(static_cast<size_t>(nnz_grad_outer));
  std::vector<int> row_hes_inner(static_cast<size_t>(nnz_hes_inner));
  std::vector<int> col_hes_inner(static_cast<size_t>(nnz_hes_inner));
  std::vector<int> row_hes_outer(static_cast<size_t>(nnz_hes_outer));
  std::vector<int> col_hes_outer(static_cast<size_t>(nnz_hes_outer));

  if (shard->get_sparse_pattern(
        grad_inner.data(), grad_outer.data(),
        row_hes_inner.data(), col_hes_inner.data(),
        row_hes_outer.data(), col_hes_outer.data()) != 0) {
    Rcpp::stop("shard get_sparse_pattern failed");
  }

  return Rcpp::List::create(
    Rcpp::Named("grad") = Rcpp::IntegerVector(
      grad_outer.begin(), grad_outer.end()),
    Rcpp::Named("grad_inner") = Rcpp::IntegerVector(
      grad_inner.begin(), grad_inner.end()),
    Rcpp::Named("row_hess") = Rcpp::IntegerVector(
      row_hes_outer.begin(), row_hes_outer.end()),
    Rcpp::Named("col_hess") = Rcpp::IntegerVector(
      col_hes_outer.begin(), col_hes_outer.end()),
    Rcpp::Named("row_hess_inner") = Rcpp::IntegerVector(
      row_hes_inner.begin(), row_hes_inner.end()),
    Rcpp::Named("col_hess_inner") = Rcpp::IntegerVector(
      col_hes_inner.begin(), col_hes_inner.end())
  );
}
