#ifndef ADLAPLACE_BACKEND_HPP
#define ADLAPLACE_BACKEND_HPP

#include <cppad/cppad.hpp>
#include <Rcpp.h>
#include <Eigen/SparseCore>
#include <cstddef>
#include <vector>

#include "adlaplace/density_data.hpp"
#include "adlaplace/rviews.hpp"

// Bump when AdTape / ad_shard / ad_pack layout or the backend
// registration contract changes. packs_to_ad_fun stamps this into ad_pack;
// adlaplace checks it when consuming a backend-built handle.
#define ADLAPLACE_ABI_VERSION 5

// Symbolic LDL pattern from hessian_map()$chol_inner_list (L1, Linv, perm,
// half_H_inv, H_inv).
struct CholPattern {
  int n = 0;
  std::vector<int> L1_p;
  std::vector<int> L1_i;
  std::vector<int> Linv_p;
  std::vector<int> Linv_i;
  std::vector<int> half_H_inv_p;
  std::vector<int> half_H_inv_i;
  std::vector<int> H_inv_p;
  std::vector<int> H_inv_i;
  std::vector<int> trace_columns_p;
  std::vector<int> trace_columns_i;
  // Per-shard slots of a quadratic form into H_inv@x. Empty range means the
  // shard has no precomputed index and trace_hinv_t walks columns.
  std::vector<int> hinv_q_index_p;
  std::vector<int> hinv_q_index_i;
  std::vector<int> perm;
  std::vector<int> perm_inv;
};

// Order-3 Forward/Reverse workspace for trace_hinv_t.
struct TraceWorkspace {
  CPPAD_TESTVECTOR(double) wthree;
  CPPAD_TESTVECTOR(double) direction_zeros;
  CPPAD_TESTVECTOR(double) direction;
};

struct AdTape {
  CppAD::ADFun<double> fun;
  CppAD::sparse_jac_work work_grad;
  CppAD::sparse_hes_work work_hess;
  CppAD::sparse_jac_work work_inner_grad;
  CppAD::sparse_hes_work work_inner_hess;
  CppAD::sparse_rcv<CPPAD_TESTVECTOR(size_t), CPPAD_TESTVECTOR(double)> pattern_grad;
  CppAD::sparse_rcv<CPPAD_TESTVECTOR(size_t), CPPAD_TESTVECTOR(double)> pattern_grad_inner;
  CppAD::sparse_rcv<CPPAD_TESTVECTOR(size_t), CPPAD_TESTVECTOR(double)> pattern_hessian;
  CppAD::sparse_rcv<CPPAD_TESTVECTOR(size_t), CPPAD_TESTVECTOR(double)> pattern_hessian_inner;

  CPPAD_TESTVECTOR(double) w;
  TraceWorkspace trace;
  // Required but ignored output argument of sparse_jac_rev / sparse_hes.
  CppAD::sparse_rc<CPPAD_TESTVECTOR(size_t)> unused_pattern;

  CPPAD_TESTVECTOR(double) x;

  std::size_t shard_index = 0;
  std::size_t owner_thread = 0;
  bool owner_thread_assigned = false;
  std::size_t n_beta = 0;
  std::size_t n_theta = 0;

  // Global parameter count (beta+gamma+theta). Tape Domain() may be smaller
  // when compact_tape is on; gather/scatter uses tape_to_global.
  std::size_t n_global = 0;
  std::vector<std::size_t> tape_to_global;
  // Global gamma row (0..n_gamma-1) -> tape position (n_tape if absent).
  std::vector<std::size_t> gamma_row_to_tape;
  // Pattern indices in global parameter coordinates (same nnz order as local).
  std::vector<std::size_t> grad_cols_global;
  std::vector<std::size_t> grad_inner_cols_global;
  std::vector<std::size_t> hes_rows_global;
  std::vector<std::size_t> hes_cols_global;
  std::vector<std::size_t> hes_inner_rows_global;
  std::vector<std::size_t> hes_inner_cols_global;
};

inline std::size_t ad_tape_n_global(const AdTape& p) {
  return p.n_global > 0 ? p.n_global : p.x.size();
}

inline AdTape clone_group_pack(const AdTape& src) {
  AdTape dst;
  dst.fun = src.fun;
  dst.work_grad = src.work_grad;
  dst.work_hess = src.work_hess;
  dst.work_inner_grad = src.work_inner_grad;
  dst.work_inner_hess = src.work_inner_hess;
  dst.pattern_grad = src.pattern_grad;
  dst.pattern_grad_inner = src.pattern_grad_inner;
  dst.pattern_hessian = src.pattern_hessian;
  dst.pattern_hessian_inner = src.pattern_hessian_inner;
  dst.x = src.x;
  dst.w = src.w;
  dst.trace = src.trace;
  dst.unused_pattern = src.unused_pattern;
  dst.shard_index = src.shard_index;
  dst.owner_thread = src.owner_thread;
  dst.owner_thread_assigned = src.owner_thread_assigned;
  dst.n_beta = src.n_beta;
  dst.n_theta = src.n_theta;
  dst.n_global = src.n_global;
  dst.tape_to_global = src.tape_to_global;
  dst.gamma_row_to_tape = src.gamma_row_to_tape;
  dst.grad_cols_global = src.grad_cols_global;
  dst.grad_inner_cols_global = src.grad_inner_cols_global;
  dst.hes_rows_global = src.hes_rows_global;
  dst.hes_cols_global = src.hes_cols_global;
  dst.hes_inner_rows_global = src.hes_inner_rows_global;
  dst.hes_inner_cols_global = src.hes_inner_cols_global;
  return dst;
}

inline void ad_tape_release_eval_buffers(AdTape& gp) {
  if (gp.fun.Domain() > 0) {
    gp.fun.capacity_order(0);
  }
  gp.trace.direction.clear();
  gp.trace.direction_zeros.clear();
  gp.trace.wthree.clear();
}

struct ad_shard;

using ShardFactory = ad_shard* (*)(AdTape&&);

struct ad_shard {
  AdTape pack;
  ShardFactory factory = nullptr;

  explicit ad_shard(AdTape&& p, ShardFactory f = nullptr)
    : pack(std::move(p)), factory(f) {}

  virtual ~ad_shard() = default;

  virtual int f(const double* x, double* out_f) = 0;
  virtual int f_grad(const double* x, bool inner, double* out_f, double* out_grad) = 0;
  virtual int f_grad_hess(
    const double* x,
    bool inner,
    double* out_f,
    double* out_grad,
    double* out_hes,
    int* map) = 0;
  virtual int get_sizes(
    int* n_inner,
    int* n_outer,
    int* n_beta,
    int* n_theta,
    int* nnz_grad_inner,
    int* nnz_grad_outer,
    int* nnz_hes_inner,
    int* nnz_hes_outer) = 0;
  virtual int get_sparse_pattern(
    int* pattern_grad_inner,
    int* pattern_grad_outer,
    int* pattern_hes_inner_row,
    int* pattern_hes_inner_col,
    int* pattern_hes_outer_row,
    int* pattern_hes_outer_col) = 0;
  virtual int assign_memory() = 0;
  // Taylor drain must run in the DSO that recorded the tape (macOS
  // thread_alloc is per shared library). Override rather than calling
  // capacity_order from adlaplace.so on a backend-built ADFun.
  virtual void release_eval_buffers() {
    ad_tape_release_eval_buffers(pack);
  }
  virtual int trace_hinv_t(
    const double* x,
    const int* LinvPt_p,
    const int* LinvPt_i,
    const double* LinvPt_x,
    std::size_t LinvPt_ncol,
    std::size_t LinvPt_p_len,
    std::size_t LinvPt_i_len,
    std::size_t LinvPt_x_len,
    const int* LinvPtColumns_p,
    const int* LinvPtColumns_i,
    std::size_t LinvPtColumns_p_len,
    std::size_t LinvPtColumns_i_len,
    double* out_trace) = 0;
  // Frobenius product of this shard's Hessian factor with H_inv.
  // Return 0 after writing out_trace. Return 1 to fall back to trace_hinv_t.
  // q_to_hinv[k] is a 0-based slot of H_inv_x, or -1 when that entry is absent.
  virtual int trace_hinv_from_h(
    const double* /*x*/,
    const double* /*H_inv_x*/,
    const int* /*q_to_hinv*/,
    std::size_t /*H_inv_x_len*/,
    std::size_t /*q_to_hinv_len*/,
    double* /*out_trace*/) {
    return 1;
  }
  virtual ad_shard* clone() const = 0;
};

using ad_vector = std::vector<ad_shard*>;

using hessian_template = Eigen::SparseMatrix<int, Eigen::ColMajor, int>;

struct ad_pack {
  int abi_version = ADLAPLACE_ABI_VERSION;
  ad_vector fun;
  hessian_template hessian_outer;
  hessian_template hessian_inner;
  hessian_map_view map_outer;
  hessian_map_view map_inner;
  IntVecView sizes;
  CholPattern chol_pattern;
  bool hessians_attached = false;
  std::size_t configured_num_threads = 1;
  bool num_threads_configured = false;
};

using LogDensObsFn = CppAD::vector<CppAD::AD<double>> (*)(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config,
  size_t Dgroup);

using LogDensSingleDataFn = CppAD::vector<CppAD::AD<double>> (*)(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config);

using LogDensSingleRandomFn = CppAD::vector<CppAD::AD<double>> (*)(
  const CppAD::vector<CppAD::AD<double>>& x,
  const density_data& model,
  const Config& config);

#endif
