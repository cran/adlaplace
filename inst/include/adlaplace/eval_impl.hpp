#ifndef ADLAPLACE_EVAL_IMPL_HPP
#define ADLAPLACE_EVAL_IMPL_HPP

// Include in exactly one .cpp per shared library (adlaplace.so or a backend .so).

#include "adlaplace/eval.hpp"

#include <algorithm>
#include <cppad/cppad.hpp>
#include <cstddef>
#include <vector>

#include "adlaplace/backend.hpp"

int pack_sparsity_sizes(
  AdTape& ad_pack,
  int* n_inner,
  int* n_outer,
  int* n_beta,
  int* n_theta,
  int* nnz_grad_inner,
  int* nnz_grad_outer,
  int* nnz_hes_inner,
  int* nnz_hes_outer) {

  // Report global parameter dimension so R-side hessian_map sizing stays
  // correct when the tape Domain() is compacted.
  const int n_global = ad_pack.n_global > 0
    ? static_cast<int>(ad_pack.n_global)
    : static_cast<int>(ad_pack.pattern_grad.nc());
  *n_inner = n_global;
  *n_outer = n_global;
  *n_beta = static_cast<int>(ad_pack.n_beta);
  *n_theta = static_cast<int>(ad_pack.n_theta);

  *nnz_grad_outer = static_cast<int>(ad_pack.pattern_grad.nnz());
  *nnz_grad_inner = static_cast<int>(ad_pack.pattern_grad_inner.nnz());
  *nnz_hes_outer = static_cast<int>(ad_pack.pattern_hessian.nnz());
  *nnz_hes_inner = static_cast<int>(ad_pack.pattern_hessian_inner.nnz());

  return 0;
}

int get_pattern(
  AdTape& ad_pack,
  int* pattern_grad_inner,
  int* pattern_grad_outer,
  int* pattern_hes_inner_row,
  int* pattern_hes_inner_col,
  int* pattern_hes_outer_row,
  int* pattern_hes_outer_col) {

  const size_t nnz_grad_inner = ad_pack.pattern_grad_inner.nnz();
  if (ad_pack.grad_inner_cols_global.size() == nnz_grad_inner) {
    for (size_t D = 0; D < nnz_grad_inner; ++D) {
      pattern_grad_inner[D] = static_cast<int>(ad_pack.grad_inner_cols_global[D]);
    }
  } else {
    const auto& cols_grad_inner = ad_pack.pattern_grad_inner.col();
    for (size_t D = 0; D < nnz_grad_inner; ++D) {
      pattern_grad_inner[D] = static_cast<int>(cols_grad_inner[D]);
    }
  }

  const size_t nnz_grad_outer = ad_pack.pattern_grad.nnz();
  if (ad_pack.grad_cols_global.size() == nnz_grad_outer) {
    for (size_t D = 0; D < nnz_grad_outer; ++D) {
      pattern_grad_outer[D] = static_cast<int>(ad_pack.grad_cols_global[D]);
    }
  } else {
    const auto& cols_grad_outer = ad_pack.pattern_grad.col();
    for (size_t D = 0; D < nnz_grad_outer; ++D) {
      pattern_grad_outer[D] = static_cast<int>(cols_grad_outer[D]);
    }
  }

  const size_t nnz_inner = ad_pack.pattern_hessian_inner.nnz();
  if (ad_pack.hes_inner_rows_global.size() == nnz_inner &&
      ad_pack.hes_inner_cols_global.size() == nnz_inner) {
    for (size_t D = 0; D < nnz_inner; ++D) {
      pattern_hes_inner_row[D] = static_cast<int>(ad_pack.hes_inner_rows_global[D]);
      pattern_hes_inner_col[D] = static_cast<int>(ad_pack.hes_inner_cols_global[D]);
    }
  } else {
    const auto& rows_hes_inner = ad_pack.pattern_hessian_inner.row();
    const auto& cols_hes_inner = ad_pack.pattern_hessian_inner.col();
    for (size_t D = 0; D < nnz_inner; ++D) {
      pattern_hes_inner_row[D] = static_cast<int>(rows_hes_inner[D]);
      pattern_hes_inner_col[D] = static_cast<int>(cols_hes_inner[D]);
    }
  }

  const size_t nnz_outer = ad_pack.pattern_hessian.nnz();
  if (ad_pack.hes_rows_global.size() == nnz_outer &&
      ad_pack.hes_cols_global.size() == nnz_outer) {
    for (size_t D = 0; D < nnz_outer; ++D) {
      pattern_hes_outer_row[D] = static_cast<int>(ad_pack.hes_rows_global[D]);
      pattern_hes_outer_col[D] = static_cast<int>(ad_pack.hes_cols_global[D]);
    }
  } else {
    const auto& rows_hes_outer = ad_pack.pattern_hessian.row();
    const auto& cols_hes_outer = ad_pack.pattern_hessian.col();
    for (size_t D = 0; D < nnz_outer; ++D) {
      pattern_hes_outer_row[D] = static_cast<int>(rows_hes_outer[D]);
      pattern_hes_outer_col[D] = static_cast<int>(cols_hes_outer[D]);
    }
  }

  return 0;
}

namespace {

struct CscView {
  const int* p;
  const int* i;
  const double* x;
  std::size_t ncol;
  std::size_t p_len;
  std::size_t i_len;
  std::size_t x_len;
};

}  // namespace

const char* trace_hinv_t_strerror(int rc) {
  switch (rc) {
    case 0: return "success";
    case 1: return "null input";
    case 2: return "empty parameter vector";
    case 5: return "LinvPtColumns p too short for shard_index";
    case 6: return "LinvPt p too short";
    case 7: return "invalid column range in LinvPtColumns";
    case 8: return "LinvPtColumns i index out of range";
    case 9: return "LinvPt column index out of range";
    case 10: return "LinvPt p too short for column";
    case 11: return "invalid LinvPt entry range";
    case 12: return "LinvPt i index out of range";
    case 13: return "LinvPt x index out of range";
    case 14: return "LinvPt row index out of range (gamma)";
    case 15: return "direction index out of range";
    case 16: return "Reverse(3) result too short";
    case 17: return "trace buffers not assigned";
    case 18: return "assign_memory API missing";
    default: return "unknown error";
  }
}

class EvalShard final : public ad_shard {
public:
  explicit EvalShard(AdTape&& p, ShardFactory f = nullptr)
    : ad_shard(std::move(p), f) {}

  int f(const double* x, double* out_f) override {
    AdTape& gp = pack;
    const size_t Nparams = gp.x.size();
    const size_t Ndomain = gp.fun.Domain();
    const size_t Nrange = gp.fun.Range();
    if (Ndomain != Nparams) return 4;
    if (Nrange < 1) return 5;

    if (gp.tape_to_global.size() == Nparams) {
      for (size_t D = 0; D < Nparams; ++D) {
        gp.x[D] = x[gp.tape_to_global[D]];
      }
    } else {
      for (size_t D = 0; D < Nparams; ++D) {
        gp.x[D] = x[D];
      }
    }
    CppAD::vector<double> y = gp.fun.Forward(0, gp.x);
    if (y.size() < 1) return 6;
    *out_f += y[0];
    return 0;
  }

  int f_grad(const double* x, bool inner, double* out_f, double* out_grad) override {
    AdTape& gp = pack;
    const size_t Nparams = gp.x.size();
    if (gp.tape_to_global.size() == Nparams) {
      for (size_t D = 0; D < Nparams; ++D) {
        gp.x[D] = x[gp.tape_to_global[D]];
      }
    } else {
      for (size_t D = 0; D < Nparams; ++D) {
        gp.x[D] = x[D];
      }
    }

    auto& pattern_here = inner ? gp.pattern_grad_inner : gp.pattern_grad;
    auto& work_here = inner ? gp.work_inner_grad : gp.work_grad;
    const auto& cols_global = inner ? gp.grad_inner_cols_global : gp.grad_cols_global;

    *out_f += gp.fun.Forward(0, gp.x)[0];
    gp.fun.sparse_jac_rev(
      gp.x,
      pattern_here,
      gp.unused_pattern,
      "cppad",
      work_here);

    const size_t NoutGrad = pattern_here.nnz();
    const auto& cols = pattern_here.col();
    const auto& vals = pattern_here.val();
    if (cols_global.size() == NoutGrad) {
      for (size_t D = 0; D < NoutGrad; ++D) {
        out_grad[cols_global[D]] += vals[D];
      }
    } else {
      for (size_t D = 0; D < NoutGrad; ++D) {
        out_grad[cols[D]] += vals[D];
      }
    }

    return 0;
  }

  int f_grad_hess(
    const double* x,
    bool inner,
    double* out_f,
    double* out_grad,
    double* out_hes,
    int* map) override {

    AdTape& gp = pack;
    const size_t Nparams = gp.x.size();
    if (gp.tape_to_global.size() == Nparams) {
      for (size_t D = 0; D < Nparams; ++D) {
        gp.x[D] = x[gp.tape_to_global[D]];
      }
    } else {
      for (size_t D = 0; D < Nparams; ++D) {
        gp.x[D] = x[D];
      }
    }

    *out_f += gp.fun.Forward(0, gp.x)[0];

    auto& pattern_here_grad = inner ? gp.pattern_grad_inner : gp.pattern_grad;
    auto& work_here_grad = inner ? gp.work_inner_grad : gp.work_grad;
    auto& pattern_here_hes = inner ? gp.pattern_hessian_inner : gp.pattern_hessian;
    auto& work_here_hes = inner ? gp.work_inner_hess : gp.work_hess;
    const auto& cols_global = inner ? gp.grad_inner_cols_global : gp.grad_cols_global;

    gp.fun.sparse_jac_rev(
      gp.x,
      pattern_here_grad,
      gp.unused_pattern,
      "cppad",
      work_here_grad);

    gp.fun.sparse_hes(
      gp.x,
      gp.w,
      pattern_here_hes,
      gp.unused_pattern,
      "cppad.symmetric",
      work_here_hes);

    const size_t NoutGrad = pattern_here_grad.nnz();
    const auto& cols = pattern_here_grad.col();
    const auto& vals = pattern_here_grad.val();

    if (cols_global.size() == NoutGrad) {
      for (size_t D = 0; D < NoutGrad; ++D) {
        out_grad[cols_global[D]] += vals[D];
      }
    } else {
      for (size_t D = 0; D < NoutGrad; ++D) {
        out_grad[cols[D]] += vals[D];
      }
    }

    const size_t n_hes = pattern_here_hes.nnz();
    const auto& vals_hes = pattern_here_hes.val();
    for (size_t Di = 0; Di < n_hes; ++Di) {
      out_hes[map[Di]] = vals_hes[Di];
    }
    return 0;
  }

  int get_sizes(
    int* n_inner,
    int* n_outer,
    int* n_beta,
    int* n_theta,
    int* nnz_grad_inner,
    int* nnz_grad_outer,
    int* nnz_hes_inner,
    int* nnz_hes_outer) override {
    return pack_sparsity_sizes(
      pack, n_inner, n_outer, n_beta, n_theta,
      nnz_grad_inner, nnz_grad_outer, nnz_hes_inner, nnz_hes_outer);
  }

  int get_sparse_pattern(
    int* pattern_grad_inner,
    int* pattern_grad_outer,
    int* pattern_hes_inner_row,
    int* pattern_hes_inner_col,
    int* pattern_hes_outer_row,
    int* pattern_hes_outer_col) override {
    return get_pattern(
      pack,
      pattern_grad_inner,
      pattern_grad_outer,
      pattern_hes_inner_row,
      pattern_hes_inner_col,
      pattern_hes_outer_row,
      pattern_hes_outer_col);
  }

  void release_eval_buffers() override {
    ad_tape_release_eval_buffers(pack);
  }

  int assign_memory() override {
    AdTape& gp = pack;
    const std::size_t n_params = gp.fun.Domain();
    if (n_params == 0) return 2;

    gp.trace.direction.clear();
    gp.trace.direction_zeros.clear();
    gp.trace.wthree.clear();

    gp.trace.direction.resize(n_params);
    gp.trace.direction_zeros.resize(n_params);
    std::fill(gp.trace.direction_zeros.begin(), gp.trace.direction_zeros.end(), 0.0);
    gp.trace.wthree.resize(3);
    gp.trace.wthree[0] = 0.0;
    gp.trace.wthree[1] = 0.0;
    gp.trace.wthree[2] = 1.0;

    gp.fun.capacity_order(3);
    return 0;
  }

  int trace_hinv_t(
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
    double* out_trace) override {

    if (x == nullptr || out_trace == nullptr) return 1;
    if (LinvPt_p == nullptr || LinvPt_i == nullptr || LinvPt_x == nullptr) return 1;
    if (LinvPtColumns_p == nullptr || LinvPtColumns_i == nullptr) return 1;

    AdTape& gp = pack;
    const std::size_t ist = gp.shard_index;
    if (ist + 1 >= LinvPtColumns_p_len) return 5;
    if (LinvPt_p_len < LinvPt_ncol + 1) return 6;

    const CscView LinvPt{
      LinvPt_p, LinvPt_i, LinvPt_x,
      LinvPt_ncol, LinvPt_p_len, LinvPt_i_len, LinvPt_x_len
    };
    const CscView LinvPtColumns{
      LinvPtColumns_p, LinvPtColumns_i, nullptr,
      LinvPtColumns_p_len > 0 ? LinvPtColumns_p_len - 1 : 0,
      LinvPtColumns_p_len, LinvPtColumns_i_len, 0
    };

    const std::size_t n_params = gp.fun.Domain();
    if (n_params == 0) return 2;
    const bool compacted =
      gp.n_global > n_params && gp.tape_to_global.size() == n_params;

#ifdef DEBUG
    if (gp.trace.direction.size() != n_params ||
        gp.trace.direction_zeros.size() != n_params ||
        gp.trace.wthree.size() != 3) {
      return 17;
    }
#endif

    int rc = 0;
    if (gp.x.size() != n_params) {
      rc = 2;
    } else {
      if (compacted) {
        for (std::size_t d = 0; d < n_params; ++d) {
          gp.x[d] = x[gp.tape_to_global[d]];
        }
      } else {
        std::copy_n(x, n_params, gp.x.begin());
      }

      const int col_start = LinvPtColumns.p[ist];
      const int col_end = LinvPtColumns.p[ist + 1];
      if (col_start < 0 || col_end < col_start) {
        rc = 7;
      } else if (static_cast<std::size_t>(col_end) > LinvPtColumns.i_len) {
        rc = 8;
      } else {
        for (int dp = col_start; dp < col_end && rc == 0; ++dp) {
          const int dcol = LinvPtColumns.i[dp];
          if (dcol < 0 || static_cast<std::size_t>(dcol) >= LinvPt.ncol) {
            rc = 9;
            break;
          }
          if (static_cast<std::size_t>(dcol) + 1 >= LinvPt.p_len) {
            rc = 10;
            break;
          }

          std::fill(gp.trace.direction.begin(), gp.trace.direction.end(), 0.0);

          const int entry_start = LinvPt.p[dcol];
          const int entry_end = LinvPt.p[dcol + 1];
          if (entry_start < 0 || entry_end < entry_start) {
            rc = 11;
            break;
          }
          if (static_cast<std::size_t>(entry_end) > LinvPt.i_len) {
            rc = 12;
            break;
          }
          if (static_cast<std::size_t>(entry_end) > LinvPt.x_len) {
            rc = 13;
            break;
          }

          for (int k = entry_start; k < entry_end; ++k) {
            const int row = LinvPt.i[k];
            std::size_t idx;
            if (compacted) {
              if (row < 0 ||
                  static_cast<std::size_t>(row) >= gp.gamma_row_to_tape.size()) {
                rc = 14;
                break;
              }
              idx = gp.gamma_row_to_tape[static_cast<std::size_t>(row)];
              if (idx >= gp.trace.direction.size()) {
                continue;
              }
            } else {
              if (row < 0 ||
                  static_cast<std::size_t>(row) >= gp.pattern_grad_inner.nc()) {
                rc = 14;
                break;
              }
              idx = gp.n_beta + static_cast<std::size_t>(row);
              if (idx >= gp.trace.direction.size()) {
                rc = 15;
                break;
              }
            }
            gp.trace.direction[idx] = LinvPt.x[k];
          }
          if (rc != 0) {
            break;
          }

          gp.fun.Forward(0, gp.x);
          gp.fun.Forward(1, gp.trace.direction);
          gp.fun.Forward(2, gp.trace.direction_zeros);
          const CppAD::vector<double> dw = gp.fun.Reverse(3, gp.trace.wthree);
          if (dw.size() < 3 * n_params) {
            rc = 16;
            break;
          }

          for (std::size_t d = 0; d < n_params; ++d) {
            if (compacted) {
              out_trace[gp.tape_to_global[d]] += dw[3 * d];
            } else {
              out_trace[d] += dw[3 * d];
            }
          }
        }
      }
    }

    gp.trace.direction.clear();
    gp.trace.direction_zeros.clear();
    gp.trace.wthree.clear();
    return rc;
  }

  ad_shard* clone() const override {
    return new EvalShard(clone_group_pack(pack), factory);
  }
};

#endif
