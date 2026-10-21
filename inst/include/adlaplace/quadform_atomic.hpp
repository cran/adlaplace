#ifndef ADLAPLACE_QUADFORM_ATOMIC_HPP
#define ADLAPLACE_QUADFORM_ATOMIC_HPP

// CppAD atomic for the scalar quadratic form
//
//   y = x^T A x
//
// with A a fixed symmetric matrix stored as upper-triangle CSC
// (dsCMatrix uplo = "U": only row <= col). A never enters the AD tape.
// Value uses FEM-style weights (1 on the diagonal, 2 off it). Taylor and
// reverse through order 2 use sparse matvecs with (A + A^T) = 2A:
//
//   y0 = x0^T A x0
//   y1 = x0^T A x1 + x1^T A x0
//   y2 = x0^T A x2 + x2^T A x0 + x1^T A x1
//
//   p_{x,m} = (A + A^T) sum_{j=0}^{q-1-m} p_{y,m+j} x_j
//
// Hessian sparsity is the support of A ∪ A^T (both triangles), matching
// CppAD's cppad.symmetric coloring which does not fill a missing triangle.
// Keep scalar multipliers such as tau(theta) outside the atomic.

#include <Rcpp.h>
#include <cppad/cppad.hpp>

#include "adlaplace/rviews.hpp"

#include <cstddef>
#include <deque>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

namespace adlaplace {
namespace quadform {

struct Payload {
  std::vector<int> p;
  std::vector<int> i;
  std::vector<double> x;
  // 1 on the diagonal, 2 off it (aligned with p/i/x).
  std::vector<double> w;
  std::size_t n = 0;
  // Support of A ∪ A^T as both triangles, for hes_sparsity.
  std::vector<std::size_t> hes_row;
  std::vector<std::size_t> hes_col;
  // Indices that appear in A (row or column), for jac / for_type / rev_depend.
  std::vector<unsigned char> touches;
};

inline void fill_patterns(Payload &pay) {
  const std::size_t n = pay.n;
  const std::size_t nnz = pay.i.size();
  if (pay.x.size() != nnz) {
    Rcpp::stop("quadform: CSC i and x length mismatch");
  }
  if (pay.p.size() != n + 1) {
    Rcpp::stop("quadform: CSC p length must be n + 1");
  }
  pay.w.resize(nnz);
  pay.touches.assign(n, 0);
  std::set<std::pair<std::size_t, std::size_t>> pairs;
  for (std::size_t col = 0; col < n; ++col) {
    for (int pos = pay.p[col]; pos < pay.p[col + 1]; ++pos) {
      const std::size_t k = static_cast<std::size_t>(pos);
      const std::size_t row = static_cast<std::size_t>(pay.i[k]);
      if (row > col) {
        Rcpp::stop(
            "quadform: expected upper-triangle CSC (row <= col); "
            "got row=%d col=%d",
            static_cast<int>(row), static_cast<int>(col));
      }
      pay.w[k] = (row == col) ? 1.0 : 2.0;
      pay.touches[row] = 1;
      pay.touches[col] = 1;
      pairs.insert({row, col});
      pairs.insert({col, row});
    }
  }
  pay.hes_row.clear();
  pay.hes_col.clear();
  pay.hes_row.reserve(pairs.size());
  pay.hes_col.reserve(pairs.size());
  for (const auto &rc : pairs) {
    pay.hes_row.push_back(rc.first);
    pay.hes_col.push_back(rc.second);
  }
}

// y = x^T A x over upper-triangle CSC (weights 1/2).
inline double quad_value(const Payload &pay, const double *x) {
  double acc = 0.0;
  for (std::size_t col = 0; col < pay.n; ++col) {
    const double xc = x[col];
    for (int pos = pay.p[col]; pos < pay.p[col + 1]; ++pos) {
      const std::size_t k = static_cast<std::size_t>(pos);
      const std::size_t row = static_cast<std::size_t>(pay.i[k]);
      acc += pay.w[k] * x[row] * pay.x[k] * xc;
    }
  }
  return acc;
}

// Bilinear form u^T A v for symmetric A stored as upper triangle.
inline double bilinear(const Payload &pay, const double *u, const double *v) {
  double acc = 0.0;
  for (std::size_t col = 0; col < pay.n; ++col) {
    for (int pos = pay.p[col]; pos < pay.p[col + 1]; ++pos) {
      const std::size_t k = static_cast<std::size_t>(pos);
      const std::size_t row = static_cast<std::size_t>(pay.i[k]);
      const double a = pay.x[k];
      if (row == col) {
        acc += a * u[row] * v[col];
      } else {
        acc += a * (u[row] * v[col] + u[col] * v[row]);
      }
    }
  }
  return acc;
}

// g = (A + A^T) s = 2 A s for symmetric A stored as upper triangle.
inline void sym_matvec(const Payload &pay, const double *s, double *g) {
  for (std::size_t j = 0; j < pay.n; ++j) {
    g[j] = 0.0;
  }
  for (std::size_t col = 0; col < pay.n; ++col) {
    const double sc = s[col];
    for (int pos = pay.p[col]; pos < pay.p[col + 1]; ++pos) {
      const std::size_t k = static_cast<std::size_t>(pos);
      const std::size_t row = static_cast<std::size_t>(pay.i[k]);
      const double two_a = 2.0 * pay.x[k];
      if (row == col) {
        g[col] += two_a * sc;
      } else {
        g[row] += two_a * sc;
        g[col] += two_a * s[row];
      }
    }
  }
}

class atomic_quadform : public CppAD::atomic_four<double> {
public:
  atomic_quadform() : CppAD::atomic_four<double>("quadform") {}

  std::size_t register_payload(Payload &&pay) {
    fill_patterns(pay);
    std::lock_guard<std::mutex> lock(mutex_);
    payloads_.push_back(std::move(pay));
    return payloads_.size() - 1;
  }

  std::size_t register_csc(const std::vector<int> &p, const std::vector<int> &i,
                           const std::vector<double> &x) {
    Payload pay;
    pay.n = p.size() > 0 ? static_cast<std::size_t>(p.size() - 1) : 0;
    pay.p = p;
    pay.i = i;
    pay.x = x;
    return register_payload(std::move(pay));
  }

  std::size_t register_csc(CscMatrix &&m) {
    Payload pay;
    pay.n = static_cast<std::size_t>(m.ncol());
    pay.p = std::move(m.p);
    pay.i = std::move(m.i);
    pay.x = std::move(m.x);
    return register_payload(std::move(pay));
  }

  std::size_t register_diagonal(const std::vector<double> &values) {
    Payload pay;
    pay.n = values.size();
    pay.p.resize(pay.n + 1);
    pay.i.resize(pay.n);
    pay.x = values;
    for (std::size_t j = 0; j < pay.n; ++j) {
      pay.p[j] = static_cast<int>(j);
      pay.i[j] = static_cast<int>(j);
    }
    pay.p[pay.n] = static_cast<int>(pay.n);
    return register_payload(std::move(pay));
  }

private:
  std::deque<Payload> payloads_;
  mutable std::mutex mutex_;

  const Payload &payload(std::size_t call_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return payloads_[call_id];
  }

  bool for_type(size_t call_id,
                const CppAD::vector<CppAD::ad_type_enum> &type_x,
                CppAD::vector<CppAD::ad_type_enum> &type_y) override {
    const Payload &pay = payload(call_id);
    if (type_x.size() != pay.n) {
      return false;
    }
    type_y.resize(1);
    CppAD::ad_type_enum t = CppAD::constant_enum;
    for (std::size_t j = 0; j < pay.n; ++j) {
      if (pay.touches[j]) {
        t = std::max(t, type_x[j]);
      }
    }
    type_y[0] = t;
    return true;
  }

  bool rev_depend(size_t call_id, const CppAD::vector<bool> &ident_zero_x,
                  CppAD::vector<bool> &depend_x,
                  const CppAD::vector<bool> &depend_y) override {
    const Payload &pay = payload(call_id);
    if (ident_zero_x.size() != pay.n) {
      return false;
    }
    depend_x.resize(pay.n);
    const bool need = depend_y.size() > 0 && depend_y[0];
    for (std::size_t j = 0; j < pay.n; ++j) {
      depend_x[j] = need && pay.touches[j] != 0;
    }
    return true;
  }

  bool jac_sparsity(
      size_t call_id, bool dependency,
      const CppAD::vector<bool> &ident_zero_x,
      const CppAD::vector<bool> &select_x,
      const CppAD::vector<bool> &select_y,
      CppAD::sparse_rc<CppAD::vector<size_t>> &pattern_out) override {
    (void)dependency;
    const Payload &pay = payload(call_id);
    const size_t n = select_x.size();
    if (n != pay.n || select_y.size() != 1) {
      return false;
    }
    size_t nnz = 0;
    if (select_y[0]) {
      for (size_t j = 0; j < n; ++j) {
        if (select_x[j] && !ident_zero_x[j] && pay.touches[j]) {
          ++nnz;
        }
      }
    }
    pattern_out.resize(1, n, nnz);
    size_t k = 0;
    if (select_y[0]) {
      for (size_t j = 0; j < n; ++j) {
        if (select_x[j] && !ident_zero_x[j] && pay.touches[j]) {
          pattern_out.set(k++, 0, j);
        }
      }
    }
    return true;
  }

  bool hes_sparsity(
      size_t call_id, const CppAD::vector<bool> &select_x,
      const CppAD::vector<bool> &select_y,
      CppAD::sparse_rc<CppAD::vector<size_t>> &pattern_out) override {
    const Payload &pay = payload(call_id);
    const size_t n = select_x.size();
    if (n != pay.n || select_y.size() != 1) {
      return false;
    }
    size_t nnz = 0;
    if (select_y[0]) {
      for (std::size_t k = 0; k < pay.hes_row.size(); ++k) {
        const size_t r = pay.hes_row[k];
        const size_t c = pay.hes_col[k];
        if (select_x[r] && select_x[c]) {
          ++nnz;
        }
      }
    }
    pattern_out.resize(n, n, nnz);
    size_t out = 0;
    if (select_y[0]) {
      for (std::size_t k = 0; k < pay.hes_row.size(); ++k) {
        const size_t r = pay.hes_row[k];
        const size_t c = pay.hes_col[k];
        if (select_x[r] && select_x[c]) {
          pattern_out.set(out++, r, c);
        }
      }
    }
    return true;
  }

  bool forward(size_t call_id, const CppAD::vector<bool> &select_y,
               size_t order_low, size_t order_up,
               const CppAD::vector<double> &tx,
               CppAD::vector<double> &ty) override {
    (void)select_y;
    if (order_up > 2) {
      return false;
    }
    const Payload &pay = payload(call_id);
    const size_t n = pay.n;
    const size_t q = order_up + 1;
    if (tx.size() < n * q) {
      return false;
    }
    ty.resize(q);

    thread_local std::vector<double> x0;
    thread_local std::vector<double> x1;
    thread_local std::vector<double> x2;
    x0.resize(n);
    for (size_t j = 0; j < n; ++j) {
      x0[j] = tx[j * q + 0];
    }

    if (order_low <= 0) {
      ty[0] = quad_value(pay, x0.data());
    }
    if (order_up >= 1) {
      x1.resize(n);
      for (size_t j = 0; j < n; ++j) {
        x1[j] = tx[j * q + 1];
      }
      if (order_low <= 1) {
        // x0^T A x1 + x1^T A x0
        ty[1] = bilinear(pay, x0.data(), x1.data()) +
                bilinear(pay, x1.data(), x0.data());
      }
    }
    if (order_up >= 2) {
      x2.resize(n);
      for (size_t j = 0; j < n; ++j) {
        x2[j] = tx[j * q + 2];
      }
      // x0^T A x2 + x2^T A x0 + x1^T A x1
      // (the 1/2 from CppAD's Taylor convention is absorbed:
      //  (1/2) x1^T (A+A^T) x1 = x1^T A x1)
      ty[2] = bilinear(pay, x0.data(), x2.data()) +
              bilinear(pay, x2.data(), x0.data()) +
              bilinear(pay, x1.data(), x1.data());
    }
    return true;
  }

  bool reverse(size_t call_id, const CppAD::vector<bool> &select_x,
               size_t order_up, const CppAD::vector<double> &tx,
               const CppAD::vector<double> &ty, CppAD::vector<double> &px,
               const CppAD::vector<double> &py) override {
    (void)select_x;
    (void)ty;
    if (order_up > 2) {
      return false;
    }
    const Payload &pay = payload(call_id);
    const size_t n = pay.n;
    const size_t q = order_up + 1;
    if (tx.size() < n * q || py.size() < q) {
      return false;
    }
    px.resize(n * q);
    for (size_t k = 0; k < px.size(); ++k) {
      px[k] = 0.0;
    }

    thread_local std::vector<double> s;
    thread_local std::vector<double> g;
    s.resize(n);
    g.resize(n);

    // p_{x,m} = (A+A^T) sum_{j=0}^{q-1-m} p_{y,m+j} x_j
    for (size_t m = 0; m < q; ++m) {
      for (size_t j = 0; j < n; ++j) {
        s[j] = 0.0;
      }
      for (size_t ell = 0; ell + m < q; ++ell) {
        const double py_m = py[m + ell];
        if (py_m == 0.0) {
          continue;
        }
        for (size_t j = 0; j < n; ++j) {
          s[j] += py_m * tx[j * q + ell];
        }
      }
      sym_matvec(pay, s.data(), g.data());
      for (size_t j = 0; j < n; ++j) {
        px[j * q + m] = g[j];
      }
    }
    return true;
  }
};

inline atomic_quadform &quadform_atomic_instance() {
  static atomic_quadform op;
  return op;
}

inline void init_quadform_atomic() {
  (void)quadform_atomic_instance();
}

inline std::size_t register_csc(const std::vector<int> &p,
                                const std::vector<int> &i,
                                const std::vector<double> &x) {
  return quadform_atomic_instance().register_csc(p, i, x);
}

inline std::size_t register_csc(CscMatrix &&m) {
  return quadform_atomic_instance().register_csc(std::move(m));
}

inline std::size_t register_diagonal(const std::vector<double> &values) {
  return quadform_atomic_instance().register_diagonal(values);
}

inline void call_quadform(std::size_t call_id,
                          const CppAD::vector<CppAD::AD<double>> &ax,
                          CppAD::vector<CppAD::AD<double>> &ay) {
  ay.resize(1);
  quadform_atomic_instance()(call_id, ax, ay);
}

} // namespace quadform
} // namespace adlaplace

#endif
