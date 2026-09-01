#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <vector>

namespace {

constexpr int C = 2;
constexpr int B = 9;
constexpr int BankWidth = 85;
constexpr int AuxWidth = 8;
constexpr int EdgeAnchorWidth = 6;
constexpr int ChannelStride = 38;
constexpr int M = 0;
constexpr int P = 3;
constexpr int X = 6;
constexpr int DM = 9;
constexpr int L = 12;
constexpr int T5 = 13;
constexpr int Q = 18;
constexpr int QP = 23;
constexpr int Same = 76;
constexpr int EdgeSj2 = 79;
constexpr int EdgeDot2 = 81;
constexpr int EdgeGate = 83;

template <typename T>
struct Dual {
  T v;
  T d;
  __device__ Dual() : v(0), d(0) {}
  __device__ Dual(T value) : v(value), d(0) {}
  __device__ Dual(T value, T tangent) : v(value), d(tangent) {}
};

template <typename T> __device__ Dual<T> operator+(Dual<T> a, Dual<T> b) {
  return {a.v + b.v, a.d + b.d};
}
template <typename T> __device__ Dual<T> operator-(Dual<T> a, Dual<T> b) {
  return {a.v - b.v, a.d - b.d};
}
template <typename T> __device__ Dual<T> operator-(Dual<T> a) {
  return {-a.v, -a.d};
}
template <typename T> __device__ Dual<T> operator*(Dual<T> a, Dual<T> b) {
  return {a.v * b.v, a.d * b.v + a.v * b.d};
}
template <typename T> __device__ Dual<T> operator/(Dual<T> a, Dual<T> b) {
  const T inverse = T(1) / b.v;
  return {a.v * inverse, (a.d * b.v - a.v * b.d) * inverse * inverse};
}
template <typename T> __device__ Dual<T>& operator+=(Dual<T>& a, Dual<T> b) {
  a.v += b.v; a.d += b.d; return a;
}

template <typename T> __device__ T value(T x) { return x; }
template <typename T> __device__ T value(Dual<T> x) { return x.v; }
template <typename T> __device__ T tangent(T) { return T(0); }
template <typename T> __device__ T tangent(Dual<T> x) { return x.d; }
template <typename T> __device__ T root(T x) { return sqrt(x); }
template <typename T> __device__ Dual<T> root(Dual<T> x) {
  const T y = sqrt(x.v);
  return {y, x.d / (T(2) * y)};
}
template <typename T> __device__ T cosine(T x) { return cos(x); }
template <typename T> __device__ Dual<T> cosine(Dual<T> x) {
  return {cos(x.v), -sin(x.v) * x.d};
}
template <typename T> __device__ T sine(T x) { return sin(x); }
template <typename T> __device__ Dual<T> sine(Dual<T> x) {
  return {sin(x.v), cos(x.v) * x.d};
}

template <typename S>
__device__ S dot3(const S* a, const S* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename S>
__device__ void cross3(const S* a, const S* b, S* out) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

template <typename S>
__device__ void add_cross_pull(
    const S* a, const S* b, const S* pull, S* grad_a, S* grad_b) {
  S b_cross_g[3], g_cross_a[3];
  cross3(b, pull, b_cross_g);
  cross3(pull, a, g_cross_a);
  for (int d = 0; d < 3; ++d) {
    grad_a[d] += b_cross_g[d];
    grad_b[d] += g_cross_a[d];
  }
}

template <typename S>
__device__ void stf5(const S* left, const S* right, S* out) {
  constexpr double InvSqrt6 = 0.40824829046386301637;
  constexpr double InvSqrt2 = 0.70710678118654752440;
  const S xx = left[0] * right[0];
  const S yy = left[1] * right[1];
  const S zz = left[2] * right[2];
  out[0] = S(InvSqrt6) * (S(2) * zz - xx - yy);
  out[1] = S(InvSqrt2) * (xx - yy);
  out[2] = S(InvSqrt2) * (
      left[0] * right[1] + left[1] * right[0]);
  out[3] = S(InvSqrt2) * (
      left[1] * right[2] + left[2] * right[1]);
  out[4] = S(InvSqrt2) * (
      left[2] * right[0] + left[0] * right[2]);
}

template <typename S>
__device__ void add_stf5_pull(
    const S* left, const S* right, const S* pull, S scale,
    S* grad_left, S* grad_right) {
  constexpr double InvSqrt6 = 0.40824829046386301637;
  constexpr double InvSqrt2 = 0.70710678118654752440;
  const S g0 = scale * pull[0];
  const S g1 = scale * pull[1];
  const S g2 = scale * pull[2];
  const S g3 = scale * pull[3];
  const S g4 = scale * pull[4];
#define STF_SIDE(target, other)                                             \
  do {                                                                      \
    (target)[0] += (-g0 * S(InvSqrt6) + g1 * S(InvSqrt2)) * (other)[0] +   \
        g2 * S(InvSqrt2) * (other)[1] + g4 * S(InvSqrt2) * (other)[2];     \
    (target)[1] += g2 * S(InvSqrt2) * (other)[0] +                         \
        (-g0 * S(InvSqrt6) - g1 * S(InvSqrt2)) * (other)[1] +              \
        g3 * S(InvSqrt2) * (other)[2];                                     \
    (target)[2] += g4 * S(InvSqrt2) * (other)[0] +                         \
        g3 * S(InvSqrt2) * (other)[1] + S(2) * g0 * S(InvSqrt6) *         \
        (other)[2];                                                         \
  } while (false)
  STF_SIDE(grad_left, right);
  STF_SIDE(grad_right, left);
#undef STF_SIDE
}

template <typename S>
__device__ void radial_basis(
    S distance, double cutoff, S* basis, S* derivative) {
  constexpr double Pi = 3.14159265358979323846;
  const S inv_rc = S(1.0 / cutoff);
  const S arg = S(Pi) * distance * inv_rc;
  const S fc = S(0.5) * cosine(arg) + S(0.5);
  const S fcp = S(-0.5 * Pi / cutoff) * sine(arg);
  const S u = distance * inv_rc - S(1);
  const S x = S(2) * u * u - S(1);
  const S dxdr = S(4.0 / cutoff) * u;
  S cheb[B], second[B];
  cheb[0] = S(1);
  cheb[1] = x;
  second[0] = S(1);
  second[1] = S(2) * x;
  for (int k = 2; k < B; ++k) {
    cheb[k] = S(2) * x * cheb[k - 1] - cheb[k - 2];
    second[k] = S(2) * x * second[k - 1] - second[k - 2];
  }
  for (int k = 0; k < B; ++k) {
    const S core = S(0.5) * (cheb[k] + S(1));
    basis[k] = core * fc;
    if (k == 0) {
      derivative[k] = fcp;
    } else {
      derivative[k] = S(0.5) * S(k) * second[k - 1] * dxdr * fc +
          core * fcp;
    }
  }
}

template <typename S>
struct EdgeState {
  S dist;
  S rhat[3];
  S si[3];
  S sj[3];
  S basis[B];
  S dbasis[B];
  S weight[C];
  S dot;
  S si2;
  S sj2;
  S omega[3];
  S axial[3];
  S qrr[5];
  S edge_stf[5];
  S longitudinal;
  bool radial_active;
};

template <typename S>
__device__ void make_state(
    const S* rij, const S* si, const S* sj, const S* coefficient,
    bool active, double cutoff, double eps, EdgeState<S>* state) {
  const S norm2 = dot3(rij, rij);
  if (value(norm2) < eps * eps) {
    state->dist = S(eps);
  } else {
    state->dist = root(norm2);
  }
  for (int d = 0; d < 3; ++d) {
    state->rhat[d] = rij[d] / state->dist;
    state->si[d] = si[d];
    state->sj[d] = sj[d];
  }
  state->radial_active = active && value(state->dist) < cutoff;
  radial_basis(state->dist, cutoff, state->basis, state->dbasis);
  for (int c = 0; c < C; ++c) {
    state->weight[c] = S(0);
    if (state->radial_active) {
      for (int b = 0; b < B; ++b) {
        state->weight[c] += coefficient[c * B + b] * state->basis[b];
      }
    }
  }
  state->dot = dot3(state->si, state->sj);
  state->si2 = dot3(state->si, state->si);
  state->sj2 = dot3(state->sj, state->sj);
  cross3(state->si, state->sj, state->omega);
  cross3(state->rhat, state->sj, state->axial);
  stf5(state->rhat, state->rhat, state->qrr);
  stf5(state->rhat, state->sj, state->edge_stf);
  state->longitudinal = dot3(state->rhat, state->sj);
}

template <typename S>
__device__ void forward_edge(
    const EdgeState<S>& state, S* bank, S* aux) {
  for (int k = 0; k < BankWidth; ++k) bank[k] = S(0);
  for (int k = 0; k < AuxWidth; ++k) aux[k] = S(0);
  for (int c = 0; c < C; ++c) {
    const int base = c * ChannelStride;
    const S w = state.weight[c];
    for (int d = 0; d < 3; ++d) {
      bank[base + M + d] = w * state.sj[d];
      bank[base + P + d] = w * state.rhat[d];
      bank[base + X + d] = w * state.axial[d];
      bank[base + DM + d] = w * state.dot * state.sj[d];
    }
    bank[base + L] = w * state.longitudinal;
    for (int k = 0; k < 5; ++k) {
      bank[base + T5 + k] = w * state.edge_stf[k];
      bank[base + Q + k] = w * state.qrr[k];
      for (int d = 0; d < 3; ++d) {
        bank[base + QP + 3 * k + d] =
            w * state.qrr[k] * state.sj[d];
      }
    }
    bank[EdgeSj2 + c] = w * state.sj2;
    bank[EdgeDot2 + c] = w * state.dot * state.dot;
    bank[EdgeGate + c] =
        w * state.dot * (state.si2 + state.sj2);
    aux[c] = w;
  }
  int pair = 0;
  for (int left = 0; left < C; ++left) {
    for (int right = left; right < C; ++right, ++pair) {
      bank[Same + pair] = state.weight[left] * state.weight[right] *
          state.dot * state.dot;
    }
  }
  for (int d = 0; d < 3; ++d) {
    aux[2 + d] = state.rhat[d];
    aux[5 + d] = state.omega[d];
  }
}

template <typename S>
__device__ void reverse_edge(
    const EdgeState<S>& state, const S* coefficient,
    const S* grad_bank, const S* grad_aux,
    S* grad_rij, S* grad_si, S* grad_sj, S* grad_coefficient) {
  S grad_weight[C] = {S(0), S(0)};
  S grad_rhat[3] = {S(0), S(0), S(0)};
  S grad_q[5] = {S(0), S(0), S(0), S(0), S(0)};
  S grad_dot = S(0);
  S grad_longitudinal = S(0);
  for (int d = 0; d < 3; ++d) {
    grad_rij[d] = S(0);
    grad_si[d] = S(0);
    grad_sj[d] = S(0);
  }
  for (int k = 0; k < C * B; ++k) grad_coefficient[k] = S(0);

  for (int c = 0; c < C; ++c) {
    const int base = c * ChannelStride;
    const S w = state.weight[c];
    const S* gm = grad_bank + base + M;
    const S* gp = grad_bank + base + P;
    const S* gx = grad_bank + base + X;
    const S* gdm = grad_bank + base + DM;
    const S gl = grad_bank[base + L];
    const S* gt = grad_bank + base + T5;
    const S* gq = grad_bank + base + Q;
    const S* gqp = grad_bank + base + QP;
    grad_weight[c] += grad_aux[c];
    grad_weight[c] += dot3(gm, state.sj) + dot3(gp, state.rhat) +
        dot3(gx, state.axial) + gl * state.longitudinal;
    for (int k = 0; k < 5; ++k) {
      grad_weight[c] += gt[k] * state.edge_stf[k] + gq[k] * state.qrr[k];
      for (int d = 0; d < 3; ++d) {
        grad_weight[c] += gqp[3 * k + d] * state.qrr[k] * state.sj[d];
      }
    }
    const S dm_contract = dot3(gdm, state.sj);
    grad_weight[c] += state.dot * dm_contract;
    for (int d = 0; d < 3; ++d) {
      grad_sj[d] += w * gm[d];
      grad_rhat[d] += w * gp[d];
      grad_sj[d] += w * state.dot * gdm[d];
    }
    grad_dot += w * dm_contract;
    grad_longitudinal += w * gl;
    S axial_r[3] = {S(0), S(0), S(0)};
    S axial_s[3] = {S(0), S(0), S(0)};
    add_cross_pull(state.rhat, state.sj, gx, axial_r, axial_s);
    for (int d = 0; d < 3; ++d) {
      grad_rhat[d] += w * axial_r[d];
      grad_sj[d] += w * axial_s[d];
    }
    S stf_r[3] = {S(0), S(0), S(0)};
    S stf_s[3] = {S(0), S(0), S(0)};
    add_stf5_pull(state.rhat, state.sj, gt, w, stf_r, stf_s);
    for (int d = 0; d < 3; ++d) {
      grad_rhat[d] += stf_r[d];
      grad_sj[d] += stf_s[d];
    }
    for (int k = 0; k < 5; ++k) {
      grad_q[k] += w * gq[k];
      for (int d = 0; d < 3; ++d) {
        grad_q[k] += w * gqp[3 * k + d] * state.sj[d];
        grad_sj[d] += w * gqp[3 * k + d] * state.qrr[k];
      }
    }

    const S a_sj2 = grad_bank[EdgeSj2 + c];
    grad_weight[c] += a_sj2 * state.sj2;
    for (int d = 0; d < 3; ++d) {
      grad_sj[d] += S(2) * w * a_sj2 * state.sj[d];
    }
    const S a_dot2 = grad_bank[EdgeDot2 + c];
    grad_weight[c] += a_dot2 * state.dot * state.dot;
    grad_dot += S(2) * w * a_dot2 * state.dot;
    const S a_gate = grad_bank[EdgeGate + c];
    const S spin_norm = state.si2 + state.sj2;
    grad_weight[c] += a_gate * state.dot * spin_norm;
    grad_dot += w * a_gate * spin_norm;
    for (int d = 0; d < 3; ++d) {
      grad_si[d] += S(2) * w * a_gate * state.dot * state.si[d];
      grad_sj[d] += S(2) * w * a_gate * state.dot * state.sj[d];
    }
  }

  int pair = 0;
  for (int left = 0; left < C; ++left) {
    for (int right = left; right < C; ++right, ++pair) {
      const S a = grad_bank[Same + pair];
      const S dot2 = state.dot * state.dot;
      if (left == right) {
        grad_weight[left] += S(2) * a * state.weight[left] * dot2;
      } else {
        grad_weight[left] += a * state.weight[right] * dot2;
        grad_weight[right] += a * state.weight[left] * dot2;
      }
      grad_dot += S(2) * a * state.weight[left] *
          state.weight[right] * state.dot;
    }
  }

  for (int d = 0; d < 3; ++d) grad_rhat[d] += grad_aux[2 + d];
  S omega_pull[3] = {grad_aux[5], grad_aux[6], grad_aux[7]};
  add_cross_pull(state.si, state.sj, omega_pull, grad_si, grad_sj);
  S qr_left[3] = {S(0), S(0), S(0)};
  S qr_right[3] = {S(0), S(0), S(0)};
  add_stf5_pull(
      state.rhat, state.rhat, grad_q, S(1), qr_left, qr_right);
  for (int d = 0; d < 3; ++d) {
    grad_rhat[d] += qr_left[d] + qr_right[d] +
        grad_longitudinal * state.sj[d];
    grad_sj[d] += grad_longitudinal * state.rhat[d];
    grad_si[d] += grad_dot * state.sj[d];
    grad_sj[d] += grad_dot * state.si[d];
  }

  S grad_dist = S(0);
  if (state.radial_active) {
    for (int c = 0; c < C; ++c) {
      for (int b = 0; b < B; ++b) {
        grad_coefficient[c * B + b] +=
            grad_weight[c] * state.basis[b];
        grad_dist += grad_weight[c] * coefficient[c * B + b] *
            state.dbasis[b];
      }
    }
  }
  const S radial_dot = dot3(grad_rhat, state.rhat);
  for (int d = 0; d < 3; ++d) {
    grad_rij[d] = grad_dist * state.rhat[d] +
        (grad_rhat[d] - radial_dot * state.rhat[d]) / state.dist;
  }
}

template <typename S>
__device__ S triple3(const S* a, const S* b, const S* c) {
  S crossed[3];
  cross3(a, b, crossed);
  return dot3(crossed, c);
}

template <typename S>
__device__ void add_triple3_pull(
    const S* a, const S* b, const S* c, S scale,
    S* grad_a, S* grad_b, S* grad_c) {
  S b_cross_c[3], c_cross_a[3], a_cross_b[3];
  cross3(b, c, b_cross_c);
  cross3(c, a, c_cross_a);
  cross3(a, b, a_cross_b);
  for (int d = 0; d < 3; ++d) {
    grad_a[d] += scale * b_cross_c[d];
    grad_b[d] += scale * c_cross_a[d];
    grad_c[d] += scale * a_cross_b[d];
  }
}

template <typename S>
__device__ void stf_to_matrix(const S* q, S* matrix) {
  constexpr double InvSqrt6 = 0.40824829046386301637;
  constexpr double InvSqrt2 = 0.70710678118654752440;
  matrix[0] = -q[0] * S(InvSqrt6) + q[1] * S(InvSqrt2);
  matrix[4] = -q[0] * S(InvSqrt6) - q[1] * S(InvSqrt2);
  matrix[8] = S(2) * q[0] * S(InvSqrt6);
  matrix[1] = matrix[3] = q[2] * S(InvSqrt2);
  matrix[5] = matrix[7] = q[3] * S(InvSqrt2);
  matrix[2] = matrix[6] = q[4] * S(InvSqrt2);
}

template <typename S>
__device__ void add_matrix_stf_pull(const S* matrix, S* grad_q) {
  constexpr double InvSqrt6 = 0.40824829046386301637;
  constexpr double InvSqrt2 = 0.70710678118654752440;
  grad_q[0] += (-matrix[0] - matrix[4] + S(2) * matrix[8]) * S(InvSqrt6);
  grad_q[1] += (matrix[0] - matrix[4]) * S(InvSqrt2);
  grad_q[2] += (matrix[1] + matrix[3]) * S(InvSqrt2);
  grad_q[3] += (matrix[5] + matrix[7]) * S(InvSqrt2);
  grad_q[4] += (matrix[2] + matrix[6]) * S(InvSqrt2);
}

template <typename S>
__device__ void stf_matvec(const S* q, const S* vector, S* output) {
  S matrix[9];
  stf_to_matrix(q, matrix);
  for (int row = 0; row < 3; ++row) {
    output[row] = S(0);
    for (int column = 0; column < 3; ++column) {
      output[row] += matrix[3 * row + column] * vector[column];
    }
  }
}

template <typename S>
__device__ void add_stf_matvec_pull(
    const S* q, const S* vector, const S* pull,
    S* grad_q, S* grad_vector) {
  S matrix[9], grad_matrix[9];
  stf_to_matrix(q, matrix);
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      grad_vector[column] += matrix[3 * row + column] * pull[row];
      grad_matrix[3 * row + column] = pull[row] * vector[column];
    }
  }
  add_matrix_stf_pull(grad_matrix, grad_q);
}

template <typename S, int D>
__device__ void zero_channels(S values[C][D]) {
  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < D; ++d) values[c][d] = S(0);
  }
}

template <typename S>
struct EdgeAnchorState {
  S weight[C];
  S rhat[3];
  S omega[3];
  S polar[C][3];
  S geometry_l2[C][5];
  S polar0[C][3];
  S polar1[C][3];
  S q0[C][5];
  S edge_weight[C];
};

template <typename S>
__device__ void make_edge_anchor_state(
    const S polar[C][3], const S geometry_l2[C][5],
    const S* auxiliary, const S* mix, EdgeAnchorState<S>* state) {
  for (int c = 0; c < C; ++c) {
    state->weight[c] = auxiliary[c];
    state->edge_weight[c] = S(0);
    for (int d = 0; d < 3; ++d) state->polar[c][d] = polar[c][d];
    for (int k = 0; k < 5; ++k) {
      state->geometry_l2[c][k] = geometry_l2[c][k];
    }
  }
  for (int d = 0; d < 3; ++d) {
    state->rhat[d] = auxiliary[2 + d];
    state->omega[d] = auxiliary[5 + d];
  }
  for (int r = 0; r < C; ++r) {
    for (int d = 0; d < 3; ++d) {
      state->polar0[r][d] = S(0);
      state->polar1[r][d] = S(0);
    }
    for (int k = 0; k < 5; ++k) state->q0[r][k] = S(0);
    for (int c = 0; c < C; ++c) {
      const S mix0 = mix[r * C + c];
      const S mix1 = mix[4 + r * C + c];
      state->edge_weight[r] += mix0 * state->weight[c];
      for (int d = 0; d < 3; ++d) {
        state->polar0[r][d] += mix0 * state->polar[c][d];
        state->polar1[r][d] += mix1 * state->polar[c][d];
      }
      for (int k = 0; k < 5; ++k) {
        state->q0[r][k] += mix0 * state->geometry_l2[c][k];
      }
    }
  }
}

template <typename S>
__device__ void edge_anchor_forward(
    const EdgeAnchorState<S>& state, S* output) {
  const S edge_response = dot3(state.rhat, state.omega);
  for (int r = 0; r < C; ++r) {
    S q_on_r[3], q_on_p1[3];
    stf_matvec(state.q0[r], state.rhat, q_on_r);
    stf_matvec(state.q0[r], state.polar1[r], q_on_p1);
    output[r] = state.edge_weight[r] * triple3(
        state.polar0[r], state.rhat, state.omega);
    output[2 + r] = state.edge_weight[r] * triple3(
        state.rhat, q_on_r, state.omega);
    output[4 + r] = state.edge_weight[r] * edge_response * triple3(
        state.polar0[r], state.polar1[r], q_on_p1);
  }
}

template <typename S>
__device__ void edge_anchor_reverse(
    const EdgeAnchorState<S>& state, const S* mix, const S* grad_output,
    S grad_polar[C][3], S grad_geometry_l2[C][5],
    S* grad_auxiliary, S* grad_mix) {
  zero_channels<S, 3>(grad_polar);
  zero_channels<S, 5>(grad_geometry_l2);
  for (int k = 0; k < AuxWidth; ++k) grad_auxiliary[k] = S(0);
  for (int k = 0; k < 16; ++k) grad_mix[k] = S(0);
  S grad_polar0[C][3], grad_polar1[C][3], grad_q0[C][5];
  zero_channels<S, 3>(grad_polar0);
  zero_channels<S, 3>(grad_polar1);
  zero_channels<S, 5>(grad_q0);
  S grad_edge_weight[C] = {S(0), S(0)};
  S grad_rhat[3] = {S(0), S(0), S(0)};
  S grad_omega[3] = {S(0), S(0), S(0)};
  const S edge_response = dot3(state.rhat, state.omega);

  for (int r = 0; r < C; ++r) {
    S q_on_r[3], q_on_p1[3];
    stf_matvec(state.q0[r], state.rhat, q_on_r);
    stf_matvec(state.q0[r], state.polar1[r], q_on_p1);

    const S l11 = triple3(state.polar0[r], state.rhat, state.omega);
    grad_edge_weight[r] += grad_output[r] * l11;
    add_triple3_pull(
        state.polar0[r], state.rhat, state.omega,
        grad_output[r] * state.edge_weight[r],
        grad_polar0[r], grad_rhat, grad_omega);

    const S l22 = triple3(state.rhat, q_on_r, state.omega);
    grad_edge_weight[r] += grad_output[2 + r] * l22;
    S grad_q_on_r[3] = {S(0), S(0), S(0)};
    add_triple3_pull(
        state.rhat, q_on_r, state.omega,
        grad_output[2 + r] * state.edge_weight[r],
        grad_rhat, grad_q_on_r, grad_omega);
    add_stf_matvec_pull(
        state.q0[r], state.rhat, grad_q_on_r,
        grad_q0[r], grad_rhat);

    const S mixed = triple3(
        state.polar0[r], state.polar1[r], q_on_p1);
    const S a112 = grad_output[4 + r];
    grad_edge_weight[r] += a112 * edge_response * mixed;
    const S grad_edge_response = a112 * state.edge_weight[r] * mixed;
    for (int d = 0; d < 3; ++d) {
      grad_rhat[d] += grad_edge_response * state.omega[d];
      grad_omega[d] += grad_edge_response * state.rhat[d];
    }
    S grad_q_on_p1[3] = {S(0), S(0), S(0)};
    add_triple3_pull(
        state.polar0[r], state.polar1[r], q_on_p1,
        a112 * state.edge_weight[r] * edge_response,
        grad_polar0[r], grad_polar1[r], grad_q_on_p1);
    add_stf_matvec_pull(
        state.q0[r], state.polar1[r], grad_q_on_p1,
        grad_q0[r], grad_polar1[r]);
  }

  for (int r = 0; r < C; ++r) {
    for (int c = 0; c < C; ++c) {
      const S mix0 = mix[r * C + c];
      const S mix1 = mix[4 + r * C + c];
      grad_auxiliary[c] += mix0 * grad_edge_weight[r];
      grad_mix[r * C + c] += grad_edge_weight[r] * state.weight[c];
      for (int d = 0; d < 3; ++d) {
        grad_polar[c][d] += mix0 * grad_polar0[r][d] +
            mix1 * grad_polar1[r][d];
        grad_mix[r * C + c] +=
            grad_polar0[r][d] * state.polar[c][d];
        grad_mix[4 + r * C + c] +=
            grad_polar1[r][d] * state.polar[c][d];
      }
      for (int k = 0; k < 5; ++k) {
        grad_geometry_l2[c][k] += mix0 * grad_q0[r][k];
        grad_mix[r * C + c] +=
            grad_q0[r][k] * state.geometry_l2[c][k];
      }
    }
  }
  for (int d = 0; d < 3; ++d) {
    grad_auxiliary[2 + d] += grad_rhat[d];
    grad_auxiliary[5 + d] += grad_omega[d];
  }
}

template <typename scalar_t>
__global__ void edge_anchor_forward_kernel(
    const scalar_t* bank, const scalar_t* auxiliary, const int64_t* pair_i,
    const scalar_t* mix, int64_t edges, scalar_t* output) {
  const int64_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= edges) return;
  const int64_t atom = pair_i[edge];
  scalar_t polar[C][3], geometry_l2[C][5];
  for (int c = 0; c < C; ++c) {
    const int base = atom * BankWidth + c * ChannelStride;
    for (int d = 0; d < 3; ++d) polar[c][d] = bank[base + P + d];
    for (int k = 0; k < 5; ++k) geometry_l2[c][k] = bank[base + Q + k];
  }
  EdgeAnchorState<scalar_t> state;
  make_edge_anchor_state(
      polar, geometry_l2, auxiliary + edge * AuxWidth, mix, &state);
  scalar_t local_output[EdgeAnchorWidth];
  edge_anchor_forward(state, local_output);
  for (int k = 0; k < EdgeAnchorWidth; ++k) {
    atomicAdd(output + atom * EdgeAnchorWidth + k, local_output[k]);
  }
}

template <typename scalar_t>
__global__ void edge_anchor_backward_kernel(
    const scalar_t* grad_output, const scalar_t* bank,
    const scalar_t* auxiliary, const int64_t* pair_i, const scalar_t* mix,
    int64_t edges, scalar_t* grad_bank, scalar_t* grad_auxiliary,
    scalar_t* grad_mix) {
  const int64_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= edges) return;
  const int64_t atom = pair_i[edge];
  scalar_t polar[C][3], geometry_l2[C][5];
  for (int c = 0; c < C; ++c) {
    const int base = atom * BankWidth + c * ChannelStride;
    for (int d = 0; d < 3; ++d) polar[c][d] = bank[base + P + d];
    for (int k = 0; k < 5; ++k) geometry_l2[c][k] = bank[base + Q + k];
  }
  EdgeAnchorState<scalar_t> state;
  make_edge_anchor_state(
      polar, geometry_l2, auxiliary + edge * AuxWidth, mix, &state);
  scalar_t grad_polar[C][3], grad_q[C][5];
  scalar_t local_aux[AuxWidth], local_mix[16];
  edge_anchor_reverse(
      state, mix, grad_output + atom * EdgeAnchorWidth,
      grad_polar, grad_q, local_aux, local_mix);
  for (int c = 0; c < C; ++c) {
    const int base = atom * BankWidth + c * ChannelStride;
    for (int d = 0; d < 3; ++d) {
      atomicAdd(grad_bank + base + P + d, grad_polar[c][d]);
    }
    for (int k = 0; k < 5; ++k) {
      atomicAdd(grad_bank + base + Q + k, grad_q[c][k]);
    }
  }
  for (int k = 0; k < AuxWidth; ++k) {
    grad_auxiliary[edge * AuxWidth + k] = local_aux[k];
  }
  for (int k = 0; k < 16; ++k) atomicAdd(grad_mix + k, local_mix[k]);
}

template <typename scalar_t>
__global__ void edge_anchor_double_backward_kernel(
    const scalar_t* grad_output, const scalar_t* bank,
    const scalar_t* auxiliary, const int64_t* pair_i, const scalar_t* mix,
    const scalar_t* h_bank, const scalar_t* h_auxiliary,
    const scalar_t* h_mix, int64_t edges, scalar_t* grad2_output,
    scalar_t* grad2_bank, scalar_t* grad2_auxiliary, scalar_t* grad2_mix) {
  const int64_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= edges) return;
  const int64_t atom = pair_i[edge];
  Dual<scalar_t> polar[C][3], geometry_l2[C][5];
  Dual<scalar_t> local_auxiliary[AuxWidth], local_mix[16];
  for (int c = 0; c < C; ++c) {
    const int base = atom * BankWidth + c * ChannelStride;
    for (int d = 0; d < 3; ++d) {
      polar[c][d] = {bank[base + P + d], h_bank[base + P + d]};
    }
    for (int k = 0; k < 5; ++k) {
      geometry_l2[c][k] = {bank[base + Q + k], h_bank[base + Q + k]};
    }
  }
  for (int k = 0; k < AuxWidth; ++k) {
    local_auxiliary[k] = {
        auxiliary[edge * AuxWidth + k], h_auxiliary[edge * AuxWidth + k]};
  }
  for (int k = 0; k < 16; ++k) local_mix[k] = {mix[k], h_mix[k]};
  EdgeAnchorState<Dual<scalar_t>> state;
  make_edge_anchor_state(
      polar, geometry_l2, local_auxiliary, local_mix, &state);
  Dual<scalar_t> local_output[EdgeAnchorWidth];
  edge_anchor_forward(state, local_output);
  for (int k = 0; k < EdgeAnchorWidth; ++k) {
    atomicAdd(
        grad2_output + atom * EdgeAnchorWidth + k, local_output[k].d);
  }
  Dual<scalar_t> local_grad_output[EdgeAnchorWidth];
  for (int k = 0; k < EdgeAnchorWidth; ++k) {
    local_grad_output[k] = {
        grad_output[atom * EdgeAnchorWidth + k], scalar_t(0)};
  }
  Dual<scalar_t> grad_polar[C][3], grad_q[C][5];
  Dual<scalar_t> local_grad_aux[AuxWidth], local_grad_mix[16];
  edge_anchor_reverse(
      state, local_mix, local_grad_output, grad_polar, grad_q,
      local_grad_aux, local_grad_mix);
  for (int c = 0; c < C; ++c) {
    const int base = atom * BankWidth + c * ChannelStride;
    for (int d = 0; d < 3; ++d) {
      atomicAdd(grad2_bank + base + P + d, grad_polar[c][d].d);
    }
    for (int k = 0; k < 5; ++k) {
      atomicAdd(grad2_bank + base + Q + k, grad_q[c][k].d);
    }
  }
  for (int k = 0; k < AuxWidth; ++k) {
    grad2_auxiliary[edge * AuxWidth + k] = local_grad_aux[k].d;
  }
  for (int k = 0; k < 16; ++k) {
    atomicAdd(grad2_mix + k, local_grad_mix[k].d);
  }
}

template <typename scalar_t>
__global__ void bank_forward_kernel(
    const scalar_t* rij, const int64_t* pair_i, const int64_t* pair_j,
    const scalar_t* spins, const int64_t* atom_types,
    const scalar_t* coefficient, const bool* dof_active,
    const bool* env_active, int64_t edges, int64_t types,
    double cutoff, double eps, scalar_t* bank, scalar_t* aux) {
  const int64_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= edges) return;
  const int64_t i = pair_i[edge];
  const int64_t j = pair_j[edge];
  const int64_t ti = atom_types[i];
  const int64_t tj = atom_types[j];
  const scalar_t* coeff = coefficient + ((ti * types + tj) * C * B);
  EdgeState<scalar_t> state;
  make_state(
      rij + 3 * edge, spins + 3 * i, spins + 3 * j, coeff,
      dof_active[ti] && env_active[tj], cutoff, eps, &state);
  scalar_t local_bank[BankWidth], local_aux[AuxWidth];
  forward_edge(state, local_bank, local_aux);
  for (int k = 0; k < BankWidth; ++k) {
    atomicAdd(bank + i * BankWidth + k, local_bank[k]);
  }
  for (int k = 0; k < AuxWidth; ++k) {
    aux[edge * AuxWidth + k] = local_aux[k];
  }
}

template <typename scalar_t>
__global__ void bank_backward_kernel(
    const scalar_t* grad_bank, const scalar_t* grad_aux,
    const scalar_t* rij, const int64_t* pair_i, const int64_t* pair_j,
    const scalar_t* spins, const int64_t* atom_types,
    const scalar_t* coefficient, const bool* dof_active,
    const bool* env_active, int64_t edges, int64_t types,
    double cutoff, double eps, scalar_t* grad_rij,
    scalar_t* grad_spins, scalar_t* grad_coefficient) {
  const int64_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= edges) return;
  const int64_t i = pair_i[edge];
  const int64_t j = pair_j[edge];
  const int64_t ti = atom_types[i];
  const int64_t tj = atom_types[j];
  const int64_t coefficient_offset = (ti * types + tj) * C * B;
  const scalar_t* coeff = coefficient + coefficient_offset;
  EdgeState<scalar_t> state;
  make_state(
      rij + 3 * edge, spins + 3 * i, spins + 3 * j, coeff,
      dof_active[ti] && env_active[tj], cutoff, eps, &state);
  scalar_t local_rij[3], local_si[3], local_sj[3], local_coeff[C * B];
  reverse_edge(
      state, coeff, grad_bank + i * BankWidth,
      grad_aux + edge * AuxWidth, local_rij, local_si, local_sj, local_coeff);
  for (int d = 0; d < 3; ++d) {
    grad_rij[3 * edge + d] = local_rij[d];
    atomicAdd(grad_spins + 3 * i + d, local_si[d]);
    atomicAdd(grad_spins + 3 * j + d, local_sj[d]);
  }
  for (int k = 0; k < C * B; ++k) {
    atomicAdd(grad_coefficient + coefficient_offset + k, local_coeff[k]);
  }
}

template <typename scalar_t>
__global__ void bank_double_backward_kernel(
    const scalar_t* grad_bank, const scalar_t* grad_aux,
    const scalar_t* rij, const int64_t* pair_i, const int64_t* pair_j,
    const scalar_t* spins, const int64_t* atom_types,
    const scalar_t* coefficient, const bool* dof_active,
    const bool* env_active, const scalar_t* h_rij,
    const scalar_t* h_spins, const scalar_t* h_coefficient,
    int64_t edges, int64_t types, double cutoff, double eps,
    scalar_t* grad2_bank, scalar_t* grad2_aux, scalar_t* grad2_rij,
    scalar_t* grad2_spins, scalar_t* grad2_coefficient) {
  const int64_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= edges) return;
  const int64_t i = pair_i[edge];
  const int64_t j = pair_j[edge];
  const int64_t ti = atom_types[i];
  const int64_t tj = atom_types[j];
  const int64_t coefficient_offset = (ti * types + tj) * C * B;
  Dual<scalar_t> rr[3], si[3], sj[3], coeff[C * B];
  for (int d = 0; d < 3; ++d) {
    rr[d] = {rij[3 * edge + d], h_rij[3 * edge + d]};
    si[d] = {spins[3 * i + d], h_spins[3 * i + d]};
    sj[d] = {spins[3 * j + d], h_spins[3 * j + d]};
  }
  for (int k = 0; k < C * B; ++k) {
    coeff[k] = {
        coefficient[coefficient_offset + k],
        h_coefficient[coefficient_offset + k]};
  }
  EdgeState<Dual<scalar_t>> state;
  make_state(
      rr, si, sj, coeff, dof_active[ti] && env_active[tj],
      cutoff, eps, &state);
  Dual<scalar_t> local_bank[BankWidth], local_aux[AuxWidth];
  forward_edge(state, local_bank, local_aux);
  for (int k = 0; k < BankWidth; ++k) {
    atomicAdd(grad2_bank + i * BankWidth + k, local_bank[k].d);
  }
  for (int k = 0; k < AuxWidth; ++k) {
    grad2_aux[edge * AuxWidth + k] = local_aux[k].d;
  }

  Dual<scalar_t> go_bank[BankWidth], go_aux[AuxWidth];
  for (int k = 0; k < BankWidth; ++k) {
    go_bank[k] = {grad_bank[i * BankWidth + k], scalar_t(0)};
  }
  for (int k = 0; k < AuxWidth; ++k) {
    go_aux[k] = {grad_aux[edge * AuxWidth + k], scalar_t(0)};
  }
  // The cotangent-direction inputs are the gradients of the VJP outputs.
  // Their pull into grad_bank/grad_aux is handled by the forward directional
  // bank above; the reverse tangent here supplies H(x)h for x plus J^T h_g.
  Dual<scalar_t> local_rij[3], local_si[3], local_sj[3];
  Dual<scalar_t> local_coeff[C * B];
  reverse_edge(
      state, coeff, go_bank, go_aux,
      local_rij, local_si, local_sj, local_coeff);
  for (int d = 0; d < 3; ++d) {
    grad2_rij[3 * edge + d] = local_rij[d].d;
    atomicAdd(grad2_spins + 3 * i + d, local_si[d].d);
    atomicAdd(grad2_spins + 3 * j + d, local_sj[d].d);
  }
  for (int k = 0; k < C * B; ++k) {
    atomicAdd(
        grad2_coefficient + coefficient_offset + k, local_coeff[k].d);
  }
}

void check_inputs(
    const torch::Tensor& rij, const torch::Tensor& pair_i,
    const torch::Tensor& pair_j, const torch::Tensor& spins,
    const torch::Tensor& atom_types, const torch::Tensor& coefficient,
    const torch::Tensor& dof_active, const torch::Tensor& env_active) {
  TORCH_CHECK(
      rij.is_cuda() && pair_i.is_cuda() && pair_j.is_cuda() &&
      spins.is_cuda() && atom_types.is_cuda() && coefficient.is_cuda() &&
      dof_active.is_cuda() && env_active.is_cuda(),
      "all density-bank tensors must be CUDA tensors");
  TORCH_CHECK(
      rij.is_contiguous() && pair_i.is_contiguous() && pair_j.is_contiguous() &&
      spins.is_contiguous() && atom_types.is_contiguous() &&
      coefficient.is_contiguous() && dof_active.is_contiguous() &&
      env_active.is_contiguous(),
      "all density-bank tensors must be contiguous");
  TORCH_CHECK(rij.dim() == 2 && rij.size(1) == 3, "rij must be [edges,3]");
  TORCH_CHECK(spins.dim() == 2 && spins.size(1) == 3, "spins must be [atoms,3]");
  TORCH_CHECK(pair_i.scalar_type() == torch::kInt64 &&
              pair_j.scalar_type() == torch::kInt64 &&
              atom_types.scalar_type() == torch::kInt64,
              "pair indices and atom types must be int64");
  TORCH_CHECK(dof_active.scalar_type() == torch::kBool &&
              env_active.scalar_type() == torch::kBool,
              "active masks must be bool");
  TORCH_CHECK(pair_i.numel() == rij.size(0) && pair_j.numel() == rij.size(0),
              "pair indices must match edge count");
  TORCH_CHECK(coefficient.scalar_type() == rij.scalar_type() &&
              spins.scalar_type() == rij.scalar_type(),
              "floating tensors must have the same dtype");
  TORCH_CHECK(coefficient.dim() == 5 && coefficient.size(0) == 1 &&
              coefficient.size(3) == C && coefficient.size(4) == B &&
              coefficient.size(1) == coefficient.size(2),
              "coefficient must be [1,T,T,2,9]");
}

void check_edge_anchor_inputs(
    const torch::Tensor& bank, const torch::Tensor& auxiliary,
    const torch::Tensor& pair_i, const torch::Tensor& mix) {
  TORCH_CHECK(
      bank.is_cuda() && auxiliary.is_cuda() && pair_i.is_cuda() && mix.is_cuda(),
      "all edge-anchor tensors must be CUDA tensors");
  TORCH_CHECK(
      bank.is_contiguous() && auxiliary.is_contiguous() &&
      pair_i.is_contiguous() && mix.is_contiguous(),
      "all edge-anchor tensors must be contiguous");
  TORCH_CHECK(bank.dim() == 2 && bank.size(1) == BankWidth,
              "bank must be [atoms,85]");
  TORCH_CHECK(auxiliary.dim() == 2 && auxiliary.size(1) == AuxWidth,
              "auxiliary must be [edges,8]");
  TORCH_CHECK(pair_i.dim() == 1 && pair_i.numel() == auxiliary.size(0) &&
              pair_i.scalar_type() == torch::kInt64,
              "pair_i must be int64 [edges]");
  TORCH_CHECK(mix.dim() == 3 && mix.size(0) == 4 && mix.size(1) == C &&
              mix.size(2) == C, "radial_leg_mix must be [4,2,2]");
  TORCH_CHECK(bank.scalar_type() == auxiliary.scalar_type() &&
              bank.scalar_type() == mix.scalar_type(),
              "edge-anchor floating tensors must share dtype");
}

}  // namespace

std::vector<torch::Tensor> density_bank_forward(
    torch::Tensor rij, torch::Tensor pair_i, torch::Tensor pair_j,
    torch::Tensor spins, torch::Tensor atom_types, torch::Tensor coefficient,
    torch::Tensor dof_active, torch::Tensor env_active,
    double cutoff, double eps) {
  check_inputs(rij, pair_i, pair_j, spins, atom_types, coefficient,
               dof_active, env_active);
  c10::cuda::CUDAGuard guard(rij.device());
  auto bank = torch::zeros({spins.size(0), BankWidth}, rij.options());
  auto aux = torch::empty({rij.size(0), AuxWidth}, rij.options());
  const int threads = 128;
  const int blocks = static_cast<int>((rij.size(0) + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(rij.scalar_type(), "density_bank_forward", [&] {
      bank_forward_kernel<scalar_t><<<
          blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          rij.data_ptr<scalar_t>(), pair_i.data_ptr<int64_t>(),
          pair_j.data_ptr<int64_t>(), spins.data_ptr<scalar_t>(),
          atom_types.data_ptr<int64_t>(), coefficient.data_ptr<scalar_t>(),
          dof_active.data_ptr<bool>(), env_active.data_ptr<bool>(),
          rij.size(0), coefficient.size(1), cutoff, eps,
          bank.data_ptr<scalar_t>(), aux.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {bank, aux};
}

std::vector<torch::Tensor> density_bank_backward(
    torch::Tensor grad_bank, torch::Tensor grad_aux,
    torch::Tensor rij, torch::Tensor pair_i, torch::Tensor pair_j,
    torch::Tensor spins, torch::Tensor atom_types, torch::Tensor coefficient,
    torch::Tensor dof_active, torch::Tensor env_active,
    double cutoff, double eps) {
  check_inputs(rij, pair_i, pair_j, spins, atom_types, coefficient,
               dof_active, env_active);
  TORCH_CHECK(grad_bank.is_contiguous() && grad_aux.is_contiguous() &&
              grad_bank.is_cuda() && grad_aux.is_cuda(),
              "bank cotangents must be contiguous CUDA tensors");
  TORCH_CHECK(grad_bank.dim() == 2 && grad_bank.size(0) == spins.size(0) &&
              grad_bank.size(1) == BankWidth,
              "grad_bank must be [atoms,85]");
  TORCH_CHECK(grad_aux.dim() == 2 && grad_aux.size(0) == rij.size(0) &&
              grad_aux.size(1) == AuxWidth,
              "grad_aux must be [edges,8]");
  TORCH_CHECK(grad_bank.scalar_type() == rij.scalar_type() &&
              grad_aux.scalar_type() == rij.scalar_type(),
              "bank cotangents must match primal dtype");
  c10::cuda::CUDAGuard guard(rij.device());
  auto grad_rij = torch::empty_like(rij);
  auto grad_spins = torch::zeros_like(spins);
  auto grad_coefficient = torch::zeros_like(coefficient);
  const int threads = 128;
  const int blocks = static_cast<int>((rij.size(0) + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(rij.scalar_type(), "density_bank_backward", [&] {
      bank_backward_kernel<scalar_t><<<
          blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          grad_bank.data_ptr<scalar_t>(), grad_aux.data_ptr<scalar_t>(),
          rij.data_ptr<scalar_t>(), pair_i.data_ptr<int64_t>(),
          pair_j.data_ptr<int64_t>(), spins.data_ptr<scalar_t>(),
          atom_types.data_ptr<int64_t>(), coefficient.data_ptr<scalar_t>(),
          dof_active.data_ptr<bool>(), env_active.data_ptr<bool>(),
          rij.size(0), coefficient.size(1), cutoff, eps,
          grad_rij.data_ptr<scalar_t>(), grad_spins.data_ptr<scalar_t>(),
          grad_coefficient.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad_rij, grad_spins, grad_coefficient};
}

std::vector<torch::Tensor> density_bank_double_backward(
    torch::Tensor grad_bank, torch::Tensor grad_aux,
    torch::Tensor rij, torch::Tensor pair_i, torch::Tensor pair_j,
    torch::Tensor spins, torch::Tensor atom_types, torch::Tensor coefficient,
    torch::Tensor dof_active, torch::Tensor env_active,
    torch::Tensor h_rij, torch::Tensor h_spins, torch::Tensor h_coefficient,
    double cutoff, double eps) {
  check_inputs(rij, pair_i, pair_j, spins, atom_types, coefficient,
               dof_active, env_active);
  TORCH_CHECK(h_rij.sizes() == rij.sizes() && h_spins.sizes() == spins.sizes() &&
              h_coefficient.sizes() == coefficient.sizes(),
              "double-backward tangents must match primal shapes");
  TORCH_CHECK(h_rij.is_cuda() && h_spins.is_cuda() && h_coefficient.is_cuda() &&
              h_rij.is_contiguous() && h_spins.is_contiguous() &&
              h_coefficient.is_contiguous(),
              "double-backward tangents must be contiguous CUDA tensors");
  TORCH_CHECK(h_rij.scalar_type() == rij.scalar_type() &&
              h_spins.scalar_type() == rij.scalar_type() &&
              h_coefficient.scalar_type() == rij.scalar_type(),
              "double-backward tangents must match primal dtype");
  c10::cuda::CUDAGuard guard(rij.device());
  auto grad2_bank = torch::zeros_like(grad_bank);
  auto grad2_aux = torch::empty_like(grad_aux);
  auto grad2_rij = torch::empty_like(rij);
  auto grad2_spins = torch::zeros_like(spins);
  auto grad2_coefficient = torch::zeros_like(coefficient);
  const int threads = 128;
  const int blocks = static_cast<int>((rij.size(0) + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(
        rij.scalar_type(), "density_bank_double_backward", [&] {
      bank_double_backward_kernel<scalar_t><<<
          blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          grad_bank.data_ptr<scalar_t>(), grad_aux.data_ptr<scalar_t>(),
          rij.data_ptr<scalar_t>(), pair_i.data_ptr<int64_t>(),
          pair_j.data_ptr<int64_t>(), spins.data_ptr<scalar_t>(),
          atom_types.data_ptr<int64_t>(), coefficient.data_ptr<scalar_t>(),
          dof_active.data_ptr<bool>(), env_active.data_ptr<bool>(),
          h_rij.data_ptr<scalar_t>(), h_spins.data_ptr<scalar_t>(),
          h_coefficient.data_ptr<scalar_t>(), rij.size(0), coefficient.size(1),
          cutoff, eps, grad2_bank.data_ptr<scalar_t>(),
          grad2_aux.data_ptr<scalar_t>(), grad2_rij.data_ptr<scalar_t>(),
          grad2_spins.data_ptr<scalar_t>(),
          grad2_coefficient.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad2_bank, grad2_aux, grad2_rij, grad2_spins, grad2_coefficient};
}

torch::Tensor edge_anchor_sector_forward(
    torch::Tensor bank, torch::Tensor auxiliary, torch::Tensor pair_i,
    torch::Tensor mix) {
  check_edge_anchor_inputs(bank, auxiliary, pair_i, mix);
  c10::cuda::CUDAGuard guard(bank.device());
  auto output = torch::zeros({bank.size(0), EdgeAnchorWidth}, bank.options());
  const int threads = 128;
  const int blocks = static_cast<int>((auxiliary.size(0) + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(
        bank.scalar_type(), "edge_anchor_sector_forward", [&] {
      edge_anchor_forward_kernel<scalar_t><<<
          blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          bank.data_ptr<scalar_t>(), auxiliary.data_ptr<scalar_t>(),
          pair_i.data_ptr<int64_t>(), mix.data_ptr<scalar_t>(),
          auxiliary.size(0), output.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

std::vector<torch::Tensor> edge_anchor_sector_backward(
    torch::Tensor grad_output, torch::Tensor bank, torch::Tensor auxiliary,
    torch::Tensor pair_i, torch::Tensor mix) {
  check_edge_anchor_inputs(bank, auxiliary, pair_i, mix);
  TORCH_CHECK(
      grad_output.is_cuda() && grad_output.is_contiguous() &&
      grad_output.dim() == 2 && grad_output.size(0) == bank.size(0) &&
      grad_output.size(1) == EdgeAnchorWidth &&
      grad_output.scalar_type() == bank.scalar_type(),
      "grad_output must be contiguous CUDA [atoms,6]");
  c10::cuda::CUDAGuard guard(bank.device());
  auto grad_bank = torch::zeros_like(bank);
  auto grad_auxiliary = torch::empty_like(auxiliary);
  auto grad_mix = torch::zeros_like(mix);
  const int threads = 128;
  const int blocks = static_cast<int>((auxiliary.size(0) + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(
        bank.scalar_type(), "edge_anchor_sector_backward", [&] {
      edge_anchor_backward_kernel<scalar_t><<<
          blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          grad_output.data_ptr<scalar_t>(), bank.data_ptr<scalar_t>(),
          auxiliary.data_ptr<scalar_t>(), pair_i.data_ptr<int64_t>(),
          mix.data_ptr<scalar_t>(), auxiliary.size(0),
          grad_bank.data_ptr<scalar_t>(), grad_auxiliary.data_ptr<scalar_t>(),
          grad_mix.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad_bank, grad_auxiliary, grad_mix};
}

std::vector<torch::Tensor> edge_anchor_sector_double_backward(
    torch::Tensor grad_output, torch::Tensor bank, torch::Tensor auxiliary,
    torch::Tensor pair_i, torch::Tensor mix, torch::Tensor h_bank,
    torch::Tensor h_auxiliary, torch::Tensor h_mix) {
  check_edge_anchor_inputs(bank, auxiliary, pair_i, mix);
  TORCH_CHECK(
      grad_output.is_cuda() && grad_output.is_contiguous() &&
      grad_output.dim() == 2 && grad_output.size(0) == bank.size(0) &&
      grad_output.size(1) == EdgeAnchorWidth &&
      grad_output.scalar_type() == bank.scalar_type(),
      "grad_output must be contiguous CUDA [atoms,6]");
  TORCH_CHECK(
      h_bank.is_cuda() && h_auxiliary.is_cuda() && h_mix.is_cuda() &&
      h_bank.is_contiguous() && h_auxiliary.is_contiguous() &&
      h_mix.is_contiguous() && h_bank.sizes() == bank.sizes() &&
      h_auxiliary.sizes() == auxiliary.sizes() && h_mix.sizes() == mix.sizes(),
      "edge-anchor tangents must be contiguous CUDA shape matches");
  TORCH_CHECK(
      h_bank.scalar_type() == bank.scalar_type() &&
      h_auxiliary.scalar_type() == bank.scalar_type() &&
      h_mix.scalar_type() == bank.scalar_type(),
      "edge-anchor tangents must match primal dtype");
  c10::cuda::CUDAGuard guard(bank.device());
  auto grad2_output = torch::zeros_like(grad_output);
  auto grad2_bank = torch::zeros_like(bank);
  auto grad2_auxiliary = torch::empty_like(auxiliary);
  auto grad2_mix = torch::zeros_like(mix);
  const int threads = 128;
  const int blocks = static_cast<int>((auxiliary.size(0) + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(
        bank.scalar_type(), "edge_anchor_sector_double_backward", [&] {
      edge_anchor_double_backward_kernel<scalar_t><<<
          blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          grad_output.data_ptr<scalar_t>(), bank.data_ptr<scalar_t>(),
          auxiliary.data_ptr<scalar_t>(), pair_i.data_ptr<int64_t>(),
          mix.data_ptr<scalar_t>(), h_bank.data_ptr<scalar_t>(),
          h_auxiliary.data_ptr<scalar_t>(), h_mix.data_ptr<scalar_t>(),
          auxiliary.size(0), grad2_output.data_ptr<scalar_t>(),
          grad2_bank.data_ptr<scalar_t>(),
          grad2_auxiliary.data_ptr<scalar_t>(), grad2_mix.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad2_output, grad2_bank, grad2_auxiliary, grad2_mix};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &density_bank_forward, "O3-C2 primitive bank (CUDA)");
  module.def("backward", &density_bank_backward, "O3-C2 primitive bank VJP (CUDA)");
  module.def(
      "double_backward", &density_bank_double_backward,
      "O3-C2 primitive bank analytic directional HVP (CUDA)");
  module.def(
      "edge_anchor_forward", &edge_anchor_sector_forward,
      "O3-C2 edge-anchored sectors (CUDA)");
  module.def(
      "edge_anchor_backward", &edge_anchor_sector_backward,
      "O3-C2 edge-anchored sector VJP (CUDA)");
  module.def(
      "edge_anchor_double_backward", &edge_anchor_sector_double_backward,
      "O3-C2 edge-anchored sector directional HVP (CUDA)");
}
