#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <vector>

namespace {

constexpr int C = 2;
constexpr int PrimitiveWidth = 85;
constexpr int CenterWidth = 29;
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
template <typename T> __device__ Dual<T>& operator+=(Dual<T>& a, Dual<T> b) {
  a.v += b.v;
  a.d += b.d;
  return a;
}
template <typename T> __device__ Dual<T>& operator-=(Dual<T>& a, Dual<T> b) {
  a.v -= b.v;
  a.d -= b.d;
  return a;
}

template <typename S>
__device__ S dot(const S* a, const S* b, int width) {
  S result = S(0);
  for (int k = 0; k < width; ++k) result += a[k] * b[k];
  return result;
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
__device__ S triple(const S* a, const S* b, const S* c) {
  S crossed[3];
  cross3(a, b, crossed);
  return dot(crossed, c, 3);
}

template <typename S>
__device__ void add_triple_pull(
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
__device__ void stf5(const S* left, const S* right, S* out) {
  constexpr double InvSqrt6 = 0.40824829046386301637;
  constexpr double InvSqrt2 = 0.70710678118654752440;
  const S xx = left[0] * right[0];
  const S yy = left[1] * right[1];
  const S zz = left[2] * right[2];
  out[0] = S(InvSqrt6) * (S(2) * zz - xx - yy);
  out[1] = S(InvSqrt2) * (xx - yy);
  out[2] = S(InvSqrt2) * (left[0] * right[1] + left[1] * right[0]);
  out[3] = S(InvSqrt2) * (left[1] * right[2] + left[2] * right[1]);
  out[4] = S(InvSqrt2) * (left[2] * right[0] + left[0] * right[2]);
}

template <typename S>
__device__ void add_stf5_pull(
    const S* left, const S* right, const S* pull,
    S* grad_left, S* grad_right) {
  constexpr double InvSqrt6 = 0.40824829046386301637;
  constexpr double InvSqrt2 = 0.70710678118654752440;
  const S g0 = pull[0];
  const S g1 = pull[1];
  const S g2 = pull[2];
  const S g3 = pull[3];
  const S g4 = pull[4];
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
__device__ void matmul3(const S* a, const S* b, S* out) {
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      S value = S(0);
      for (int k = 0; k < 3; ++k) value += a[3 * row + k] * b[3 * k + col];
      out[3 * row + col] = value;
    }
  }
}

template <typename S>
__device__ void commutator_axial(const S* left5, const S* right5, S* out) {
  S left[9], right[9], lr[9], rl[9];
  stf_to_matrix(left5, left);
  stf_to_matrix(right5, right);
  matmul3(left, right, lr);
  matmul3(right, left, rl);
  out[0] = S(0.5) * ((lr[7] - rl[7]) - (lr[5] - rl[5]));
  out[1] = S(0.5) * ((lr[2] - rl[2]) - (lr[6] - rl[6]));
  out[2] = S(0.5) * ((lr[3] - rl[3]) - (lr[1] - rl[1]));
}

template <typename S>
__device__ void add_commutator_pull(
    const S* left5, const S* right5, const S* pull,
    S* grad_left5, S* grad_right5) {
  S left[9], right[9], h[9];
  stf_to_matrix(left5, left);
  stf_to_matrix(right5, right);
  for (int k = 0; k < 9; ++k) h[k] = S(0);
  h[7] = S(0.5) * pull[0]; h[5] = -S(0.5) * pull[0];
  h[2] = S(0.5) * pull[1]; h[6] = -S(0.5) * pull[1];
  h[3] = S(0.5) * pull[2]; h[1] = -S(0.5) * pull[2];
  S grad_left[9], grad_right[9];
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      S gl = S(0), gr = S(0);
      for (int k = 0; k < 3; ++k) {
        gl += h[3 * row + k] * right[3 * k + col]
            - right[3 * row + k] * h[3 * k + col];
        gr += left[3 * row + k] * h[3 * k + col]
            - h[3 * row + k] * left[3 * k + col];
      }
      grad_left[3 * row + col] = gl;
      grad_right[3 * row + col] = gr;
    }
  }
  add_matrix_stf_pull(grad_left, grad_left5);
  add_matrix_stf_pull(grad_right, grad_right5);
}

template <typename S>
__device__ void qp_on_spin(const S* qp, S* result) {
  for (int row = 0; row < 3; ++row) result[row] = S(0);
  for (int column = 0; column < 3; ++column) {
    S q[5], matrix[9];
    for (int k = 0; k < 5; ++k) q[k] = qp[3 * k + column];
    stf_to_matrix(q, matrix);
    for (int row = 0; row < 3; ++row) result[row] += matrix[3 * row + column];
  }
}

template <typename S>
__device__ void add_qp_on_spin_pull(const S* pull, S* grad_qp) {
  for (int column = 0; column < 3; ++column) {
    S matrix[9], grad_q[5] = {S(0), S(0), S(0), S(0), S(0)};
    for (int k = 0; k < 9; ++k) matrix[k] = S(0);
    for (int row = 0; row < 3; ++row) matrix[3 * row + column] = pull[row];
    add_matrix_stf_pull(matrix, grad_q);
    for (int k = 0; k < 5; ++k) grad_qp[3 * k + column] += grad_q[k];
  }
}

template <typename S, int D>
__device__ void project(
    const S* mix, int leg, const S source[C][D], S output[C][D]) {
  for (int r = 0; r < C; ++r) {
    for (int d = 0; d < D; ++d) {
      output[r][d] = S(0);
      for (int c = 0; c < C; ++c) {
        output[r][d] += mix[leg * 4 + r * C + c] * source[c][d];
      }
    }
  }
}

template <typename S, int D>
__device__ void add_project_pull(
    const S* mix, int leg, const S source[C][D], const S grad_output[C][D],
    S grad_source[C][D], S* grad_mix) {
  for (int r = 0; r < C; ++r) {
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        grad_source[c][d] +=
            mix[leg * 4 + r * C + c] * grad_output[r][d];
        grad_mix[leg * 4 + r * C + c] +=
            grad_output[r][d] * source[c][d];
      }
    }
  }
}

template <typename S>
struct CenterState {
  S spin[3];
  S m[C][3], p[C][3], x[C][3], dm[C][3];
  S longitudinal[C][1], tensor[C][5], q[C][5], qp[C][15];
  S omega[C][3], dot_omega[C][3], qs[C][3];
  S p0[C][3], p1[C][3], omega0[C][3], dot_omega0[C][3];
  S m1[C][3], x2[C][3], qs1[C][3];
  S q0[C][5], q1[C][5];
};

template <typename S>
__device__ void make_state(
    const S* primitive, const S* spin, const S* mix, CenterState<S>* state) {
  for (int d = 0; d < 3; ++d) state->spin[d] = spin[d];
  for (int c = 0; c < C; ++c) {
    const int base = c * ChannelStride;
    for (int d = 0; d < 3; ++d) {
      state->m[c][d] = primitive[base + M + d];
      state->p[c][d] = primitive[base + P + d];
      state->x[c][d] = primitive[base + X + d];
      state->dm[c][d] = primitive[base + DM + d];
    }
    state->longitudinal[c][0] = primitive[base + L];
    for (int k = 0; k < 5; ++k) {
      state->tensor[c][k] = primitive[base + T5 + k];
      state->q[c][k] = primitive[base + Q + k];
      for (int d = 0; d < 3; ++d) {
        state->qp[c][3 * k + d] = primitive[base + QP + 3 * k + d];
      }
    }
    cross3(state->spin, state->m[c], state->omega[c]);
    cross3(state->spin, state->dm[c], state->dot_omega[c]);
    qp_on_spin(state->qp[c], state->qs[c]);
  }
  project<S, 3>(mix, 0, state->p, state->p0);
  project<S, 3>(mix, 1, state->p, state->p1);
  project<S, 3>(mix, 0, state->omega, state->omega0);
  project<S, 3>(mix, 0, state->dot_omega, state->dot_omega0);
  project<S, 3>(mix, 1, state->m, state->m1);
  project<S, 3>(mix, 2, state->x, state->x2);
  project<S, 3>(mix, 1, state->qs, state->qs1);
  project<S, 5>(mix, 0, state->q, state->q0);
  project<S, 5>(mix, 1, state->q, state->q1);
}

template <typename S>
__device__ void center_forward(
    const CenterState<S>& state, const S* primitive, S* output) {
  S local_q[5];
  stf5(state.spin, state.spin, local_q);
  for (int c = 0; c < C; ++c) {
    output[c] = dot(state.q[c], local_q, 5);
    output[2 + c] = dot(state.m[c], state.m[c], 3);
    output[4 + c] = state.longitudinal[c][0] * state.longitudinal[c][0];
    output[6 + c] = dot(state.x[c], state.x[c], 3);
    output[8 + c] = dot(state.tensor[c], state.tensor[c], 5);
    output[10 + c] = dot(state.qp[c], state.qp[c], 15);
    output[12 + c] = dot(state.m[c], state.dm[c], 3);
  }
  const S first_dot[C] = {
      dot(state.spin, state.m[0], 3), dot(state.spin, state.m[1], 3)};
  int pair = 0;
  for (int left = 0; left < C; ++left) {
    for (int right = left; right < C; ++right, ++pair) {
      output[14 + pair] = first_dot[left] * first_dot[right]
          - primitive[Same + pair];
    }
  }
  for (int r = 0; r < C; ++r) {
    S axis[3];
    output[17 + r] = triple(state.p0[r], state.p1[r], state.omega0[r]);
    commutator_axial(state.q0[r], state.q1[r], axis);
    output[19 + r] = dot(axis, state.omega0[r], 3);
    output[21 + r] = triple(
        state.p0[r], state.p1[r], state.dot_omega0[r]);
    output[23 + r] = triple(state.p0[r], state.m1[r], state.x2[r]);
    output[25 + r] = dot(axis, state.dot_omega0[r], 3);
    output[27 + r] = triple(state.p0[r], state.qs1[r], state.x2[r]);
  }
}

template <typename S, int D>
__device__ void zero2(S value[C][D]) {
  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < D; ++d) value[c][d] = S(0);
  }
}

template <typename S>
__device__ void center_reverse(
    const CenterState<S>& state, const S* primitive, const S* mix,
    const S* grad_output, S* grad_primitive, S* grad_spin, S* grad_mix) {
  for (int k = 0; k < PrimitiveWidth; ++k) grad_primitive[k] = S(0);
  for (int d = 0; d < 3; ++d) grad_spin[d] = S(0);
  for (int k = 0; k < 16; ++k) grad_mix[k] = S(0);

  S gm[C][3], gp[C][3], gx[C][3], gdm[C][3];
  S gl[C][1], gt[C][5], gq[C][5], gqp[C][15];
  S gomega[C][3], gdot_omega[C][3], gqs[C][3];
  S gp0[C][3], gp1[C][3], gomega0[C][3], gdot_omega0[C][3];
  S gm1[C][3], gx2[C][3], gqs1[C][3];
  S gq0[C][5], gq1[C][5];
  zero2<S, 3>(gm); zero2<S, 3>(gp); zero2<S, 3>(gx); zero2<S, 3>(gdm);
  zero2<S, 1>(gl); zero2<S, 5>(gt); zero2<S, 5>(gq); zero2<S, 15>(gqp);
  zero2<S, 3>(gomega); zero2<S, 3>(gdot_omega); zero2<S, 3>(gqs);
  zero2<S, 3>(gp0); zero2<S, 3>(gp1); zero2<S, 3>(gomega0);
  zero2<S, 3>(gdot_omega0); zero2<S, 3>(gm1); zero2<S, 3>(gx2);
  zero2<S, 3>(gqs1); zero2<S, 5>(gq0); zero2<S, 5>(gq1);

  S local_q[5], grad_local_q[5] = {S(0), S(0), S(0), S(0), S(0)};
  stf5(state.spin, state.spin, local_q);
  for (int c = 0; c < C; ++c) {
    const S a_l2 = grad_output[c];
    for (int k = 0; k < 5; ++k) {
      gq[c][k] += a_l2 * local_q[k];
      grad_local_q[k] += a_l2 * state.q[c][k];
    }
    const S a_m = S(2) * grad_output[2 + c];
    const S a_l = S(2) * grad_output[4 + c] * state.longitudinal[c][0];
    const S a_x = S(2) * grad_output[6 + c];
    const S a_t = S(2) * grad_output[8 + c];
    const S a_qp = S(2) * grad_output[10 + c];
    for (int d = 0; d < 3; ++d) {
      gm[c][d] += a_m * state.m[c][d]
          + grad_output[12 + c] * state.dm[c][d];
      gdm[c][d] += grad_output[12 + c] * state.m[c][d];
      gx[c][d] += a_x * state.x[c][d];
    }
    gl[c][0] += a_l;
    for (int k = 0; k < 5; ++k) gt[c][k] += a_t * state.tensor[c][k];
    for (int k = 0; k < 15; ++k) gqp[c][k] += a_qp * state.qp[c][k];
  }
  S grad_spin_left[3] = {S(0), S(0), S(0)};
  S grad_spin_right[3] = {S(0), S(0), S(0)};
  add_stf5_pull(
      state.spin, state.spin, grad_local_q, grad_spin_left, grad_spin_right);
  for (int d = 0; d < 3; ++d) {
    grad_spin[d] += grad_spin_left[d] + grad_spin_right[d];
  }

  const S first_dot[C] = {
      dot(state.spin, state.m[0], 3), dot(state.spin, state.m[1], 3)};
  S grad_first_dot[C] = {S(0), S(0)};
  int pair = 0;
  for (int left = 0; left < C; ++left) {
    for (int right = left; right < C; ++right, ++pair) {
      const S a = grad_output[14 + pair];
      grad_first_dot[left] += a * first_dot[right];
      grad_first_dot[right] += a * first_dot[left];
      grad_primitive[Same + pair] -= a;
    }
  }
  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < 3; ++d) {
      grad_spin[d] += grad_first_dot[c] * state.m[c][d];
      gm[c][d] += grad_first_dot[c] * state.spin[d];
    }
  }

  for (int r = 0; r < C; ++r) {
    add_triple_pull(
        state.p0[r], state.p1[r], state.omega0[r], grad_output[17 + r],
        gp0[r], gp1[r], gomega0[r]);
    S axis[3], axis_pull[3];
    commutator_axial(state.q0[r], state.q1[r], axis);
    for (int d = 0; d < 3; ++d) {
      axis_pull[d] = grad_output[19 + r] * state.omega0[r][d]
          + grad_output[25 + r] * state.dot_omega0[r][d];
      gomega0[r][d] += grad_output[19 + r] * axis[d];
      gdot_omega0[r][d] += grad_output[25 + r] * axis[d];
    }
    add_commutator_pull(
        state.q0[r], state.q1[r], axis_pull, gq0[r], gq1[r]);
    add_triple_pull(
        state.p0[r], state.p1[r], state.dot_omega0[r],
        grad_output[21 + r], gp0[r], gp1[r], gdot_omega0[r]);
    add_triple_pull(
        state.p0[r], state.m1[r], state.x2[r], grad_output[23 + r],
        gp0[r], gm1[r], gx2[r]);
    add_triple_pull(
        state.p0[r], state.qs1[r], state.x2[r], grad_output[27 + r],
        gp0[r], gqs1[r], gx2[r]);
  }

  add_project_pull<S, 3>(mix, 0, state.p, gp0, gp, grad_mix);
  add_project_pull<S, 3>(mix, 1, state.p, gp1, gp, grad_mix);
  add_project_pull<S, 3>(mix, 0, state.omega, gomega0, gomega, grad_mix);
  add_project_pull<S, 3>(
      mix, 0, state.dot_omega, gdot_omega0, gdot_omega, grad_mix);
  add_project_pull<S, 3>(mix, 1, state.m, gm1, gm, grad_mix);
  add_project_pull<S, 3>(mix, 2, state.x, gx2, gx, grad_mix);
  add_project_pull<S, 3>(mix, 1, state.qs, gqs1, gqs, grad_mix);
  add_project_pull<S, 5>(mix, 0, state.q, gq0, gq, grad_mix);
  add_project_pull<S, 5>(mix, 1, state.q, gq1, gq, grad_mix);

  for (int c = 0; c < C; ++c) {
    add_cross_pull(state.spin, state.m[c], gomega[c], grad_spin, gm[c]);
    add_cross_pull(
        state.spin, state.dm[c], gdot_omega[c], grad_spin, gdm[c]);
    add_qp_on_spin_pull(gqs[c], gqp[c]);
    const int base = c * ChannelStride;
    for (int d = 0; d < 3; ++d) {
      grad_primitive[base + M + d] += gm[c][d];
      grad_primitive[base + P + d] += gp[c][d];
      grad_primitive[base + X + d] += gx[c][d];
      grad_primitive[base + DM + d] += gdm[c][d];
    }
    grad_primitive[base + L] += gl[c][0];
    for (int k = 0; k < 5; ++k) {
      grad_primitive[base + T5 + k] += gt[c][k];
      grad_primitive[base + Q + k] += gq[c][k];
      for (int d = 0; d < 3; ++d) {
        grad_primitive[base + QP + 3 * k + d] += gqp[c][3 * k + d];
      }
    }
  }
}

template <typename scalar_t>
__global__ void center_forward_kernel(
    const scalar_t* primitive, const scalar_t* spins, const scalar_t* mix,
    int64_t atoms, scalar_t* output) {
  const int64_t atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atoms) return;
  CenterState<scalar_t> state;
  make_state(
      primitive + atom * PrimitiveWidth, spins + atom * 3, mix, &state);
  center_forward(
      state, primitive + atom * PrimitiveWidth, output + atom * CenterWidth);
}

template <typename scalar_t>
__global__ void center_backward_kernel(
    const scalar_t* grad_output, const scalar_t* primitive,
    const scalar_t* spins, const scalar_t* mix, int64_t atoms,
    scalar_t* grad_primitive, scalar_t* grad_spins, scalar_t* grad_mix) {
  const int64_t atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atoms) return;
  CenterState<scalar_t> state;
  make_state(
      primitive + atom * PrimitiveWidth, spins + atom * 3, mix, &state);
  scalar_t local_primitive[PrimitiveWidth], local_spin[3], local_mix[16];
  center_reverse(
      state, primitive + atom * PrimitiveWidth, mix,
      grad_output + atom * CenterWidth,
      local_primitive, local_spin, local_mix);
  for (int k = 0; k < PrimitiveWidth; ++k) {
    grad_primitive[atom * PrimitiveWidth + k] = local_primitive[k];
  }
  for (int d = 0; d < 3; ++d) grad_spins[atom * 3 + d] = local_spin[d];
  for (int k = 0; k < 16; ++k) atomicAdd(grad_mix + k, local_mix[k]);
}

template <typename scalar_t>
__global__ void center_double_backward_kernel(
    const scalar_t* grad_output, const scalar_t* primitive,
    const scalar_t* spins, const scalar_t* mix,
    const scalar_t* h_primitive, const scalar_t* h_spins,
    const scalar_t* h_mix, int64_t atoms, scalar_t* grad2_output,
    scalar_t* grad2_primitive, scalar_t* grad2_spins, scalar_t* grad2_mix) {
  const int64_t atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atoms) return;
  Dual<scalar_t> local_primitive[PrimitiveWidth], local_spin[3], local_mix[16];
  for (int k = 0; k < PrimitiveWidth; ++k) {
    local_primitive[k] = {
        primitive[atom * PrimitiveWidth + k],
        h_primitive[atom * PrimitiveWidth + k]};
  }
  for (int d = 0; d < 3; ++d) {
    local_spin[d] = {spins[atom * 3 + d], h_spins[atom * 3 + d]};
  }
  for (int k = 0; k < 16; ++k) local_mix[k] = {mix[k], h_mix[k]};
  CenterState<Dual<scalar_t>> state;
  make_state(local_primitive, local_spin, local_mix, &state);
  Dual<scalar_t> local_output[CenterWidth];
  center_forward(state, local_primitive, local_output);
  for (int k = 0; k < CenterWidth; ++k) {
    grad2_output[atom * CenterWidth + k] = local_output[k].d;
  }
  Dual<scalar_t> local_grad_output[CenterWidth];
  for (int k = 0; k < CenterWidth; ++k) {
    local_grad_output[k] = {grad_output[atom * CenterWidth + k], scalar_t(0)};
  }
  Dual<scalar_t> local_grad_primitive[PrimitiveWidth];
  Dual<scalar_t> local_grad_spin[3], local_grad_mix[16];
  center_reverse(
      state, local_primitive, local_mix, local_grad_output,
      local_grad_primitive, local_grad_spin, local_grad_mix);
  for (int k = 0; k < PrimitiveWidth; ++k) {
    grad2_primitive[atom * PrimitiveWidth + k] = local_grad_primitive[k].d;
  }
  for (int d = 0; d < 3; ++d) {
    grad2_spins[atom * 3 + d] = local_grad_spin[d].d;
  }
  for (int k = 0; k < 16; ++k) {
    atomicAdd(grad2_mix + k, local_grad_mix[k].d);
  }
}

void check_inputs(
    const torch::Tensor& primitive, const torch::Tensor& spins,
    const torch::Tensor& mix) {
  TORCH_CHECK(primitive.is_cuda() && spins.is_cuda() && mix.is_cuda(),
              "center-sector tensors must be CUDA tensors");
  TORCH_CHECK(primitive.is_contiguous() && spins.is_contiguous() &&
              mix.is_contiguous(), "center-sector tensors must be contiguous");
  TORCH_CHECK(primitive.dim() == 2 && primitive.size(1) == PrimitiveWidth,
              "primitive must be [atoms,85]");
  TORCH_CHECK(spins.dim() == 2 && spins.size(0) == primitive.size(0) &&
              spins.size(1) == 3, "spins must be [atoms,3]");
  TORCH_CHECK(mix.dim() == 3 && mix.size(0) == 4 && mix.size(1) == C &&
              mix.size(2) == C, "radial_leg_mix must be [4,2,2]");
  TORCH_CHECK(primitive.scalar_type() == spins.scalar_type() &&
              primitive.scalar_type() == mix.scalar_type(),
              "center-sector floating tensors must share dtype");
}

}  // namespace

torch::Tensor center_sector_forward(
    torch::Tensor primitive, torch::Tensor spins, torch::Tensor mix) {
  check_inputs(primitive, spins, mix);
  c10::cuda::CUDAGuard guard(primitive.device());
  auto output = torch::empty(
      {primitive.size(0), CenterWidth}, primitive.options());
  const int threads = 64;
  const int blocks = static_cast<int>((primitive.size(0) + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(
        primitive.scalar_type(), "center_sector_forward", [&] {
      center_forward_kernel<scalar_t><<<
          blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          primitive.data_ptr<scalar_t>(), spins.data_ptr<scalar_t>(),
          mix.data_ptr<scalar_t>(), primitive.size(0),
          output.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

std::vector<torch::Tensor> center_sector_backward(
    torch::Tensor grad_output, torch::Tensor primitive,
    torch::Tensor spins, torch::Tensor mix) {
  check_inputs(primitive, spins, mix);
  TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous() &&
              grad_output.dim() == 2 &&
              grad_output.size(0) == primitive.size(0) &&
              grad_output.size(1) == CenterWidth &&
              grad_output.scalar_type() == primitive.scalar_type(),
              "grad_output must be contiguous CUDA [atoms,29]");
  c10::cuda::CUDAGuard guard(primitive.device());
  auto grad_primitive = torch::empty_like(primitive);
  auto grad_spins = torch::empty_like(spins);
  auto grad_mix = torch::zeros_like(mix);
  const int threads = 64;
  const int blocks = static_cast<int>((primitive.size(0) + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(
        primitive.scalar_type(), "center_sector_backward", [&] {
      center_backward_kernel<scalar_t><<<
          blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          grad_output.data_ptr<scalar_t>(), primitive.data_ptr<scalar_t>(),
          spins.data_ptr<scalar_t>(), mix.data_ptr<scalar_t>(),
          primitive.size(0), grad_primitive.data_ptr<scalar_t>(),
          grad_spins.data_ptr<scalar_t>(), grad_mix.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad_primitive, grad_spins, grad_mix};
}

std::vector<torch::Tensor> center_sector_double_backward(
    torch::Tensor grad_output, torch::Tensor primitive,
    torch::Tensor spins, torch::Tensor mix, torch::Tensor h_primitive,
    torch::Tensor h_spins, torch::Tensor h_mix) {
  check_inputs(primitive, spins, mix);
  TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous() &&
              grad_output.dim() == 2 &&
              grad_output.size(0) == primitive.size(0) &&
              grad_output.size(1) == CenterWidth &&
              grad_output.scalar_type() == primitive.scalar_type(),
              "grad_output must be contiguous CUDA [atoms,29]");
  TORCH_CHECK(h_primitive.is_cuda() && h_spins.is_cuda() && h_mix.is_cuda() &&
              h_primitive.is_contiguous() && h_spins.is_contiguous() &&
              h_mix.is_contiguous() &&
              h_primitive.sizes() == primitive.sizes() &&
              h_spins.sizes() == spins.sizes() && h_mix.sizes() == mix.sizes() &&
              h_primitive.scalar_type() == primitive.scalar_type() &&
              h_spins.scalar_type() == primitive.scalar_type() &&
              h_mix.scalar_type() == primitive.scalar_type(),
              "center-sector tangents must be contiguous CUDA shape matches");
  c10::cuda::CUDAGuard guard(primitive.device());
  auto grad2_output = torch::empty_like(grad_output);
  auto grad2_primitive = torch::empty_like(primitive);
  auto grad2_spins = torch::empty_like(spins);
  auto grad2_mix = torch::zeros_like(mix);
  const int threads = 64;
  const int blocks = static_cast<int>((primitive.size(0) + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(
        primitive.scalar_type(), "center_sector_double_backward", [&] {
      center_double_backward_kernel<scalar_t><<<
          blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          grad_output.data_ptr<scalar_t>(), primitive.data_ptr<scalar_t>(),
          spins.data_ptr<scalar_t>(), mix.data_ptr<scalar_t>(),
          h_primitive.data_ptr<scalar_t>(), h_spins.data_ptr<scalar_t>(),
          h_mix.data_ptr<scalar_t>(), primitive.size(0),
          grad2_output.data_ptr<scalar_t>(),
          grad2_primitive.data_ptr<scalar_t>(),
          grad2_spins.data_ptr<scalar_t>(), grad2_mix.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad2_output, grad2_primitive, grad2_spins, grad2_mix};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &center_sector_forward, "O3-C2 center sector (CUDA)");
  module.def("backward", &center_sector_backward, "O3-C2 center VJP (CUDA)");
  module.def(
      "double_backward", &center_sector_double_backward,
      "O3-C2 center analytic directional HVP (CUDA)");
}
