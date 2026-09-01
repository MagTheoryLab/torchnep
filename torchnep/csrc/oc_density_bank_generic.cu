#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <vector>

namespace {

constexpr int ChannelStride = 47;
constexpr int M = 0;
constexpr int P = 3;
constexpr int X = 6;
constexpr int DM = 9;
constexpr int L = 12;
constexpr int T5 = 13;
constexpr int Q = 18;
constexpr int QP = 23;
constexpr int Raw1 = 38;

template <typename T> struct Dual {
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
template <typename T> __device__ Dual<T> &operator+=(Dual<T> &a, Dual<T> b) {
  a.v += b.v;
  a.d += b.d;
  return a;
}

template <typename T> __device__ T primal(T x) { return x; }
template <typename T> __device__ T primal(Dual<T> x) { return x.v; }
template <typename T> __device__ T tangent(T) { return T(0); }
template <typename T> __device__ T tangent(Dual<T> x) { return x.d; }
template <typename T> __device__ T root(T x) { return sqrt(x); }
template <typename T> __device__ Dual<T> root(Dual<T> x) {
  const T value = sqrt(x.v);
  return {value, x.d / (T(2) * value)};
}
template <typename T> __device__ T cosine(T x) { return cos(x); }
template <typename T> __device__ Dual<T> cosine(Dual<T> x) {
  return {cos(x.v), -sin(x.v) * x.d};
}
template <typename T> __device__ T sine(T x) { return sin(x); }
template <typename T> __device__ Dual<T> sine(Dual<T> x) {
  return {sin(x.v), cos(x.v) * x.d};
}

template <typename S, typename T> struct Seed;
template <typename T> struct Seed<T, T> {
  __device__ static T make(T value, T) { return value; }
};
template <typename T> struct Seed<Dual<T>, T> {
  __device__ static Dual<T> make(T value, T direction) {
    return {value, direction};
  }
};
template <typename S, typename T>
__device__ S seeded(T value, const T *direction) {
  return Seed<S, T>::make(value, direction == nullptr ? T(0) : *direction);
}

template <bool Directional, typename T, typename S>
__device__ void emit_add(T *output, int64_t index, S value) {
  if constexpr (Directional) {
    atomicAdd(output + index, tangent(value));
  } else {
    atomicAdd(output + index, primal(value));
  }
}
template <bool Directional, typename T, typename S>
__device__ void emit_set(T *output, int64_t index, S value) {
  if constexpr (Directional) {
    output[index] = tangent(value);
  } else {
    output[index] = primal(value);
  }
}

template <typename S> __device__ S dot3(const S *a, const S *b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
template <typename S> __device__ void cross3(const S *a, const S *b, S *out) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}
template <typename S>
__device__ void add_cross_pull(const S *a, const S *b, const S *pull, S *grad_a,
                               S *grad_b) {
  S b_cross_g[3], g_cross_a[3];
  cross3(b, pull, b_cross_g);
  cross3(pull, a, g_cross_a);
  for (int d = 0; d < 3; ++d) {
    grad_a[d] += b_cross_g[d];
    grad_b[d] += g_cross_a[d];
  }
}
template <typename S>
__device__ void stf5(const S *left, const S *right, S *out) {
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
__device__ void add_stf5_pull(const S *left, const S *right, const S *pull,
                              S scale, S *grad_left, S *grad_right) {
  constexpr double InvSqrt6 = 0.40824829046386301637;
  constexpr double InvSqrt2 = 0.70710678118654752440;
  const S g0 = scale * pull[0];
  const S g1 = scale * pull[1];
  const S g2 = scale * pull[2];
  const S g3 = scale * pull[3];
  const S g4 = scale * pull[4];
#define STF_SIDE(target, other)                                                \
  do {                                                                         \
    (target)[0] += (-g0 * S(InvSqrt6) + g1 * S(InvSqrt2)) * (other)[0] +       \
                   g2 * S(InvSqrt2) * (other)[1] +                             \
                   g4 * S(InvSqrt2) * (other)[2];                              \
    (target)[1] += g2 * S(InvSqrt2) * (other)[0] +                             \
                   (-g0 * S(InvSqrt6) - g1 * S(InvSqrt2)) * (other)[1] +       \
                   g3 * S(InvSqrt2) * (other)[2];                              \
    (target)[2] += g4 * S(InvSqrt2) * (other)[0] +                             \
                   g3 * S(InvSqrt2) * (other)[1] +                             \
                   S(2) * g0 * S(InvSqrt6) * (other)[2];                       \
  } while (false)
  STF_SIDE(grad_left, right);
  STF_SIDE(grad_right, left);
#undef STF_SIDE
}

template <typename S> struct EdgeState {
  S dist;
  S rhat[3];
  S si[3];
  S sj[3];
  S dot;
  S si2;
  S sj2;
  S omega[3];
  S axial[3];
  S qrr[5];
  S edge_stf[5];
  S longitudinal;
  bool active;
};

template <typename S, typename T>
__device__ void make_state(const T *rij, const T *h_rij, const T *si,
                           const T *h_si, const T *sj, const T *h_sj,
                           bool active, double cutoff, double eps,
                           EdgeState<S> *state) {
  S rr[3];
  for (int d = 0; d < 3; ++d) {
    rr[d] = seeded<S>(rij[d], h_rij == nullptr ? nullptr : h_rij + d);
    state->si[d] = seeded<S>(si[d], h_si == nullptr ? nullptr : h_si + d);
    state->sj[d] = seeded<S>(sj[d], h_sj == nullptr ? nullptr : h_sj + d);
  }
  const S norm2 = dot3(rr, rr);
  state->dist = primal(norm2) < eps * eps ? S(eps) : root(norm2);
  for (int d = 0; d < 3; ++d)
    state->rhat[d] = rr[d] / state->dist;
  state->active = active && primal(state->dist) < cutoff;
  state->dot = dot3(state->si, state->sj);
  state->si2 = dot3(state->si, state->si);
  state->sj2 = dot3(state->sj, state->sj);
  cross3(state->si, state->sj, state->omega);
  cross3(state->rhat, state->sj, state->axial);
  stf5(state->rhat, state->rhat, state->qrr);
  stf5(state->rhat, state->sj, state->edge_stf);
  state->longitudinal = dot3(state->rhat, state->sj);
}

template <typename S, typename T>
__device__ void radial_weight(S distance, const T *coefficient,
                              const T *h_coefficient, int64_t basis_width,
                              double cutoff, S *weight, S *derivative) {
  *weight = S(0);
  *derivative = S(0);
  const S inv_rc = S(1.0 / cutoff);
  const S arg = S(3.14159265358979323846) * distance * inv_rc;
  const S fc = S(0.5) * cosine(arg) + S(0.5);
  const S fcp = S(-0.5 * 3.14159265358979323846 / cutoff) * sine(arg);
  const S u = distance * inv_rc - S(1);
  const S x = S(2) * u * u - S(1);
  const S dxdr = S(4.0 / cutoff) * u;
  S t0 = S(1), dt0 = S(0);
  S t1 = x, dt1 = dxdr;
  for (int64_t b = 0; b < basis_width; ++b) {
    S tb, dtb;
    if (b == 0) {
      tb = t0;
      dtb = dt0;
    } else if (b == 1) {
      tb = t1;
      dtb = dt1;
    } else {
      const S next = S(2) * x * t1 - t0;
      const S dnext = S(2) * (dxdr * t1 + x * dt1) - dt0;
      t0 = t1;
      dt0 = dt1;
      t1 = next;
      dt1 = dnext;
      tb = next;
      dtb = dnext;
    }
    const S core = S(0.5) * (tb + S(1));
    const S basis = core * fc;
    const S dbasis = S(0.5) * dtb * fc + core * fcp;
    const S coeff = seeded<S>(
        coefficient[b], h_coefficient == nullptr ? nullptr : h_coefficient + b);
    *weight += coeff * basis;
    *derivative += coeff * dbasis;
  }
}

__device__ int64_t pair_index(int64_t left, int64_t right, int64_t channels) {
  return left * channels - left * (left - 1) / 2 + (right - left);
}

template <bool Directional, typename S, typename T>
__device__ void
emit_main_forward(const EdgeState<S> &state, const T *coefficient,
                  const T *h_coefficient, int64_t channels, int64_t basis_width,
                  int64_t atom, int64_t edge, int64_t bank_width,
                  int64_t aux_width, double cutoff, T *bank, T *auxiliary) {
  const int64_t pair_width = channels * (channels + 1) / 2;
  const int64_t same = ChannelStride * channels;
  const int64_t edge_sj2 = same + pair_width;
  const int64_t edge_dot2 = edge_sj2 + channels;
  const int64_t edge_gate = edge_dot2 + channels;
  for (int64_t c = 0; c < channels; ++c) {
    S weight, derivative;
    radial_weight(state.dist, coefficient + c * basis_width,
                  h_coefficient == nullptr ? nullptr
                                           : h_coefficient + c * basis_width,
                  basis_width, cutoff, &weight, &derivative);
    if (!state.active)
      weight = S(0);
    const int64_t base = c * ChannelStride;
    for (int d = 0; d < 3; ++d) {
      emit_add<Directional>(bank, atom * bank_width + base + M + d,
                            weight * state.sj[d]);
      emit_add<Directional>(bank, atom * bank_width + base + P + d,
                            weight * state.rhat[d]);
      emit_add<Directional>(bank, atom * bank_width + base + X + d,
                            weight * state.axial[d]);
      emit_add<Directional>(bank, atom * bank_width + base + DM + d,
                            weight * state.dot * state.sj[d]);
    }
    emit_add<Directional>(bank, atom * bank_width + base + L,
                          weight * state.longitudinal);
    for (int k = 0; k < 5; ++k) {
      emit_add<Directional>(bank, atom * bank_width + base + T5 + k,
                            weight * state.edge_stf[k]);
      emit_add<Directional>(bank, atom * bank_width + base + Q + k,
                            weight * state.qrr[k]);
      for (int d = 0; d < 3; ++d) {
        emit_add<Directional>(bank, atom * bank_width + base + QP + 3 * k + d,
                              weight * state.qrr[k] * state.sj[d]);
      }
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        emit_add<Directional>(bank, atom * bank_width + base + Raw1 + 3 * a + b,
                              weight * state.rhat[a] * state.sj[b]);
      }
    }
    emit_add<Directional>(bank, atom * bank_width + edge_sj2 + c,
                          weight * state.sj2);
    emit_add<Directional>(bank, atom * bank_width + edge_dot2 + c,
                          weight * state.dot * state.dot);
    emit_add<Directional>(bank, atom * bank_width + edge_gate + c,
                          weight * state.dot * (state.si2 + state.sj2));
    emit_set<Directional>(auxiliary, edge * aux_width + c, weight);
  }
  for (int d = 0; d < 3; ++d) {
    emit_set<Directional>(auxiliary, edge * aux_width + channels + d,
                          state.rhat[d]);
    emit_set<Directional>(auxiliary, edge * aux_width + channels + 3 + d,
                          state.omega[d]);
  }
}

template <bool Directional, typename S, typename T>
__device__ void emit_pair_forward(const EdgeState<S> &state,
                                  const T *coefficient, const T *h_coefficient,
                                  int64_t channels, int64_t basis_width,
                                  int64_t pair, int64_t atom,
                                  int64_t bank_width, double cutoff, T *bank) {
  int64_t left = 0;
  int64_t cursor = pair;
  while (cursor >= channels - left) {
    cursor -= channels - left;
    ++left;
  }
  const int64_t right = left + cursor;
  S wl, dwl, wr, dwr;
  radial_weight(state.dist, coefficient + left * basis_width,
                h_coefficient == nullptr ? nullptr
                                         : h_coefficient + left * basis_width,
                basis_width, cutoff, &wl, &dwl);
  radial_weight(state.dist, coefficient + right * basis_width,
                h_coefficient == nullptr ? nullptr
                                         : h_coefficient + right * basis_width,
                basis_width, cutoff, &wr, &dwr);
  if (!state.active) {
    wl = S(0);
    wr = S(0);
  }
  emit_add<Directional>(bank,
                        atom * bank_width + ChannelStride * channels + pair,
                        wl * wr * state.dot * state.dot);
}

template <bool Directional, typename S, typename T>
__device__ void
reverse_edge(const EdgeState<S> &state, const T *coefficient,
             const T *h_coefficient, const T *grad_bank, const T *grad_aux,
             int64_t channels, int64_t basis_width, int64_t types,
             int64_t coefficient_offset, int64_t atom, int64_t neighbor,
             int64_t edge, int64_t bank_width, int64_t aux_width, double cutoff,
             T *out_rij, T *out_spins, T *out_coefficient) {
  const int64_t pair_width = channels * (channels + 1) / 2;
  const int64_t same = ChannelStride * channels;
  const int64_t edge_sj2 = same + pair_width;
  const int64_t edge_dot2 = edge_sj2 + channels;
  const int64_t edge_gate = edge_dot2 + channels;
  S grad_rhat[3] = {S(0), S(0), S(0)};
  S grad_q[5] = {S(0), S(0), S(0), S(0), S(0)};
  S grad_si[3] = {S(0), S(0), S(0)};
  S grad_sj[3] = {S(0), S(0), S(0)};
  S grad_dot = S(0);
  S grad_longitudinal = S(0);
  S grad_dist = S(0);

  for (int64_t c = 0; c < channels; ++c) {
    S weight, dweight;
    radial_weight(state.dist, coefficient + c * basis_width,
                  h_coefficient == nullptr ? nullptr
                                           : h_coefficient + c * basis_width,
                  basis_width, cutoff, &weight, &dweight);
    if (!state.active) {
      weight = S(0);
      dweight = S(0);
    }
    const int64_t base = c * ChannelStride;
    auto gb = [&](int64_t index) {
      return S(grad_bank[atom * bank_width + index]);
    };
    S grad_weight = S(grad_aux[edge * aux_width + c]);
    S gm[3], gp[3], gx[3], gdm[3];
    for (int d = 0; d < 3; ++d) {
      gm[d] = gb(base + M + d);
      gp[d] = gb(base + P + d);
      gx[d] = gb(base + X + d);
      gdm[d] = gb(base + DM + d);
    }
    const S gl = gb(base + L);
    grad_weight += dot3(gm, state.sj) + dot3(gp, state.rhat) +
                   dot3(gx, state.axial) + gl * state.longitudinal;
    const S dm_contract = dot3(gdm, state.sj);
    grad_weight += state.dot * dm_contract;
    grad_dot += weight * dm_contract;
    grad_longitudinal += weight * gl;
    for (int d = 0; d < 3; ++d) {
      grad_sj[d] += weight * gm[d] + weight * state.dot * gdm[d];
      grad_rhat[d] += weight * gp[d];
    }
    S axial_r[3] = {S(0), S(0), S(0)};
    S axial_s[3] = {S(0), S(0), S(0)};
    add_cross_pull(state.rhat, state.sj, gx, axial_r, axial_s);
    for (int d = 0; d < 3; ++d) {
      grad_rhat[d] += weight * axial_r[d];
      grad_sj[d] += weight * axial_s[d];
    }
    S gt[5], gq[5];
    for (int k = 0; k < 5; ++k) {
      gt[k] = gb(base + T5 + k);
      gq[k] = gb(base + Q + k);
      grad_weight += gt[k] * state.edge_stf[k] + gq[k] * state.qrr[k];
      grad_q[k] += weight * gq[k];
      for (int d = 0; d < 3; ++d) {
        const S g = gb(base + QP + 3 * k + d);
        grad_weight += g * state.qrr[k] * state.sj[d];
        grad_q[k] += weight * g * state.sj[d];
        grad_sj[d] += weight * g * state.qrr[k];
      }
    }
    S stf_r[3] = {S(0), S(0), S(0)};
    S stf_s[3] = {S(0), S(0), S(0)};
    add_stf5_pull(state.rhat, state.sj, gt, weight, stf_r, stf_s);
    for (int d = 0; d < 3; ++d) {
      grad_rhat[d] += stf_r[d];
      grad_sj[d] += stf_s[d];
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        const S g = gb(base + Raw1 + 3 * a + b);
        grad_weight += g * state.rhat[a] * state.sj[b];
        grad_rhat[a] += weight * g * state.sj[b];
        grad_sj[b] += weight * g * state.rhat[a];
      }
    }
    const S a_sj2 = gb(edge_sj2 + c);
    grad_weight += a_sj2 * state.sj2;
    for (int d = 0; d < 3; ++d)
      grad_sj[d] += S(2) * weight * a_sj2 * state.sj[d];
    const S a_dot2 = gb(edge_dot2 + c);
    grad_weight += a_dot2 * state.dot * state.dot;
    grad_dot += S(2) * weight * a_dot2 * state.dot;
    const S a_gate = gb(edge_gate + c);
    const S spin_norm = state.si2 + state.sj2;
    grad_weight += a_gate * state.dot * spin_norm;
    grad_dot += weight * a_gate * spin_norm;
    for (int d = 0; d < 3; ++d) {
      grad_si[d] += S(2) * weight * a_gate * state.dot * state.si[d];
      grad_sj[d] += S(2) * weight * a_gate * state.dot * state.sj[d];
    }

    for (int64_t other = 0; other < channels; ++other) {
      const int64_t left = c < other ? c : other;
      const int64_t right = c < other ? other : c;
      const S a = gb(same + pair_index(left, right, channels));
      S wo, dwo;
      radial_weight(state.dist, coefficient + other * basis_width,
                    h_coefficient == nullptr
                        ? nullptr
                        : h_coefficient + other * basis_width,
                    basis_width, cutoff, &wo, &dwo);
      if (!state.active)
        wo = S(0);
      grad_weight +=
          (other == c ? S(2) : S(1)) * a * wo * state.dot * state.dot;
    }
    if (state.active) {
      const S inv_rc = S(1.0 / cutoff);
      const S arg = S(3.14159265358979323846) * state.dist * inv_rc;
      const S fc = S(0.5) * cosine(arg) + S(0.5);
      const S u = state.dist * inv_rc - S(1);
      const S x = S(2) * u * u - S(1);
      S t0 = S(1), t1 = x;
      for (int64_t b = 0; b < basis_width; ++b) {
        S tb;
        if (b == 0)
          tb = t0;
        else if (b == 1)
          tb = t1;
        else {
          const S next = S(2) * x * t1 - t0;
          t0 = t1;
          t1 = next;
          tb = next;
        }
        const S basis = S(0.5) * (tb + S(1)) * fc;
        emit_add<Directional>(out_coefficient,
                              coefficient_offset + c * basis_width + b,
                              grad_weight * basis);
      }
      grad_dist += grad_weight * dweight;
    }
  }

  for (int64_t left = 0; left < channels; ++left) {
    S wl, dwl;
    radial_weight(state.dist, coefficient + left * basis_width,
                  h_coefficient == nullptr ? nullptr
                                           : h_coefficient + left * basis_width,
                  basis_width, cutoff, &wl, &dwl);
    if (!state.active)
      wl = S(0);
    for (int64_t right = left; right < channels; ++right) {
      S wr, dwr;
      radial_weight(state.dist, coefficient + right * basis_width,
                    h_coefficient == nullptr
                        ? nullptr
                        : h_coefficient + right * basis_width,
                    basis_width, cutoff, &wr, &dwr);
      if (!state.active)
        wr = S(0);
      const S a = S(grad_bank[atom * bank_width + same +
                              pair_index(left, right, channels)]);
      grad_dot += S(2) * a * wl * wr * state.dot;
    }
  }
  for (int d = 0; d < 3; ++d)
    grad_rhat[d] += S(grad_aux[edge * aux_width + channels + d]);
  S omega_pull[3];
  for (int d = 0; d < 3; ++d)
    omega_pull[d] = S(grad_aux[edge * aux_width + channels + 3 + d]);
  add_cross_pull(state.si, state.sj, omega_pull, grad_si, grad_sj);
  S qr_left[3] = {S(0), S(0), S(0)};
  S qr_right[3] = {S(0), S(0), S(0)};
  add_stf5_pull(state.rhat, state.rhat, grad_q, S(1), qr_left, qr_right);
  for (int d = 0; d < 3; ++d) {
    grad_rhat[d] += qr_left[d] + qr_right[d] + grad_longitudinal * state.sj[d];
    grad_sj[d] += grad_longitudinal * state.rhat[d];
    grad_si[d] += grad_dot * state.sj[d];
    grad_sj[d] += grad_dot * state.si[d];
  }
  const S radial_dot = dot3(grad_rhat, state.rhat);
  for (int d = 0; d < 3; ++d) {
    const S grad_rr = grad_dist * state.rhat[d] +
                      (grad_rhat[d] - radial_dot * state.rhat[d]) / state.dist;
    emit_set<Directional>(out_rij, 3 * edge + d, grad_rr);
    emit_add<Directional>(out_spins, 3 * atom + d, grad_si[d]);
    emit_add<Directional>(out_spins, 3 * neighbor + d, grad_sj[d]);
  }
}

template <typename scalar_t>
__global__ void bank_forward_kernel(
    const scalar_t *rij, const int64_t *pair_i, const int64_t *pair_j,
    const scalar_t *spins, const int64_t *atom_types,
    const scalar_t *coefficient, const bool *dof_active, const bool *env_active,
    int64_t edges, int64_t types, int64_t channels, int64_t basis_width,
    int64_t bank_width, int64_t aux_width, double cutoff, double eps,
    scalar_t *bank, scalar_t *auxiliary) {
  const int64_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= edges)
    return;
  const int64_t i = pair_i[edge], j = pair_j[edge];
  const int64_t ti = atom_types[i], tj = atom_types[j];
  const int64_t offset = (ti * types + tj) * channels * basis_width;
  EdgeState<scalar_t> state;
  make_state(rij + 3 * edge, static_cast<const scalar_t *>(nullptr),
             spins + 3 * i, static_cast<const scalar_t *>(nullptr),
             spins + 3 * j, static_cast<const scalar_t *>(nullptr),
             dof_active[ti] && env_active[tj], cutoff, eps, &state);
  emit_main_forward<false>(state, coefficient + offset,
                           static_cast<const scalar_t *>(nullptr), channels,
                           basis_width, i, edge, bank_width, aux_width, cutoff,
                           bank, auxiliary);
}

template <typename scalar_t>
__global__ void
pair_forward_kernel(const scalar_t *rij, const int64_t *pair_i,
                    const int64_t *pair_j, const scalar_t *spins,
                    const int64_t *atom_types, const scalar_t *coefficient,
                    const bool *dof_active, const bool *env_active,
                    int64_t edges, int64_t types, int64_t channels,
                    int64_t basis_width, int64_t pair_width, int64_t bank_width,
                    double cutoff, double eps, scalar_t *bank) {
  const int64_t item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= edges * pair_width)
    return;
  const int64_t edge = item / pair_width, pair = item % pair_width;
  const int64_t i = pair_i[edge], j = pair_j[edge];
  const int64_t ti = atom_types[i], tj = atom_types[j];
  const int64_t offset = (ti * types + tj) * channels * basis_width;
  EdgeState<scalar_t> state;
  make_state(rij + 3 * edge, static_cast<const scalar_t *>(nullptr),
             spins + 3 * i, static_cast<const scalar_t *>(nullptr),
             spins + 3 * j, static_cast<const scalar_t *>(nullptr),
             dof_active[ti] && env_active[tj], cutoff, eps, &state);
  emit_pair_forward<false>(state, coefficient + offset,
                           static_cast<const scalar_t *>(nullptr), channels,
                           basis_width, pair, i, bank_width, cutoff, bank);
}

template <typename scalar_t>
__global__ void
bank_directional_kernel(const scalar_t *rij, const int64_t *pair_i,
                        const int64_t *pair_j, const scalar_t *spins,
                        const int64_t *atom_types, const scalar_t *coefficient,
                        const bool *dof_active, const bool *env_active,
                        const scalar_t *h_rij, const scalar_t *h_spins,
                        const scalar_t *h_coefficient, int64_t edges,
                        int64_t types, int64_t channels, int64_t basis_width,
                        int64_t bank_width, int64_t aux_width, double cutoff,
                        double eps, scalar_t *bank, scalar_t *auxiliary) {
  const int64_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= edges)
    return;
  const int64_t i = pair_i[edge], j = pair_j[edge];
  const int64_t ti = atom_types[i], tj = atom_types[j];
  const int64_t offset = (ti * types + tj) * channels * basis_width;
  EdgeState<Dual<scalar_t>> state;
  make_state(rij + 3 * edge, h_rij + 3 * edge, spins + 3 * i, h_spins + 3 * i,
             spins + 3 * j, h_spins + 3 * j, dof_active[ti] && env_active[tj],
             cutoff, eps, &state);
  emit_main_forward<true>(state, coefficient + offset, h_coefficient + offset,
                          channels, basis_width, i, edge, bank_width, aux_width,
                          cutoff, bank, auxiliary);
}

template <typename scalar_t>
__global__ void pair_directional_kernel(
    const scalar_t *rij, const int64_t *pair_i, const int64_t *pair_j,
    const scalar_t *spins, const int64_t *atom_types,
    const scalar_t *coefficient, const bool *dof_active, const bool *env_active,
    const scalar_t *h_rij, const scalar_t *h_spins,
    const scalar_t *h_coefficient, int64_t edges, int64_t types,
    int64_t channels, int64_t basis_width, int64_t pair_width,
    int64_t bank_width, double cutoff, double eps, scalar_t *bank) {
  const int64_t item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= edges * pair_width)
    return;
  const int64_t edge = item / pair_width, pair = item % pair_width;
  const int64_t i = pair_i[edge], j = pair_j[edge];
  const int64_t ti = atom_types[i], tj = atom_types[j];
  const int64_t offset = (ti * types + tj) * channels * basis_width;
  EdgeState<Dual<scalar_t>> state;
  make_state(rij + 3 * edge, h_rij + 3 * edge, spins + 3 * i, h_spins + 3 * i,
             spins + 3 * j, h_spins + 3 * j, dof_active[ti] && env_active[tj],
             cutoff, eps, &state);
  emit_pair_forward<true>(state, coefficient + offset, h_coefficient + offset,
                          channels, basis_width, pair, i, bank_width, cutoff,
                          bank);
}

template <typename scalar_t>
__global__ void bank_backward_kernel(
    const scalar_t *grad_bank, const scalar_t *grad_aux, const scalar_t *rij,
    const int64_t *pair_i, const int64_t *pair_j, const scalar_t *spins,
    const int64_t *atom_types, const scalar_t *coefficient,
    const bool *dof_active, const bool *env_active, int64_t edges,
    int64_t types, int64_t channels, int64_t basis_width, int64_t bank_width,
    int64_t aux_width, double cutoff, double eps, scalar_t *grad_rij,
    scalar_t *grad_spins, scalar_t *grad_coefficient) {
  const int64_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= edges)
    return;
  const int64_t i = pair_i[edge], j = pair_j[edge];
  const int64_t ti = atom_types[i], tj = atom_types[j];
  const int64_t offset = (ti * types + tj) * channels * basis_width;
  EdgeState<scalar_t> state;
  make_state(rij + 3 * edge, static_cast<const scalar_t *>(nullptr),
             spins + 3 * i, static_cast<const scalar_t *>(nullptr),
             spins + 3 * j, static_cast<const scalar_t *>(nullptr),
             dof_active[ti] && env_active[tj], cutoff, eps, &state);
  reverse_edge<false>(
      state, coefficient + offset, static_cast<const scalar_t *>(nullptr),
      grad_bank, grad_aux, channels, basis_width, types, offset, i, j, edge,
      bank_width, aux_width, cutoff, grad_rij, grad_spins, grad_coefficient);
}

template <typename scalar_t>
__global__ void bank_reverse_directional_kernel(
    const scalar_t *grad_bank, const scalar_t *grad_aux, const scalar_t *rij,
    const int64_t *pair_i, const int64_t *pair_j, const scalar_t *spins,
    const int64_t *atom_types, const scalar_t *coefficient,
    const bool *dof_active, const bool *env_active, const scalar_t *h_rij,
    const scalar_t *h_spins, const scalar_t *h_coefficient, int64_t edges,
    int64_t types, int64_t channels, int64_t basis_width, int64_t bank_width,
    int64_t aux_width, double cutoff, double eps, scalar_t *grad_rij,
    scalar_t *grad_spins, scalar_t *grad_coefficient) {
  const int64_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= edges)
    return;
  const int64_t i = pair_i[edge], j = pair_j[edge];
  const int64_t ti = atom_types[i], tj = atom_types[j];
  const int64_t offset = (ti * types + tj) * channels * basis_width;
  EdgeState<Dual<scalar_t>> state;
  make_state(rij + 3 * edge, h_rij + 3 * edge, spins + 3 * i, h_spins + 3 * i,
             spins + 3 * j, h_spins + 3 * j, dof_active[ti] && env_active[tj],
             cutoff, eps, &state);
  reverse_edge<true>(state, coefficient + offset, h_coefficient + offset,
                     grad_bank, grad_aux, channels, basis_width, types, offset,
                     i, j, edge, bank_width, aux_width, cutoff, grad_rij,
                     grad_spins, grad_coefficient);
}

void check_inputs(const torch::Tensor &rij, const torch::Tensor &pair_i,
                  const torch::Tensor &pair_j, const torch::Tensor &spins,
                  const torch::Tensor &atom_types,
                  const torch::Tensor &coefficient,
                  const torch::Tensor &dof_active,
                  const torch::Tensor &env_active) {
  TORCH_CHECK(rij.is_cuda() && pair_i.is_cuda() && pair_j.is_cuda() &&
                  spins.is_cuda() && atom_types.is_cuda() &&
                  coefficient.is_cuda() && dof_active.is_cuda() &&
                  env_active.is_cuda(),
              "all generic density-bank tensors must be CUDA");
  TORCH_CHECK(rij.is_contiguous() && pair_i.is_contiguous() &&
                  pair_j.is_contiguous() && spins.is_contiguous() &&
                  atom_types.is_contiguous() && coefficient.is_contiguous() &&
                  dof_active.is_contiguous() && env_active.is_contiguous(),
              "all generic density-bank tensors must be contiguous");
  TORCH_CHECK(rij.dim() == 2 && rij.size(1) == 3, "rij must be [edges,3]");
  TORCH_CHECK(spins.dim() == 2 && spins.size(1) == 3,
              "spins must be [atoms,3]");
  TORCH_CHECK(pair_i.scalar_type() == torch::kInt64 &&
                  pair_j.scalar_type() == torch::kInt64 &&
                  atom_types.scalar_type() == torch::kInt64,
              "indices and atom types must be int64");
  TORCH_CHECK(dof_active.scalar_type() == torch::kBool &&
                  env_active.scalar_type() == torch::kBool,
              "active masks must be bool");
  TORCH_CHECK(coefficient.dim() == 5 && coefficient.size(0) == 1 &&
                  coefficient.size(1) == coefficient.size(2) &&
                  coefficient.size(3) >= 1 && coefficient.size(4) >= 1,
              "coefficient must be [1,T,T,C,B] with C,B >= 1");
  TORCH_CHECK(coefficient.scalar_type() == rij.scalar_type() &&
                  spins.scalar_type() == rij.scalar_type(),
              "floating tensors must have one dtype");
}

} // namespace

std::vector<torch::Tensor> density_bank_generic_forward(
    torch::Tensor rij, torch::Tensor pair_i, torch::Tensor pair_j,
    torch::Tensor spins, torch::Tensor atom_types, torch::Tensor coefficient,
    torch::Tensor dof_active, torch::Tensor env_active, double cutoff,
    double eps) {
  check_inputs(rij, pair_i, pair_j, spins, atom_types, coefficient, dof_active,
               env_active);
  c10::cuda::CUDAGuard guard(rij.device());
  const int64_t channels = coefficient.size(3);
  const int64_t basis_width = coefficient.size(4);
  const int64_t pair_width = channels * (channels + 1) / 2;
  const int64_t bank_width =
      ChannelStride * channels + pair_width + 3 * channels;
  const int64_t aux_width = channels + 6;
  auto bank = torch::zeros({spins.size(0), bank_width}, rij.options());
  auto auxiliary = torch::empty({rij.size(0), aux_width}, rij.options());
  const int threads = 128;
  const int blocks = static_cast<int>((rij.size(0) + threads - 1) / threads);
  const int pair_blocks =
      static_cast<int>((rij.size(0) * pair_width + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(
        rij.scalar_type(), "density_bank_generic_forward", [&] {
          bank_forward_kernel<scalar_t>
              <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                  rij.data_ptr<scalar_t>(), pair_i.data_ptr<int64_t>(),
                  pair_j.data_ptr<int64_t>(), spins.data_ptr<scalar_t>(),
                  atom_types.data_ptr<int64_t>(),
                  coefficient.data_ptr<scalar_t>(), dof_active.data_ptr<bool>(),
                  env_active.data_ptr<bool>(), rij.size(0), coefficient.size(1),
                  channels, basis_width, bank_width, aux_width, cutoff, eps,
                  bank.data_ptr<scalar_t>(), auxiliary.data_ptr<scalar_t>());
          pair_forward_kernel<scalar_t>
              <<<pair_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                  rij.data_ptr<scalar_t>(), pair_i.data_ptr<int64_t>(),
                  pair_j.data_ptr<int64_t>(), spins.data_ptr<scalar_t>(),
                  atom_types.data_ptr<int64_t>(),
                  coefficient.data_ptr<scalar_t>(), dof_active.data_ptr<bool>(),
                  env_active.data_ptr<bool>(), rij.size(0), coefficient.size(1),
                  channels, basis_width, pair_width, bank_width, cutoff, eps,
                  bank.data_ptr<scalar_t>());
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {bank, auxiliary};
}

std::vector<torch::Tensor> density_bank_generic_backward(
    torch::Tensor grad_bank, torch::Tensor grad_auxiliary, torch::Tensor rij,
    torch::Tensor pair_i, torch::Tensor pair_j, torch::Tensor spins,
    torch::Tensor atom_types, torch::Tensor coefficient,
    torch::Tensor dof_active, torch::Tensor env_active, double cutoff,
    double eps) {
  check_inputs(rij, pair_i, pair_j, spins, atom_types, coefficient, dof_active,
               env_active);
  c10::cuda::CUDAGuard guard(rij.device());
  const int64_t channels = coefficient.size(3);
  const int64_t basis_width = coefficient.size(4);
  const int64_t pair_width = channels * (channels + 1) / 2;
  const int64_t bank_width =
      ChannelStride * channels + pair_width + 3 * channels;
  const int64_t aux_width = channels + 6;
  TORCH_CHECK(grad_bank.sizes() ==
                  torch::IntArrayRef({spins.size(0), bank_width}),
              "grad_bank has the wrong runtime-C shape");
  TORCH_CHECK(grad_auxiliary.sizes() ==
                  torch::IntArrayRef({rij.size(0), aux_width}),
              "grad_auxiliary has the wrong runtime-C shape");
  TORCH_CHECK(grad_bank.scalar_type() == rij.scalar_type() &&
                  grad_auxiliary.scalar_type() == rij.scalar_type(),
              "pull tensors must have the input dtype");
  auto grad_rij = torch::empty_like(rij);
  auto grad_spins = torch::zeros_like(spins);
  auto grad_coefficient = torch::zeros_like(coefficient);
  const int threads = 128;
  const int blocks = static_cast<int>((rij.size(0) + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(
        rij.scalar_type(), "density_bank_generic_backward", [&] {
          bank_backward_kernel<scalar_t>
              <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                  grad_bank.data_ptr<scalar_t>(),
                  grad_auxiliary.data_ptr<scalar_t>(), rij.data_ptr<scalar_t>(),
                  pair_i.data_ptr<int64_t>(), pair_j.data_ptr<int64_t>(),
                  spins.data_ptr<scalar_t>(), atom_types.data_ptr<int64_t>(),
                  coefficient.data_ptr<scalar_t>(), dof_active.data_ptr<bool>(),
                  env_active.data_ptr<bool>(), rij.size(0), coefficient.size(1),
                  channels, basis_width, bank_width, aux_width, cutoff, eps,
                  grad_rij.data_ptr<scalar_t>(),
                  grad_spins.data_ptr<scalar_t>(),
                  grad_coefficient.data_ptr<scalar_t>());
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad_rij, grad_spins, grad_coefficient};
}

std::vector<torch::Tensor> density_bank_generic_double_backward(
    torch::Tensor grad_bank, torch::Tensor grad_auxiliary, torch::Tensor rij,
    torch::Tensor pair_i, torch::Tensor pair_j, torch::Tensor spins,
    torch::Tensor atom_types, torch::Tensor coefficient,
    torch::Tensor dof_active, torch::Tensor env_active, torch::Tensor h_rij,
    torch::Tensor h_spins, torch::Tensor h_coefficient, double cutoff,
    double eps) {
  check_inputs(rij, pair_i, pair_j, spins, atom_types, coefficient, dof_active,
               env_active);
  c10::cuda::CUDAGuard guard(rij.device());
  const int64_t channels = coefficient.size(3);
  const int64_t basis_width = coefficient.size(4);
  const int64_t pair_width = channels * (channels + 1) / 2;
  const int64_t bank_width =
      ChannelStride * channels + pair_width + 3 * channels;
  const int64_t aux_width = channels + 6;
  TORCH_CHECK(h_rij.sizes() == rij.sizes() &&
                  h_spins.sizes() == spins.sizes() &&
                  h_coefficient.sizes() == coefficient.sizes(),
              "double-backward directions must match primal shapes");
  auto grad2_bank = torch::zeros_like(grad_bank);
  auto grad2_auxiliary = torch::empty_like(grad_auxiliary);
  auto grad2_rij = torch::empty_like(rij);
  auto grad2_spins = torch::zeros_like(spins);
  auto grad2_coefficient = torch::zeros_like(coefficient);
  const int threads = 128;
  const int blocks = static_cast<int>((rij.size(0) + threads - 1) / threads);
  const int pair_blocks =
      static_cast<int>((rij.size(0) * pair_width + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(
        rij.scalar_type(), "density_bank_generic_double_backward", [&] {
          bank_directional_kernel<scalar_t>
              <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                  rij.data_ptr<scalar_t>(), pair_i.data_ptr<int64_t>(),
                  pair_j.data_ptr<int64_t>(), spins.data_ptr<scalar_t>(),
                  atom_types.data_ptr<int64_t>(),
                  coefficient.data_ptr<scalar_t>(), dof_active.data_ptr<bool>(),
                  env_active.data_ptr<bool>(), h_rij.data_ptr<scalar_t>(),
                  h_spins.data_ptr<scalar_t>(),
                  h_coefficient.data_ptr<scalar_t>(), rij.size(0),
                  coefficient.size(1), channels, basis_width, bank_width,
                  aux_width, cutoff, eps, grad2_bank.data_ptr<scalar_t>(),
                  grad2_auxiliary.data_ptr<scalar_t>());
          pair_directional_kernel<scalar_t>
              <<<pair_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                  rij.data_ptr<scalar_t>(), pair_i.data_ptr<int64_t>(),
                  pair_j.data_ptr<int64_t>(), spins.data_ptr<scalar_t>(),
                  atom_types.data_ptr<int64_t>(),
                  coefficient.data_ptr<scalar_t>(), dof_active.data_ptr<bool>(),
                  env_active.data_ptr<bool>(), h_rij.data_ptr<scalar_t>(),
                  h_spins.data_ptr<scalar_t>(),
                  h_coefficient.data_ptr<scalar_t>(), rij.size(0),
                  coefficient.size(1), channels, basis_width, pair_width,
                  bank_width, cutoff, eps, grad2_bank.data_ptr<scalar_t>());
          bank_reverse_directional_kernel<scalar_t>
              <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                  grad_bank.data_ptr<scalar_t>(),
                  grad_auxiliary.data_ptr<scalar_t>(), rij.data_ptr<scalar_t>(),
                  pair_i.data_ptr<int64_t>(), pair_j.data_ptr<int64_t>(),
                  spins.data_ptr<scalar_t>(), atom_types.data_ptr<int64_t>(),
                  coefficient.data_ptr<scalar_t>(), dof_active.data_ptr<bool>(),
                  env_active.data_ptr<bool>(), h_rij.data_ptr<scalar_t>(),
                  h_spins.data_ptr<scalar_t>(),
                  h_coefficient.data_ptr<scalar_t>(), rij.size(0),
                  coefficient.size(1), channels, basis_width, bank_width,
                  aux_width, cutoff, eps, grad2_rij.data_ptr<scalar_t>(),
                  grad2_spins.data_ptr<scalar_t>(),
                  grad2_coefficient.data_ptr<scalar_t>());
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad2_bank, grad2_auxiliary, grad2_rij, grad2_spins,
          grad2_coefficient};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &density_bank_generic_forward,
             "Runtime-C/B canonical density-bank forward (CUDA)");
  module.def("backward", &density_bank_generic_backward,
             "Runtime-C/B canonical density-bank VJP (CUDA)");
  module.def("double_backward", &density_bank_generic_double_backward,
             "Runtime-C/B canonical density-bank mixed pullback (CUDA)");
}
