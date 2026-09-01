#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

namespace {
constexpr int Stride = 47;
constexpr int Polar = 3;
constexpr int Q = 18;

template <typename T> struct Dual {
  T v, d;
  __device__ Dual() : v(0), d(0) {}
  __device__ Dual(T x) : v(x), d(0) {}
  __device__ Dual(T x, T h) : v(x), d(h) {}
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
template <typename T> __device__ Dual<T> &operator+=(Dual<T> &a, Dual<T> b) {
  a.v += b.v;
  a.d += b.d;
  return a;
}
template <typename T> __device__ T value(T x) { return x; }
template <typename T> __device__ T value(Dual<T> x) { return x.v; }
template <typename T> __device__ T deriv(T) { return T(0); }
template <typename T> __device__ T deriv(Dual<T> x) { return x.d; }
template <typename S, typename T> struct Seed;
template <typename T> struct Seed<T, T> {
  __device__ static T make(T x, T) { return x; }
};
template <typename T> struct Seed<Dual<T>, T> {
  __device__ static Dual<T> make(T x, T h) { return {x, h}; }
};
template <typename S, typename T> __device__ S seed(T x, const T *h) {
  return Seed<S, T>::make(x, h ? *h : T(0));
}

template <typename S> __device__ S dot3(const S *a, const S *b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
template <typename S> __device__ void cross3(const S *a, const S *b, S *o) {
  o[0] = a[1] * b[2] - a[2] * b[1];
  o[1] = a[2] * b[0] - a[0] * b[2];
  o[2] = a[0] * b[1] - a[1] * b[0];
}
template <typename S> __device__ S triple3(const S *a, const S *b, const S *c) {
  S x[3];
  cross3(a, b, x);
  return dot3(x, c);
}
template <typename S>
__device__ void triple_pull(const S *a, const S *b, const S *c, S scale, S *ga,
                            S *gb, S *gc) {
  S x[3], y[3], z[3];
  cross3(b, c, x);
  cross3(c, a, y);
  cross3(a, b, z);
  for (int d = 0; d < 3; ++d) {
    ga[d] += scale * x[d];
    gb[d] += scale * y[d];
    gc[d] += scale * z[d];
  }
}
template <typename S> __device__ void stf_matrix(const S *q, S *m) {
  const S a = S(0.40824829046386301637), b = S(0.70710678118654752440);
  m[0] = -q[0] * a + q[1] * b;
  m[4] = -q[0] * a - q[1] * b;
  m[8] = S(2) * q[0] * a;
  m[1] = m[3] = q[2] * b;
  m[5] = m[7] = q[3] * b;
  m[2] = m[6] = q[4] * b;
}
template <typename S> __device__ void stf_matvec(const S *q, const S *v, S *o) {
  S m[9];
  stf_matrix(q, m);
  for (int r = 0; r < 3; ++r) {
    o[r] = S(0);
    for (int c = 0; c < 3; ++c)
      o[r] += m[3 * r + c] * v[c];
  }
}
template <typename S> __device__ void matrix_stf_pull(const S *m, S *gq) {
  const S a = S(0.40824829046386301637), b = S(0.70710678118654752440);
  gq[0] += (-m[0] - m[4] + S(2) * m[8]) * a;
  gq[1] += (m[0] - m[4]) * b;
  gq[2] += (m[1] + m[3]) * b;
  gq[3] += (m[5] + m[7]) * b;
  gq[4] += (m[2] + m[6]) * b;
}
template <typename S>
__device__ void stf_matvec_pull(const S *q, const S *v, const S *pull, S *gq,
                                S *gv) {
  S m[9], gm[9];
  stf_matrix(q, m);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) {
      gv[c] += m[3 * r + c] * pull[r];
      gm[3 * r + c] = pull[r] * v[c];
    }
  matrix_stf_pull(gm, gq);
}

template <typename S, typename T>
__device__ void project(const T *bank, const T *aux, const T *mix,
                        const T *h_bank, const T *h_aux, const T *h_mix,
                        int64_t atom, int64_t edge, int64_t C,
                        int64_t bank_width, int64_t r, S *ew, S *p0, S *p1,
                        S *q0, S *rhat, S *omega) {
  *ew = S(0);
  for (int d = 0; d < 3; ++d) {
    p0[d] = S(0);
    p1[d] = S(0);
    rhat[d] = seed<S>(aux[edge * (C + 6) + C + d],
                      h_aux ? h_aux + edge * (C + 6) + C + d : nullptr);
    omega[d] = seed<S>(aux[edge * (C + 6) + C + 3 + d],
                       h_aux ? h_aux + edge * (C + 6) + C + 3 + d : nullptr);
  }
  for (int k = 0; k < 5; ++k)
    q0[k] = S(0);
  const int64_t mix_plane = C * C;
  for (int64_t c = 0; c < C; ++c) {
    const int64_t base = atom * bank_width + c * Stride;
    S m0 = seed<S>(mix[r * C + c], h_mix ? h_mix + r * C + c : nullptr);
    S m1 = seed<S>(mix[mix_plane + r * C + c],
                   h_mix ? h_mix + mix_plane + r * C + c : nullptr);
    S w = seed<S>(aux[edge * (C + 6) + c],
                  h_aux ? h_aux + edge * (C + 6) + c : nullptr);
    *ew += m0 * w;
    for (int d = 0; d < 3; ++d) {
      S p = seed<S>(bank[base + Polar + d],
                    h_bank ? h_bank + base + Polar + d : nullptr);
      p0[d] += m0 * p;
      p1[d] += m1 * p;
    }
    for (int k = 0; k < 5; ++k) {
      S q =
          seed<S>(bank[base + Q + k], h_bank ? h_bank + base + Q + k : nullptr);
      q0[k] += m0 * q;
    }
  }
}

template <typename S, typename T, bool Tangent>
__device__ void edge_item(const T *grad_output, const T *bank, const T *aux,
                          const T *mix, const T *h_bank, const T *h_aux,
                          const T *h_mix, int64_t atom, int64_t edge, int64_t C,
                          int64_t bank_width, int64_t r, T *out, T *grad_bank,
                          T *grad_aux, T *grad_mix) {
  S ew, p0[3], p1[3], q0[5], rhat[3], omega[3];
  project<S>(bank, aux, mix, h_bank, h_aux, h_mix, atom, edge, C, bank_width, r,
             &ew, p0, p1, q0, rhat, omega);
  S qr[3], qp[3];
  stf_matvec(q0, rhat, qr);
  stf_matvec(q0, p1, qp);
  const S er = dot3(rhat, omega);
  S vals[3] = {ew * triple3(p0, rhat, omega), ew * triple3(rhat, qr, omega),
               ew * er * triple3(p0, p1, qp)};
  if (out)
    for (int f = 0; f < 3; ++f)
      atomicAdd(out + atom * (3 * C) + f * C + r,
                Tangent ? deriv(vals[f]) : value(vals[f]));
  if (!grad_output)
    return;
  S go[3] = {S(grad_output[atom * (3 * C) + r]),
             S(grad_output[atom * (3 * C) + C + r]),
             S(grad_output[atom * (3 * C) + 2 * C + r])};
  S gew = S(0), gp0[3] = {S(0), S(0), S(0)}, gp1[3] = {S(0), S(0), S(0)},
    gq[5] = {S(0), S(0), S(0), S(0), S(0)}, gr[3] = {S(0), S(0), S(0)},
    gom[3] = {S(0), S(0), S(0)};
  S a = triple3(p0, rhat, omega);
  gew += go[0] * a;
  triple_pull(p0, rhat, omega, go[0] * ew, gp0, gr, gom);
  S b = triple3(rhat, qr, omega), gqr[3] = {S(0), S(0), S(0)};
  gew += go[1] * b;
  triple_pull(rhat, qr, omega, go[1] * ew, gr, gqr, gom);
  stf_matvec_pull(q0, rhat, gqr, gq, gr);
  S m = triple3(p0, p1, qp), ger = go[2] * ew * m;
  gew += go[2] * er * m;
  for (int d = 0; d < 3; ++d) {
    gr[d] += ger * omega[d];
    gom[d] += ger * rhat[d];
  }
  S gqp[3] = {S(0), S(0), S(0)};
  triple_pull(p0, p1, qp, go[2] * ew * er, gp0, gp1, gqp);
  stf_matvec_pull(q0, p1, gqp, gq, gp1);
  const int64_t plane = C * C;
  for (int64_t c = 0; c < C; ++c) {
    const int64_t base = atom * bank_width + c * Stride;
    S m0 = seed<S>(mix[r * C + c], h_mix ? h_mix + r * C + c : nullptr);
    S m1 = seed<S>(mix[plane + r * C + c],
                   h_mix ? h_mix + plane + r * C + c : nullptr);
    S w = seed<S>(aux[edge * (C + 6) + c],
                  h_aux ? h_aux + edge * (C + 6) + c : nullptr);
    S gm0 = gew * w, gm1 = S(0);
    S gw = gew * m0;
    for (int d = 0; d < 3; ++d) {
      S p = seed<S>(bank[base + Polar + d],
                    h_bank ? h_bank + base + Polar + d : nullptr);
      S g = m0 * gp0[d] + m1 * gp1[d];
      gm0 += gp0[d] * p;
      gm1 += gp1[d] * p;
      atomicAdd(grad_bank + base + Polar + d, Tangent ? deriv(g) : value(g));
    }
    for (int k = 0; k < 5; ++k) {
      S q =
          seed<S>(bank[base + Q + k], h_bank ? h_bank + base + Q + k : nullptr);
      S g = m0 * gq[k];
      gm0 += gq[k] * q;
      atomicAdd(grad_bank + base + Q + k, Tangent ? deriv(g) : value(g));
    }
    atomicAdd(grad_aux + edge * (C + 6) + c, Tangent ? deriv(gw) : value(gw));
    atomicAdd(grad_mix + r * C + c, Tangent ? deriv(gm0) : value(gm0));
    atomicAdd(grad_mix + plane + r * C + c, Tangent ? deriv(gm1) : value(gm1));
  }
  for (int d = 0; d < 3; ++d) {
    atomicAdd(grad_aux + edge * (C + 6) + C + d,
              Tangent ? deriv(gr[d]) : value(gr[d]));
    atomicAdd(grad_aux + edge * (C + 6) + C + 3 + d,
              Tangent ? deriv(gom[d]) : value(gom[d]));
  }
}

template <typename T>
__global__ void forward_kernel(const T *bank, const T *aux,
                               const int64_t *pair_i, const T *mix, int64_t E,
                               int64_t C, int64_t W, T *out) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= E * C)
    return;
  int64_t e = x / C, r = x % C;
  edge_item<T, T, false>(nullptr, bank, aux, mix, nullptr, nullptr, nullptr,
                         pair_i[e], e, C, W, r, out, nullptr, nullptr, nullptr);
}
template <typename T>
__global__ void backward_kernel(const T *go, const T *bank, const T *aux,
                                const int64_t *pair_i, const T *mix, int64_t E,
                                int64_t C, int64_t W, T *gb, T *ga, T *gm) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= E * C)
    return;
  int64_t e = x / C, r = x % C;
  edge_item<T, T, false>(go, bank, aux, mix, nullptr, nullptr, nullptr,
                         pair_i[e], e, C, W, r, nullptr, gb, ga, gm);
}
template <typename T>
__global__ void double_kernel(const T *go, const T *bank, const T *aux,
                              const int64_t *pair_i, const T *mix, const T *hb,
                              const T *ha, const T *hm, int64_t E, int64_t C,
                              int64_t W, T *ggo, T *gb, T *ga, T *gm) {
  int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= E * C)
    return;
  int64_t e = x / C, r = x % C;
  edge_item<Dual<T>, T, true>(go, bank, aux, mix, hb, ha, hm, pair_i[e], e, C,
                              W, r, ggo, gb, ga, gm);
}

void check(const torch::Tensor &b, const torch::Tensor &a,
           const torch::Tensor &i, const torch::Tensor &m) {
  TORCH_CHECK(b.is_cuda() && a.is_cuda() && i.is_cuda() && m.is_cuda(),
              "generic edge tensors must be CUDA");
  TORCH_CHECK(b.is_contiguous() && a.is_contiguous() && i.is_contiguous() &&
                  m.is_contiguous(),
              "generic edge tensors must be contiguous");
  TORCH_CHECK(m.dim() == 3 && m.size(0) == 4 && m.size(1) == m.size(2),
              "mix must be [4,C,C]");
  int64_t C = m.size(1);
  TORCH_CHECK(b.size(1) == Stride * C + C * (C + 1) / 2 + 3 * C &&
                  a.size(1) == C + 6,
              "bank/aux runtime-C width mismatch");
}
} // namespace

torch::Tensor edge_generic_forward(torch::Tensor b, torch::Tensor a,
                                   torch::Tensor i, torch::Tensor m) {
  check(b, a, i, m);
  c10::cuda::CUDAGuard g(b.device());
  int64_t C = m.size(1), n = a.size(0) * C;
  auto o = torch::zeros({b.size(0), 3 * C}, b.options());
  int t = 128, bl = (n + t - 1) / t;
  if (bl) {
    AT_DISPATCH_FLOATING_TYPES(b.scalar_type(), "edge_generic_forward", [&] {
      forward_kernel<scalar_t><<<bl, t, 0, at::cuda::getCurrentCUDAStream()>>>(
          b.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), i.data_ptr<int64_t>(),
          m.data_ptr<scalar_t>(), a.size(0), C, b.size(1),
          o.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return o;
}
std::vector<torch::Tensor>
edge_generic_backward(torch::Tensor go, torch::Tensor b, torch::Tensor a,
                      torch::Tensor i, torch::Tensor m) {
  check(b, a, i, m);
  c10::cuda::CUDAGuard g(b.device());
  int64_t C = m.size(1), n = a.size(0) * C;
  auto gb = torch::zeros_like(b), ga = torch::zeros_like(a),
       gm = torch::zeros_like(m);
  int t = 128, bl = (n + t - 1) / t;
  if (bl) {
    AT_DISPATCH_FLOATING_TYPES(b.scalar_type(), "edge_generic_backward", [&] {
      backward_kernel<scalar_t><<<bl, t, 0, at::cuda::getCurrentCUDAStream()>>>(
          go.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
          a.data_ptr<scalar_t>(), i.data_ptr<int64_t>(), m.data_ptr<scalar_t>(),
          a.size(0), C, b.size(1), gb.data_ptr<scalar_t>(),
          ga.data_ptr<scalar_t>(), gm.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {gb, ga, gm};
}
std::vector<torch::Tensor>
edge_generic_double(torch::Tensor go, torch::Tensor b, torch::Tensor a,
                    torch::Tensor i, torch::Tensor m, torch::Tensor hb,
                    torch::Tensor ha, torch::Tensor hm) {
  check(b, a, i, m);
  c10::cuda::CUDAGuard g(b.device());
  int64_t C = m.size(1), n = a.size(0) * C;
  auto ggo = torch::zeros_like(go), gb = torch::zeros_like(b),
       ga = torch::zeros_like(a), gm = torch::zeros_like(m);
  int t = 128, bl = (n + t - 1) / t;
  if (bl) {
    AT_DISPATCH_FLOATING_TYPES(b.scalar_type(), "edge_generic_double", [&] {
      double_kernel<scalar_t><<<bl, t, 0, at::cuda::getCurrentCUDAStream()>>>(
          go.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
          a.data_ptr<scalar_t>(), i.data_ptr<int64_t>(), m.data_ptr<scalar_t>(),
          hb.data_ptr<scalar_t>(), ha.data_ptr<scalar_t>(),
          hm.data_ptr<scalar_t>(), a.size(0), C, b.size(1),
          ggo.data_ptr<scalar_t>(), gb.data_ptr<scalar_t>(),
          ga.data_ptr<scalar_t>(), gm.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {ggo, gb, ga, gm};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &edge_generic_forward);
  m.def("backward", &edge_generic_backward);
  m.def("double_backward", &edge_generic_double);
}
