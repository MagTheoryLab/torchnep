#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

template <typename T>
__device__ T monomial(const T p[3][5], int a, int b, int c, T coefficient) {
  return coefficient * p[0][a] * p[1][b] * p[2][c];
}

template <typename T, int Mode>
__global__ void harmonics_kernel(
    const T* xyz, const T* coefficients, const int64_t* powers,
    const int64_t* offsets, const T* pull, const T* tangent,
    int64_t n, int channels, T* output, T* grad_xyz) {
  const int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= n) return;
  T p[3][5];
  for (int k = 0; k < 3; ++k) {
    p[k][0] = T(1);
    p[k][1] = xyz[3 * row + k];
    p[k][2] = p[k][1] * p[k][1];
    p[k][3] = p[k][2] * p[k][1];
    p[k][4] = p[k][2] * p[k][2];
  }
  T gx[3] = {T(0), T(0), T(0)};
  for (int d = 0; d < channels; ++d) {
    T value = T(0), jacobian[3] = {T(0), T(0), T(0)};
    T hessian_tangent[3] = {T(0), T(0), T(0)};
    for (int64_t t = offsets[d]; t < offsets[d + 1]; ++t) {
      int e[3] = {int(powers[3*t]), int(powers[3*t+1]), int(powers[3*t+2])};
      const T c = coefficients[t];
      if (Mode == 0) {
        value += monomial(p, e[0], e[1], e[2], c);
      } else {
        for (int k = 0; k < 3; ++k) {
          const int ek = e[k];
          if (!ek) continue;
          --e[k];
          jacobian[k] += monomial(p, e[0], e[1], e[2], c * T(ek));
          if (Mode == 2) {
            for (int l = 0; l < 3; ++l) {
              const int el = e[l];
              if (!el) continue;
              --e[l];
              hessian_tangent[k] += monomial(
                  p, e[0], e[1], e[2], c * T(ek * el)) * tangent[3 * row + l];
              ++e[l];
            }
          }
          ++e[k];
        }
      }
    }
    if (Mode == 0) {
      output[row * channels + d] = value;
    } else {
      for (int k = 0; k < 3; ++k) {
        gx[k] += pull[row * channels + d] *
            (Mode == 1 ? jacobian[k] : hessian_tangent[k]);
        if (Mode == 2) value += jacobian[k] * tangent[3 * row + k];
      }
      if (Mode == 2) output[row * channels + d] = value;
    }
  }
  if (Mode != 0) {
    for (int k = 0; k < 3; ++k) grad_xyz[3 * row + k] = gx[k];
  }
}

void check(at::Tensor xyz, at::Tensor c, at::Tensor p, at::Tensor o) {
  TORCH_CHECK(xyz.is_cuda() && xyz.is_contiguous() && xyz.dim() == 2 && xyz.size(1) == 3,
              "xyz must be contiguous CUDA [N,3]");
  TORCH_CHECK(xyz.scalar_type() == at::kFloat || xyz.scalar_type() == at::kDouble,
              "harmonics requires float32 or float64");
  for (const auto& t : {c, p, o}) {
    TORCH_CHECK(t.device() == xyz.device() && t.is_contiguous(), "harmonic table device/layout mismatch");
  }
  TORCH_CHECK(c.scalar_type() == xyz.scalar_type() && c.dim() == 1 &&
              p.scalar_type() == at::kLong && p.dim() == 2 &&
              p.size(0) == c.numel() && p.size(1) == 3 &&
              o.scalar_type() == at::kLong && o.dim() == 1 &&
              o.numel() >= 2 && o.numel() <= 26, "invalid c516 harmonic tables");
}

void check_derivative(at::Tensor t, at::Tensor xyz, int64_t width) {
  TORCH_CHECK(t.device() == xyz.device() && t.scalar_type() == xyz.scalar_type() &&
              t.is_contiguous() && t.dim() == 2 && t.size(0) == xyz.size(0) &&
              t.size(1) == width, "harmonic derivative shape/device/dtype mismatch");
}

template <int Mode>
std::vector<at::Tensor> evaluate(
    at::Tensor xyz, at::Tensor c, at::Tensor p, at::Tensor o,
    at::Tensor pull = {}, at::Tensor tangent = {}) {
  check(xyz, c, p, o);
  const int channels = o.numel() - 1;
  if (Mode != 0) check_derivative(pull, xyz, channels);
  if (Mode == 2) check_derivative(tangent, xyz, 3);
  c10::cuda::CUDAGuard guard(xyz.device());
  auto output = Mode == 1 ? at::Tensor() : at::empty({xyz.size(0), channels}, xyz.options());
  auto grad = Mode == 0 ? at::Tensor() : at::empty_like(xyz);
  const int threads = 128;
  const int blocks = (xyz.size(0) + threads - 1) / threads;
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(xyz.scalar_type(), "c516_harmonics", [&] {
      harmonics_kernel<scalar_t, Mode><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          xyz.data_ptr<scalar_t>(), c.data_ptr<scalar_t>(), p.data_ptr<int64_t>(),
          o.data_ptr<int64_t>(), Mode ? pull.data_ptr<scalar_t>() : nullptr,
          Mode == 2 ? tangent.data_ptr<scalar_t>() : nullptr,
          xyz.size(0), channels, Mode == 1 ? nullptr : output.data_ptr<scalar_t>(),
          Mode ? grad.data_ptr<scalar_t>() : nullptr);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output, grad};
}
}  // namespace

at::Tensor harmonics_forward(at::Tensor x, at::Tensor c, at::Tensor p, at::Tensor o) {
  return evaluate<0>(x,c,p,o)[0];
}

at::Tensor harmonics_backward(at::Tensor g, at::Tensor x, at::Tensor c, at::Tensor p, at::Tensor o) {
  return evaluate<1>(x,c,p,o,g)[1];
}

std::vector<at::Tensor> harmonics_double_backward(
    at::Tensor g, at::Tensor x, at::Tensor c,
    at::Tensor p, at::Tensor o, at::Tensor h) {
  return evaluate<2>(x,c,p,o,g,h);
}
