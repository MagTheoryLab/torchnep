#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <vector>

namespace {

constexpr int RadialChannels = 5;
constexpr int MomentWidth = 24;
constexpr int OutputWidth = 25;

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
template <typename T> __device__ Dual<T> operator*(Dual<T> a, Dual<T> b) {
  return {a.v * b.v, a.d * b.v + a.v * b.d};
}
template <typename T> __device__ Dual<T>& operator+=(Dual<T>& a, Dual<T> b) {
  a.v += b.v;
  a.d += b.d;
  return a;
}

template <typename S>
__device__ void evaluate(const S* moments, const S* c3b, const S* c4b,
                         S* output) {
  for (int li = 0; li < 4; ++li) {
    const int order = li + 1;
    const int width = 2 * order + 1;
    const int start = order * order - 1;
    for (int n = 0; n < RadialChannels; ++n) {
      const S* s = moments + n * MomentWidth + start;
      S value = c3b[start] * s[0] * s[0];
      for (int m = 1; m < width; ++m) {
        value += S(2) * c3b[start + m] * s[m] * s[m];
      }
      output[li * RadialChannels + n] = value;
    }
  }
  for (int n = 0; n < RadialChannels; ++n) {
    const S* s = moments + n * MomentWidth;
    const S s20 = s[3], s21r = s[4], s21i = s[5];
    const S s22r = s[6], s22i = s[7];
    output[20 + n] =
        c4b[0] * s20 * s20 * s20
        + c4b[1] * s20 * (s21r * s21r + s21i * s21i)
        + c4b[2] * s20 * (s22r * s22r + s22i * s22i)
        + c4b[3] * s22r * (s21i * s21i - s21r * s21r)
        + c4b[4] * s21r * s21i * s22i;
  }
}

template <typename S>
__device__ void reverse(const S* moments, const S* c3b, const S* c4b,
                        const S* grad_output, S* grad_moments) {
  for (int k = 0; k < RadialChannels * MomentWidth; ++k) {
    grad_moments[k] = S(0);
  }
  for (int li = 0; li < 4; ++li) {
    const int order = li + 1;
    const int width = 2 * order + 1;
    const int start = order * order - 1;
    for (int n = 0; n < RadialChannels; ++n) {
      const S pull = grad_output[li * RadialChannels + n];
      const S* s = moments + n * MomentWidth + start;
      S* grad = grad_moments + n * MomentWidth + start;
      grad[0] += S(2) * pull * c3b[start] * s[0];
      for (int m = 1; m < width; ++m) {
        grad[m] += S(4) * pull * c3b[start + m] * s[m];
      }
    }
  }
  for (int n = 0; n < RadialChannels; ++n) {
    const S pull = grad_output[20 + n];
    const S* s = moments + n * MomentWidth;
    S* grad = grad_moments + n * MomentWidth;
    const S a = s[3], b = s[4], c = s[5], d = s[6], e = s[7];
    grad[3] += pull * (S(3) * c4b[0] * a * a
        + c4b[1] * (b * b + c * c)
        + c4b[2] * (d * d + e * e));
    grad[4] += pull * (S(2) * c4b[1] * a * b
        - S(2) * c4b[3] * d * b + c4b[4] * c * e);
    grad[5] += pull * (S(2) * c4b[1] * a * c
        + S(2) * c4b[3] * d * c + c4b[4] * b * e);
    grad[6] += pull * (S(2) * c4b[2] * a * d
        + c4b[3] * (c * c - b * b));
    grad[7] += pull * (S(2) * c4b[2] * a * e + c4b[4] * b * c);
  }
}

template <typename scalar_t>
__global__ void forward_kernel(const scalar_t* moments, const scalar_t* c3b,
                               const scalar_t* c4b, int64_t atoms,
                               scalar_t* output) {
  const int64_t atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atoms) return;
  evaluate(moments + atom * RadialChannels * MomentWidth, c3b, c4b,
           output + atom * OutputWidth);
}

template <typename scalar_t>
__global__ void backward_kernel(const scalar_t* grad_output,
                                const scalar_t* moments,
                                const scalar_t* c3b, const scalar_t* c4b,
                                int64_t atoms, scalar_t* grad_moments) {
  const int64_t atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atoms) return;
  reverse(moments + atom * RadialChannels * MomentWidth, c3b, c4b,
          grad_output + atom * OutputWidth,
          grad_moments + atom * RadialChannels * MomentWidth);
}

template <typename scalar_t>
__global__ void double_backward_kernel(
    const scalar_t* grad_output, const scalar_t* moments,
    const scalar_t* c3b, const scalar_t* c4b,
    const scalar_t* h_moments, int64_t atoms,
    scalar_t* grad2_output, scalar_t* grad2_moments) {
  const int64_t atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atoms) return;
  Dual<scalar_t> local_moments[RadialChannels * MomentWidth];
  for (int k = 0; k < RadialChannels * MomentWidth; ++k) {
    local_moments[k] = {
        moments[atom * RadialChannels * MomentWidth + k],
        h_moments[atom * RadialChannels * MomentWidth + k]};
  }
  Dual<scalar_t> local_c3b[MomentWidth], local_c4b[5];
  for (int k = 0; k < MomentWidth; ++k) local_c3b[k] = c3b[k];
  for (int k = 0; k < 5; ++k) local_c4b[k] = c4b[k];
  Dual<scalar_t> local_output[OutputWidth];
  evaluate(local_moments, local_c3b, local_c4b, local_output);
  for (int k = 0; k < OutputWidth; ++k) {
    grad2_output[atom * OutputWidth + k] = local_output[k].d;
  }
  Dual<scalar_t> local_pull[OutputWidth];
  for (int k = 0; k < OutputWidth; ++k) {
    local_pull[k] = {grad_output[atom * OutputWidth + k], scalar_t(0)};
  }
  Dual<scalar_t> local_grad[RadialChannels * MomentWidth];
  reverse(local_moments, local_c3b, local_c4b, local_pull, local_grad);
  for (int k = 0; k < RadialChannels * MomentWidth; ++k) {
    grad2_moments[atom * RadialChannels * MomentWidth + k] = local_grad[k].d;
  }
}

void check_inputs(const torch::Tensor& moments, const torch::Tensor& c3b,
                  const torch::Tensor& c4b) {
  TORCH_CHECK(moments.is_cuda() && c3b.is_cuda() && c4b.is_cuda(),
              "structural-center tensors must be CUDA tensors");
  TORCH_CHECK(moments.is_contiguous() && c3b.is_contiguous() &&
              c4b.is_contiguous(), "structural-center tensors must be contiguous");
  TORCH_CHECK(moments.dim() == 3 && moments.size(1) == RadialChannels &&
              moments.size(2) == MomentWidth, "moments must be [atoms,5,24]");
  TORCH_CHECK(c3b.numel() == MomentWidth && c4b.numel() >= 5,
              "coefficient widths must be 24 and at least 5");
  TORCH_CHECK(moments.scalar_type() == c3b.scalar_type() &&
              moments.scalar_type() == c4b.scalar_type(),
              "structural-center tensors must share dtype");
}

}  // namespace

torch::Tensor structural_center_forward(torch::Tensor moments,
                                        torch::Tensor c3b,
                                        torch::Tensor c4b) {
  check_inputs(moments, c3b, c4b);
  c10::cuda::CUDAGuard guard(moments.device());
  auto output = torch::empty({moments.size(0), OutputWidth}, moments.options());
  const int threads = 64;
  const int blocks = static_cast<int>((moments.size(0) + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(moments.scalar_type(),
                               "structural_center_forward", [&] {
      forward_kernel<scalar_t><<<blocks, threads, 0,
          at::cuda::getCurrentCUDAStream()>>>(
          moments.data_ptr<scalar_t>(), c3b.data_ptr<scalar_t>(),
          c4b.data_ptr<scalar_t>(), moments.size(0), output.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

torch::Tensor structural_center_backward(torch::Tensor grad_output,
                                         torch::Tensor moments,
                                         torch::Tensor c3b,
                                         torch::Tensor c4b) {
  check_inputs(moments, c3b, c4b);
  TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous() &&
              grad_output.dim() == 2 &&
              grad_output.size(0) == moments.size(0) &&
              grad_output.size(1) == OutputWidth &&
              grad_output.scalar_type() == moments.scalar_type(),
              "grad_output must be contiguous CUDA [atoms,25]");
  c10::cuda::CUDAGuard guard(moments.device());
  auto grad_moments = torch::empty_like(moments);
  const int threads = 64;
  const int blocks = static_cast<int>((moments.size(0) + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(moments.scalar_type(),
                               "structural_center_backward", [&] {
      backward_kernel<scalar_t><<<blocks, threads, 0,
          at::cuda::getCurrentCUDAStream()>>>(
          grad_output.data_ptr<scalar_t>(), moments.data_ptr<scalar_t>(),
          c3b.data_ptr<scalar_t>(), c4b.data_ptr<scalar_t>(), moments.size(0),
          grad_moments.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return grad_moments;
}

std::vector<torch::Tensor> structural_center_double_backward(
    torch::Tensor grad_output, torch::Tensor moments, torch::Tensor c3b,
    torch::Tensor c4b, torch::Tensor h_moments) {
  check_inputs(moments, c3b, c4b);
  TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous() &&
              grad_output.dim() == 2 &&
              grad_output.size(0) == moments.size(0) &&
              grad_output.size(1) == OutputWidth &&
              grad_output.scalar_type() == moments.scalar_type(),
              "grad_output must be contiguous CUDA [atoms,25]");
  TORCH_CHECK(h_moments.is_cuda() && h_moments.is_contiguous() &&
              h_moments.sizes() == moments.sizes() &&
              h_moments.scalar_type() == moments.scalar_type(),
              "moment tangent must match moments");
  c10::cuda::CUDAGuard guard(moments.device());
  auto grad2_output = torch::empty_like(grad_output);
  auto grad2_moments = torch::empty_like(moments);
  const int threads = 64;
  const int blocks = static_cast<int>((moments.size(0) + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(moments.scalar_type(),
                               "structural_center_double_backward", [&] {
      double_backward_kernel<scalar_t><<<blocks, threads, 0,
          at::cuda::getCurrentCUDAStream()>>>(
          grad_output.data_ptr<scalar_t>(), moments.data_ptr<scalar_t>(),
          c3b.data_ptr<scalar_t>(), c4b.data_ptr<scalar_t>(),
          h_moments.data_ptr<scalar_t>(), moments.size(0),
          grad2_output.data_ptr<scalar_t>(), grad2_moments.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad2_output, grad2_moments};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &structural_center_forward,
             "O3-C2 structural center forward (CUDA)");
  module.def("backward", &structural_center_backward,
             "O3-C2 structural center VJP (CUDA)");
  module.def("double_backward", &structural_center_double_backward,
             "O3-C2 structural center directional HVP (CUDA)");
}
