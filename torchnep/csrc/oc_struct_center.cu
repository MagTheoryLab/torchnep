#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <vector>

namespace {

constexpr int MaxLm = 24;
constexpr int MaxBlocks = 10;

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

template <typename scalar_t>
struct TermSet {
  const scalar_t* coefficients;
  const int64_t* indices;
  int count;
};

template <typename S, typename scalar_t>
__device__ S evaluate_extra(const S* moments, TermSet<scalar_t> terms) {
  S value = S(0);
  for (int term = 0; term < terms.count; ++term) {
    const int64_t* index = terms.indices + 3 * term;
    value += S(terms.coefficients[term]) * moments[index[0]] *
        moments[index[1]] * moments[index[2]];
  }
  return value;
}

template <typename S, typename scalar_t>
__device__ void reverse_extra(
    const S* moments, TermSet<scalar_t> terms, S pull, S* grad) {
  for (int term = 0; term < terms.count; ++term) {
    const int64_t* index = terms.indices + 3 * term;
    const S coefficient = S(terms.coefficients[term]) * pull;
    const S a = moments[index[0]];
    const S b = moments[index[1]];
    const S c = moments[index[2]];
    grad[index[0]] += coefficient * b * c;
    grad[index[1]] += coefficient * a * c;
    grad[index[2]] += coefficient * a * b;
  }
}

template <typename S, typename scalar_t>
__device__ int evaluate_channel(
    const S* moments, const scalar_t* c3b, const scalar_t* c4b,
    const scalar_t* c5b, const scalar_t* c4b2,
    TermSet<scalar_t> q123, TermSet<scalar_t> q233,
    TermSet<scalar_t> q134, int l_max,
    int has_q222, int has_q1111, int has_q112,
    int has_q123, int has_q233, int has_q134, S* output) {
  int block = 0;
  for (int order = 1; order <= l_max; ++order) {
    const int width = 2 * order + 1;
    const int start = order * order - 1;
    S value = S(c3b[start]) * moments[start] * moments[start];
    for (int m = 1; m < width; ++m) {
      value += S(2 * c3b[start + m]) * moments[start + m] *
          moments[start + m];
    }
    output[block++] = value;
  }
  if (has_q222) {
    const S a = moments[3], b = moments[4], c = moments[5];
    const S d = moments[6], e = moments[7];
    output[block++] =
        S(c4b[0]) * a * a * a
        + S(c4b[1]) * a * (b * b + c * c)
        + S(c4b[2]) * a * (d * d + e * e)
        + S(c4b[3]) * d * (c * c - b * b)
        + S(c4b[4]) * b * c * e;
  }
  if (has_q1111) {
    const S a = moments[0], b = moments[1], c = moments[2];
    const S aa = a * a, ss = b * b + c * c;
    output[block++] = S(c5b[0]) * aa * aa
        + S(c5b[1]) * aa * ss + S(c5b[2]) * ss * ss;
  }
  if (has_q112) {
    const S a = moments[0], b = moments[1], c = moments[2];
    const S d = moments[3], e = moments[4], f = moments[5];
    const S g = moments[6], h = moments[7];
    output[block++] =
        S(c4b2[0]) * a * a * d
        + S(c4b2[1]) * a * (b * e + c * f)
        + S(c4b2[2]) * d * (b * b + c * c)
        + S(c4b2[3]) * g * (b * b - c * c)
        + S(c4b2[4]) * b * c * h;
  }
  if (has_q123) output[block++] = evaluate_extra(moments, q123);
  if (has_q233) output[block++] = evaluate_extra(moments, q233);
  if (has_q134) output[block++] = evaluate_extra(moments, q134);
  return block;
}

template <typename S, typename scalar_t>
__device__ void reverse_channel(
    const S* moments, const scalar_t* c3b, const scalar_t* c4b,
    const scalar_t* c5b, const scalar_t* c4b2,
    TermSet<scalar_t> q123, TermSet<scalar_t> q233,
    TermSet<scalar_t> q134, int num_lm, int l_max,
    int has_q222, int has_q1111, int has_q112,
    int has_q123, int has_q233, int has_q134,
    const S* pull, S* grad) {
  for (int k = 0; k < num_lm; ++k) grad[k] = S(0);
  int block = 0;
  for (int order = 1; order <= l_max; ++order) {
    const int width = 2 * order + 1;
    const int start = order * order - 1;
    grad[start] += S(2 * c3b[start]) * pull[block] * moments[start];
    for (int m = 1; m < width; ++m) {
      grad[start + m] += S(4 * c3b[start + m]) * pull[block] *
          moments[start + m];
    }
    ++block;
  }
  if (has_q222) {
    const S p = pull[block++];
    const S a = moments[3], b = moments[4], c = moments[5];
    const S d = moments[6], e = moments[7];
    grad[3] += p * (S(3 * c4b[0]) * a * a
        + S(c4b[1]) * (b * b + c * c)
        + S(c4b[2]) * (d * d + e * e));
    grad[4] += p * (S(2 * c4b[1]) * a * b
        - S(2 * c4b[3]) * d * b + S(c4b[4]) * c * e);
    grad[5] += p * (S(2 * c4b[1]) * a * c
        + S(2 * c4b[3]) * d * c + S(c4b[4]) * b * e);
    grad[6] += p * (S(2 * c4b[2]) * a * d
        + S(c4b[3]) * (c * c - b * b));
    grad[7] += p * (S(2 * c4b[2]) * a * e + S(c4b[4]) * b * c);
  }
  if (has_q1111) {
    const S p = pull[block++];
    const S a = moments[0], b = moments[1], c = moments[2];
    const S aa = a * a, ss = b * b + c * c;
    grad[0] += p * (S(4 * c5b[0]) * a * aa
        + S(2 * c5b[1]) * a * ss);
    grad[1] += p * (S(2 * c5b[1]) * aa * b
        + S(4 * c5b[2]) * ss * b);
    grad[2] += p * (S(2 * c5b[1]) * aa * c
        + S(4 * c5b[2]) * ss * c);
  }
  if (has_q112) {
    const S p = pull[block++];
    const S a = moments[0], b = moments[1], c = moments[2];
    const S d = moments[3], e = moments[4], f = moments[5];
    const S g = moments[6], h = moments[7];
    grad[0] += p * (S(2 * c4b2[0]) * a * d
        + S(c4b2[1]) * (b * e + c * f));
    grad[1] += p * (S(c4b2[1]) * a * e
        + S(2 * c4b2[2]) * d * b
        + S(2 * c4b2[3]) * g * b + S(c4b2[4]) * c * h);
    grad[2] += p * (S(c4b2[1]) * a * f
        + S(2 * c4b2[2]) * d * c
        - S(2 * c4b2[3]) * g * c + S(c4b2[4]) * b * h);
    grad[3] += p * (S(c4b2[0]) * a * a
        + S(c4b2[2]) * (b * b + c * c));
    grad[4] += p * S(c4b2[1]) * a * b;
    grad[5] += p * S(c4b2[1]) * a * c;
    grad[6] += p * S(c4b2[3]) * (b * b - c * c);
    grad[7] += p * S(c4b2[4]) * b * c;
  }
  if (has_q123) reverse_extra(moments, q123, pull[block++], grad);
  if (has_q233) reverse_extra(moments, q233, pull[block++], grad);
  if (has_q134) reverse_extra(moments, q134, pull[block++], grad);
}

template <typename scalar_t>
__global__ void forward_kernel(
    const scalar_t* moments, int64_t atoms, int radial, int num_lm,
    const scalar_t* c3b, const scalar_t* c4b, const scalar_t* c5b,
    const scalar_t* c4b2, TermSet<scalar_t> q123, TermSet<scalar_t> q233,
    TermSet<scalar_t> q134, int l_max, int q222, int q1111, int q112,
    int f123, int f233, int f134, int blocks, scalar_t* output) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= atoms * radial) return;
  const int64_t atom = linear / radial;
  const int channel = linear % radial;
  scalar_t local[MaxBlocks];
  evaluate_channel(
      moments + linear * num_lm, c3b, c4b, c5b, c4b2,
      q123, q233, q134, l_max, q222, q1111, q112,
      f123, f233, f134, local);
  for (int block = 0; block < blocks; ++block) {
    output[atom * blocks * radial + block * radial + channel] = local[block];
  }
}

template <typename scalar_t>
__global__ void backward_kernel(
    const scalar_t* grad_output, const scalar_t* moments,
    int64_t atoms, int radial, int num_lm,
    const scalar_t* c3b, const scalar_t* c4b, const scalar_t* c5b,
    const scalar_t* c4b2, TermSet<scalar_t> q123, TermSet<scalar_t> q233,
    TermSet<scalar_t> q134, int l_max, int q222, int q1111, int q112,
    int f123, int f233, int f134, int blocks, scalar_t* grad_moments) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= atoms * radial) return;
  const int64_t atom = linear / radial;
  const int channel = linear % radial;
  scalar_t pull[MaxBlocks];
  for (int block = 0; block < blocks; ++block) {
    pull[block] = grad_output[atom * blocks * radial + block * radial + channel];
  }
  reverse_channel(
      moments + linear * num_lm, c3b, c4b, c5b, c4b2,
      q123, q233, q134, num_lm, l_max, q222, q1111, q112,
      f123, f233, f134, pull, grad_moments + linear * num_lm);
}

template <typename scalar_t>
__global__ void double_backward_kernel(
    const scalar_t* grad_output, const scalar_t* moments,
    const scalar_t* h_moments, int64_t atoms, int radial, int num_lm,
    const scalar_t* c3b, const scalar_t* c4b, const scalar_t* c5b,
    const scalar_t* c4b2, TermSet<scalar_t> q123, TermSet<scalar_t> q233,
    TermSet<scalar_t> q134, int l_max, int q222, int q1111, int q112,
    int f123, int f233, int f134, int blocks,
    scalar_t* grad2_output, scalar_t* grad2_moments) {
  const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= atoms * radial) return;
  const int64_t atom = linear / radial;
  const int channel = linear % radial;
  Dual<scalar_t> local_moments[MaxLm];
  for (int k = 0; k < num_lm; ++k) {
    local_moments[k] = {
        moments[linear * num_lm + k], h_moments[linear * num_lm + k]};
  }
  Dual<scalar_t> local_output[MaxBlocks];
  evaluate_channel(
      local_moments, c3b, c4b, c5b, c4b2, q123, q233, q134,
      l_max, q222, q1111, q112, f123, f233, f134, local_output);
  Dual<scalar_t> pull[MaxBlocks];
  for (int block = 0; block < blocks; ++block) {
    const int64_t offset = atom * blocks * radial + block * radial + channel;
    grad2_output[offset] = local_output[block].d;
    pull[block] = {grad_output[offset], scalar_t(0)};
  }
  Dual<scalar_t> local_grad[MaxLm];
  reverse_channel(
      local_moments, c3b, c4b, c5b, c4b2, q123, q233, q134,
      num_lm, l_max, q222, q1111, q112, f123, f233, f134,
      pull, local_grad);
  for (int k = 0; k < num_lm; ++k) {
    grad2_moments[linear * num_lm + k] = local_grad[k].d;
  }
}

void check_float_tensor(const torch::Tensor& value, const torch::Tensor& moments,
                        const char* name) {
  TORCH_CHECK(value.is_cuda() && value.is_contiguous(), name,
              " must be contiguous CUDA");
  TORCH_CHECK(value.scalar_type() == moments.scalar_type(), name,
              " must match moment dtype");
}

void check_terms(const torch::Tensor& coefficients, const torch::Tensor& indices,
                 const torch::Tensor& moments, const char* name) {
  check_float_tensor(coefficients, moments, name);
  TORCH_CHECK(indices.is_cuda() && indices.is_contiguous() &&
              indices.scalar_type() == torch::kInt64 && indices.dim() == 2 &&
              indices.size(1) == 3 && indices.size(0) == coefficients.numel(),
              name, " indices must be contiguous CUDA int64 [terms,3]");
}

void check_inputs(
    const torch::Tensor& moments, const torch::Tensor& c3b,
    const torch::Tensor& c4b, const torch::Tensor& c5b,
    const torch::Tensor& c4b2, const torch::Tensor& q123c,
    const torch::Tensor& q123i, const torch::Tensor& q233c,
    const torch::Tensor& q233i, const torch::Tensor& q134c,
    const torch::Tensor& q134i, int64_t l_max) {
  TORCH_CHECK(moments.is_cuda() && moments.is_contiguous() &&
              moments.dim() == 3, "moments must be contiguous CUDA [N,R,lm]");
  TORCH_CHECK(l_max >= 1 && l_max <= 4 &&
              moments.size(2) == l_max * (l_max + 2),
              "moment width must match l_max in [1,4]");
  check_float_tensor(c3b, moments, "c3b");
  check_float_tensor(c4b, moments, "c4b");
  check_float_tensor(c5b, moments, "c5b");
  check_float_tensor(c4b2, moments, "c4b2");
  TORCH_CHECK(c3b.numel() >= moments.size(2) && c4b.numel() >= 5 &&
              c5b.numel() >= 3 && c4b2.numel() >= 5,
              "structural coefficient buffers are too short");
  check_terms(q123c, q123i, moments, "q123");
  check_terms(q233c, q233i, moments, "q233");
  check_terms(q134c, q134i, moments, "q134");
}

int output_blocks(int64_t l_max, int64_t q222, int64_t q1111, int64_t q112,
                  int64_t q123, int64_t q233, int64_t q134) {
  return static_cast<int>(l_max + q222 + q1111 + q112 + q123 + q233 + q134);
}

}  // namespace

torch::Tensor structural_center_forward(
    torch::Tensor moments, torch::Tensor c3b, torch::Tensor c4b,
    torch::Tensor c5b, torch::Tensor c4b2,
    torch::Tensor q123c, torch::Tensor q123i,
    torch::Tensor q233c, torch::Tensor q233i,
    torch::Tensor q134c, torch::Tensor q134i,
    int64_t l_max, int64_t q222, int64_t q1111, int64_t q112,
    int64_t q123f, int64_t q233f, int64_t q134f) {
  check_inputs(moments, c3b, c4b, c5b, c4b2, q123c, q123i,
               q233c, q233i, q134c, q134i, l_max);
  c10::cuda::CUDAGuard guard(moments.device());
  const int blocks_out = output_blocks(
      l_max, q222, q1111, q112, q123f, q233f, q134f);
  auto output = torch::empty(
      {moments.size(0), blocks_out * moments.size(1)}, moments.options());
  const int threads = 128;
  const int64_t work = moments.size(0) * moments.size(1);
  const int blocks = static_cast<int>((work + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(moments.scalar_type(), "struct_center_forward", [&] {
      TermSet<scalar_t> t123{q123c.data_ptr<scalar_t>(), q123i.data_ptr<int64_t>(),
                             static_cast<int>(q123c.numel())};
      TermSet<scalar_t> t233{q233c.data_ptr<scalar_t>(), q233i.data_ptr<int64_t>(),
                             static_cast<int>(q233c.numel())};
      TermSet<scalar_t> t134{q134c.data_ptr<scalar_t>(), q134i.data_ptr<int64_t>(),
                             static_cast<int>(q134c.numel())};
      forward_kernel<scalar_t><<<blocks, threads, 0,
          at::cuda::getCurrentCUDAStream()>>>(
          moments.data_ptr<scalar_t>(), moments.size(0), moments.size(1),
          moments.size(2), c3b.data_ptr<scalar_t>(), c4b.data_ptr<scalar_t>(),
          c5b.data_ptr<scalar_t>(), c4b2.data_ptr<scalar_t>(), t123, t233, t134,
          l_max, q222, q1111, q112, q123f, q233f, q134f, blocks_out,
          output.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

torch::Tensor structural_center_backward(
    torch::Tensor grad_output, torch::Tensor moments,
    torch::Tensor c3b, torch::Tensor c4b, torch::Tensor c5b,
    torch::Tensor c4b2, torch::Tensor q123c, torch::Tensor q123i,
    torch::Tensor q233c, torch::Tensor q233i,
    torch::Tensor q134c, torch::Tensor q134i,
    int64_t l_max, int64_t q222, int64_t q1111, int64_t q112,
    int64_t q123f, int64_t q233f, int64_t q134f) {
  check_inputs(moments, c3b, c4b, c5b, c4b2, q123c, q123i,
               q233c, q233i, q134c, q134i, l_max);
  const int blocks_out = output_blocks(
      l_max, q222, q1111, q112, q123f, q233f, q134f);
  check_float_tensor(grad_output, moments, "grad_output");
  TORCH_CHECK(grad_output.dim() == 2 && grad_output.size(0) == moments.size(0) &&
              grad_output.size(1) == blocks_out * moments.size(1),
              "grad_output shape mismatch");
  c10::cuda::CUDAGuard guard(moments.device());
  auto grad_moments = torch::empty_like(moments);
  const int threads = 128;
  const int64_t work = moments.size(0) * moments.size(1);
  const int blocks = static_cast<int>((work + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(moments.scalar_type(), "struct_center_backward", [&] {
      TermSet<scalar_t> t123{q123c.data_ptr<scalar_t>(), q123i.data_ptr<int64_t>(),
                             static_cast<int>(q123c.numel())};
      TermSet<scalar_t> t233{q233c.data_ptr<scalar_t>(), q233i.data_ptr<int64_t>(),
                             static_cast<int>(q233c.numel())};
      TermSet<scalar_t> t134{q134c.data_ptr<scalar_t>(), q134i.data_ptr<int64_t>(),
                             static_cast<int>(q134c.numel())};
      backward_kernel<scalar_t><<<blocks, threads, 0,
          at::cuda::getCurrentCUDAStream()>>>(
          grad_output.data_ptr<scalar_t>(), moments.data_ptr<scalar_t>(),
          moments.size(0), moments.size(1), moments.size(2),
          c3b.data_ptr<scalar_t>(), c4b.data_ptr<scalar_t>(),
          c5b.data_ptr<scalar_t>(), c4b2.data_ptr<scalar_t>(), t123, t233, t134,
          l_max, q222, q1111, q112, q123f, q233f, q134f, blocks_out,
          grad_moments.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return grad_moments;
}

std::vector<torch::Tensor> structural_center_double_backward(
    torch::Tensor grad_output, torch::Tensor moments,
    torch::Tensor c3b, torch::Tensor c4b, torch::Tensor c5b,
    torch::Tensor c4b2, torch::Tensor q123c, torch::Tensor q123i,
    torch::Tensor q233c, torch::Tensor q233i,
    torch::Tensor q134c, torch::Tensor q134i, torch::Tensor h_moments,
    int64_t l_max, int64_t q222, int64_t q1111, int64_t q112,
    int64_t q123f, int64_t q233f, int64_t q134f) {
  check_inputs(moments, c3b, c4b, c5b, c4b2, q123c, q123i,
               q233c, q233i, q134c, q134i, l_max);
  const int blocks_out = output_blocks(
      l_max, q222, q1111, q112, q123f, q233f, q134f);
  check_float_tensor(grad_output, moments, "grad_output");
  check_float_tensor(h_moments, moments, "h_moments");
  TORCH_CHECK(grad_output.dim() == 2 && grad_output.size(0) == moments.size(0) &&
              grad_output.size(1) == blocks_out * moments.size(1) &&
              h_moments.sizes() == moments.sizes(), "double backward shape mismatch");
  c10::cuda::CUDAGuard guard(moments.device());
  auto grad2_output = torch::empty_like(grad_output);
  auto grad2_moments = torch::empty_like(moments);
  const int threads = 128;
  const int64_t work = moments.size(0) * moments.size(1);
  const int blocks = static_cast<int>((work + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(
        moments.scalar_type(), "struct_center_double_backward", [&] {
      TermSet<scalar_t> t123{q123c.data_ptr<scalar_t>(), q123i.data_ptr<int64_t>(),
                             static_cast<int>(q123c.numel())};
      TermSet<scalar_t> t233{q233c.data_ptr<scalar_t>(), q233i.data_ptr<int64_t>(),
                             static_cast<int>(q233c.numel())};
      TermSet<scalar_t> t134{q134c.data_ptr<scalar_t>(), q134i.data_ptr<int64_t>(),
                             static_cast<int>(q134c.numel())};
      double_backward_kernel<scalar_t><<<blocks, threads, 0,
          at::cuda::getCurrentCUDAStream()>>>(
          grad_output.data_ptr<scalar_t>(), moments.data_ptr<scalar_t>(),
          h_moments.data_ptr<scalar_t>(), moments.size(0), moments.size(1),
          moments.size(2), c3b.data_ptr<scalar_t>(), c4b.data_ptr<scalar_t>(),
          c5b.data_ptr<scalar_t>(), c4b2.data_ptr<scalar_t>(), t123, t233, t134,
          l_max, q222, q1111, q112, q123f, q233f, q134f, blocks_out,
          grad2_output.data_ptr<scalar_t>(), grad2_moments.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad2_output, grad2_moments};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &structural_center_forward,
             "generic structural center forward (CUDA)");
  module.def("backward", &structural_center_backward,
             "generic structural center VJP (CUDA)");
  module.def("double_backward", &structural_center_double_backward,
             "generic structural center directional HVP (CUDA)");
}
