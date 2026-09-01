#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void moment_gate_forward_kernel(
    const scalar_t* __restrict__ weights,
    const int64_t* __restrict__ pair_i,
    const int64_t* __restrict__ pair_j,
    const scalar_t* __restrict__ spins,
    int64_t edges,
    scalar_t* __restrict__ output) {
  const int64_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= edges) return;
  const int64_t i = pair_i[edge];
  const int64_t j = pair_j[edge];
  const scalar_t* si = spins + 3 * i;
  const scalar_t* sj = spins + 3 * j;
  const scalar_t dot = si[0] * sj[0] + si[1] * sj[1] + si[2] * sj[2];
  const scalar_t norm =
      si[0] * si[0] + si[1] * si[1] + si[2] * si[2] +
      sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
  const scalar_t moment = dot * norm;
  atomicAdd(output + 2 * i, weights[2 * edge] * moment);
  atomicAdd(output + 2 * i + 1, weights[2 * edge + 1] * moment);
}

template <typename scalar_t>
__global__ void moment_gate_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ weights,
    const int64_t* __restrict__ pair_i,
    const int64_t* __restrict__ pair_j,
    const scalar_t* __restrict__ spins,
    int64_t edges,
    scalar_t* __restrict__ grad_weights,
    scalar_t* __restrict__ grad_spins) {
  const int64_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= edges) return;
  const int64_t i = pair_i[edge];
  const int64_t j = pair_j[edge];
  const scalar_t* si = spins + 3 * i;
  const scalar_t* sj = spins + 3 * j;
  const scalar_t si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  const scalar_t sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
  const scalar_t dot = si[0] * sj[0] + si[1] * sj[1] + si[2] * sj[2];
  const scalar_t norm = si2 + sj2;
  const scalar_t moment = dot * norm;
  const scalar_t g0 = grad_output[2 * i];
  const scalar_t g1 = grad_output[2 * i + 1];
  grad_weights[2 * edge] = g0 * moment;
  grad_weights[2 * edge + 1] = g1 * moment;
  const scalar_t scale = g0 * weights[2 * edge] + g1 * weights[2 * edge + 1];
  for (int d = 0; d < 3; ++d) {
    const scalar_t gi = scale * (sj[d] * norm + scalar_t(2) * dot * si[d]);
    const scalar_t gj = scale * (si[d] * norm + scalar_t(2) * dot * sj[d]);
    atomicAdd(grad_spins + 3 * i + d, gi);
    atomicAdd(grad_spins + 3 * j + d, gj);
  }
}

void check_inputs(
    const torch::Tensor& weights,
    const torch::Tensor& pair_i,
    const torch::Tensor& pair_j,
    const torch::Tensor& spins) {
  TORCH_CHECK(weights.is_cuda() && pair_i.is_cuda() && pair_j.is_cuda() &&
              spins.is_cuda(), "all moment-gate tensors must be CUDA tensors");
  TORCH_CHECK(weights.is_contiguous() && pair_i.is_contiguous() &&
              pair_j.is_contiguous() && spins.is_contiguous(),
              "all moment-gate tensors must be contiguous");
  TORCH_CHECK(weights.dim() == 2 && weights.size(1) == 2,
              "weights must have shape [edges, 2]");
  TORCH_CHECK(spins.dim() == 2 && spins.size(1) == 3,
              "spins must have shape [atoms, 3]");
  TORCH_CHECK(pair_i.scalar_type() == torch::kInt64 &&
              pair_j.scalar_type() == torch::kInt64,
              "pair indices must be int64");
  TORCH_CHECK(weights.scalar_type() == spins.scalar_type(),
              "weights and spins must have the same dtype");
  TORCH_CHECK(pair_i.numel() == weights.size(0) &&
              pair_j.numel() == weights.size(0),
              "pair indices and weights must have the same edge count");
}

}  // namespace

torch::Tensor moment_gate_forward(
    torch::Tensor weights,
    torch::Tensor pair_i,
    torch::Tensor pair_j,
    torch::Tensor spins) {
  check_inputs(weights, pair_i, pair_j, spins);
  c10::cuda::CUDAGuard guard(weights.device());
  auto output = torch::zeros({spins.size(0), 2}, weights.options());
  const int threads = 256;
  const int blocks = static_cast<int>((weights.size(0) + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "moment_gate_forward", [&] {
      moment_gate_forward_kernel<scalar_t><<<
          blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          weights.data_ptr<scalar_t>(),
          pair_i.data_ptr<int64_t>(),
          pair_j.data_ptr<int64_t>(),
          spins.data_ptr<scalar_t>(),
          weights.size(0),
          output.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

std::vector<torch::Tensor> moment_gate_backward(
    torch::Tensor grad_output,
    torch::Tensor weights,
    torch::Tensor pair_i,
    torch::Tensor pair_j,
    torch::Tensor spins) {
  check_inputs(weights, pair_i, pair_j, spins);
  TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous(),
              "grad_output must be a contiguous CUDA tensor");
  TORCH_CHECK(grad_output.dim() == 2 &&
              grad_output.size(0) == spins.size(0) &&
              grad_output.size(1) == 2,
              "grad_output must have shape [atoms, 2]");
  TORCH_CHECK(grad_output.scalar_type() == weights.scalar_type(),
              "grad_output must match weights dtype");
  c10::cuda::CUDAGuard guard(weights.device());
  auto grad_weights = torch::empty_like(weights);
  auto grad_spins = torch::zeros_like(spins);
  const int threads = 256;
  const int blocks = static_cast<int>((weights.size(0) + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "moment_gate_backward", [&] {
      moment_gate_backward_kernel<scalar_t><<<
          blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          grad_output.data_ptr<scalar_t>(),
          weights.data_ptr<scalar_t>(),
          pair_i.data_ptr<int64_t>(),
          pair_j.data_ptr<int64_t>(),
          spins.data_ptr<scalar_t>(),
          weights.size(0),
          grad_weights.data_ptr<scalar_t>(),
          grad_spins.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad_weights, grad_spins};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &moment_gate_forward, "O3 moment-gate forward (CUDA)");
  module.def("backward", &moment_gate_backward, "O3 moment-gate VJP (CUDA)");
}
