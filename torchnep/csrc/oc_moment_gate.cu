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

template <typename scalar_t>
__global__ void moment_gate_double_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ weights,
    const int64_t* __restrict__ pair_i,
    const int64_t* __restrict__ pair_j,
    const scalar_t* __restrict__ spins,
    const scalar_t* __restrict__ grad_grad_weights,
    const scalar_t* __restrict__ grad_grad_spins,
    int64_t edges,
    scalar_t* __restrict__ grad2_output,
    scalar_t* __restrict__ grad2_weights,
    scalar_t* __restrict__ grad2_spins) {
  const int64_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= edges) return;
  const int64_t i = pair_i[edge];
  const int64_t j = pair_j[edge];
  const scalar_t* si = spins + 3 * i;
  const scalar_t* sj = spins + 3 * j;
  const scalar_t* hi = grad_grad_spins + 3 * i;
  const scalar_t* hj = grad_grad_spins + 3 * j;
  const scalar_t si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  const scalar_t sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
  const scalar_t dot = si[0] * sj[0] + si[1] * sj[1] + si[2] * sj[2];
  const scalar_t norm = si2 + sj2;
  const scalar_t moment = dot * norm;

  scalar_t scale = scalar_t(0);
  scalar_t weight_cotangent = scalar_t(0);
  scalar_t beta = scalar_t(0);
  scalar_t gi[3];
  scalar_t gj[3];
  for (int d = 0; d < 3; ++d) {
    gi[d] = sj[d] * norm + scalar_t(2) * dot * si[d];
    gj[d] = si[d] * norm + scalar_t(2) * dot * sj[d];
    beta += hi[d] * gi[d] + hj[d] * gj[d];
  }
  for (int c = 0; c < 2; ++c) {
    const scalar_t go = grad_output[2 * i + c];
    const scalar_t weight = weights[2 * edge + c];
    const scalar_t uw = grad_grad_weights[2 * edge + c];
    scale += go * weight;
    weight_cotangent += uw * go;
    atomicAdd(grad2_output + 2 * i + c, uw * moment + weight * beta);
    grad2_weights[2 * edge + c] = go * beta;
  }

  // Directional derivative of grad(m) along (hi, hj), i.e. H(m) [hi,hj].
  const scalar_t delta_dot =
      hi[0] * sj[0] + hi[1] * sj[1] + hi[2] * sj[2] +
      si[0] * hj[0] + si[1] * hj[1] + si[2] * hj[2];
  const scalar_t delta_norm = scalar_t(2) * (
      si[0] * hi[0] + si[1] * hi[1] + si[2] * hi[2] +
      sj[0] * hj[0] + sj[1] * hj[1] + sj[2] * hj[2]);
  for (int d = 0; d < 3; ++d) {
    const scalar_t hess_i =
        hj[d] * norm + sj[d] * delta_norm +
        scalar_t(2) * delta_dot * si[d] + scalar_t(2) * dot * hi[d];
    const scalar_t hess_j =
        hi[d] * norm + si[d] * delta_norm +
        scalar_t(2) * delta_dot * sj[d] + scalar_t(2) * dot * hj[d];
    atomicAdd(
        grad2_spins + 3 * i + d,
        weight_cotangent * gi[d] + scale * hess_i);
    atomicAdd(
        grad2_spins + 3 * j + d,
        weight_cotangent * gj[d] + scale * hess_j);
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

std::vector<torch::Tensor> moment_gate_double_backward(
    torch::Tensor grad_output,
    torch::Tensor weights,
    torch::Tensor pair_i,
    torch::Tensor pair_j,
    torch::Tensor spins,
    torch::Tensor grad_grad_weights,
    torch::Tensor grad_grad_spins) {
  check_inputs(weights, pair_i, pair_j, spins);
  TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous(),
              "grad_output must be a contiguous CUDA tensor");
  TORCH_CHECK(grad_grad_weights.is_cuda() && grad_grad_weights.is_contiguous(),
              "grad_grad_weights must be a contiguous CUDA tensor");
  TORCH_CHECK(grad_grad_spins.is_cuda() && grad_grad_spins.is_contiguous(),
              "grad_grad_spins must be a contiguous CUDA tensor");
  TORCH_CHECK(grad_output.dim() == 2 &&
              grad_output.size(0) == spins.size(0) &&
              grad_output.size(1) == 2,
              "grad_output must have shape [atoms, 2]");
  TORCH_CHECK(grad_grad_weights.sizes() == weights.sizes(),
              "grad_grad_weights must match weights shape");
  TORCH_CHECK(grad_grad_spins.sizes() == spins.sizes(),
              "grad_grad_spins must match spins shape");
  TORCH_CHECK(grad_output.scalar_type() == weights.scalar_type() &&
              grad_grad_weights.scalar_type() == weights.scalar_type() &&
              grad_grad_spins.scalar_type() == weights.scalar_type(),
              "double-backward tensors must match weights dtype");
  c10::cuda::CUDAGuard guard(weights.device());
  auto grad2_output = torch::zeros_like(grad_output);
  auto grad2_weights = torch::empty_like(weights);
  auto grad2_spins = torch::zeros_like(spins);
  const int threads = 256;
  const int blocks = static_cast<int>((weights.size(0) + threads - 1) / threads);
  if (blocks) {
    AT_DISPATCH_FLOATING_TYPES(
        weights.scalar_type(), "moment_gate_double_backward", [&] {
      moment_gate_double_backward_kernel<scalar_t><<<
          blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          grad_output.data_ptr<scalar_t>(),
          weights.data_ptr<scalar_t>(),
          pair_i.data_ptr<int64_t>(),
          pair_j.data_ptr<int64_t>(),
          spins.data_ptr<scalar_t>(),
          grad_grad_weights.data_ptr<scalar_t>(),
          grad_grad_spins.data_ptr<scalar_t>(),
          weights.size(0),
          grad2_output.data_ptr<scalar_t>(),
          grad2_weights.data_ptr<scalar_t>(),
          grad2_spins.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad2_output, grad2_weights, grad2_spins};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &moment_gate_forward, "O3 moment-gate forward (CUDA)");
  module.def("backward", &moment_gate_backward, "O3 moment-gate VJP (CUDA)");
  module.def(
      "double_backward", &moment_gate_double_backward,
      "O3 moment-gate analytic double backward (CUDA)");
}
