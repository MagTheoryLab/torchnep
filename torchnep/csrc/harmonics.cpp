#include <torch/extension.h>

torch::Tensor harmonics_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
torch::Tensor harmonics_backward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> harmonics_double_backward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> harmonics_jacobian(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &harmonics_forward);
  m.def("backward", &harmonics_backward);
  m.def("double_backward", &harmonics_double_backward);
  m.def("jacobian", &harmonics_jacobian);
}
