# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# TorchNEP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with TorchNEP.  If not, see <http://www.gnu.org/licenses/>.

"""Checkpoint metadata and parameter layout for the O/C spin protocol."""

from typing import List

import torch


def make_spin_descriptor(spin_mode: int, **kwargs):
    """Construct the descriptor implementation selected by the file format."""
    if spin_mode == 2:
        from .magnetic_polynomial import MagneticPolynomialDescriptor
        return MagneticPolynomialDescriptor(**kwargs)
    if spin_mode == 3:
        from .magnetic_polynomial_v3 import MagneticPolynomialDescriptorV3
        return MagneticPolynomialDescriptorV3(**kwargs)
    raise ValueError(f"spin descriptor requires spin_mode 2 or 3, got {spin_mode}")


def resolve_gpumd_qscaler(spin_mode: int, requested):
    """Use self-consistent scaling for the trainable O/C radial frame."""
    if requested is None:
        return spin_mode not in (2, 3)
    enabled = bool(requested)
    if spin_mode in (2, 3) and enabled:
        raise ValueError(
            f"spin_mode {spin_mode} does not support "
            "use_gpumd_qscaler=True: "
            "the all-one radial coefficients collapse axial descriptor "
            "channels and make q_scaler singular; use the self-consistent "
            "scaler (False)"
        )
    return enabled


def spin_checkpoint_lines(model) -> List[str]:
    """Return the counted ``spin_mode`` metadata block for ``nep.txt``."""
    baseline = " ".join(
        f"{value:.16e}"
        for value in model.energy_baseline.detach().cpu().numpy()
    )
    if model.spin_mode not in (2, 3):
        raise ValueError("spin checkpoint metadata requires spin_mode 2 or 3")
    lines = [
        f"spin_baseline {baseline}",
        f"spin_basis_size {model.spin_basis_size[0]}",
        f"spin_l_max {model.spin_l_max[0]}",
        f"spin_compress {model.spin_compress}",
        f"spin_cutoff {model.spin_cutoff[0]}",
        f"spin_order {model.spin_order}",
        f"spin_soc {model.spin_soc}",
        f"spin_projection_size {spin_projection_size(model)}",
        "spin_scaler 1",
    ]
    if (model.is_spin_dof_type_set
            or not bool(model.spin_dof_type_active.all().item())):
        names = [
            model.type_names[index]
            for index in range(model.num_types)
            if bool(model.spin_dof_type_active[index].item())
        ]
        lines.append("spin_dof_type " + " ".join(names))
    if (model.is_spin_env_type_set
            or not torch.equal(
                model.spin_env_type_active, model.spin_dof_type_active)):
        names = [
            model.type_names[index]
            for index in range(model.num_types)
            if bool(model.spin_env_type_active[index].item())
        ]
        lines.append("spin_env_type " + " ".join(names))
    return lines


def spin_projection_parameters(model):
    """Return O/C descriptor tensors in stable text-format order."""
    if model.spin_mode not in (2, 3):
        return ()
    return (model.spin_descriptor.radial_leg_mix,)


def spin_projection_size(model) -> int:
    return sum(
        parameter.numel()
        for parameter in spin_projection_parameters(model)
    )
