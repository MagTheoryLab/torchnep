"""Checkpoint metadata for the supported Spin NEP Lite protocol."""

from typing import List

import torch


def spin_checkpoint_lines(model) -> List[str]:
    """Return the counted ``spin_mode`` metadata block for ``nep.txt``."""
    baseline = " ".join(
        f"{value:.16e}"
        for value in model.energy_baseline.detach().cpu().numpy()
    )
    lines = [
        f"spin_baseline {baseline}",
        f"spin_n_max {model.spin_n_max[0]} {model.spin_n_max[1]}",
        f"spin_basis_size {model.spin_basis_size[0]} "
        f"{model.spin_basis_size[1]}",
        f"spin_l_max {model.spin_l_max[0]} {model.spin_l_max[1]} "
        f"{model.spin_l_max[2]}",
        f"spin_compress {model.spin_compress}",
        f"spin_cutoff {model.spin_cutoff[0]} {model.spin_cutoff[1]}",
        f"spin_chiral {model.spin_chiral}",
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
