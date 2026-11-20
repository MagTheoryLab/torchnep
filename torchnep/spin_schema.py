"""The sole magnetic protocol: GPUMD c516 model nep_spin / spin_version 1."""

import math
from pathlib import Path

import numpy as np
import torch


def normalize_spin_cutoff(values, num_types):
    cutoffs = [float(value) for value in values]
    if len(cutoffs) == 1:
        cutoffs *= num_types
    if len(cutoffs) != num_types:
        raise ValueError(f"spin_cutoff requires one value or one value per element type ({num_types})")
    if not cutoffs or any(not math.isfinite(value) or value < 3 for value in cutoffs):
        raise ValueError("spin_cutoff values must be finite and >= 3 A")
    return cutoffs


def normalize_spin_wall(values, num_types, active_mask, name):
    """Normalize spin_wall_high / spin_wall_low into per-type (s, k) lists."""
    if values is None:
        return [0.0] * num_types, [0.0] * num_types
    nums = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in nums):
        raise ValueError(f"{name} values must be finite and non-negative")
    if len(nums) == 2:
        s_val, k_val = nums
        if k_val == 0.0:
            return [0.0] * num_types, [0.0] * num_types
        if k_val > 0.0 and s_val <= 0.0:
            raise ValueError(f"{name} requires positive threshold when stiffness > 0")
        s_list = [s_val if active else 0.0 for active in active_mask]
        k_list = [k_val if active else 0.0 for active in active_mask]
        return s_list, k_list
    if len(nums) == 2 * num_types:
        s_list = nums[0::2]
        k_list = nums[1::2]
        for idx, (s_val, k_val, active) in enumerate(zip(s_list, k_list, active_mask)):
            if k_val > 0.0:
                if not active:
                    raise ValueError(f"{name} stiffness must be 0 for inactive spin type index {idx}")
                if s_val <= 0.0:
                    raise ValueError(f"{name} requires positive threshold when stiffness > 0")
            else:
                s_list[idx] = 0.0
        return s_list, k_list
    raise ValueError(
        f"{name} requires 2 values (<s_bound> <k>) or {2 * num_types} values "
        f"(one <s_bound> <k> pair per element type)"
    )


def compute_spin_wall(spins, atom_types, hi_s, hi_k, lo_s, lo_k):
    """C^2 smooth one-sided longitudinal spin walls: high (s - s_hi)^3 and
    dimensionless even-function low (1 - s^2 / s_lo^2)^3.

    Returns (e_wall, mforce_wall) where mforce_wall = -d(e_wall)/d(spins).
    Purely longitudinal (zero spin torque), zero atomic force and virial,
    and smooth/finite everywhere including spins == 0.
    """
    dtype, device = spins.dtype, spins.device
    s_hi = hi_s[atom_types].to(dtype=dtype, device=device)
    k_hi = hi_k[atom_types].to(dtype=dtype, device=device)
    s_lo = lo_s[atom_types].to(dtype=dtype, device=device)
    k_lo = lo_k[atom_types].to(dtype=dtype, device=device)

    # Rescale before squaring, and keep sqrt away from zero on the inactive branch.
    scale = spins.detach().abs().amax(dim=-1)
    nonzero = scale > 0.0
    scaled = spins / torch.where(nonzero, scale, 1.0).unsqueeze(-1)
    root = torch.sqrt(torch.where(nonzero, scaled.square().sum(dim=-1), 1.0))
    norm = scale * root

    high = (k_hi > 0.0) & (s_hi > 0.0)
    excess = torch.where(high, torch.relu(norm - s_hi), 0.0)
    f_hi = (k_hi * excess) * excess
    e_hi = f_hi * (excess / 3.0)
    mf_hi = -f_hi.unsqueeze(-1) * (scaled / root.unsqueeze(-1))

    # Only scale the low-side inputs inside the wall, where |S / S_lo| <= 1.
    low = (k_lo > 0.0) & (s_lo > 0.0) & (norm < s_lo)
    safe_lo = torch.where(low, s_lo, 1.0)
    scaled_lo = torch.where(low.unsqueeze(-1), spins, 0.0) / safe_lo.unsqueeze(-1)
    deficit = torch.where(low, torch.relu(1.0 - scaled_lo.square().sum(dim=-1)), 0.0)
    f_lo = k_lo * deficit.square()
    e_lo = (f_lo * (safe_lo / 3.0)) * deficit
    mf_lo = 2.0 * (f_lo.unsqueeze(-1) * scaled_lo)

    return e_hi + e_lo, mf_hi + mf_lo


def normalize_spin_config(config):
    config = dict(config)
    config.setdefault("_explicit", set(config))
    retired = {"spin_descriptor", "spin_dof_type", "spin_env_type", "spin_curriculum"} & config.keys()
    if retired or config.get("spin_mode", 0) not in (0, 1):
        raise ValueError("unsupported spin configuration; use model nep_spin, spin_version 1, spin_type")
    kind = config.get("model", "nep")
    if kind not in ("nep", "nep_spin"):
        raise ValueError("model must be nep or nep_spin")
    config["model"] = kind
    config["spin_mode"] = int(kind == "nep_spin")
    spin_keys = {
        "spin_version", "spin_basis_size", "spin_compress", "spin_l_max",
        "spin_order", "spin_soc", "spin_cutoff", "spin_type",
        "spin_wall_high", "spin_wall_low",
    }
    explicit = config.get("_explicit", set(config))
    if not config["spin_mode"]:
        if set(explicit) & spin_keys:
            raise ValueError("spin parameters require model nep_spin")
        config["spin_cutoff"] = [config["cutoff_radial"], config["cutoff_angular"]]
        config["spin_cutoff_by_type"] = [config["cutoff_radial"]]*config["num_types"]
        zeros = [0.0] * config["num_types"]
        config["spin_wall_high_s"] = list(zeros)
        config["spin_wall_high_k"] = list(zeros)
        config["spin_wall_low_s"] = list(zeros)
        config["spin_wall_low_k"] = list(zeros)
        return config
    if config.setdefault("spin_version", 1) != 1:
        raise ValueError("only spin_version 1 is supported")
    for key, default in (("spin_compress", 3), ("spin_l_max", [2, 0, 0]), ("spin_order", 2), ("spin_soc", 1)):
        config.setdefault(key, default)
    lmax = config["spin_l_max"]
    if isinstance(lmax, int):
        lmax = [lmax, 0, 0]
    if len(lmax) != 3 or lmax[1:] != [0, 0] or not 0 <= lmax[0] <= 4:
        raise ValueError("spin_l_max must be in [0, 4]")
    config["spin_l_max"] = lmax
    for key, lo, hi in (("spin_compress", 1, 9), ("spin_order", 1, 4), ("spin_soc", 0, 1)):
        value = config[key]
        if not isinstance(value, int) or not lo <= value <= hi:
            raise ValueError(f"{key} must be an integer in [{lo}, {hi}]")
    basis = config.get("spin_basis_size", config["basis_size_angular"])
    if isinstance(basis, (list, tuple)):
        if len(basis) != 2 or basis[1] != 0:
            raise ValueError("spin_basis_size must equal angular basis_size")
        basis = basis[0]
    if basis != config["basis_size_angular"]:
        raise ValueError("spin_basis_size must equal angular basis_size")
    config["spin_basis_size"] = [basis, 0]
    cutoffs = config.get("spin_cutoff_by_type", config.get("spin_cutoff", [config["cutoff_radial"]]))
    config["spin_cutoff_by_type"] = normalize_spin_cutoff(cutoffs, config["num_types"])
    if max(config["spin_cutoff_by_type"]) > config["cutoff_radial"]:
        raise ValueError("spin_cutoff must not exceed the structural radial cutoff")
    config["spin_cutoff"] = [max(config["spin_cutoff_by_type"])] * 2
    names = config.get("spin_type")
    if names is not None and (not names or len(set(names)) != len(names) or set(names) - set(config["type_names"])):
        raise ValueError("spin_type requires distinct declared element names")
    active_set = set(config["type_names"] if names is None else names)
    active_mask = [name in active_set for name in config["type_names"]]
    hi_s, hi_k = normalize_spin_wall(
        config.get("spin_wall_high"), config["num_types"], active_mask, "spin_wall_high"
    )
    lo_s, lo_k = normalize_spin_wall(
        config.get("spin_wall_low"), config["num_types"], active_mask, "spin_wall_low"
    )
    for idx, (sl, kl, sh, kh) in enumerate(zip(lo_s, lo_k, hi_s, hi_k)):
        if kl > 0.0 and kh > 0.0 and sl >= sh:
            raise ValueError(
                f"spin_wall_low threshold ({sl}) must be strictly less than "
                f"spin_wall_high threshold ({sh}) for element index {idx}"
            )
    config["spin_wall_high_s"] = hi_s
    config["spin_wall_high_k"] = hi_k
    config["spin_wall_low_s"] = lo_s
    config["spin_wall_low_k"] = lo_k
    if config["l_max"][0] < 1:
        raise ValueError("nep_spin requires structural angular descriptors")
    return config


def slim_type_config(config, keep):
    """Slice every per-element spin setting in the same order as the types."""
    indices = [config["type_names"].index(name) for name in keep]
    result = dict(config, type_names=keep, num_types=len(keep),
                  spin_cutoff_by_type=[config["spin_cutoff_by_type"][i] for i in indices])
    active = config.get("spin_type")
    if active is not None:
        result["spin_type"] = [name for name in active if name in keep]
        if config["spin_mode"] and not result["spin_type"]:
            raise ValueError("slim_types removed every active spin_type")
    for key in ("spin_wall_high", "spin_wall_low"):
        if key in config:
            result[key] = [config[f"{key}_{field}"][i] for i in indices for field in ("s", "k")]
    return normalize_spin_config(result)


def resolve_gpumd_qscaler(spin_mode, requested):
    if spin_mode and requested:
        raise ValueError("nep_spin requires self-consistent scaling; use_gpumd_qscaler=True is unsupported")
    return not spin_mode if requested is None else bool(requested)


def spin_projection_parameters(model):
    if not model.spin_mode:
        return ()
    descriptor = model.spin_descriptor
    return descriptor.radial_mix, descriptor.weights, descriptor.center_mix


def spin_projection_size(model):
    return sum(parameter.numel() for parameter in spin_projection_parameters(model))


def read_nep_model(path):
    """Parse named blocks before touching any model weights; reject unknown fields."""
    lines = [line.split('#', 1)[0].strip() for line in Path(path).read_text().splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError("empty NEP model")
    header = lines[0].split()
    if header[0] != "nep" or len(header) < 3 or len(header) != int(header[1]) + 2 or int(header[1]) < 1 or len(set(header[2:])) != int(header[1]):
        raise ValueError("expected nep <type count> <elements> header")
    fields, blocks = {}, {}
    i = 1
    while i < len(lines):
        parts = lines[i].split()
        if parts[0] == '@':
            if len(parts) != 3 or parts[1] in blocks:
                raise ValueError("malformed or duplicate model block")
            name, count = parts[1], int(parts[2])
            if count < 0 or i+1+count > len(lines):
                raise ValueError("truncated model block")
            try:
                values = np.asarray([float(value) for value in lines[i+1:i+1+count]], dtype=np.float64)
            except ValueError as error:
                raise ValueError("malformed numeric model block") from error
            if not np.isfinite(values).all():
                raise ValueError("model blocks must be finite")
            blocks[name] = values
            i += count + 1
        else:
            key = parts[0].lower()
            if key in fields:
                raise ValueError(f"duplicate model metadata: {key}")
            fields[key] = parts[1:]
            i += 1
    required = {"format_version", "model", "cutoff", "n_max", "basis_size", "l_max", "ann"}
    spin = fields.get("model") == ["nep_spin"]
    if spin:
        required |= {"spin_version", "spin_basis_size", "spin_l_max", "spin_compress", "spin_cutoff", "spin_order", "spin_soc", "spin_scaler", "energy_reference"}
    optional = {"zbl", "energy_reference"} | ({"spin_type", "spin_wall_high", "spin_wall_low"} if spin else set())
    if required - fields.keys() or fields.keys() - required - optional:
        raise ValueError("missing or unknown model metadata: " + str(sorted((required-fields.keys()) | (fields.keys()-required-optional))))
    if fields["format_version"] != ['1'] or fields["model"] not in (['nep'], ['nep_spin']):
        raise ValueError("requires format_version 1 and model nep or nep_spin")
    if spin and (fields["spin_version"] != ['1'] or fields["spin_scaler"] != ['1']):
        raise ValueError("requires spin_version 1, spin_scaler 1")
    widths = {"cutoff": (4,), "n_max": (2,), "basis_size": (2,), "l_max": tuple(range(3,8)), "ann": (2,), }
    for key in ("spin_basis_size", "spin_l_max", "spin_compress", "spin_order", "spin_soc"):
        if spin:
            widths[key] = (1,)
    if "energy_reference" in fields:
        widths["energy_reference"] = (int(header[1]),)
    for key in ("spin_wall_high", "spin_wall_low"):
        if key in fields:
            widths[key] = (2 * int(header[1]),)
    if any(len(fields[key]) not in sizes for key, sizes in widths.items()):
        raise ValueError("malformed model metadata width")
    if blocks.keys() != {"parameters", "q_scaler"} or (blocks['q_scaler'] <= 0).any():
        raise ValueError("requires parameters and positive q_scaler blocks")
    config = dict(model=fields['model'][0], num_types=int(header[1]), type_names=header[2:],
                  cutoff_radial=float(fields['cutoff'][0]), cutoff_angular=float(fields['cutoff'][1]),
                  n_max_radial=int(fields['n_max'][0]), n_max_angular=int(fields['n_max'][1]),
                  basis_size_radial=int(fields['basis_size'][0]), basis_size_angular=int(fields['basis_size'][1]),
                  l_max=list(map(int, fields['l_max'])), neuron=int(fields['ann'][0]))
    if int(fields['ann'][1]) != 0:
        raise ValueError("nep_spin requires one hidden ANN layer")
    for key in ("spin_basis_size", "spin_l_max", "spin_compress", "spin_order", "spin_soc"):
        if spin:
            config[key] = int(fields[key][0])
    if spin:
        config['spin_version'] = 1
        config['spin_cutoff'] = list(map(float, fields['spin_cutoff']))
    if 'spin_type' in fields:
        config['spin_type'] = fields['spin_type']
    for key in ('spin_wall_high', 'spin_wall_low'):
        if key in fields:
            config[key] = list(map(float, fields[key]))
    baseline = np.asarray(fields.get('energy_reference', [0]*int(header[1])), dtype=np.float64)
    if not np.isfinite(baseline).all():
        raise ValueError("energy_reference must be finite")
    if 'zbl' in fields:
        zbl = list(map(float, fields['zbl']))
        if len(zbl) not in (2,3) or not np.isfinite(zbl).all() or zbl[0] <= 0 or zbl[1] != 2*zbl[0]:
            raise ValueError("TorchNEP ZBL requires outer = 2 * inner > 0")
        config['zbl'] = zbl[1]
        if len(zbl) == 3:
            config['typewise_cutoff_zbl_factor'] = zbl[2]
    return normalize_spin_config(config), blocks, baseline


def load_nep_weights(model, path):
    config, blocks, baseline = read_nep_model(path)
    expected = {
        'num_types': model.num_types, 'type_names': model.type_names,
        'model': 'nep_spin' if model.spin_mode else 'nep',
        'cutoff_radial': model.rc_radial, 'cutoff_angular': model.rc_angular,
        'n_max_radial': model.n_max_radial, 'n_max_angular': model.n_max_angular,
        'basis_size_radial': model.basis_size_radial, 'basis_size_angular': model.basis_size_angular,
        'neuron': model.num_neurons,
        'zbl': model.zbl, 'typewise_cutoff_zbl_factor': model.zbl_typewise_factor,
    }
    if model.spin_mode:
        expected.update(spin_version=1, spin_compress=model.spin_compress,
                        spin_l_max=model.spin_l_max, spin_order=model.spin_order,
                        spin_soc=model.spin_soc, spin_cutoff_by_type=model.spin_cutoff_by_type,
                        spin_wall_high_s=model.spin_wall_high_s, spin_wall_high_k=model.spin_wall_high_k,
                        spin_wall_low_s=model.spin_wall_low_s, spin_wall_low_k=model.spin_wall_low_k)
        file_types = config.get('spin_type', model.type_names)
        expected_types = [name for i, name in enumerate(model.type_names) if model.spin_dof_type_active[i]]
        if set(file_types) != set(expected_types):
            raise ValueError('spin_type does not match target config')
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'{key.removesuffix("_by_type")} does not match target config')
    lmax = (config['l_max'] + [0]*7)[:7]
    lmax = [lmax[0]] + [int(value > 0) for value in lmax[1:]]
    if lmax != [model.l_max_3b, model.has_q_222, model.has_q_1111, model.has_q_112, model.has_q_123, model.has_q_233, model.has_q_134]:
        raise ValueError('l_max does not match target config')
    tensors = []
    for net in model.fitting_nets:
        tensors.extend((net.w0.T, net.b0, net.w1))
    tensors.extend((model.b1, model.c_param_2))
    if model.c_param_3 is not None:
        tensors.append(model.c_param_3)
    tensors.extend(spin_projection_parameters(model))
    values = blocks['parameters']
    if values.size != sum(t.numel() for t in tensors) or blocks['q_scaler'].size != model.dim:
        raise ValueError('incorrect parameters or q_scaler block count')
    # Validate all metadata and counts before mutating the target.
    with torch.no_grad():
        offset = 0
        for tensor in tensors:
            size = tensor.numel()
            tensor.copy_(torch.as_tensor(values[offset:offset+size].copy(), dtype=tensor.dtype, device=tensor.device).reshape(tensor.shape))
            offset += size
        model.q_scaler.copy_(torch.as_tensor(blocks['q_scaler'], dtype=model.q_scaler.dtype, device=model.q_scaler.device))
        model.energy_baseline.copy_(torch.as_tensor(baseline, dtype=model.energy_baseline.dtype, device=model.energy_baseline.device))


def save_nep_model(model, path, max_radial, max_angular):
    if max_radial <= 0 or max_angular <= 0:
        raise ValueError('save_nep_txt requires max_NN_radial > 0 and max_NN_angular > 0')
    lines = [f"nep {model.num_types} " + ' '.join(model.type_names), 'format_version 1',
             'model nep_spin' if model.spin_mode else 'model nep',
             f'cutoff {model.rc_radial:.17g} {model.rc_angular:.17g} {max_radial} {max_angular}',
             f'n_max {model.n_max_radial} {model.n_max_angular}',
             f'basis_size {model.basis_size_radial} {model.basis_size_angular}']
    lmax = [model.l_max_3b, 2 if model.has_q_222 else 0, model.has_q_1111,
            model.has_q_112, model.has_q_123, model.has_q_233, model.has_q_134]
    lines.extend(['l_max ' + ' '.join(map(str, lmax)), f'ann {model.num_neurons} 0',
                  'energy_reference ' + ' '.join(f'{value:.17g}' for value in model.energy_baseline.tolist())])
    if model.zbl is not None:
        line = f'zbl {model.zbl/2:.17g} {model.zbl:.17g}'
        if model.zbl_typewise_factor is not None:
            line += f' {model.zbl_typewise_factor:.17g}'
        lines.append(line)
    if model.spin_mode:
        lines.extend(['spin_version 1', f'spin_basis_size {model.spin_basis_size[0]}',
                      f'spin_l_max {model.spin_l_max[0]}', f'spin_compress {model.spin_compress}',
                      f'spin_order {model.spin_order}', f'spin_soc {model.spin_soc}', 'spin_scaler 1',
                      'spin_cutoff ' + ' '.join(map(str, model.spin_cutoff_by_type)),
                      'spin_type ' + ' '.join(name for i, name in enumerate(model.type_names) if model.spin_dof_type_active[i])])
        if model.has_spin_wall_high:
            pairs = [v for pair in zip(model.spin_wall_high_s, model.spin_wall_high_k) for v in pair]
            lines.append('spin_wall_high ' + ' '.join(f'{v:.17g}' for v in pairs))
        if model.has_spin_wall_low:
            pairs = [v for pair in zip(model.spin_wall_low_s, model.spin_wall_low_k) for v in pair]
            lines.append('spin_wall_low ' + ' '.join(f'{v:.17g}' for v in pairs))
    tensors = []
    for net in model.fitting_nets:
        tensors.extend((net.w0.T, net.b0, net.w1))
    tensors.extend((model.b1, model.c_param_2))
    if model.c_param_3 is not None:
        tensors.append(model.c_param_3)
    tensors.extend(spin_projection_parameters(model))
    lines.append('@ parameters ' + str(sum(t.numel() for t in tensors)))
    for tensor in tensors:
        lines.extend(f'{value:.17g}' for value in tensor.detach().cpu().reshape(-1).tolist())
    lines.append(f'@ q_scaler {model.dim}')
    lines.extend(f'{value:.17g}' for value in model.q_scaler.tolist())
    Path(path).write_text('\n'.join(lines) + '\n')
