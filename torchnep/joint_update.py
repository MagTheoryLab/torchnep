"""Joint residual-Jacobian preconditioning of an Adam/AdamW displacement.

The four sketch columns span *all* parameter blocks. This is not a rank-four
model or a rescaling of the structural gradient. No model/export changes.
"""
import math

import torch
import torch.distributed as dist

from .virial import virial6_for_labels as virial6_for_loss
from .response_training import response_from_store


def physical_residual(model, batch, weights, backend, counts=None):
    """Local residual; sum of squared norms over ranks equals E/F/V/MF/Tau loss.

    ``counts`` are global label counts for DDP, local counts otherwise. There
    is deliberately no DDP world-size factor: sketch vectors are SUM-reduced.
    Masked padding contributes zero, including ranks without any labels.
    """
    e, f, v, m, t = weights
    batch = dict(batch)
    batch['need_mforce'] = m > 0
    batch['need_tau'] = t > 0
    result = model.compute_properties_cached(
        batch, need_forces=f > 0, need_virial=v > 0,
        backend=backend, create_graph=True)
    active = model.spin_dof_type_active[batch['atom_types']]
    masks = [batch['energy_mask'], batch['force_mask'], batch['virial_mask'],
             batch['mforce_mask'] & active, batch['tau_mask'] & active]
    if counts is None:
        counts = torch.stack([mask.sum() for mask in masks])
    counts = counts.to(result['Etot']).clamp_min(1)
    parts = [result['Etot'].reshape(-1) * 0]  # differentiable on empty shards
    for index, (weight, mask, dof) in enumerate(zip(
            weights, masks, (1, 3, 6, 3, 2 if model.spin_mode == 3 else 3))):
        if weight <= 0:
            continue
        if index == 0:
            diff = (result['Etot'] - batch['energy']) / batch['natoms']
        elif index == 2:
            if 'virial' not in result or batch['virial'].shape[1] != 9:
                continue
            total = result['virial'].new_zeros((batch['num_structures'], 9))
            total.index_add_(0, batch['struct_idx'], result['virial'])
            diff = (virial6_for_loss(total, model.spin_mode)
                    - virial6_for_loss(batch['virial'], model.spin_mode))
            diff = diff / batch['natoms'][:, None]
        else:
            key = ('', 'forces', '', 'mforces', 'tau')[index]
            diff = result[key] - batch[key]
        parts.append((diff[mask] * (weight / (dof * counts[index])).sqrt()).reshape(-1))
    return torch.cat(parts)


class JointRank4:
    """Cached Woodbury correction, refreshed every 24 accepted optimizer steps.

    If M is Adam's bias-corrected diagonal denominator and V is a rank-four
    sketch of J.T, U=M**(-1/2)V. The proposed Adam displacement d becomes
    M**(-1/2) [I-U(U.T U+lambda I)**(-1)U.T] M**(1/2)d.
    lambda=lambda_max(U.T U), so eigenvalues in whitened coordinates lie in
    [1/2, 1]. Adam moments are updated normally; all parameter blocks remain
    active from the first update.
    """
    rank = 4
    interval = 24

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.params = [p for group in optimizer.param_groups for p in group['params']]
        self.steps = 0
        self.cache = None
        self.refreshes = 0
        self.last_norm_ratio = 1.0

    def state_dict(self):
        return dict(version=1, steps=self.steps, cache=self.cache,
                    refreshes=self.refreshes)

    def load_state_dict(self, state):
        if state['version'] != 1:
            raise ValueError('unsupported joint_rank4 checkpoint version')
        self.steps, self.cache = state['steps'], state['cache']
        self.refreshes = state['refreshes']
        # Preserve the sketch when resuming an old, fully open schedule.
        if self.cache is not None:
            token = self.cache['token']
            if isinstance(token, tuple) and len(token) == 4:
                if token[1] != 1.0:
                    self.cache = None
                else:
                    self.cache = dict(self.cache, token=(token[0], token[3]))

    @staticmethod
    def _flat(items):
        return torch.cat([x.reshape(-1) for x in items])

    @torch.no_grad()
    def _assign(self, values):
        offset = 0
        for p in self.params:
            p.copy_(values[offset:offset+p.numel()].view_as(p))
            offset += p.numel()

    def _refresh(self, residual_fn, token, step):
        with torch.enable_grad():
            residual = residual_fn()
            rank = dist.get_rank() if dist.is_initialized() else 0
            generator = torch.Generator(device=residual.device)
            generator.manual_seed(734101 + 1009 * step + rank)
            columns = []
            for j in range(self.rank):
                signs = torch.randint(0, 2, residual.shape, device=residual.device,
                                      generator=generator).to(residual.dtype) * 2 - 1
                grads = torch.autograd.grad(residual.dot(signs), self.params,
                                            allow_unused=True, retain_graph=j < self.rank-1)
                column = self._flat([g.detach() if g is not None else torch.zeros_like(p)
                                     for g, p in zip(grads, self.params)])
                if dist.is_initialized():
                    dist.all_reduce(column)
                columns.append(column / math.sqrt(self.rank))
        diagonals = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                if 'exp_avg_sq' not in state:
                    diagonals.append(torch.full_like(p, group['eps']))
                    continue
                second = state['max_exp_avg_sq'] if group['amsgrad'] else state['exp_avg_sq']
                diagonal = second.sqrt() / math.sqrt(1 - group['betas'][1] ** float(state['step']))
                diagonals.append(diagonal + group['eps'])
        metric = self._flat(diagonals).detach()
        metric = metric.clamp_min(metric.max() * 1e-10 + 1e-30)
        root = metric.sqrt()
        u = torch.stack(columns, dim=1) / root[:, None]
        # Only the small factorisation uses double in float32 training.
        gram = u.double().T @ u.double()
        damping = torch.linalg.eigvalsh(gram)[-1].clamp_min(1e-30)
        chol = torch.linalg.cholesky(gram + damping * torch.eye(
            self.rank, device=gram.device, dtype=gram.dtype))
        self.cache = dict(root=root, u=u, chol=chol, token=token, step=step)
        self.refreshes += 1

    def step(self, residual_fn, token):
        """Called after finite-gradient checks and clipping.

        ``token`` encodes objective weights and response contract;
        a change invalidates the cache immediately. VJPs do not alter .grad or
        the training RNG. Cached tensors and step counters are checkpointed.
        """
        before = self._flat([p.detach().clone() for p in self.params])
        self.optimizer.step()
        proposal = self._flat([p.detach() for p in self.params]) - before
        step = self.steps + 1
        if (self.cache is None or step - self.cache['step'] >= self.interval
                or token != self.cache['token']):
            self._assign(before)
            self._refresh(residual_fn, token, step)
        root, u, chol = (self.cache[k] for k in ('root', 'u', 'chol'))
        y = root * proposal
        coef = torch.cholesky_solve((u.double().T @ y.double())[:, None], chol).squeeze(1)
        corrected = y - (u.double() @ coef).to(y.dtype)
        if not torch.isfinite(corrected).all():
            self._assign(before)
            raise FloatingPointError('non-finite joint_rank4 correction; checkpoint not advanced')
        self.last_norm_ratio = float(corrected.norm() / y.norm().clamp_min(1e-30))
        self._assign(before + corrected / root)
        self.steps = step


def configure_joint_update(optimizer, config):
    optimizer.joint_rank4 = JointRank4(optimizer) if config.get('joint_rank4', False) else None


def optimizer_step(optimizer, model, batch, weights, backend, schedule, counts=None,
                   responses=()):
    update = getattr(optimizer, 'joint_rank4', None)
    if update is None:
        optimizer.step()
        return
    response_contract = tuple(
        (getattr(store, "response_target", "mforce"),
         getattr(store, "response_energy_scale", 1e-6), float(weight))
        for _, store, weight in responses)
    token = (tuple(weights), response_contract)
    def residual():
        parts = [physical_residual(model, batch, weights, backend, counts)]
        world = dist.get_world_size() if dist.is_initialized() else 1
        # Replicated response branch: independent rank sketches, 1/sqrt(world)
        # normalisation prevents multiplying its Gram contribution by world.
        for response_batch, store, weight in responses:
            parts.append(response_from_store(
                model, response_batch, store, backend) * math.sqrt(weight/world))
        return torch.cat(parts)
    update.step(residual, token)
