#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import random
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


MAIN_METRICS = ('mixing_speed', 'clockwise_cycle_alignment', 'skew_asymmetry')
METRIC_LABEL = {
    'mixing_speed': 'Mixing Speed',
    'clockwise_cycle_alignment': 'Triangle Shape',
    'skew_asymmetry': 'Jacobian Skewness',
    'distribution_violation': 'Distribution Violation',
}


class Config:
    outdir = 'controllable_vector_field_frontier_output'
    seed = 123
    device = 'cpu'
    dtype = 'float64'

    n_components = 3
    mixture_radius = 2.3
    component_std = 0.65
    sigma = 0.45

    triangle_window_width = 0.24
    triangle_smoothness = 10.0
    triangle_edge_gate_sharpness = 5.0
    triangle_interior_weight = 0.12
    triangle_outer_falloff_width = 0.9
    triangle_density_floor = 1e-12
    skew_sharpness = 1.75
    skew_angle_deg = -20.0
    skew_center_radius_frac = 0.78
    skew_radial_width = 0.60
    skew_tangential_width = 1.05
    skew_bias_weight = 0.45
    display_skew_cycle_tie_tol = 0.995
    orthonorm_batch = 1200
    orthonorm_eps = 1e-7

    grid_n = 25
    lim = 5.5
    theta_box_radius = 2.5
    theta_grid_n = 3
    display_axis_scan_radius = 8.0
    display_axis_scan_n = 33
    compatibility_bins = 12

    flux_mc_batch_size = 256
    flux_mc_repeats = 64
    flux_mc_horizon = 4.0
    flux_mc_substeps = 4

    stationary_samples = 1800
    dpi = 180


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def torch_dtype(name: str) -> torch.dtype:
    return {'float32': torch.float32, 'float64': torch.float64}[name]


def rotate90(v: torch.Tensor) -> torch.Tensor:
    return torch.stack([-v[:, 1], v[:, 0]], dim=1)


class TriangleGaussianMixture2D:
    def __init__(self, cfg: Config, device: torch.device, dtype: torch.dtype):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        angles = torch.tensor([math.pi / 2, 7 * math.pi / 6, 11 * math.pi / 6], device=device, dtype=dtype)
        self.centers = cfg.mixture_radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        self.center_dirs = self.centers / cfg.mixture_radius
        self.weights = torch.full((cfg.n_components,), 1.0 / cfg.n_components, device=device, dtype=dtype)

    def noisy_variance(self, sigma: float) -> float:
        return self.cfg.component_std ** 2 + float(sigma) ** 2

    def sample_noisy(self, n: int, sigma: float) -> torch.Tensor:
        idx = torch.multinomial(self.weights, n, replacement=True)
        std = math.sqrt(self.noisy_variance(sigma))
        return self.centers[idx] + std * torch.randn(n, 2, device=self.device, dtype=self.dtype)

    def log_density_noisy(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        var = self.noisy_variance(sigma)
        x2 = (x * x).sum(dim=1, keepdim=True)
        mu2 = (self.centers * self.centers).sum(dim=1, keepdim=True).T
        dist2 = x2 + mu2 - 2.0 * (x @ self.centers.T)
        logits = -0.5 * dist2 / var + torch.log(self.weights).view(1, -1)
        return torch.logsumexp(logits, dim=1) - math.log(2.0 * math.pi * var)

    def score_noisy(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        var = self.noisy_variance(sigma)
        x2 = (x * x).sum(dim=1, keepdim=True)
        mu2 = (self.centers * self.centers).sum(dim=1, keepdim=True).T
        dist2 = x2 + mu2 - 2.0 * (x @ self.centers.T)
        logits = -0.5 * dist2 / var + torch.log(self.weights).view(1, -1)
        w = torch.softmax(logits, dim=1)
        mean = w @ self.centers
        return (mean - x) / var

    def density_noisy_numpy(self, xy: np.ndarray, sigma: float) -> np.ndarray:
        var = self.noisy_variance(sigma)
        centers = self.centers.detach().cpu().numpy()
        diff = xy[:, None, :] - centers[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        norm = (2.0 * math.pi * var) ** (-1.0)
        return norm * np.mean(np.exp(-0.5 * dist2 / var), axis=1)


class ExactFieldFamily:
    def __init__(self, mixture: TriangleGaussianMixture2D, sigma: float, cfg: Config):
        self.mixture = mixture
        self.sigma = float(sigma)
        self.cfg = cfg
        self.device = mixture.device
        self.dtype = mixture.dtype
        self.triangle_radius = mixture.cfg.mixture_radius
        self.triangle_vertices = mixture.centers
        self.triangle_edges = torch.roll(self.triangle_vertices, shifts=-1, dims=0) - self.triangle_vertices
        self.triangle_edge_lengths = torch.linalg.norm(self.triangle_edges, dim=1)
        self.triangle_edge_unit = self.triangle_edges / self.triangle_edge_lengths[:, None]
        signed_area = 0.5 * torch.sum(
            self.triangle_vertices[:, 0] * torch.roll(self.triangle_vertices[:, 1], shifts=-1)
            - torch.roll(self.triangle_vertices[:, 0], shifts=-1) * self.triangle_vertices[:, 1]
        )
        if float(signed_area.detach().cpu()) >= 0.0:
            outward = torch.stack([self.triangle_edge_unit[:, 1], -self.triangle_edge_unit[:, 0]], dim=1)
        else:
            outward = torch.stack([-self.triangle_edge_unit[:, 1], self.triangle_edge_unit[:, 0]], dim=1)
        self.triangle_edge_normals = outward
        self.triangle_edge_offsets = (self.triangle_edge_normals * self.triangle_vertices).sum(dim=1)
        self.skew_direction = torch.tensor(
            [math.cos(math.radians(cfg.skew_angle_deg)), math.sin(math.radians(cfg.skew_angle_deg))],
            device=self.device,
            dtype=self.dtype,
        )
        self.skew_direction = self.skew_direction / torch.clamp(torch.linalg.norm(self.skew_direction), min=1e-12)
        self.skew_center = self.cfg.skew_center_radius_frac * self.triangle_radius * self.skew_direction
        self.skew_radial_dir = self.skew_direction
        self.skew_tangential_dir = rotate90(self.skew_radial_dir[None, :])[0]
        self.transform = self._estimate_mode_scaling()

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixture.score_noisy(x, self.sigma)

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixture.log_density_noisy(x, self.sigma)

    def density(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_density(x))

    def _triangle_level(self, x: torch.Tensor) -> torch.Tensor:
        signed_edge_dist = x @ self.triangle_edge_normals.T - self.triangle_edge_offsets[None, :]
        return torch.logsumexp(self.cfg.triangle_smoothness * signed_edge_dist, dim=1) / self.cfg.triangle_smoothness

    def _triangle_edge_ridges(self, x: torch.Tensor) -> torch.Tensor:
        rel = x[:, None, :] - self.triangle_vertices[None, :, :]
        along = (rel * self.triangle_edge_unit[None, :, :]).sum(dim=-1)
        dist = (rel * self.triangle_edge_normals[None, :, :]).sum(dim=-1)
        width = max(self.cfg.triangle_window_width, 1e-8)
        sharp = self.cfg.triangle_edge_gate_sharpness
        enter_gate = torch.sigmoid(sharp * along / width)
        exit_gate = torch.sigmoid(sharp * (self.triangle_edge_lengths[None, :] - along) / width)
        segment_gate = enter_gate * exit_gate
        edge_ridges = torch.exp(-0.5 * (dist / width) ** 2) * segment_gate
        return edge_ridges

    def _psi_cycle(self, x: torch.Tensor) -> torch.Tensor:
        level = self._triangle_level(x)
        edge_ridges = self._triangle_edge_ridges(x)
        edge_mean = edge_ridges.mean(dim=1)
        interior_fill = torch.sigmoid(-self.cfg.triangle_smoothness * level)
        outer_dist = torch.relu(level)
        outer_envelope = torch.exp(-0.5 * (outer_dist / max(self.cfg.triangle_outer_falloff_width, 1e-8)) ** 2)
        return outer_envelope * (edge_mean + self.cfg.triangle_interior_weight * interior_fill)

    def _psi_skew(self, x: torch.Tensor) -> torch.Tensor:
        rel = x - self.skew_center[None, :]
        radial = (rel * self.skew_radial_dir[None, :]).sum(dim=1)
        tangential = (rel * self.skew_tangential_dir[None, :]).sum(dim=1)
        radial_scale = max(self.cfg.skew_radial_width, 1e-8)
        tangential_scale = max(self.cfg.skew_tangential_width, 1e-8)
        core = torch.exp(-0.5 * ((radial / radial_scale) ** 2 + (tangential / tangential_scale) ** 2))
        directional_bias = torch.sigmoid(self.cfg.skew_sharpness * (x @ self.skew_direction) / max(self.triangle_radius, 1e-8))
        return core * (1.0 + self.cfg.skew_bias_weight * directional_bias)

    def _raw_modes(self, x: torch.Tensor, preserving: bool, preserve_graph: bool = False) -> torch.Tensor:
        s = self.score(x)
        if preserve_graph:
            if not x.requires_grad:
                raise ValueError('preserve_graph=True requires x.requires_grad=True.')
            psi_cycle = self._psi_cycle(x)
            grad_psi_cycle = torch.autograd.grad(psi_cycle.sum(), x, create_graph=True, retain_graph=True)[0]
            psi_skew = self._psi_skew(x)
            grad_psi_skew = torch.autograd.grad(psi_skew.sum(), x, create_graph=True, retain_graph=True)[0]
            p = self.density(x)
        else:
            x1 = x.detach().clone().requires_grad_(True)
            psi_cycle = self._psi_cycle(x1)
            grad_psi_cycle = torch.autograd.grad(psi_cycle.sum(), x1)[0].detach()
            x2 = x.detach().clone().requires_grad_(True)
            psi_skew = self._psi_skew(x2)
            grad_psi_skew = torch.autograd.grad(psi_skew.sum(), x2)[0].detach()
            p = self.density(x.detach()).detach()
        p_safe = p.clamp_min(self.cfg.triangle_density_floor)

        if preserving:
            b0 = rotate90(s)
            b1 = -rotate90(grad_psi_cycle) / p_safe[:, None]
            b2 = rotate90(grad_psi_skew) / p_safe[:, None]
        else:
            b0 = s
            b1 = grad_psi_cycle / p_safe[:, None]
            b2 = grad_psi_skew / p_safe[:, None]
        return torch.stack([b0, b1, b2], dim=1)

    def _estimate_mode_scaling(self) -> torch.Tensor:
        x = self.mixture.sample_noisy(self.cfg.orthonorm_batch, self.sigma)
        raw = self._raw_modes(x, preserving=True)
        mode_energy = torch.einsum('nki,nki->k', raw, raw) / raw.shape[0]
        scale = torch.rsqrt(torch.clamp(mode_energy, min=self.cfg.orthonorm_eps))
        return torch.diag(scale).detach()

    def basis(self, x: torch.Tensor, preserving: bool, preserve_graph: bool = False) -> torch.Tensor:
        raw = self._raw_modes(x, preserving=preserving, preserve_graph=preserve_graph)
        return torch.einsum('nki,kl->nli', raw, self.transform)

    def field(self, x: torch.Tensor, theta: torch.Tensor, preserving: bool, preserve_graph: bool = False) -> torch.Tensor:
        s = self.score(x)
        basis = self.basis(x, preserving=preserving, preserve_graph=preserve_graph)
        if theta.ndim == 1:
            theta = theta[None, :].expand(x.shape[0], -1)
        return s + torch.einsum('nk,nki->ni', theta.to(x), basis)

    def residual(self, x: torch.Tensor, theta: torch.Tensor, preserving: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if theta.ndim == 1:
            theta = theta[None, :].expand(x.shape[0], -1)
        s = self.score(x)
        f = self.field(x, theta, preserving=preserving, preserve_graph=True)
        u = f - s
        div_terms = []
        for d in range(u.shape[1]):
            grad_ud = torch.autograd.grad(u[:, d].sum(), x, create_graph=True, retain_graph=True)[0]
            div_terms.append(grad_ud[:, d])
        div_u = torch.stack(div_terms, dim=1).sum(dim=1)
        r = div_u + (u * s).sum(dim=1)
        return r, u

def detached_flux_loss_mc(
    family: ExactFieldFamily,
    mixture: TriangleGaussianMixture2D,
    theta: np.ndarray,
    batch_size: int,
    repeats: int,
    sigma: float,
    horizon: float,
    n_substeps: int,
    preserving: bool,
) -> float:
    eps = 1e-12
    device = family.device
    dtype = family.dtype
    theta_t = torch.tensor(theta, device=device, dtype=dtype)
    sigma2_scalar = torch.tensor(float(sigma) ** 2, device=device, dtype=dtype).clamp_min(eps)

    def score_fn(x: torch.Tensor) -> torch.Tensor:
        return mixture.score_noisy(x, float(sigma))

    def ou_step(x: torch.Tensor, tau_step: torch.Tensor) -> torch.Tensor:
        rho = torch.exp(-tau_step)
        mu = x + sigma2_scalar * score_fn(x)
        noise = torch.sqrt((sigma2_scalar * (1.0 - rho.square())).clamp_min(0.0)) * torch.randn_like(x)
        return rho * x + (1.0 - rho) * mu + noise

    def rollout(x0: torch.Tensor, tau_scalar: torch.Tensor) -> torch.Tensor:
        x = x0
        tau_step = tau_scalar / float(n_substeps)
        for _ in range(n_substeps):
            x = ou_step(x, tau_step)
        return x

    def responsibilities(x_anchor: torch.Tensor, x_end: torch.Tensor, tau_scalar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x_anchor.shape[0]
        rho_scalar = torch.exp(-tau_scalar)
        var_scalar = (sigma2_scalar * (1.0 - rho_scalar.square())).clamp_min(eps)
        s_anchor = score_fn(x_anchor)
        means_anchor = x_anchor + (1.0 - rho_scalar) * sigma2_scalar * s_anchor
        means_flat = means_anchor.reshape(b, -1)
        end_flat = x_end.reshape(b, -1)
        dist2 = (
            means_flat.square().sum(dim=-1, keepdim=True)
            + end_flat.square().sum(dim=-1, keepdim=True).T
            - 2.0 * means_flat @ end_flat.T
        )
        logits = -0.5 * dist2 / var_scalar
        return torch.softmax(logits, dim=0), rho_scalar

    batch_means = []
    for _ in range(repeats):
        x_anchor = mixture.sample_noisy(batch_size, float(sigma))
        tau_scalar = torch.rand((), device=device, dtype=dtype) * float(horizon)
        q_tau_scalar = torch.as_tensor(1.0 / float(horizon), device=device, dtype=dtype)
        x_tau = rollout(x_anchor.detach(), tau_scalar.detach())

        x_tau_grad = x_tau.detach().requires_grad_(True)
        r_tau, _ = family.residual(x_tau_grad, theta_t, preserving=preserving)
        grad_r_tau = torch.autograd.grad(r_tau.sum(), x_tau_grad, create_graph=False, retain_graph=False)[0].detach()

        resp, rho_scalar = responsibilities(x_anchor.detach(), x_tau.detach(), tau_scalar.detach())
        grad_r_tau_flat = grad_r_tau.reshape(batch_size, -1)
        q_hat_flat = (rho_scalar / q_tau_scalar) * (resp @ grad_r_tau_flat)
        q_hat = q_hat_flat.reshape_as(x_anchor).detach()

        _, u0 = family.residual(x_anchor.detach().requires_grad_(True), theta_t, preserving=preserving)
        losses = -2.0 * (u0 * q_hat).sum(dim=1)
        batch_means.append(float(losses.mean().detach().cpu()))
    return float(np.mean(batch_means))


def second_derivative_matrix(n: int, h: float) -> sp.csr_matrix:
    h2 = h ** 2
    d = sp.diags([1/h2, -2/h2, 1/h2], [-1, 0, 1], shape=(n, n), format='lil')
    d[0, 0], d[0, 1] = -2/h2, 2/h2
    d[-1, -1], d[-1, -2] = -2/h2, 2/h2
    return d.tocsr()


def first_derivative_matrix(n: int, h: float) -> sp.csr_matrix:
    d = sp.diags([-0.5/h, 0.5/h], [-1, 1], shape=(n, n), format='lil')
    d[0, 0], d[0, 1] = -1/h, 1/h
    d[-1, -2], d[-1, -1] = -1/h, 1/h
    return d.tocsr()


class GridDiscretization:
    def __init__(self, grid_n: int, lim: float):
        self.n = grid_n
        self.lim = lim
        self.xs = np.linspace(-lim, lim, grid_n)
        self.h = float(self.xs[1] - self.xs[0])
        self.X, self.Y = np.meshgrid(self.xs, self.xs, indexing='ij')
        self.points = np.stack([self.X.ravel(), self.Y.ravel()], axis=1)
        d2 = second_derivative_matrix(grid_n, self.h)
        d1 = first_derivative_matrix(grid_n, self.h)
        eye = sp.eye(grid_n, format='csr')
        self.lap = (sp.kron(d2, eye) + sp.kron(eye, d2)).tocsc()
        self.dx = sp.kron(d1, eye).tocsc()
        self.dy = sp.kron(eye, d1).tocsc()

    def build_generator(self, u: np.ndarray) -> sp.csc_matrix:
        ux, uy = u[:, 0], u[:, 1]
        return self.lap + sp.diags(ux) @ self.dx + sp.diags(uy) @ self.dy


class MixingIATCalculator:
    def __init__(self, grid: GridDiscretization, weights: np.ndarray):
        self.grid = grid
        self.w = weights / weights.sum()
        self.mean_constraint = sp.csc_matrix(np.outer(np.ones_like(self.w), self.w))
        pts = grid.points
        x, y = pts[:, 0], pts[:, 1]
        r = np.sqrt(x * x + y * y + 1e-12)
        th = np.arctan2(y, x)
        self.observables = [x, y, r, np.cos(th), np.sin(th)]

    def mixing_speed(self, u: np.ndarray) -> float:
        a = -self.grid.build_generator(u) + self.mean_constraint
        taus = []
        for obs in self.observables:
            f = obs - float(self.w @ obs)
            var = float(self.w @ (f * f))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', spla.MatrixRankWarning)
                psi = spla.spsolve(a, f)
            tau_raw = 2.0 * float(self.w @ (f * psi)) / max(var, 1e-12)
            tau = tau_raw if np.isfinite(tau_raw) else 1e8
            taus.append(max(float(np.real(tau)), 1e-8))
        return 1.0 / max(float(np.mean(taus)), 1e-8)


def cosine_alignment(delta_u: np.ndarray, template: np.ndarray, w: np.ndarray) -> float:
    num = float(w @ np.sum(delta_u * template, axis=1))
    den1 = math.sqrt(max(float(w @ np.sum(delta_u * delta_u, axis=1)), 1e-12))
    den2 = math.sqrt(max(float(w @ np.sum(template * template, axis=1)), 1e-12))
    return num / max(den1 * den2, 1e-12)


def skew_asymmetry(delta_u: np.ndarray, grid: GridDiscretization, w: np.ndarray) -> float:
    dux_dy = np.asarray(grid.dy @ delta_u[:, 0]).reshape(-1)
    duy_dx = np.asarray(grid.dx @ delta_u[:, 1]).reshape(-1)
    return float(w @ (0.5 * (dux_dy - duy_dx) ** 2))


def distribution_violation(delta_u: np.ndarray, score_grid: np.ndarray, grid: GridDiscretization, w: np.ndarray) -> float:
    div_u = np.asarray(grid.dx @ delta_u[:, 0] + grid.dy @ delta_u[:, 1]).reshape(-1)
    r = div_u + np.sum(delta_u * score_grid, axis=1)
    return float(np.sqrt(max(float(w @ (r * r)), 0.0)))


def binned_mean_and_se(rows: list[dict], x_key: str, y_key: str, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([r[x_key] for r in rows], dtype=float)
    y = np.array([r[y_key] for r in rows], dtype=float)
    edges = np.linspace(float(x.min()), float(x.max()), bins + 1)
    xs, ys, es = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (x >= a) & (x < b if b < edges[-1] else x <= b)
        if not mask.any():
            continue
        vals = y[mask]
        xs.append(0.5 * (a + b))
        ys.append(float(vals.mean()))
        es.append(float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)
    return np.asarray(xs), np.asarray(ys), np.asarray(es)


def _insert_score_point(xs: np.ndarray, ys: np.ndarray, es: np.ndarray, score_x: float, score_y: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs2, ys2, es2 = np.append(xs, score_x), np.append(ys, score_y), np.append(es, 0.0)
    order = np.argsort(xs2)
    return xs2[order], ys2[order], es2[order]



def plot_compatibility_curves(
    path: str,
    preserving_rows: list[dict],
    violating_rows: list[dict],
    score_points: dict[str, tuple[float, float]],
    bins: int,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(21.0, 4.8))
    for ax, metric in zip(axes[:3], MAIN_METRICS):
        xs_d, ys_d, es_d = binned_mean_and_se(preserving_rows, metric, 'dsm_loss_normalized', bins)
        xs_f, ys_f, es_f = binned_mean_and_se(preserving_rows, metric, 'flux_loss_normalized', bins)
        score_x, score_y = score_points[metric]
        xs_d, ys_d, es_d = _insert_score_point(xs_d, ys_d, es_d, score_x, score_y)
        xs_f, ys_f, es_f = _insert_score_point(xs_f, ys_f, es_f, score_x, score_y)
        ax.errorbar(xs_d, ys_d, yerr=es_d, marker='o', ms=6.0, lw=1.7, capsize=2.0, label='Score Matching')
        ax.errorbar(xs_f, ys_f, yerr=es_f, marker='o', ms=6.0, lw=1.7, capsize=2.0, label='Flux Matching')
        ax.scatter([score_x], [score_y], marker='*', s=140, c='black', zorder=6, label=r'$\nabla \log p$')
        ax.set_xlabel(METRIC_LABEL[metric], fontweight='bold', fontsize=18)
        if ax is axes[0]:
            ax.set_ylabel('Standardized Loss', fontweight='bold', fontsize=18)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(alpha=0.12)

    xs_d, ys_d, es_d = binned_mean_and_se(violating_rows, 'distribution_violation', 'dsm_loss_normalized', bins)
    xs_f, ys_f, es_f = binned_mean_and_se(violating_rows, 'distribution_violation', 'flux_loss_normalized', bins)
    score_x, score_y = score_points['distribution_violation']
    xs_d, ys_d, es_d = _insert_score_point(xs_d, ys_d, es_d, score_x, score_y)
    xs_f, ys_f, es_f = _insert_score_point(xs_f, ys_f, es_f, score_x, score_y)
    axes[3].errorbar(xs_d, ys_d, yerr=es_d, marker='o', ms=6.0, lw=1.7, capsize=2.0, label='DSM')
    axes[3].errorbar(xs_f, ys_f, yerr=es_f, marker='o', ms=6.0, lw=1.7, capsize=2.0, label='Flux')
    axes[3].scatter([score_x], [score_y], marker='*', s=140, c='black', zorder=6, label='Score')
    axes[3].set_xlabel(METRIC_LABEL['distribution_violation'], fontweight='bold', fontsize=18)
    axes[3].tick_params(axis='both', labelsize=12)
    axes[3].grid(alpha=0.12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=18, prop={'weight': 'bold', 'size': 18})
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _add_panel_divider(fig: plt.Figure, ax_left: plt.Axes, ax_right: plt.Axes) -> None:
    bbox0, bbox1 = ax_left.get_position(), ax_right.get_position()
    fig.add_artist(plt.Line2D(
        [(bbox0.x1 + bbox1.x0) / 2] * 2, [0.05, 0.95],
        transform=fig.transFigure, color='black', linewidth=2.5,
    ))


def plot_fixed_length_vector_fields(
    path: str,
    grid: GridDiscretization,
    fields: list[np.ndarray],
    titles: list[str],
    stationary_samples: np.ndarray,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.8))
    speeds = [np.sqrt((u[:, 0] ** 2 + u[:, 1] ** 2)).reshape(grid.n, grid.n) for u in fields]
    vmax = max(max(float(np.quantile(s, 0.95)) for s in speeds), 1e-8)
    for ax, u, speed, title in zip(axes, fields, speeds, titles):
        u = u.reshape(grid.n, grid.n, 2)
        uu = u[:, :, 0]
        vv = u[:, :, 1]
        norm = np.sqrt(uu * uu + vv * vv)
        uu = uu / np.maximum(norm, 1e-12)
        vv = vv / np.maximum(norm, 1e-12)
        ax.quiver(
            grid.X,
            grid.Y,
            uu,
            vv,
            np.clip(speed, 0.0, vmax),
            cmap='viridis',
            alpha=0.9,
            angles='xy',
            scale_units='xy',
            scale=3.1,
            pivot='mid',
            width=0.004,
            headwidth=3.2,
            headlength=4.2,
            headaxislength=3.6,
        )
        ax.scatter(stationary_samples[:, 0], stationary_samples[:, 1], s=5, alpha=0.2, c='red', linewidths=0, zorder=0)
        ax.set_title(title, fontweight='bold', fontsize=18)
        ax.set_xlim(-grid.lim, grid.lim)
        ax.set_ylim(-grid.lim, grid.lim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.subplots_adjust(wspace=0.02)
    _add_panel_divider(fig, axes[0], axes[1])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_streamline_vector_fields(
    path: str,
    grid: GridDiscretization,
    fields: list[np.ndarray],
    titles: list[str],
    stationary_samples: np.ndarray,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.8))
    speeds = [np.sqrt((u[:, 0] ** 2 + u[:, 1] ** 2)).reshape(grid.n, grid.n) for u in fields]
    xs = grid.X[:, 0]
    ys = grid.Y[0, :]
    for ax, u, speed, title in zip(axes, fields, speeds, titles):
        u = u.reshape(grid.n, grid.n, 2)
        uu = u[:, :, 0].T
        vv = u[:, :, 1].T
        speed = speed.T
        panel_vmax = max(float(np.quantile(speed, 0.95)), 1e-8)
        panel_norm = plt.Normalize(0.0, panel_vmax)
        ax.streamplot(
            xs,
            ys,
            uu,
            vv,
            color=np.clip(speed, 0.0, panel_vmax),
            cmap='viridis',
            linewidth=1.2,
            density=1.2,
            arrowsize=1.1,
            norm=panel_norm,
        )
        ax.scatter(stationary_samples[:, 0], stationary_samples[:, 1], s=5, alpha=0.2, c='red', linewidths=0, zorder=0)
        ax.set_title(title, fontweight='bold', fontsize=18)
        ax.set_xlim(-grid.lim, grid.lim)
        ax.set_ylim(-grid.lim, grid.lim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.subplots_adjust(wspace=0.02)
    _add_panel_divider(fig, axes[0], axes[1])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def run_experiment(cfg: Config) -> str:
    ensure_dir(cfg.outdir)
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    dtype = torch_dtype(cfg.dtype)
    mixture = TriangleGaussianMixture2D(cfg, device, dtype)
    family = ExactFieldFamily(mixture, cfg.sigma, cfg)
    grid = GridDiscretization(cfg.grid_n, cfg.lim)

    pts_t = torch.tensor(grid.points, device=device, dtype=dtype)
    with torch.no_grad():
        score_grid = family.score(pts_t).cpu().numpy()
    basis_grid = family.basis(pts_t, preserving=True).detach().cpu().numpy()
    nonpres_basis_grid = family.basis(pts_t, preserving=False).detach().cpu().numpy()

    p_grid = mixture.density_noisy_numpy(grid.points, cfg.sigma)
    w = p_grid / p_grid.sum()
    mixer = MixingIATCalculator(grid, w)

    gram = np.zeros((3, 3), dtype=float)
    gram_bad = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            gram[i, j] = float(w @ np.sum(basis_grid[:, i, :] * basis_grid[:, j, :], axis=1))
            gram_bad[i, j] = float(w @ np.sum(nonpres_basis_grid[:, i, :] * nonpres_basis_grid[:, j, :], axis=1))

    vals = np.linspace(-cfg.theta_box_radius, cfg.theta_box_radius, cfg.theta_grid_n)
    theta_scan = np.stack(np.meshgrid(vals, vals, vals, indexing='ij'), axis=-1).reshape(-1, 3)
    if not np.any(np.all(np.isclose(theta_scan, 0.0), axis=1)):
        theta_scan = np.concatenate([theta_scan, np.zeros((1, 3), dtype=float)], axis=0)

    preserving_rows = []
    violating_rows = []
    metric_axis = {
        'mixing_speed': 0,
        'clockwise_cycle_alignment': 1,
        'skew_asymmetry': 2,
    }
    best = {metric: (-float('inf'), None) for metric in MAIN_METRICS}

    for theta in theta_scan:
        delta_u = np.tensordot(basis_grid, theta, axes=([1], [0]))
        u = score_grid + delta_u
        mix = mixer.mixing_speed(u)
        cycle = cosine_alignment(delta_u, basis_grid[:, 1, :], w)
        skew = skew_asymmetry(delta_u, grid, w)
        dsm_loss = 0.5 * float(theta @ gram @ theta)
        flux_loss = detached_flux_loss_mc(
            family, mixture, theta,
            batch_size=cfg.flux_mc_batch_size,
            repeats=cfg.flux_mc_repeats,
            sigma=cfg.sigma,
            horizon=cfg.flux_mc_horizon,
            n_substeps=cfg.flux_mc_substeps,
            preserving=True,
        )
        row = {
            'theta': theta.copy(),
            'field': u.copy(),
            'mixing_speed': float(mix),
            'clockwise_cycle_alignment': float(cycle),
            'skew_asymmetry': float(skew),
            'dsm_loss': abs(float(dsm_loss)),
            'flux_loss': abs(float(flux_loss)),
        }
        preserving_rows.append(row)
        for metric in MAIN_METRICS:
            axis = metric_axis[metric]
            other_axes_zero = np.all(np.isclose(np.delete(theta, axis), 0.0))
            if other_axes_zero and row[metric] > best[metric][0]:
                best[metric] = (row[metric], u.copy())

        delta_u_bad = np.tensordot(nonpres_basis_grid, theta, axes=([1], [0]))
        dsm_bad = 0.5 * float(theta @ gram_bad @ theta)
        flux_bad = detached_flux_loss_mc(
            family, mixture, theta,
            batch_size=cfg.flux_mc_batch_size,
            repeats=cfg.flux_mc_repeats,
            sigma=cfg.sigma,
            horizon=cfg.flux_mc_horizon,
            n_substeps=cfg.flux_mc_substeps,
            preserving=False,
        )
        violating_rows.append({
            'distribution_violation': distribution_violation(delta_u_bad, score_grid, grid, w),
            'dsm_loss': abs(float(dsm_bad)),
            'flux_loss': abs(float(flux_bad)),
        })

    dsm_denom = max(float(np.mean([r['dsm_loss'] for r in violating_rows])), 1e-12)
    flux_denom = max(float(np.mean([r['flux_loss'] for r in violating_rows])), 1e-12)
    for row in preserving_rows:
        row['dsm_loss_normalized'] = row['dsm_loss'] / dsm_denom
        row['flux_loss_normalized'] = row['flux_loss'] / flux_denom
    for row in violating_rows:
        row['dsm_loss_normalized'] = row['dsm_loss'] / dsm_denom
        row['flux_loss_normalized'] = row['flux_loss'] / flux_denom

    score_points = {
        'mixing_speed': (mixer.mixing_speed(score_grid), 0.0),
        'clockwise_cycle_alignment': (0.0, 0.0),
        'skew_asymmetry': (0.0, 0.0),
        'distribution_violation': (0.0, 0.0),
    }

    plot_compatibility_curves(
        os.path.join(cfg.outdir, 'compatibility_curves.png'),
        preserving_rows,
        violating_rows,
        score_points,
        cfg.compatibility_bins,
        cfg.dpi,
    )

    display_best = {}
    display_axis_vals = np.linspace(-cfg.display_axis_scan_radius, cfg.display_axis_scan_radius, cfg.display_axis_scan_n)
    for metric in MAIN_METRICS:
        axis = metric_axis[metric]
        candidates = []
        for val in display_axis_vals:
            theta = np.zeros(3, dtype=float)
            theta[axis] = float(val)
            delta_u = np.tensordot(basis_grid, theta, axes=([1], [0]))
            u = score_grid + delta_u
            if metric == 'mixing_speed':
                metric_value = mixer.mixing_speed(u)
            elif metric == 'clockwise_cycle_alignment':
                metric_value = cosine_alignment(delta_u, basis_grid[:, 1, :], w)
            elif metric == 'skew_asymmetry':
                metric_value = skew_asymmetry(delta_u, grid, w)
            else:
                raise RuntimeError(f'Unknown metric: {metric}')
            cycle_alignment = cosine_alignment(delta_u, basis_grid[:, 1, :], w)
            candidates.append({
                'metric_value': float(metric_value),
                'cycle_alignment': float(cycle_alignment),
                'abs_theta': abs(float(val)),
                'field': u.copy(),
            })
        if metric == 'skew_asymmetry':
            max_metric = max(c['metric_value'] for c in candidates)
            kept = [c for c in candidates if c['metric_value'] >= cfg.display_skew_cycle_tie_tol * max_metric]
            kept.sort(key=lambda c: (abs(c['cycle_alignment']), -c['metric_value'], -c['abs_theta']))
            display_best[metric] = kept[0]['field']
        else:
            candidates.sort(key=lambda c: (c['metric_value'], c['abs_theta']))
            display_best[metric] = candidates[-1]['field']

    stationary_samples = mixture.sample_noisy(cfg.stationary_samples, cfg.sigma).cpu().numpy()
    vector_fields = [score_grid] + [display_best[m] for m in MAIN_METRICS]
    titles = [
        r'Score Field ($\nabla \log p$)',
        'Max Mixing Field',
        'Max Triangle Shape Field',
        'Max Jacobian Skewness Field',
    ]
    plot_streamline_vector_fields(
        os.path.join(cfg.outdir, 'vector_fields_streamlines.png'),
        grid,
        vector_fields,
        titles,
        stationary_samples,
        dpi=cfg.dpi,
    )
    return cfg.outdir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', type=str, default=None)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--dtype', type=str, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--sigma', type=float, default=None)
    p.add_argument('--grid-n', type=int, default=None)
    p.add_argument('--lim', type=float, default=None)
    p.add_argument('--theta-box-radius', type=float, default=None)
    p.add_argument('--theta-grid-n', type=int, default=None)
    p.add_argument('--compatibility-bins', type=int, default=None)
    p.add_argument('--flux-mc-batch-size', type=int, default=None)
    p.add_argument('--flux-mc-repeats', type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    for key, val in vars(args).items():
        if val is not None:
            setattr(cfg, key, val)
    print(f'Saved outputs to: {run_experiment(cfg)}')


if __name__ == '__main__':
    main()
