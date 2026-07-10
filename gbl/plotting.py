"""Standardised plotting helpers for the FYL_Gap companion notebook."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as _np
import torch
from matplotlib.axes import Axes
from torch import Tensor

from .fy import FYTracker


# -----------------------------------------------------------------------------
# Generic styling
# -----------------------------------------------------------------------------

def style_log(ax: Axes) -> None:
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)


# -----------------------------------------------------------------------------
# FY decomposition plot (Figure A)
# -----------------------------------------------------------------------------

def plot_fy_decomposition(
    tracker: FYTracker,
    lam: float,
    ax: Optional[Axes] = None,
    title: str = "Fenchel-Young decomposition",
    L_F_floor: float = 1e-16,
    show_L_F: bool = True,
    J_star: Optional[float] = None,
) -> Axes:
    r"""Frank-Wolfe duality-gap decomposition along a CPGD trajectory.

    Plots, at every recorded iterate, the components of the honest duality gap
    of Theorem 1 evaluated at the rescaled Mirror-Alignment dual
        tilde h_k = tau_k * nabla F(Phi mu_k),  tau_k = min(1, 1/||eta_k||_inf):

      * J(mu_k) - hat J^*  (hat J^* = min_k J(mu_k))  — primal optimality gap
      * L_F(Phi mu_k, tilde h_k)                     — data-fidelity FY discrepancy
                                                       (> 0 whenever the un-rescaled
                                                       dual is infeasible, tau_k < 1)
      * lam * L_R(mu_k, tilde eta_k)                 — regulariser FY discrepancy
      * Delta_k = L_F + lam * L_R                    — total FW duality gap

    `L_F_floor` is the visual floor used when L_F is numerically zero on a log axis.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))

    iters = tracker.array("iteration")
    J_arr = tracker.array("J")
    L_F_arr = tracker.array("L_F")
    L_R_lam = lam * tracker.array("L_R")
    delta_arr = tracker.array("delta")

    # Blue curve: primal optimality gap J(mu_k) - hat J^*, a proxy for J(mu_k) - inf J.
    # Since delta_k = J(mu_k) - J_H(tilde h_k) >= J(mu_k) - inf J >= J(mu_k) - hat J^*
    # for any hat J^* >= inf J, the red Delta_k curve provably upper-bounds the blue
    # optimality gap. If `J_star` is not supplied we fall back to the running min
    # min_k J(mu_k); but that min is *attained* at one plotted iterate, whose gap is
    # then exactly zero and nosedives to machine precision on the log axis. Passing a
    # `J_star` computed from a *longer* reference run (strictly below every plotted
    # J(mu_k)) keeps the optimality gap positive and faithful.
    if J_star is None:
        J_star = float(J_arr.min())
    opt_gap = (J_arr - J_star).clone()
    opt_gap[opt_gap <= 0.0] = L_F_floor

    L_F_plot = L_F_arr.clone()
    L_F_plot[L_F_plot <= 0.0] = L_F_floor

    ax.plot(iters, opt_gap, color="C0", linewidth=1.8,
            label=r"$J(\mu_k) - \hat J^\star$ (optimality gap)")
    ax.plot(iters, delta_arr, color="C3", linewidth=2.0,
            label=r"$\Delta_k = L_F + \lambda L_R$ (FY gap)")
    ax.plot(iters, L_R_lam, color="C1", linewidth=1.5,
            linestyle="--", label=r"$\lambda L_R(\mu_k\,\|\,\tilde\eta_k)$")
    if show_L_F:
        ax.plot(iters, L_F_plot, color="C2", linewidth=1.5, linestyle=":",
                label=r"$L_F(\Phi\mu_k\,\|\,\tilde h_k)$")

    ax.set_xlabel("CPGD iteration $k$")
    ax.set_ylabel("value")
    ax.set_title(title)
    style_log(ax)
    ax.legend(loc="best", fontsize=9)
    return ax


# -----------------------------------------------------------------------------
# Sup-norm of the certificate vs iteration (Figure B)
# -----------------------------------------------------------------------------

def plot_sup_norm(
    tracker: FYTracker,
    ax: Optional[Axes] = None,
    title: str = r"$\|\eta_k\|_\infty$ along the trajectory",
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))

    iters = tracker.array("iteration")
    sup = tracker.array("sup_norm_eta")
    ax.plot(iters, sup, color="C0", linewidth=2)
    ax.axhline(1.0, color="black", linestyle=":", label=r"$\|\eta\|_\infty = 1$")
    ax.set_xlabel("CPGD iteration $k$")
    ax.set_ylabel(r"$\|\eta_k\|_\infty$")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax


# -----------------------------------------------------------------------------
# Active particles vs FY-loss (Figure C)
# -----------------------------------------------------------------------------

def plot_active_vs_fy(
    tracker: FYTracker,
    lam: float,
    ax: Optional[Axes] = None,
    title: str = "Active particles and FW gap",
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))
    iters = tracker.array("iteration")
    n_active = tracker.array("n_active")
    delta = tracker.array("delta")

    ax.plot(iters, n_active, color="C0", linewidth=2, label="active particles")
    ax.set_xlabel("CPGD iteration $k$")
    ax.set_ylabel("# active particles", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    ax2 = ax.twinx()
    ax2.plot(iters, delta, color="C3", linewidth=2,
             label=r"$\Delta_k$ (FW gap)")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$\Delta_k$", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    return ax


# -----------------------------------------------------------------------------
# 1D certificate heatmap with active particles (Figure D)
# -----------------------------------------------------------------------------

def plot_certificate_1d(
    eta_grid: Tensor,
    grid: Tensor,
    particles_positions: Tensor,
    particles_signed_amps: Tensor,
    truth_positions: Optional[Tensor] = None,
    truth_amps: Optional[Tensor] = None,
    cluster_radius: Optional[float] = None,
    ax: Optional[Axes] = None,
    title: str = "Pre-dual certificate $\\eta_k$",
) -> Axes:
    """Plot eta_k together with the recovered and truth signed measures.

    If `cluster_radius` is provided AND `truth_positions` is given, the
    recovered atoms are *cluster-aggregated*: for each truth spike we sum
    the signed amplitudes of all particles within `cluster_radius` and
    plot a single representative dot at the cluster centre, on a scale
    directly comparable to `truth_amps`. Without `cluster_radius` we fall
    back to the per-particle scatter.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))

    g = grid.squeeze().detach().cpu().numpy()
    e = eta_grid.detach().cpu().numpy()
    ax.plot(g, e, color="C0", linewidth=2, label=r"$\eta_k(x)$")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
    ax.axhline(-1.0, color="black", linestyle=":", linewidth=1)
    ax.fill_between(g, -1.0, 1.0, color="black", alpha=0.05)

    pp = particles_positions.squeeze().detach().cpu().numpy()
    pa = particles_signed_amps.detach().cpu().numpy()
    if pp.ndim == 0:
        pp = [float(pp)]

    if cluster_radius is not None and truth_positions is not None:
        # Cluster-aggregated stems: one bar per truth spike at the cluster
        # mass; plus a thin tick per particle (visual "atomic spread") at
        # zero so the underlying particle distribution stays visible.
        tp_t = truth_positions.squeeze().detach().cpu()
        pp_t = particles_positions.squeeze().detach().cpu()
        pa_t = particles_signed_amps.detach().cpu()
        for j in range(tp_t.numel()):
            x_j = float(tp_t[j])
            d = (pp_t - x_j).abs()
            mask = d < cluster_radius
            if not mask.any():
                continue
            cluster_a = float(pa_t[mask].sum())
            # Cluster centre weighted by absolute amplitude
            cluster_x = float((pp_t[mask] * pa_t[mask].abs()).sum()
                              / (pa_t[mask].abs().sum() + 1e-12))
            ax.vlines(cluster_x, 0, cluster_a, color="C2", linewidth=2.0,
                      alpha=0.9, zorder=4)
            ax.scatter(cluster_x, cluster_a, color="C2", s=70,
                       zorder=5, marker="o",
                       label=(r"$\hat\mu_k$ (cluster)" if j == 0 else None))
        # Faint tick marks for individual particle positions
        ax.scatter(pp, [0.0] * len(pp), color="C2", s=8, alpha=0.35, zorder=3)
    else:
        # Per-particle mini-Diracs: one signed stem per active atom, marker size
        # growing with |amplitude| so concentrating atoms stand out from the
        # background cloud (no cluster aggregation).
        pp_arr = _np.atleast_1d(_np.asarray(pp, dtype=float))
        pa_arr = _np.atleast_1d(pa.astype(float))
        active = _np.abs(pa_arr) > 1e-4
        if active.any():
            xa, ya = pp_arr[active], pa_arr[active]
            sizes = 6.0 + 120.0 * _np.minimum(_np.abs(ya), 3.0)
            ax.vlines(xa, 0.0, _np.clip(ya, -3.0, 3.0), color="C2",
                      linewidth=1.0, alpha=0.5, zorder=3)
            ax.scatter(xa, _np.clip(ya, -3.0, 3.0), color="C2", s=sizes,
                       alpha=0.8, zorder=5, label=r"$\hat\mu_k$ (particles)")

    if truth_positions is not None:
        tp = truth_positions.squeeze().detach().cpu().numpy()
        if tp.ndim == 0:
            tp = [float(tp)]
        for j, x_j in enumerate(tp):
            ax.axvline(float(x_j), color="C3", alpha=0.5, linewidth=1.2,
                       linestyle="--")
        if truth_amps is not None:
            ta = truth_amps.detach().cpu().numpy()
            ax.scatter(tp, ta, color="C3", marker="*", s=80, zorder=6,
                       label=r"$\mu^\star$")

    ax.set_xlabel("$x \\in \\mathcal{X}$")
    ax.set_ylabel(r"$\eta_k(x)$ / amplitudes")
    ax.set_ylim(-3.0, 3.0)
    ax.set_title(title)
    ax.legend(loc="upper right")
    return ax


# -----------------------------------------------------------------------------
# Localization radius vs BR-predicted bound (Figure E)
# -----------------------------------------------------------------------------

def plot_localization(
    iters: Tensor,
    radius: Tensor,
    delta: Tensor,
    tau: float,
    ax: Optional[Axes] = None,
    title: str = "Localization radius vs BR bound",
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(iters, radius, color="C0", linewidth=2, label="empirical $r_k$")
    ax.plot(iters, torch.sqrt(tau * delta), color="C3",
            linewidth=2, linestyle="--",
            label=rf"BR bound $\sqrt{{\tau\,\Delta_k}}$ ($\tau={tau:.2f}$)")
    ax.set_xlabel("CPGD iteration $k$")
    ax.set_ylabel("radius")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title(title)
    return ax
