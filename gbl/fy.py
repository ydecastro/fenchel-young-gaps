"""Fenchel-Young loss tracking via the Frank-Wolfe dual-rescaling certificate.

Implements the honest, finite, computable duality gap of Theorem 1:

    Delta_k = L_F(Phi mu_k || tilde h_k) + lam * L_R(mu_k || tilde eta_k)

at the *rescaled* dual

    tilde h_k = tau_k * h_k,  tau_k = min(1, lam / ||Phi^* h_k||_inf),
    tilde eta_k = -Phi^* tilde h_k / lam = tau_k * eta_k,

obtained from the Mirror-Alignment candidate h_k = nabla F(Phi mu_k) by a single
scalar contraction onto the dual feasible set K_lam = { h : ||Phi^* h||_inf <= lam }.

Why rescaling and not pointwise clipping? Because the bilinear part
||mu||_TV - <bar_eta, mu> with bar_eta = clip(eta, -1, 1) is *not* the
duality-gap term lam * L_R(mu || -Phi^* h / lam) of Theorem 1 at any feasible
dual point h: the clipped function bar_eta does not in general live in
ran(Phi^*) (e.g. a clipped Gaussian-RKHS function is not in the Gaussian RKHS),
and lam * (||mu||_TV - <bar_eta, mu>) systematically *under-bounds* the true
optimality gap J(mu) - inf J. The rescaling tilde eta_k = tau_k * eta_k is the
cheapest legitimate construction that lives in ran(Phi^*) and yields a valid
upper bound on J(mu_k) - inf J at every iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor

from .particles import Particles
from .problems import BlassoProblem, GaussianBlasso, NNBlasso


# -----------------------------------------------------------------------------
# Dual rescaling: cheap contraction onto K_lam
# -----------------------------------------------------------------------------

@dataclass
class DualRescaling:
    """Scalar contraction onto K_lam = { h : ||Phi^* h||_inf <= lam }.

    Given sup_norm_eta = ||eta_k||_inf estimated on a finite `grid`, returns
    tau_k = min(1, 1 / sup_norm_eta) so that tau_k * eta_k is dual-feasible
    by construction. The rescaled certificate tilde eta_k = tau_k * eta_k
    lives in ran(Phi^*) automatically (it is a scalar multiple of eta_k).
    """

    grid: Tensor

    @staticmethod
    def tau_from_sup_norm(sup_norm_eta: float, eps: float = 1e-30) -> float:
        return min(1.0, 1.0 / max(sup_norm_eta, eps))

    def sup_norm(self, eta_values: Tensor) -> float:
        return float(eta_values.abs().max())


# -----------------------------------------------------------------------------
# Per-iteration record
# -----------------------------------------------------------------------------

@dataclass
class FYRecord:
    iteration: int
    F_val: float                # F(Phi mu)
    TV: float                   # ||mu||_TV
    J: float                    # F + lam * TV  (primal objective)
    tau_k: float                # rescaling factor min(1, 1/||eta||_inf)
    L_F: float                  # L_F(Phi mu, tilde h_k) >= 0; vanishes iff tau_k = 1
    L_R: float                  # L_R(mu, tilde eta_k) = TV - <tau_k * eta, mu> >= 0
    delta: float                # Delta_k = L_F + lam * L_R  (Theorem 1 duality gap)
    L_R_raw: float              # TV - <eta, mu>  (diagnostic; signed)
    sup_norm_eta: float         # ||eta||_inf on the discretisation grid
    n_active: int               # # particles with amplitude > activity_threshold
    extra: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# FY tracker
# -----------------------------------------------------------------------------

@dataclass
class FYTracker:
    """Per-iteration recorder of the Frank-Wolfe duality gap on a BLASSO trajectory.

    At every iterate mu_k the tracker computes the rescaled Mirror-Alignment
    dual tilde h_k = tau_k * nabla F(Phi mu_k) with tau_k = min(1, 1/||eta_k||_inf),
    then evaluates Theorem 1 *exactly* at the pair (mu_k, tilde h_k):

        Delta_k = L_F(Phi mu_k || tilde h_k) + lam * L_R(mu_k || tilde eta_k).

    Both pieces are non-negative, finite, computable from the iterate alone,
    and the sum upper-bounds J(mu_k) - inf J. The data-fidelity piece L_F is
    generically non-zero: it is the price paid for contracting h_k onto K_lam.

    `problem.L_F_rescaled(particles, tau)` is required: it returns
    L_F(Phi mu, tau * nabla F(Phi mu)). Closed-form for LSQ (= F(Phi mu) * (1-tau)^2)
    and for the logistic loss; trivial for any new strictly convex F.
    """

    problem: BlassoProblem
    grid: Tensor
    eta_evaluator: Optional[Callable[[Particles, Tensor, float], Tensor]] = None
    L_F_rescaled: Optional[Callable[[Particles, float], float]] = None
    activity_threshold: float = 1e-3

    def __post_init__(self):
        self.history: list[FYRecord] = []
        self.rescaler = DualRescaling(self.grid)

        if self.eta_evaluator is None:
            if hasattr(self.problem, "eta_on_grid"):
                self.eta_evaluator = self.problem.eta_on_grid
            else:
                raise ValueError(
                    "Provide `eta_evaluator` or use a problem class that "
                    "implements `eta_on_grid(particles, grid, lam)`."
                )
        if self.L_F_rescaled is None:
            if hasattr(self.problem, "L_F_rescaled"):
                self.L_F_rescaled = self.problem.L_F_rescaled
            else:
                raise ValueError(
                    "Provide `L_F_rescaled` or use a problem class that "
                    "implements `L_F_rescaled(particles, tau)`."
                )

    def record(self, iteration: int, particles: Particles, lam: float) -> FYRecord:
        with torch.no_grad():
            ev = self.problem(particles)
            F_val = float(ev.proximity)
            TV = float(particles.tv_norm())
            J = F_val + lam * TV

            # Raw certificate eta_k = -Phi^* h_k / lam evaluated on grid + atoms
            eta_grid = self.eta_evaluator(particles, self.grid, lam)
            sup_norm_eta = self.rescaler.sup_norm(eta_grid)
            eta_at_particles = self.eta_evaluator(
                particles, particles.positions, lam
            )
            a_signed = particles.signs * particles.amplitudes
            inner_eta_mu = float(eta_at_particles @ a_signed)

            # Dual rescaling: tau_k * eta_k is feasible by construction.
            tau_k = self.rescaler.tau_from_sup_norm(sup_norm_eta)
            L_R = TV - tau_k * inner_eta_mu                 # = TV - <tilde eta, mu>
            L_R_raw = TV - inner_eta_mu                      # diagnostic

            # Honest L_F at the rescaled dual: vanishes iff tau_k = 1 (dual feasibility).
            L_F = float(self.L_F_rescaled(particles, tau_k))
            delta = L_F + lam * L_R                          # Theorem 1 duality gap

            n_active = int((particles.amplitudes > self.activity_threshold).sum())

        rec = FYRecord(
            iteration=iteration,
            F_val=F_val, TV=TV, J=J,
            tau_k=tau_k,
            L_F=L_F,
            L_R=L_R,
            delta=delta,
            L_R_raw=L_R_raw,
            sup_norm_eta=sup_norm_eta,
            n_active=n_active,
        )
        self.history.append(rec)
        return rec

    # -- convenience accessors ---------------------------------------------------

    def array(self, attr: str) -> Tensor:
        return torch.tensor([getattr(r, attr) for r in self.history])

    def first_certified(self, eps: float) -> Optional[int]:
        """Return the first iteration with `delta_k <= eps`, or None."""
        for r in self.history:
            if r.delta <= eps:
                return r.iteration
        return None


# -----------------------------------------------------------------------------
# Mirror Alignment / KKT diagnostic (used "approximately" near convergence)
# -----------------------------------------------------------------------------

def mirror_alignment_check(
    problem: BlassoProblem,
    particles: Particles,
    lam: float,
    grid: Tensor,
    activity_threshold: float = 1e-3,
    activity_threshold_rel: float = 0.0,
    eta_evaluator: Optional[Callable[[Particles, Tensor, float], Tensor]] = None,
) -> dict:
    r"""Numerical Mirror-Alignment / KKT check at a (near-)stationary point.

    For a TV-regularised BLASSO with optimum mu^*, the dual certificate
    eta = -Phi^* nabla F(Phi mu^*) / lam must satisfy

        (i)  ||eta||_\infty <= 1                         (off-support bound)
        (ii) eta(theta_i^*) = sign(a_i^*),  i in supp(mu^*)   (saturation
              on support — Mirror Alignment / KKT).

    We evaluate (i) on a fine `grid` and (ii) at every "active" particle
    (|a_i| > activity_threshold) and return a dictionary of diagnostics.

    `eta_evaluator` overrides `problem.eta_on_grid` (needed e.g. for
    `NNBlasso` whose certificate evaluator is constructed via
    `gbl.utils.nn_eta_factory`).

    Note: for finite-iteration CPGD trajectories the equalities hold
    only *approximately* — see the printed `ma_residual_max`.
    """
    if eta_evaluator is None:
        if not hasattr(problem, "eta_on_grid"):
            raise ValueError(
                "Pass `eta_evaluator=...` (problem has no `eta_on_grid`)."
            )
        eta_evaluator = problem.eta_on_grid

    with torch.no_grad():
        eta_grid = eta_evaluator(particles, grid, lam)
        sup_norm_eta = float(eta_grid.abs().max())

        eta_at_p = eta_evaluator(particles, particles.positions, lam)
        signs = particles.signs
        amps = particles.amplitudes
        max_amp = float(amps.max()) if amps.numel() > 0 else 0.0
        rel_thresh = activity_threshold_rel * max_amp
        active = (amps > activity_threshold) & (amps > rel_thresh)
        active_idx = torch.where(active)[0]

        eta_active = eta_at_p[active_idx]
        signs_active = signs[active_idx]
        amps_active = amps[active_idx]
        ma_residual = (eta_active - signs_active).abs()
        ma_residual_max = float(ma_residual.max()) if ma_residual.numel() > 0 else 0.0

    return {
        "sup_norm_eta": sup_norm_eta,
        "n_active": int(active.sum()),
        "active_idx": active_idx,
        "eta_active": eta_active,
        "signs_active": signs_active,
        "amps_active": amps_active,
        "ma_residual": ma_residual,
        "ma_residual_max": ma_residual_max,
        "off_support_violation": max(0.0, sup_norm_eta - 1.0),
    }


def print_mirror_alignment(check: dict, label: str = "") -> None:
    """Pretty-print the dict returned by `mirror_alignment_check`."""
    print(f"Mirror-Alignment check {label}".rstrip())
    print(f"  ||eta||_inf (sup over grid) = {check['sup_norm_eta']:.4f}")
    print(f"  off-support violation       = max(0, sup_norm-1) = "
          f"{check['off_support_violation']:.2e}")
    print(f"  # active particles          = {check['n_active']}")
    print(f"  max |eta(theta_i*)-sign_i*| = {check['ma_residual_max']:.4f}  "
          f"(should be small ≈ 0)")
    n_to_print = min(check['n_active'], 25)
    if n_to_print > 0:
        amps = check['amps_active']
        idx_order = torch.argsort(amps, descending=True)[:n_to_print]
        idxs = check['active_idx'][idx_order].tolist()
        ais = amps[idx_order].tolist()
        sis = check['signs_active'][idx_order].tolist()
        etis = check['eta_active'][idx_order].tolist()
        ris = check['ma_residual'][idx_order].tolist()
        print(f"  Top-{n_to_print} active particles by amplitude:")
        print(f"    {'idx':>5}  {'|a_i|':>9}  {'sign_i':>7}  "
              f"{'eta(theta_i)':>13}  {'residual':>10}")
        for i, ai, si, eti, ri in zip(idxs, ais, sis, etis, ris):
            print(f"    {i:>5}  {ai:>9.4f}  {si:>+7.0f}  "
                  f"{eti:>+13.4f}  {ri:>10.4f}")
        if check['n_active'] > n_to_print:
            print(f"    (... {check['n_active'] - n_to_print} more particles "
                  f"with smaller amplitudes omitted)")
