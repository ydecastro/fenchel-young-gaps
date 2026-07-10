"""Utility helpers: particle initialisation, dataset loading, synthetic ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .particles import Particles


# -----------------------------------------------------------------------------
# Particle initialisation
# -----------------------------------------------------------------------------

def initialize_nn_particles(
    n_particles: int,
    theta_dim: int,
    seed: Optional[int] = None,
    weight_scale: float = 10.0,
    all_positive: bool = True,
) -> Particles:
    """Initialise particles on the unit sphere of R^{theta_dim} (FastPart-style).

    Matches `experiments_fastpart_nn_california.ipynb`:
    - random isotropic positions, then projected to the unit sphere,
    - small uniform weights `weight_scale / n_particles`,
    - signs all `+1` (the antipodal symmetry of the unit sphere covers the
      negative-amplitude part of the dual measure for free).
    """
    if seed is not None:
        torch.manual_seed(seed)

    positions = torch.randn(n_particles, theta_dim)
    positions = positions / positions.norm(dim=1, keepdim=True)
    weights = (weight_scale / n_particles) * torch.ones(n_particles)
    if all_positive:
        signs = torch.ones(n_particles)
    else:
        signs = 2.0 * torch.randint(2, size=(n_particles,)).float() - 1.0

    return Particles(h=lambda r: r, positions=positions, weights=weights, signs=signs)


def initialize_gaussian_particles(
    n_particles: int,
    domain: tuple[float, float] = (0.0, 1.0),
    dim: int = 1,
    seed: Optional[int] = None,
    weight_scale: float = 1.0,
    signed: bool = True,
) -> Particles:
    """Initialise particles uniformly in a box [a, b]^dim with random signs.

    Used for the synthetic 1D Gaussian super-resolution problem. Positions
    are *unconstrained* during CPGD (no projection to a sphere).
    """
    if seed is not None:
        torch.manual_seed(seed)

    a, b = domain
    positions = a + (b - a) * torch.rand(n_particles, dim)
    weights = (weight_scale / n_particles) * torch.ones(n_particles)
    if signed:
        signs = 2.0 * torch.randint(2, size=(n_particles,)).float() - 1.0
    else:
        signs = torch.ones(n_particles)

    return Particles(h=lambda r: r, positions=positions, weights=weights, signs=signs)


# -----------------------------------------------------------------------------
# Synthetic 1D problem: ground truth and observations
# -----------------------------------------------------------------------------

@dataclass
class Synth1D:
    design_points: Tensor      # (N, 1)
    y: Tensor                  # (N,)
    sigma_phi: float
    spike_locations: Tensor    # (k, 1)
    spike_amplitudes: Tensor   # (k,)
    spike_signs: Tensor        # (k,)
    sigma_w: float


def synth_1d_problem(
    n_samples: int = 200,
    sigma_phi: float = 0.04,
    spike_locations: tuple[float, ...] = (0.18, 0.34, 0.55, 0.79),
    spike_amplitudes: tuple[float, ...] = (1.0, 0.6, 0.8, 1.2),
    spike_signs: tuple[int, ...] = (+1, -1, +1, -1),
    sigma_w: float = 1e-2,
    seed: int = 314,
) -> Synth1D:
    """Synthetic 1D Gaussian super-resolution problem.

    Returns the design points (uniform in [0,1]), the observations y =
    Phi mu^* + w, and the ground truth.
    """
    torch.manual_seed(seed)

    s = torch.linspace(0.0, 1.0, n_samples).unsqueeze(-1)        # (N, 1)
    x_star = torch.tensor(spike_locations).unsqueeze(-1)         # (k, 1)
    a_star = torch.tensor(spike_amplitudes)                      # (k,)
    sgn_star = torch.tensor(spike_signs).float()                 # (k,)

    # Phi mu* at design points
    d2 = ((s.unsqueeze(1) - x_star.unsqueeze(0)) ** 2).sum(dim=-1)  # (N, k)
    phi = torch.exp(-d2 / (2.0 * sigma_phi ** 2))                    # (N, k)
    Phi_mu_star = phi @ (sgn_star * a_star)                          # (N,)

    w = sigma_w * torch.randn(n_samples)
    y = Phi_mu_star + w

    return Synth1D(
        design_points=s,
        y=y,
        sigma_phi=sigma_phi,
        spike_locations=x_star,
        spike_amplitudes=a_star,
        spike_signs=sgn_star,
        sigma_w=sigma_w,
    )


# -----------------------------------------------------------------------------
# California Housing loader (matches the FastPart 2024 split)
# -----------------------------------------------------------------------------

def load_california_housing(
    test_size: float = 0.1, random_state: int = 314, dtype: torch.dtype = torch.float32,
):
    """Load California Housing using the FastPart 2024 split & scaling.

    Returns a dict with X_train_tensor, X_test_tensor, y_train_tensor,
    y_test_tensor, feature_names, and metadata. Falls back to local-cached
    sklearn data; will raise if sklearn is not installed.
    """
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    out = dict(
        X_train_tensor=torch.tensor(X_train, dtype=dtype),
        X_test_tensor=torch.tensor(X_test, dtype=dtype),
        y_train_tensor=torch.tensor(y_train, dtype=dtype).unsqueeze(-1),
        y_test_tensor=torch.tensor(y_test, dtype=dtype).unsqueeze(-1),
        feature_names=housing.feature_names,
        n_train=X_train.shape[0],
        n_test=X_test.shape[0],
        input_dim=X_train.shape[1],
    )
    return out


# -----------------------------------------------------------------------------
# Eta evaluator for NNBlasso (Monte-Carlo sup-norm on the unit sphere)
# -----------------------------------------------------------------------------

def nn_eta_evaluator(particles: Particles, grid: Tensor, lam: float) -> Tensor:
    """Evaluate eta = -(Phi^* h)/lam at points in `grid` (each row a theta on S^{d}).

    For an NNBlasso-style problem with affine ReLU features, eta(theta) is
    *not* a closed form but can be evaluated point-by-point via the residual
    representation:
        eta(theta) = -(1/lam) * (1/N) sum_k (Phi mu(s_k) - y_k) * ReLU(<w, s_k> + b)
    where theta = (w, b).

    This helper is closure-bound to a particular `NNBlasso` problem in
    practice (call from a wrapper); the signature matches `FYTracker`.
    """
    # NOTE: This function intentionally raises; the actual evaluation requires
    # the NNBlasso problem instance, which we close over in a wrapper at the
    # notebook level. See `nn_eta_factory` below.
    raise NotImplementedError(
        "Use nn_eta_factory(problem) to construct a problem-bound evaluator."
    )


def nn_eta_factory(problem) -> "callable":
    """Return an `eta_evaluator(particles, grid, lam)` for the NNBlasso problem."""

    def eta_eval(particles, grid, lam):
        with torch.no_grad():
            ev = problem(particles)
            yhat = (ev.signed_amps[:, None] * ev.phi).sum(dim=0)         # (N,)
            residual = yhat - problem.y.squeeze()                         # (N,)
            # Now eta(theta) for a row theta = (w, b) of `grid`:
            if grid.dim() == 1:
                grid = grid.unsqueeze(0)
            w_grid = grid[:, :-1]
            b_grid = grid[:, -1:]
            phi_grid = torch.relu(w_grid @ problem.inputs.T + b_grid)    # (G, N)
            # (Phi^* h)(theta) = (1/N) sum_k h_k * phi(theta, s_k)
            phistar_h = (phi_grid * residual[None, :]).mean(dim=1)       # (G,)
            return -phistar_h / lam

    return eta_eval


def sample_unit_sphere(n_samples: int, dim: int, seed: Optional[int] = None) -> Tensor:
    """Monte-Carlo sample of `n_samples` points on the unit sphere of R^dim."""
    if seed is not None:
        torch.manual_seed(seed)
    pts = torch.randn(n_samples, dim)
    return pts / pts.norm(dim=1, keepdim=True)


# -----------------------------------------------------------------------------
# Logistic-GBL sample synthesis (paper §3.6)
# -----------------------------------------------------------------------------

@dataclass
class LogisticSample:
    design_points: Tensor      # (N, d), s_i drawn from rho_s
    y: Tensor                  # (N,) in {-1, +1}, y_i ~ Bernoulli on {0,1} mapped to {-1,+1}
    g_star: Tensor             # (N,), g^*(s_i) = (Phi mu^*)(s_i) — the noiseless logit
    p_star: Tensor             # (N,), sigmoid(g_star) = P(y_i=1 | s_i)
    spike_locations: Tensor    # (k, d)
    spike_amplitudes: Tensor   # (k,)
    spike_signs: Tensor        # (k,) in {-1, +1}
    sigma_phi: float


def sample_logistic_gbl(
    spike_locations: Tensor,            # (k, d) or (k,) — locations
    spike_amplitudes: Tensor,           # (k,) — non-negative amplitudes
    spike_signs: Tensor,                # (k,) in {-1, +1}
    sigma_phi: float,
    n_samples: int,
    domain: tuple[float, float] = (0.0, 1.0),
    seed: Optional[int] = None,
) -> LogisticSample:
    r"""Synthesise a well-specified population logistic GBL sample.

    Steps:
      1. Draw  s_i  i.i.d.\ uniformly in `domain`^d.
      2. Compute the noiseless logit  g^*(s_i) = sum_j a_j sgn_j phi(s_i, x_j^*).
      3. Draw  y_i in {-1, +1}  with  P(y_i = +1 | s_i) = sigma(g^*(s_i)).

    Returns a `LogisticSample` dataclass holding both the data (`design_points`,
    `y`) and the ground-truth quantities (`g_star`, `p_star`, ...) needed for
    downstream sanity checks.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if spike_locations.dim() == 1:
        spike_locations = spike_locations.unsqueeze(-1)
    d = spike_locations.shape[1]

    a, b = domain
    s = a + (b - a) * torch.rand(n_samples, d)                   # (N, d)

    d2 = ((s.unsqueeze(1) - spike_locations.unsqueeze(0)) ** 2).sum(dim=-1)  # (N, k)
    phi = torch.exp(-d2 / (2.0 * sigma_phi ** 2))                # (N, k)
    g_star = phi @ (spike_signs * spike_amplitudes)              # (N,)
    p_star = torch.sigmoid(g_star)                                # (N,)

    u = torch.rand(n_samples)
    y = torch.where(u < p_star, torch.ones(n_samples), -torch.ones(n_samples))

    return LogisticSample(
        design_points=s, y=y,
        g_star=g_star, p_star=p_star,
        spike_locations=spike_locations,
        spike_amplitudes=spike_amplitudes,
        spike_signs=spike_signs,
        sigma_phi=sigma_phi,
    )


# -----------------------------------------------------------------------------
# Greedy matching for recovery error
# -----------------------------------------------------------------------------

def matched_recovery_error(
    particles,
    truth_locations: Tensor,
    truth_amplitudes: Tensor,
    truth_signs: Tensor,
    activity_threshold: float = 1e-3,
) -> dict:
    """Match active particles to ground-truth spikes (greedy nearest-neighbour)
    and return the L^1 recovery error in amplitudes and positions.

    Returns a dict with keys
      - `amp_err`     : sum_j |hat a_j - a_j^*|
      - `pos_err`     : sum_j ||hat x_j - x_j^*||_2
      - `total_err`   : amp_err + pos_err
      - `matched_idx` : indices of the matched active particles (into the active list)
      - `n_active`    : number of active particles
    """
    a = particles.amplitudes.detach()
    mask = a > activity_threshold
    n_active = int(mask.sum())

    if n_active == 0:
        # No active particles — every spike is unmatched
        truth_amp_signed = truth_signs.float() * truth_amplitudes
        amp_err = float(truth_amp_signed.abs().sum())
        pos_err = 0.0  # ill-defined — return amp_err only
        return dict(amp_err=amp_err, pos_err=pos_err, total_err=amp_err,
                    matched_idx=[], n_active=0)

    pos = particles.positions[mask].detach()                   # (n_active, d)
    amp_signed = (particles.amplitudes[mask].detach()
                  * particles.signs[mask].float())             # (n_active,)

    if truth_locations.dim() == 1:
        truth_locations = truth_locations.unsqueeze(-1)
    truth_amp_signed = truth_signs.float() * truth_amplitudes  # (k,)

    used = torch.zeros(n_active, dtype=torch.bool)
    matched_idx = []
    amp_err = 0.0
    pos_err = 0.0
    for j in range(truth_locations.shape[0]):
        target_pos = truth_locations[j]
        target_amp = float(truth_amp_signed[j])
        dist = (pos - target_pos.unsqueeze(0)).norm(dim=-1)    # (n_active,)
        dist = torch.where(used, torch.full_like(dist, float("inf")), dist)
        idx = int(dist.argmin())
        matched_idx.append(idx)
        used[idx] = True
        pos_err += float(dist[idx])
        amp_err += float(abs(float(amp_signed[idx]) - target_amp))

    return dict(amp_err=amp_err, pos_err=pos_err, total_err=amp_err + pos_err,
                matched_idx=matched_idx, n_active=n_active)


# -----------------------------------------------------------------------------
# Cluster-aggregated recovery error
# -----------------------------------------------------------------------------

def cluster_recovery_error(
    particles, truth_locs, truth_amps, truth_signs,
    cluster_radius, activity_threshold=1e-3,
):
    r"""Recovery error after aggregating active particles within a `cluster_radius` of each truth spike.

    For each ground-truth spike $(x_j^\star, a_j^\star, s_j^\star)$:
    - aggregate the *signed* amplitudes of all active particles within
      `cluster_radius` of $x_j^\star$;
    - record the amplitude-weighted mean position as the cluster centroid;
    - compute the per-spike $|\hat a_j - a_j^\star|$ and $|\hat x_j - x_j^\star|$.

    Particles outside every truth-spike cluster contribute to `spurious_mass`.

    This is the natural metric when CPGD spreads the recovery of each spike
    over many active particles (typical in continuous BLASSO with Gaussian
    features).
    """
    a = particles.amplitudes.detach()
    mask = a > activity_threshold
    if mask.sum() == 0:
        truth_amp_signed = truth_signs.float() * truth_amps
        return dict(amp_err=float(truth_amp_signed.abs().sum()), pos_err=0.0,
                    total_err=float(truth_amp_signed.abs().sum()),
                    spurious_mass=0.0, cluster_amps=[], cluster_pos=[])

    pos = (particles.positions[mask].detach().squeeze(-1)
           if particles.positions.shape[1] == 1
           else particles.positions[mask].detach())
    amp_signed = (particles.amplitudes[mask] * particles.signs[mask].float()).detach()

    if truth_locs.dim() > 1:
        truth_locs_squeezed = truth_locs.squeeze(-1) if truth_locs.shape[-1] == 1 else truth_locs
    else:
        truth_locs_squeezed = truth_locs
    truth_amp_signed = truth_signs.float() * truth_amps

    cluster_amps, cluster_pos, claimed = [], [], torch.zeros_like(amp_signed, dtype=torch.bool)
    pos_err, amp_err = 0.0, 0.0
    for j in range(truth_locs_squeezed.shape[0]):
        d = ((pos - truth_locs_squeezed[j]).abs()
             if pos.dim() == 1
             else (pos - truth_locs_squeezed[j].unsqueeze(0)).norm(dim=-1))
        in_cluster = (d <= cluster_radius) & ~claimed
        if in_cluster.sum() > 0:
            cluster_a = float(amp_signed[in_cluster].sum())
            tot_abs_a = float(amp_signed[in_cluster].abs().sum()) + 1e-12
            cluster_p = (float((pos[in_cluster] * amp_signed[in_cluster].abs()).sum() / tot_abs_a)
                         if pos.dim() == 1 else 0.0)
            claimed = claimed | in_cluster
        else:
            cluster_a = 0.0
            cluster_p = float(truth_locs_squeezed[j])
        cluster_amps.append(cluster_a)
        cluster_pos.append(cluster_p)
        amp_err += abs(cluster_a - float(truth_amp_signed[j]))
        pos_err += abs(cluster_p - float(truth_locs_squeezed[j]))

    spurious_mass = float(amp_signed[~claimed].abs().sum())
    return dict(amp_err=amp_err, pos_err=pos_err,
                total_err=amp_err + pos_err,
                spurious_mass=spurious_mass,
                cluster_amps=cluster_amps, cluster_pos=cluster_pos)
