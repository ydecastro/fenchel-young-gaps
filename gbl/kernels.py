"""Kernels and the Kernel-Switch constant.

This module provides:
  - `gaussian_kernel(u, sigma)`        : the population model kernel
                                          K_model(z) = (phi * phi)(z) for
                                          phi(s) = exp(-||s||^2/(2 sigma^2)),
                                          which equals sigma sqrt(pi) exp(-||z||^2/(4 sigma^2))
                                          on R (1D, the synthetic setting).
  - `sinc4_kernel(u, kappa)`           : the sinc-4 pivot kernel
                                          Psi_kappa(u) = prod_j sinc^4(u_j/(4 kappa)).
  - `compute_C_switch_squared(...)`    : the kernel-switch constant
                                          C_switch^2 = sup_omega (psi_hat / K_hat),
                                          computed via FFT on a fine grid.

These are used by the notebook's Phase 3 (kernel-switch demonstration).
The sinc-4 pivot is the canonical LPC kernel of the paper (cf. §3.6); the
Gaussian model kernel does not satisfy LPC directly, so its dual certificate
is constructed via the Kernel Switch through the sinc-4 pivot.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from torch import Tensor


# -----------------------------------------------------------------------------
# Closed-form kernels (1D)
# -----------------------------------------------------------------------------

def gaussian_kernel(u: Tensor, sigma: float) -> Tensor:
    r"""Population model kernel for Gaussian features (1D).

    For the feature map  phi(s) = exp(-||s||^2/(2 sigma^2)) and
    rho_s = Lebesgue, the model kernel is the autocorrelation
        K_model(z) = (phi * phi)(z) = sigma sqrt(pi) exp(-||z||^2 / (4 sigma^2)).

    Returns a tensor of the same shape as `u` (broadcast-friendly for grids).
    """
    z2 = u.pow(2)
    if u.dim() > 1:
        z2 = z2.sum(dim=-1)
    return float(math.sqrt(math.pi) * sigma) * torch.exp(-z2 / (4.0 * sigma ** 2))


def sinc4_kernel(u: Tensor, kappa: float, d: int = 1) -> Tensor:
    r"""Sinc-4 pivot kernel  Psi_kappa(u) = prod_j sinc^4(u_j / (4 kappa)).

    The numerical convention is sinc(t) = sin(pi t)/(pi t). The sinc^4
    factor's Fourier transform is supported on |omega_j| <= 1/(2 kappa)
    (equivalently the bandlimited cube B_kappa of paper §3.6).

    For `u` of shape (N,) [d=1] returns a (N,) tensor.
    For `u` of shape (N, d) returns a (N,) tensor (product over coordinates).
    """
    if u.dim() == 1:
        u = u.unsqueeze(-1)
    a = 1.0 / (4.0 * kappa)
    arg = u * a                                # (..., d)
    pi_arg = math.pi * arg
    s = torch.where(
        arg.abs() < 1e-12,
        torch.ones_like(pi_arg),
        torch.sin(pi_arg) / pi_arg,
    )
    return torch.prod(s.pow(4), dim=-1)


# -----------------------------------------------------------------------------
# Fourier-ratio kernel-switch constant
# -----------------------------------------------------------------------------

def _fft_freqs(u: Tensor) -> Tensor:
    """Centred angular frequencies (rad/length) corresponding to a uniform grid `u`."""
    n = u.numel()
    du = float(u[1] - u[0])
    omega = 2.0 * math.pi * torch.fft.fftshift(torch.fft.fftfreq(n, d=du))
    return omega


def compute_C_switch_squared(
    kappa: float,
    sigma_phi: float,
    n_grid: int = 8192,
    u_max: float = 50.0,
    eps_psi: float = 1e-10,
) -> Tuple[float, Tensor, Tensor, Tensor]:
    r"""Numerically estimate the kernel-switch constant C_switch^2 (1D).

    Uses
        C_switch^2 = sup_{omega in B_kappa}  hat_Psi(omega) / hat_K_model(omega) ,
    where the supremum is restricted to the support of `hat_Psi`
    (the bandlimited cube B_kappa = [-1/(2 kappa), 1/(2 kappa)] in 1D
    under the standard sinc convention).

    The computation discretises u in [-u_max, u_max] with `n_grid` samples,
    computes the FFTs of Psi_kappa and K_model, and evaluates the ratio on
    the bins where `hat_Psi > eps_psi * max(hat_Psi)`.

    Returns
    -------
    C_switch_sq : float
    omega       : Tensor   (centred angular frequencies)
    psi_hat     : Tensor   (real-valued Fourier transform of Psi_kappa)
    K_hat       : Tensor   (real-valued Fourier transform of K_model)
    """
    u = torch.linspace(-u_max, u_max, n_grid)
    du = float(u[1] - u[0])

    psi = sinc4_kernel(u, kappa).double()
    K = gaussian_kernel(u, sigma_phi).double()

    # Continuous Fourier transform via FFT, with the (-i omega) convention:
    psi_hat = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(psi))) * du
    K_hat = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(K))) * du

    # Both kernels are real and even -> Fourier transforms are real
    psi_hat = psi_hat.real
    K_hat = K_hat.real

    omega = _fft_freqs(u)

    # Restrict to the support of psi_hat
    psi_max = float(psi_hat.abs().max())
    mask = psi_hat.abs() > eps_psi * psi_max
    if mask.sum() == 0:
        return float("inf"), omega, psi_hat, K_hat

    ratio = (psi_hat[mask].clamp(min=0.0)) / (K_hat[mask].clamp(min=1e-30))
    C_switch_sq = float(ratio.max())
    return C_switch_sq, omega, psi_hat, K_hat


def fisher_rao_step_sinc4(kappa: float) -> float:
    r"""Pivot Fisher-Rao infinitesimal step for the sinc-4 kernel:
        d_pivot(x, x') = (1 / (2 sqrt(3) kappa)) * ||x - x'||_2.
    Returns the inverse 2 sqrt(3) kappa, i.e.\ the Euclidean distance per
    pivot-Fisher-Rao unit (cf. paper eq:sinc4_fisher_rao).
    """
    return 2.0 * math.sqrt(3.0) * kappa


def lpc_separation_sinc4(k_spikes: int, d: int = 1) -> float:
    r"""Sinc-4 LPC minimal pivot separation (paper eq:sinc4_LPC):
        Delta_0 = 42.66 * k^{1/4} * d^{7/4}     (in pivot-Fisher-Rao units).
    """
    return 42.66 * (k_spikes ** 0.25) * (d ** 1.75)


def euclidean_separation_sinc4(k_spikes: int, kappa: float, d: int = 1) -> float:
    r"""Equivalent Euclidean minimal separation for sinc-4 LPC at scale `kappa`.

    Combines `lpc_separation_sinc4` with `fisher_rao_step_sinc4`:
        Delta^{eucl} = 2 sqrt(3) * kappa * Delta_0.
    """
    return fisher_rao_step_sinc4(kappa) * lpc_separation_sinc4(k_spikes, d)
