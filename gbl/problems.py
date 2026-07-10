"""BLASSO problem classes (squared-L2 fidelity) and concrete instantiations.

Two concrete instantiations are provided:

* `GaussianBlasso`  — synthetic 1D super-resolution with Gaussian feature
                      map  phi_x(s) = exp(-||s-x||^2 / (2 sigma^2))
                      sampled at design points s_1,...,s_N. Positions
                      `theta_i = x_i in R^d` (d typically = 1) are
                      *unconstrained*.

* `NNBlasso`        — affine ReLU neurons  phi_theta(x) = ReLU(<w,x>+b)
                      with `theta = (w, b) in R^{d+1}`. Positions are
                      *projected* on the unit sphere by the optimizer.
                      Identical to the FastPart-2024 California setup.

In both, the L2 cost is
    proximity(mu) = (1/2) (a^T K a) - a^T sp + offset,
where  a = signed_amplitudes,  K = (1/N) Phi^T Phi  is the empirical model
kernel,  sp_i = (1/N) <y, phi_{theta_i}>  is the inner product against the
observations, and `offset = (1/2) ||y||^2 / N` to align with the squared-MSE.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar, Generic, TypeVar

import torch
from jaxtyping import Float
from torch import Tensor

from .particles import Particles

TProblem = TypeVar("TProblem", bound="BlassoProblem", covariant=True)
Observation = TypeVar("Observation")


@dataclass
class ModelEvaluation(Generic[TProblem]):
    """Cached lazy evaluation of a BLASSO problem at a given particle set."""

    problem: TProblem
    particles: Particles

    def set_particles_gradient(self, dweights, dpositions) -> None:
        self.particles.weights.grad = dweights
        self.particles.positions.grad = dpositions
        self.particles.gradient_evaluation = self

    @cached_property
    def proximity(self) -> Float[Tensor, "1"]:
        raise NotImplementedError

    @cached_property
    def lasso(self) -> Float[Tensor, "1"]:
        return self.problem.kappa * self.particles.tv_norm()

    @cached_property
    def loss(self) -> Float[Tensor, "1"]:
        return self.proximity + self.lasso

    @cached_property
    def amplitudes(self) -> Float[Tensor, " n_particles"]:
        return self.particles.amplitudes

    @cached_property
    def amplitudes_deriv(self) -> Float[Tensor, " n_particles"]:
        return self.particles.amplitudes_deriv

    @cached_property
    def Jp(self) -> Float[Tensor, " n_particles"]:
        raise NotImplementedError

    @cached_property
    def Jp_grad(self) -> Float[Tensor, "n_particles dim"]:
        raise NotImplementedError

    @cached_property
    def Jp_backward(self) -> None:
        self.set_particles_gradient(
            dweights=self.amplitudes_deriv * self.Jp,
            dpositions=self.amplitudes.unsqueeze(-1) * self.Jp_grad,
        )

    def backward(self) -> None:
        self.Jp_backward

    def item(self) -> float:
        return float(self.loss)


@dataclass
class BlassoProblem(ABC):
    """Base class: BLASSO with regularizer kappa * ||mu||_TV."""

    Evaluation: ClassVar[type[ModelEvaluation]] = ModelEvaluation
    kappa: float

    def __call__(self, particles: Particles) -> ModelEvaluation:
        return self.Evaluation(self, particles)


@dataclass
class L2Blasso(BlassoProblem, Generic[Observation]):
    """L2-fidelity BLASSO: F(h) = (1/2N) ||h - y||^2 + offset_to_align_with_sq_mse."""

    y: Observation
    loss_offset: float = field(init=False, default=0.0)

    @dataclass
    class Evaluation(ModelEvaluation["L2Blasso"]):
        @cached_property
        @abstractmethod
        def kernel(self) -> Float[Tensor, "n_particles n_particles"]:
            ...

        @cached_property
        def _kernel(self) -> Float[Tensor, "n_particles n_particles"]:
            return self.kernel

        @cached_property
        @abstractmethod
        def scalar_products(self) -> Float[Tensor, " n_particles"]:
            ...

        @cached_property
        def signed_amps(self) -> Float[Tensor, " n_particles"]:
            return self.amplitudes * self.particles.signs

        @cached_property
        def l2_cost(self) -> Float[Tensor, "1"]:
            a = self.signed_amps
            quad = 0.5 * a @ (self._kernel @ a)
            lin = a @ self.scalar_products
            return quad - lin + self.problem.loss_offset

        @cached_property
        def proximity(self) -> Float[Tensor, "1"]:
            return self.l2_cost

    def L_F_rescaled(self, particles: Particles, tau: float) -> float:
        r"""L_F(Phi mu || tau * nabla F(Phi mu)) for the (1/2N) ||.-y||^2 fidelity.

        Closed form: F(h) = (1/2N) ||h - y||^2 has F^*(g) = (N/2) ||g||^2 + <g, y>
        and nabla F(Phi mu) = (Phi mu - y)/N; algebra yields the identity
            L_F(Phi mu, tau * nabla F(Phi mu)) = F(Phi mu) * (1 - tau)^2.
        Vanishes iff tau = 1 (i.e. the un-rescaled Mirror-Alignment dual is
        already dual-feasible).
        """
        with torch.no_grad():
            return float(self(particles).proximity.detach()) * (1.0 - tau) ** 2


# -----------------------------------------------------------------------------
# Synthetic 1D Gaussian super-resolution
# -----------------------------------------------------------------------------

@dataclass
class GaussianBlasso(L2Blasso[Tensor]):
    r"""1D super-resolution with a Gaussian feature map phi_x(s) = exp(-||s-x||^2/(2 sigma^2)).

    Inputs (positional, set after construction via __post_init__-style wrappers):
      - design_points: Tensor of shape (N, d), the sample locations s_k
      - sigma_phi:     bandwidth sigma_phi of the Gaussian feature map
      - y:             observations Tensor of shape (N,)
      - kappa:         the BLASSO regularization weight lambda
    """

    design_points: Tensor = field(default=None)
    sigma_phi: float = 1.0

    def __post_init__(self):
        # Ensure design_points is a 2D tensor (N, d)
        if self.design_points is None:
            raise ValueError("design_points must be provided")
        if self.design_points.dim() == 1:
            self.design_points = self.design_points.unsqueeze(-1)
        N = self.design_points.shape[0]
        # Loss offset to align with the (1/2N)||y||^2 normalization used by the L2 cost
        self.loss_offset = 0.5 * (self.y ** 2).sum() / N

    @dataclass
    class Evaluation(L2Blasso.Evaluation):
        @cached_property
        def phi(self) -> Float[Tensor, "n_particles n_samples"]:
            r"""Gaussian features  phi[i,k] = exp(-||s_k - theta_i||^2 / (2 sigma^2))."""
            theta = self.particles.positions          # (P, d)
            s = self.problem.design_points            # (N, d)
            d2 = ((theta.unsqueeze(1) - s.unsqueeze(0)) ** 2).sum(dim=-1)  # (P, N)
            return torch.exp(-d2 / (2.0 * self.problem.sigma_phi ** 2))

        @cached_property
        def kernel(self) -> Float[Tensor, "n_particles n_particles"]:
            return self.phi @ self.phi.T / self.phi.shape[1]

        @cached_property
        def scalar_products(self) -> Float[Tensor, " n_particles"]:
            return (self.phi @ self.problem.y.view(-1)) / self.phi.shape[1]

        @cached_property
        def Jp(self) -> Float[Tensor, " n_particles"]:
            return (
                self.problem.kappa
                + self._kernel @ self.signed_amps
                - self.scalar_products
            )

        @cached_property
        def Jp_grad(self) -> Float[Tensor, "n_particles dim"]:
            r"""Position gradient  d/dtheta J' at every particle.

            With phi = exp(-||s-theta||^2/(2 sigma^2)), the relevant derivative is
                d_theta phi(theta_i, s_k) = (s_k - theta_i)/sigma^2 * phi[i,k]
            and J'(theta) = (kappa) + (1/N) sum_k r_k phi(theta, s_k) - (1/N) sum_k y_k phi(theta, s_k)
            so grad_theta J'(theta_i)
              = (1/(N sigma^2)) sum_k (Phi mu(s_k) - y_k) (s_k - theta_i) phi[i,k].
            """
            theta = self.particles.positions          # (P, d)
            s = self.problem.design_points            # (N, d)
            P, d = theta.shape
            N = s.shape[0]

            phi = self.phi                              # (P, N)
            a_signed = self.signed_amps                 # (P,)
            yhat = (a_signed[:, None] * phi).sum(dim=0)  # (N,)
            residual = yhat - self.problem.y.view(-1)   # (N,)

            # Vectorised: (s_k - theta_i) shape (P, N, d).  We sum over k.
            diff = s.unsqueeze(0) - theta.unsqueeze(1)  # (P, N, d)
            weight = phi * residual.unsqueeze(0)        # (P, N)
            grad = (weight.unsqueeze(-1) * diff).sum(dim=1) / (N * self.problem.sigma_phi ** 2)
            return grad

    def eta_on_grid(
        self, particles: Particles, grid: Tensor, lam: float
    ) -> Tensor:
        r"""Evaluate the pre-dual certificate eta(x) = -(Phi^* h)(x)/lam on a grid.

        Here  h = nabla F(Phi mu) = (1/N) (Phi mu - y)   (the LSQ residual),
        so   (Phi^* h)(x) = (1/N) sum_k h_k phi(x, s_k)
        and  eta(x) = -(Phi^* h)(x) / lam.

        Equivalently (and what the code uses), eta(theta) = -J'(theta)/lam
        plus the constant kappa/lam that comes from the lasso term in J':
            eta(x) = - (J'(x) - kappa) / lam.
        We use the explicit Phi^* h form below for clarity.
        """
        if grid.dim() == 1:
            grid = grid.unsqueeze(-1)
        ev = self(particles)
        a_signed = ev.signed_amps                                # (P,)
        s = self.design_points                                    # (N, d)
        N = s.shape[0]
        # phi at grid (G, N): phi[g, k] = exp(-||g - s_k||^2/(2 sig^2))
        d2 = ((grid.unsqueeze(1) - s.unsqueeze(0)) ** 2).sum(dim=-1)
        phi_g = torch.exp(-d2 / (2.0 * self.sigma_phi ** 2))     # (G, N)
        # phi at particles (P, N)
        phi_p = ev.phi                                            # (P, N)
        # Phi mu at samples
        yhat = (a_signed[:, None] * phi_p).sum(dim=0)             # (N,)
        h = (yhat - self.y.view(-1)) / N                          # (N,)
        Phi_star_h = phi_g @ h                                    # (G,)
        return -Phi_star_h / lam


# -----------------------------------------------------------------------------
# California-Housing-style Affine ReLU Neuron BLASSO
# -----------------------------------------------------------------------------

@dataclass
class NNBlasso(L2Blasso[Tensor]):
    """BLASSO over affine ReLU neurons  phi_theta(x) = ReLU(<w, x> + b).

    `theta = (w, b) in R^{d+1}` lives on the unit sphere (enforced by the
    optimizer's projection step), and `inputs` are the design covariates.
    Identical to the FastPart-2024 California Housing setup.
    """

    inputs: Tensor = field(default=None)
    R_X: float = 1.0

    def __post_init__(self):
        if self.inputs is None:
            raise ValueError("inputs must be provided")
        n, d = self.inputs.shape
        self.input_dim = d
        self.loss_offset = 0.5 * (self.y ** 2).mean()

    @dataclass
    class Evaluation(L2Blasso.Evaluation):
        @cached_property
        def phi(self) -> Float[Tensor, "n_particles n_samples"]:
            w = self.particles.positions[:, :-1]
            b = self.particles.positions[:, -1:]
            return torch.relu(w @ self.problem.inputs.T + b)

        @cached_property
        def prediction(self) -> Float[Tensor, "n_samples 1"]:
            a_signed = self.signed_amps
            return (a_signed[:, None] * self.phi).sum(dim=0).unsqueeze(-1)

        @cached_property
        def kernel(self) -> Float[Tensor, "n_particles n_particles"]:
            return self.phi @ self.phi.T / self.phi.shape[1]

        @cached_property
        def scalar_products(self) -> Float[Tensor, " n_particles"]:
            return (self.phi @ self.problem.y).squeeze() / self.phi.shape[1]

        @cached_property
        def proximity(self) -> Float[Tensor, "1"]:
            return 0.5 * ((self.problem.y - self.prediction) ** 2).mean()

        @cached_property
        def Jp(self) -> Float[Tensor, " n_particles"]:
            return (
                self.problem.kappa
                + self.kernel @ self.signed_amps
                - self.scalar_products
            )

        @cached_property
        def Jp_grad(self) -> Float[Tensor, "n_particles input_dim+1"]:
            relu_mask = (self.phi > 0).float()
            a_signed = self.signed_amps
            yhat = (a_signed[:, None] * self.phi).sum(dim=0)
            residual = yhat - self.problem.y.squeeze()
            grad_w = (relu_mask * residual[None, :]) @ self.problem.inputs / relu_mask.shape[1]
            grad_b = (relu_mask * residual[None, :]).sum(dim=1, keepdim=True) / relu_mask.shape[1]
            return torch.cat([grad_w, grad_b], dim=1)


# -----------------------------------------------------------------------------
# Logistic GBL with Gaussian features (1D worked example, paper §3.6)
# -----------------------------------------------------------------------------

@dataclass
class LogisticGaussianBlasso(BlassoProblem):
    r"""Empirical logistic GBL with Gaussian feature map (1D, scalar input).

    Observations: design points  s_i in R^d,  binary labels  y_i in {-1, +1}.
    Feature:      phi(s, x) = exp(-||s-x||^2 / (2 sigma^2)).
    Loss:         F_n(h) = (1/N) sum_i  log(1 + exp(-y_i h_i)).
    Penalty:      kappa * ||mu||_TV.

    Under the well-specified model assumption  P(y=1 | s) = sigma(g^*(s)),
    where  g^* = Phi mu^*  is the noiseless logit, this is the population
    logistic GBL of paper §3.6 (Corollary 6 / Proposition 5 setup).
    """

    design_points: Tensor = field(default=None)
    y: Tensor = field(default=None)
    sigma_phi: float = 1.0

    def __post_init__(self):
        if self.design_points is None or self.y is None:
            raise ValueError("design_points and y must both be provided")
        if self.design_points.dim() == 1:
            self.design_points = self.design_points.unsqueeze(-1)

    @dataclass
    class Evaluation(ModelEvaluation["LogisticGaussianBlasso"]):
        @cached_property
        def phi(self) -> Float[Tensor, "n_particles n_samples"]:
            theta = self.particles.positions          # (P, d)
            s = self.problem.design_points            # (N, d)
            d2 = ((theta.unsqueeze(1) - s.unsqueeze(0)) ** 2).sum(dim=-1)  # (P, N)
            return torch.exp(-d2 / (2.0 * self.problem.sigma_phi ** 2))

        @cached_property
        def signed_amps(self) -> Float[Tensor, " n_particles"]:
            return self.particles.signs * self.amplitudes

        @cached_property
        def h_pred(self) -> Float[Tensor, " n_samples"]:
            r"""h_i = (Phi mu)(s_i) = sum_j a_j s_j phi(s_i, x_j)."""
            return (self.signed_amps[:, None] * self.phi).sum(dim=0)

        @cached_property
        def proximity(self) -> Float[Tensor, "1"]:
            r"""Empirical logistic loss F_n(Phi mu) = mean_i  softplus(-y_i h_i)."""
            yh = self.problem.y * self.h_pred
            return torch.nn.functional.softplus(-yh).mean()

        @cached_property
        def grad_F(self) -> Float[Tensor, " n_samples"]:
            r"""nabla_h F_n(h)_i = -y_i sigma(-y_i h_i) / N."""
            yh = self.problem.y * self.h_pred
            return -self.problem.y * torch.sigmoid(-yh) / self.problem.y.numel()

        @cached_property
        def Jp(self) -> Float[Tensor, " n_particles"]:
            r"""J'(theta) = kappa + (Phi^* nabla F)(theta).

            Note: the 1/N factor lives inside `grad_F` already, so Phi^* is
            the unweighted sum  sum_i grad_F[i] * phi(s_i, theta).
            """
            return self.problem.kappa + self.phi @ self.grad_F

        @cached_property
        def Jp_grad(self) -> Float[Tensor, "n_particles dim"]:
            r"""grad_theta J'(theta_i) = sum_i  grad_F[i] * (s_i - theta) / sigma^2 * phi[i]."""
            grad_F = self.grad_F                        # (N,)
            phi = self.phi                              # (P, N)
            theta = self.particles.positions            # (P, d)
            s = self.problem.design_points              # (N, d)
            diff = s.unsqueeze(0) - theta.unsqueeze(1)  # (P, N, d)
            weight = phi * grad_F.unsqueeze(0)          # (P, N)
            grad = (weight.unsqueeze(-1) * diff).sum(dim=1) / (self.problem.sigma_phi ** 2)
            return grad

    def eta_on_grid(
        self, particles: Particles, grid: Tensor, lam: float,
    ) -> Tensor:
        r"""Pre-dual certificate eta(x) = -(Phi^* nabla F_n)(x) / lam on `grid`."""
        if grid.dim() == 1:
            grid = grid.unsqueeze(-1)
        ev = self(particles)
        s = self.design_points
        d2 = ((grid.unsqueeze(1) - s.unsqueeze(0)) ** 2).sum(dim=-1)
        phi_g = torch.exp(-d2 / (2.0 * self.sigma_phi ** 2))   # (G, N)
        Phi_star_nablaF = phi_g @ ev.grad_F                    # (G,)
        return -Phi_star_nablaF / lam

    def L_F_rescaled(self, particles: Particles, tau: float) -> float:
        r"""L_F(Phi mu || tau * nabla F_n(Phi mu)) for the empirical logistic loss.

        F_n(h) = (1/N) sum_i log(1 + exp(-y_i h_i)). With p_i := sigma(-y_i (Phi mu)_i),
        the rescaled dual satisfies -N y_i tilde h_i = tau p_i in [0,1], so
            F^*(tilde h) = (1/N) sum_i [ tau p_i log(tau p_i) + (1-tau p_i) log(1-tau p_i) ]
        (binary-entropy form, with 0 log 0 = 0). The identity
            L_F = F(Phi mu) + F^*(tilde h) - <Phi mu, tilde h>
        is non-negative and vanishes iff tau = 1.
        """
        with torch.no_grad():
            ev = self(particles)
            F_val = float(ev.proximity)
            h_pred = ev.h_pred                                # (N,)
            grad_F = ev.grad_F                                # nabla F_n(Phi mu)
            h_tilde = tau * grad_F
            N = self.y.numel()
            p = torch.sigmoid(-self.y * h_pred)                # in (0,1)
            tp = tau * p
            # F^*(tilde h) via binary entropy; xlogy handles 0 log 0 = 0.
            F_star = float(
                (torch.special.xlogy(tp, tp)
                 + torch.special.xlogy(1.0 - tp, 1.0 - tp)).sum() / N
            )
            inner = float((h_pred * h_tilde).sum())
            return max(0.0, F_val + F_star - inner)
