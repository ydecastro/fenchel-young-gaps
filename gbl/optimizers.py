"""CPGD-style optimizers (full-batch and stochastic).

Two classes are provided:

  ConicOptimizer   - generic CPGD with parameters (alpha, beta) where
                     alpha is the amplitude (mirror) step size and beta
                     is the position (Euclidean) step size, matching the
                     FastPart 2024 convention. Uses `Jp` and `Jp_grad`
                     directly when available, else falls back to autograd.

  NNOptimizer      - the California-Housing-style update used in
                     FastPart 2024:
                       weights *= exp(-eta_w * (signs * Jp + kappa * (1 - signs)))
                       positions -= eta * (signs * Jp_grad)
                       positions /= max(1, ||positions||) (project to unit ball)
                     The (eta_w, eta) here equal (alpha, beta) of ConicOptimizer
                     with the convention eta_w = alpha = eta_input,
                     eta = beta = alpha / d_input.

  StochasticNNOptimizer - mini-batch variant of NNOptimizer that re-builds
                     a new `NNBlasso` problem on each batch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .particles import Particles, Retraction
from .problems import NNBlasso


@dataclass
class Optimizer(ABC):
    particles: Particles

    def zero_grad(self) -> None:
        self.particles.zero_grad()

    @abstractmethod
    def step(self) -> None: ...


@dataclass
class ConicOptimizer(Optimizer):
    """Generic CPGD with retraction.

    Update rule (positive measure):
      weights  *= exp(-alpha * Jp)            (mirror step)
      positions -= beta * Jp_grad             (Euclidean step)

    For a signed measure, signs are fixed at construction; the gradient
    seen by each particle is `signs_i * Jp_i` (resp. `Jp_grad`).
    """

    alpha: float
    beta: float
    retraction: Retraction = field(default_factory=Retraction)

    def step(self) -> None:
        positions = self.particles.positions
        weights = self.particles.weights
        amplitudes = self.particles.amplitudes
        amplitudes_deriv = self.particles.amplitudes_deriv

        ev = self.particles.gradient_evaluation
        assert ev is not None, "Call problem(particles).Jp_backward (or .backward()) first"

        with torch.no_grad():
            Jp = ev.Jp.detach()
            Jp_grad = ev.Jp_grad.detach()
            signs = self.particles.signs

            # Mirror-style amplitude gradient: alpha * (signs * Jp + kappa * (1 - signs))
            # Reduces to alpha * Jp for non-negative measures (signs == +1).
            kappa = ev.problem.kappa
            grad_weights = self.alpha * (signs * Jp + kappa * (1.0 - signs))
            grad_positions = self.beta * (signs.unsqueeze(-1) * Jp_grad)

            self.retraction(positions, weights, -grad_positions, -grad_weights)


@dataclass
class NNOptimizer(Optimizer):
    """Full-batch CPGD for affine-ReLU neuron BLASSO (matches FastPart-2024).

    The update reads, with `eta_w` the amplitude (mirror) step size and `eta`
    the position step size:
        grad_omega    = signs * Jp + kappa * (1 - signs)         # signed mirror grad
        weights      *= exp(-eta_w * grad_omega)
        weights       = clamp(weights, min=min_weight)
        g             = signs * Jp_grad
        positions    -= eta * g
        positions    /= max(1, ||positions||)                    # unit-ball projection
    """

    problem: NNBlasso = None
    eta: float = 0.01
    eta_w: float = 0.1
    min_weight: float = 1e-6
    project_unit_ball: bool = True

    def step(self) -> float:
        positions = self.particles.positions
        weights = self.particles.weights
        signs = self.particles.signs

        ev = self.problem(self.particles)

        with torch.no_grad():
            jp = ev.Jp.clone()
            jp_grad = ev.Jp_grad.clone()

            grad_omega = signs * jp + self.problem.kappa * (1.0 - signs)
            weights.mul_(torch.exp(-self.eta_w * grad_omega))
            weights.clamp_(min=self.min_weight)

            g = signs.unsqueeze(-1) * jp_grad
            positions.sub_(self.eta * g)
            if self.project_unit_ball:
                positions.div_(positions.norm(dim=1, keepdim=True).clamp(min=1.0))

        return float(ev.loss)


@dataclass
class StochasticNNOptimizer(Optimizer):
    """Mini-batch stochastic CPGD for `NNBlasso`.

    On each step, draws a fresh batch of `batch_size` indices from the dataset
    held by the parent `problem`, builds a small `NNBlasso` on that batch,
    and applies the same multiplicative-weight + Euclidean-position update.
    The averaged kernel (and scalar products) is the standard FastPart noisy
    estimator. See `gbl.problems.NNBlasso` for the underlying expressions.
    """

    problem: NNBlasso = None
    eta: float = 0.01
    eta_w: float = 0.1
    batch_size: int = 256
    min_weight: float = 1e-6
    project_unit_ball: bool = True
    rng: Any = field(default_factory=lambda: torch.Generator())

    def __post_init__(self):
        if self.problem is None:
            raise ValueError("`problem` must be provided")
        # The full-data problem is held; mini-batch replicas are made on the fly
        self.n_total = self.problem.inputs.shape[0]

    def get_batch(self) -> NNBlasso:
        idx = torch.randint(0, self.n_total, (self.batch_size,), generator=self.rng)
        mini = NNBlasso(
            kappa=self.problem.kappa,
            y=self.problem.y[idx],
            inputs=self.problem.inputs[idx, :],
            R_X=self.problem.R_X,
        )
        return mini

    def step(self) -> float:
        positions = self.particles.positions
        weights = self.particles.weights
        signs = self.particles.signs

        mini = self.get_batch()
        ev = mini(self.particles)

        with torch.no_grad():
            jp = ev.Jp.clone()
            jp_grad = ev.Jp_grad.clone()

            grad_omega = signs * jp + mini.kappa * (1.0 - signs)
            weights.mul_(torch.exp(-self.eta_w * grad_omega))
            weights.clamp_(min=self.min_weight)

            g = signs.unsqueeze(-1) * jp_grad
            positions.sub_(self.eta * g)
            if self.project_unit_ball:
                positions.div_(positions.norm(dim=1, keepdim=True).clamp(min=1.0))

        return float(ev.loss)
