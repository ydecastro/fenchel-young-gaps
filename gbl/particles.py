"""Particles and retraction operators (CPGD-style discrete signed measures).

Adapted from the FastPart-2024 codebase: `notebook_python/experiments_fastpart_nn_california.ipynb`.

A `Particles` instance encodes a discrete signed measure
    mu = sum_i  signs_i * h(weights_i) * delta_{positions_i}
with `signs_i in {-1, +1}` fixed at construction, `weights_i >= 0` learnable
through a (typically multiplicative) update, and `positions_i in R^d`
learnable through a Euclidean update. The reparametrisation `a_i = h(r_i)`
allows a `MirrorRetraction` to act multiplicatively on `weights` while
keeping `amplitudes = h(weights)` (so the cone of nonneg amplitudes is
respected by construction).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import Module, Parameter


class Retraction:
    """Default Euclidean retraction: positions += dpositions, weights += dweights."""

    def __call__(
        self,
        positions: Tensor,
        weights: Tensor,
        dpositions: Tensor,
        dweights: Tensor,
    ) -> None:
        positions += dpositions
        weights += dweights


class MirrorRetraction(Retraction):
    """Mirror (multiplicative) retraction on weights: weights *= exp(dweights).

    This is the natural retraction for a KL-geometry on the cone of nonneg
    amplitudes. Combined with the position Euclidean step it implements
    Conic Particle Gradient Descent (CPGD). It also coincides with the
    Mirror Alignment update on the regularizer side for the TV norm
    (cf. paper §1.1, bullet "Mirror Alignment").
    """

    def __call__(
        self,
        positions: Tensor,
        weights: Tensor,
        dpositions: Tensor,
        dweights: Tensor,
    ) -> None:
        positions += dpositions
        weights *= torch.exp(dweights)


class Particles(Module):
    r"""A discrete signed measure mu = sum_i signs_i * h(r_i) * delta_{theta_i}."""

    def __init__(
        self,
        h: Callable[[Float[Tensor, " n_particles"]], Float[Tensor, " n_particles"]],
        positions: Float[Tensor, "n_particles dim"],
        weights: Float[Tensor, " n_particles"],
        signs: Int[Tensor, " n_particles"],
    ):
        n_particles, _ = positions.shape
        assert weights.shape == (n_particles,)
        assert signs.shape == (n_particles,)
        super().__init__()
        self.h = h
        self.positions = Parameter(positions)
        self.weights = Parameter(weights)
        self.signs = signs
        self.gradient_evaluation: Optional[object] = None

    @staticmethod
    def from_randn(
        h: Callable,
        dimension: int,
        n_particles: int,
        position_sigma: float = 1.0,
        weight_sigma: float = 1.0,
        **kwargs: Any,
    ) -> "Particles":
        positions = position_sigma * torch.randn((n_particles, dimension), **kwargs)
        weights = weight_sigma * torch.randn(n_particles, **kwargs).abs()
        signs = 2.0 * torch.randint(2, size=(n_particles,)).float() - 1.0
        return Particles(h, positions, weights, signs)

    @property
    def amplitudes(self) -> Float[Tensor, " n_particles"]:
        """The non-negative amplitudes a_i = h(r_i)."""
        return self.h(self.weights)

    @property
    def amplitudes_deriv(self) -> Float[Tensor, " n_particles"]:
        """The Jacobian of h at weights, used by ConicOptimizer for chain-rule scaling."""
        with torch.enable_grad():
            amps = self.amplitudes
            return torch.autograd.grad(
                amps, self.weights, grad_outputs=torch.ones_like(amps),
            )[0]

    def __len__(self) -> int:
        return self.positions.size(0)

    def __getitem__(self, idx: Any) -> "Particles":
        if isinstance(idx, int):
            idx = [idx]
        return Particles(
            self.h,
            self.positions[idx, ...].detach().clone(),
            self.weights[idx, ...].detach().clone(),
            self.signs[idx, ...].clone(),
        )

    def tv_norm(self) -> Float[Tensor, "1"]:
        return self.amplitudes.abs().sum()

    def signed_amplitudes(self) -> Float[Tensor, " n_particles"]:
        return self.signs * self.amplitudes

    def clone(self, *, detach: bool = True) -> "Particles":
        if detach:
            positions = self.positions.detach().clone()
            weights = self.weights.detach().clone()
        else:
            positions = self.positions.clone()
            weights = self.weights.clone()
        return Particles(self.h, positions, weights, self.signs.clone())

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n={len(self)}, "
            f"TV={float(self.tv_norm()):.4f})"
        )
