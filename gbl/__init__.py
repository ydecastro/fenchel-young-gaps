"""gbl: Generalized Beurling-Lasso companion code for the FYL_Gap paper.

This package provides the minimum machinery needed to illustrate the paper's
contributions on two BLASSO problems:

  1) a synthetic 1D Gaussian super-resolution problem (continuous setting,
     visualizable certificate), and
  2) a real-data fit on California Housing with affine ReLU neurons (finite-
     dimensional CPGD setting matching the FastPart 2024 notebook).

Sub-modules
-----------
particles   : Particles, Retraction, MirrorRetraction
problems    : ModelEvaluation, BlassoProblem, L2Blasso, GaussianBlasso, NNBlasso
optimizers  : Optimizer, ConicOptimizer, NNOptimizer
fy          : FYTracker, box_project, fy_loss components
utils       : initialization helpers, dataset loaders, ground-truth synthesis
plotting    : standard log-scale and certificate plotters
"""

from . import fy, kernels, optimizers, particles, plotting, problems, utils
from .fy import DualRescaling, FYRecord, FYTracker, mirror_alignment_check, print_mirror_alignment
from .kernels import (
    compute_C_switch_squared,
    euclidean_separation_sinc4,
    fisher_rao_step_sinc4,
    gaussian_kernel,
    lpc_separation_sinc4,
    sinc4_kernel,
)
from .optimizers import ConicOptimizer, NNOptimizer, Optimizer
from .particles import MirrorRetraction, Particles, Retraction
from .problems import (
    BlassoProblem,
    GaussianBlasso,
    L2Blasso,
    LogisticGaussianBlasso,
    ModelEvaluation,
    NNBlasso,
)
from .utils import (
    cluster_recovery_error,
    initialize_gaussian_particles,
    initialize_nn_particles,
    load_california_housing,
    matched_recovery_error,
    sample_logistic_gbl,
    synth_1d_problem,
)

__all__ = [
    # particles
    "Particles", "Retraction", "MirrorRetraction",
    # problems
    "ModelEvaluation", "BlassoProblem", "L2Blasso", "GaussianBlasso", "NNBlasso",
    "LogisticGaussianBlasso",
    # optimizers
    "Optimizer", "ConicOptimizer", "NNOptimizer",
    # fy
    "FYTracker", "FYRecord", "DualRescaling",
    "mirror_alignment_check", "print_mirror_alignment",
    # kernels
    "gaussian_kernel", "sinc4_kernel", "compute_C_switch_squared",
    "fisher_rao_step_sinc4", "lpc_separation_sinc4", "euclidean_separation_sinc4",
    # utils
    "initialize_nn_particles", "initialize_gaussian_particles",
    "load_california_housing", "synth_1d_problem",
    "sample_logistic_gbl", "matched_recovery_error", "cluster_recovery_error",
    # submodules
    "particles", "problems", "optimizers", "fy", "kernels", "utils", "plotting",
]
