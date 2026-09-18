# Fenchel–Young duality gaps — companion experiments

Numerical experiments for the two-paper series by **Pierre-Cyril
Aubin-Frankowski and Yohann De Castro (2026)**:

1. **Paper I** — *Fenchel–Young Duality Gaps: Certified Early Stopping for
   Regularized Inverse Problems*.
2. **Paper II** — *Localization and Exact Support Recovery for the
   Generalized Beurling–Lasso*.

One notebook, `experiments.ipynb` (source of truth: `experiments.py` in
[jupytext](https://jupytext.readthedocs.io/) percent format), reproduces all
four publication figures and the in-text verification checks quoted in the
papers' experiment sections.

## Figures ↔ papers

| Part | Paper | Figure | File |
|---|---|---|---|
| I | Paper I, Fig 1 (§5.2) | Certifiability of Lion-K and Muon | `fig_lion_muon_delta.pdf` |
| I | Paper I, Fig 2 (§5.1) | Dual certificate evolution along CPGD | `fig_certificate_evolution.pdf` |
| I | Paper I, Fig 3 (§5.1) | Fenchel–Young duality-gap decomposition | `fig_fy_decomposition_synthetic.pdf` |
| II | Paper II, Fig 1 (§4.2) | Well-specified logistic GBL, parametric rate | `fig_logistic_well_spec.pdf` |

## Structure

```
fenchel-young-gaps/
├── README.md
├── LICENSE                  # MIT
├── requirements.txt
├── experiments.py           # the notebook in jupytext "percent" format (source of truth)
├── experiments.ipynb        # the same notebook in .ipynb form
├── figs/                    # generated figures (created on first run)
└── gbl/                     # shared utilities imported by the notebook
    ├── particles.py         # Particles, MirrorRetraction
    ├── problems.py          # GaussianBlasso, LogisticGaussianBlasso, ...
    ├── optimizers.py        # ConicOptimizer (CPGD)
    ├── fy.py                # FYTracker: the Fenchel–Young gap diagnostic
    ├── kernels.py           # feature maps / kernels
    ├── utils.py             # data synthesis, initialisation, recovery metrics
    └── plotting.py          # figure helpers
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook experiments.ipynb        # or: open experiments.py in VSCode/PyCharm
```

Figures are written to `./figs/` by default; set `FYG_FIGS_DIR` to redirect
them (e.g. into a LaTeX project):

```bash
FYG_FIGS_DIR=/path/to/latex/figs jupyter nbconvert --to notebook --execute experiments.ipynb
```

A full run takes roughly 10–40 minutes on a laptop (the CPGD trajectories use
30 000 iterations, plus a doubled-horizon reference run, and the logistic
sweep runs five sample sizes up to n = 10 000).

## Citation

```bibtex
@misc{aubinfrankowski2026fenchelyoungdualitygapscertified,
      title={Fenchel-Young Duality Gaps: Certified Early Stopping for Regularized Inverse Problems},
      author={Pierre-Cyril Aubin-Frankowski and Yohann de Castro},
      year={2026},
      eprint={2609.17629},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2609.17629},
}

@article{aubin2026localization,
  author  = {Aubin-Frankowski, Pierre-Cyril and De Castro, Yohann},
  title   = {Localization and Exact Support Recovery for the Generalized Beurling--Lasso},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {arXiv identifier to appear}
}
```

## Disclosure of AI use

In line with the Leiden Declaration on Artificial Intelligence and Mathematics, AI agents (Claude Code, Anthropic) were used in preparing this notebook and its documentation, to audit the code and cross-references and for editorial tasks. The authors produced and checked the experiments themselves and take sole responsibility for the code and its results.

## License

MIT — see [LICENSE](LICENSE).
