# Changelog

## v0.2.3 (2026-03-25)
- feat: add TailVariableImportance to EVT module — adapted lasso with tail-weighted residuals for identifying extreme-claim drivers (arXiv:2504.06984)
- Standard Gini/SHAP importance reflects bulk behaviour; TailVariableImportance isolates what matters above the configurable tail threshold
- 15 tests covering: signal recovery, importances sum to 1, feature name handling, summary keys, plot rendering, and input validation

## v0.2.1 (2026-03-22) [unreleased]
- Add Databricks benchmark script and expand benchmark results section
- fix: use plain string license field for setuptools compatibility
- fix: use importlib.metadata for __version__ (prevents drift from pyproject.toml)

## v0.2.1 (2026-03-21)
- Add cross-links to related libraries in README
- docs: replace pip install with uv add in README
- Add extreme-tail benchmark: Pareto alpha=1.5 (xi=0.667), n=20,000
- Add blog post link and community CTA to README
- bench: add EVT module benchmark — TruncatedGPD, CensoredHill, WeibullTemperedPareto vs naive baselines
- docs: add EVT module section to README
- feat: add EVT submodule (TruncatedGPD, CensoredHillEstimator, WeibullTemperedPareto)
- Add insurance_severity.evt subpackage for EVT with truncation and censoring corrections (v0.2.0)
- Add MIT license
- Fix CI: guard all torch-dependent tests with pytest.importorskip
- fix: guard DRN tests with pytest.importorskip for optional torch dep
- Fix P0/P1 issues from QA audit: correct Burr XII mode docstrings, fix quick-start attributes, use fitted pi in predict, optional torch dependency
- Add PyPI classifiers for financial/insurance audience
- Update Performance section with post-Phase-98 benchmark results
- Fix P0/P1 bugs: DRN NLL path, predict() mixture, pi_mean_, score() completeness
- Add benchmark: LognormalGPDComposite vs single Lognormal on spliced DGP
- fix: make test_scr_aware_cutpoints deterministic by supplying explicit val data
- pin statsmodels>=0.14.5 for scipy compat
- Add shields.io badge row to README
- Add Quick Start section to README
- docs: add Databricks notebook link
- Add Related Libraries section to README
- fix: replace np.trapz with compat shim for NumPy 2.0

