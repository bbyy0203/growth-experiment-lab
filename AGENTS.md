# Repository instructions

- This repository is a personal open-source portfolio project using synthetic data only.
- Never describe it as a real company production system or imply that generated results are real.
- Do not hard-code, prefill, or fabricate analytical results. Results must come from executable code.
- Target Python 3.11 and prefer commands in the form `python -m ...`.
- Do not create, remove, or switch Conda environments.
- Do not install, upgrade, or uninstall dependencies without explicit user confirmation.
- Read configuration with Python's standard-library `tomllib`; do not add PyYAML in v1.
- Preserve fixed seeds and document statistical assumptions, applicability, and limitations.
- Keep the architecture small and readable. Add modules only when their implementation stage begins.
- The experimental unit is `user_id`; use first assignment and intention-to-treat analysis.
- Only pre-treatment covariates may be used by CUPED or propensity-score models.
- Bootstrap sampling must occur at user level. IPTW bootstrap samples must refit propensity scores.
- Generated full datasets belong under ignored data directories. Small test fixtures may be committed.
- Before completing a change, run relevant tests and Ruff when the active environment supports them.

