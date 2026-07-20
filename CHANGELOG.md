# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-07-20
### Fixed
- Stabilized and unified the persistent figure saving architecture across the GUI and CLI. Jet Reversal plots are now properly saved into structured directories (e.g., `interactive_figures/jet_reversal_checks/mms<probe>/`) rather than root directories.
- Enabled automatic, unconditional plot saving for all generated visualization figures during GUI Single Event mode exploration.
- Fixed a bug causing PyTplot Jet Reversal PNGs to overwrite a single `mms_jet_reversal_check.png` default file instead of following the structured directory naming convention.

## [0.2.2] - 2026-07-20
### Added
- Added a new "Show Summary Statistics" toggle in the GUI that dynamically calculates and renders an interactive markdown table for the filtered dataset. The table gracefully presents Mean, Min, Max, and Percentiles using LaTeX equations.

### Changed
- Relaxed dependency constraints in `pyproject.toml` (changing `^` to `>=`) to maximize compatibility for users installing alongside other complex Python packages, while maintaining `python >= 3.11`.
- Cleaned up Streamlit UI deprecation warnings by replacing `use_container_width=True` with `width="stretch"` across the application.

## [0.2.1] - 2026-07-20
### Fixed
- Fixed an issue in `app.py` where interactive Seaborn joint-plots crashed due to a duplicate element ID (`StreamlitDuplicateElementId`).
- Corrected the interactive Plotly engine to properly accept and scale marker sizes (`marker_size_var`) in joint-plots.
- Enforced the custom "Deep Ocean" UI theme globally for all PyPI installations by passing theme parameters directly through the `rxn-location-gui` entry point CLI arguments instead of relying on `.streamlit/config.toml`.

## [0.2.0] - 2026-07-20
### Added
- PyPI release configuration with GitHub Actions Trusted Publishing.
- Bundled `data/potential_jet_reconnection_times.txt` and `sample_files/` in the package distribution.
- Added advanced Plotly Interactive vs Matplotlib Static engine toggle for statistical figures.
- Introduced interactive 2D density histograms (Hexbins) and Matplotlib Box/Violin plots.
- Integrated a custom "Deep Ocean" Dark Mode UI theme via Streamlit's `config.toml`.
- Replaced programmatic Streamlit facecolor injections with direct Matplotlib context styling.

## [0.1.0] - 2026-07-19
### Added
- Initial package release containing the Streamlit GUI (`rxn-location-gui`) and batch CLI (`rxn-batch`).
- Full support for Jet Reversal checking, MMS telemetry fetching, and Reconnection Models (Bisection, Exhaust, Shear, Energy).
