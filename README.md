# Dark Energy Particle CATHODE Time-Dependent Study

This repository contains the final GitHub-ready release of the CATHODE time-dependent toy study.

The main entry point is:

```bash
python cathode_mcjinbopromax_v2_T1e4.py
```

## Contents

- `cathode_mcjinbopromax_v2_T1e4.py`: main analysis and scan script.
- `sk_cathode/`: vendored local CATHODE helper package used by the main script.
- `phijj.txt.gz`: signal mass sample input.
- `ppuu_background-8D-mmjj400-mass.txt.gz`: background mass sample input.
- `results/max_repro_bundle_20260307/`: compact reproducibility bundle for the current figures and CSV outputs.
- `requirements.txt`: Python dependencies for this release.

Large generated directories, virtual environments, logs, checkpoints, and historical scan outputs are intentionally not included.

## Setup

Use Python 3.10 or newer. A virtual environment is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If a LaTeX executable is available, matplotlib will use it for plot text. Otherwise the script automatically falls back to matplotlib mathtext.

## Quick Smoke Test

The full scan is computationally expensive. Use a small configuration first to confirm the environment:

```bash
SCAN_OUTPUT_DIR=smoke_T1e4 \
LOG_T_N=2 \
PHI_N=1 \
N_SIG_EVENTS=200 \
N_BG_EVENTS=2000 \
N_NULL_TOYS=5 \
POI_SCAN_N=101 \
SAVE_PICKLE_FIGS=0 \
python cathode_mcjinbopromax_v2_T1e4.py
```

Outputs are written under the directory named by `SCAN_OUTPUT_DIR`.

## Full Run

The default configuration performs a logarithmic period scan from `T=0.1` to `T=1e4` months with 100 period points and 20 phase values:

```bash
python cathode_mcjinbopromax_v2_T1e4.py
```

Common environment variables:

- `SCAN_OUTPUT_DIR`: output directory name.
- `SCAN_MODE`: `log`, `piecewise`, or `narrow`.
- `LOG_T_MIN`, `LOG_T_MAX`, `LOG_T_N`: period grid for log scans.
- `OSC_PERIODS`: comma-separated explicit period list, overriding `SCAN_MODE`.
- `PHI_MODE`: `random` or `linspace`.
- `PHI_N`: number of phase values.
- `PHIS_PI`: comma-separated phase values in units of pi, overriding phase sampling.
- `N_SIG_EVENTS`, `N_BG_EVENTS`: simulated event counts.
- `N_NULL_TOYS`: null toys for global p-value calibration.
- `POI_SCAN_MIN`, `POI_SCAN_MAX`, `POI_SCAN_N`: pyhf limit scan grid.
- `SAVE_PICKLE_FIGS`: set to `0` to save only figures/CSV without matplotlib pickle files.
- `GLOBAL_SEED`: optional random seed for reproducibility.

## Notes

The main script expects the two `.txt.gz` input files to live in the same directory as the script. The local `sk_cathode` source is included because the final script imports it directly.

The vendored `sk_cathode` code keeps its original license at `sk_cathode/LICENSE.txt`.
