#conda activate sk_cathode_env10
"""
CATHODE time-dependent toy (period-agnostic time features)

Key change vs the original "folded" implementation:
- We do NOT fold time by a single hypothesized period.
- Instead we give the NN a fixed dictionary ("bank") of sinusoidal features spanning
  a range of candidate periods and harmonics, plus a linear time feature.
This matches the paper text: CATHODE can learn the time dependence without being
told the true period.

Practical note:
- A bank of sin/cos features is still a *choice of function class* and therefore
  still requires calibration for discovery (look-elsewhere / model flexibility),
  but it avoids retraining or transforming by a single assumed period.
"""
import numpy as np
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl
import gzip
import sys
import os
import random
import pickle
import time
from datetime import datetime

import pyhf

from os.path import exists
from pathlib import Path
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# add local sk_cathode package to path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))
sys.path.append(str(CURRENT_DIR / "sk_cathode"))
sys.path.append(str(CURRENT_DIR / "sk_cathode" / "sk_cathode"))

SIGNAL_MASS_FILE = CURRENT_DIR / "phijj.txt.gz"
BACKGROUND_MASS_FILE = CURRENT_DIR / "ppuu_background-8D-mmjj400-mass.txt.gz"

from sk_cathode.classifier_models.neural_network_classifier import NeuralNetworkClassifier
from sk_cathode.utils.preprocessing import LogitScaler

from scipy.signal import lombscargle

pyhf.set_backend("numpy")

GLOBAL_SEED_RAW = os.environ.get("GLOBAL_SEED", "").strip()
if GLOBAL_SEED_RAW != "":
    GLOBAL_SEED = int(GLOBAL_SEED_RAW)
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    try:
        import torch  # optional: only if available in the runtime

        torch.manual_seed(GLOBAL_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(GLOBAL_SEED)
    except Exception:
        pass
    print(f"[Seed] GLOBAL_SEED={GLOBAL_SEED}")

# fall back to mathtext if latex is unavailable; keep a serif font available to avoid warnings
USE_TEX = shutil.which("latex") is not None
if not USE_TEX:
    print("[Warn] latex not found on PATH; using matplotlib mathtext instead.")

SERIF_FONTS = ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"]

# use LaTeX-style fonts and larger labels for plots
plt.rcParams.update(
    {
        "text.usetex": USE_TEX,
        "font.family": "serif",
        "font.serif": SERIF_FONTS,
        "mathtext.fontset": "cm",
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 12,
    }
)

# =============================================================================
# Period-agnostic time feature bank
# =============================================================================
SCORE_EPS = 1e-6
SAVE_PICKLE_FIGS = os.environ.get("SAVE_PICKLE_FIGS", "1").strip().lower() not in ("0", "false", "no")

# =============================================================================
# Limit-setting config (aligned with Sec. 4A pyhf setup)
# =============================================================================
MASS_RANGE = (500.0, 1250.0)
N_MASS_BINS = 40
MASS_BINS = np.linspace(MASS_RANGE[0], MASS_RANGE[1], N_MASS_BINS + 1)

REL_BKG_UNC = float(os.environ.get("REL_BKG_UNC", "0.10"))
if REL_BKG_UNC < 0.0:
    REL_BKG_UNC = 0.0
POI_SCAN_MIN = float(os.environ.get("POI_SCAN_MIN", "0.0"))
POI_SCAN_MAX = float(os.environ.get("POI_SCAN_MAX", "5.0"))
POI_SCAN_N = int(os.environ.get("POI_SCAN_N", "2001"))
if POI_SCAN_N < 2:
    POI_SCAN_N = 2
POI_SCAN = np.linspace(POI_SCAN_MIN, POI_SCAN_MAX, POI_SCAN_N)
POI_SCAN_MAX_TRIES = 6


def parse_float_list_env(env_key):
    """Parse comma-separated float list from an env var."""
    raw = os.environ.get(env_key, "").strip()
    if raw == "":
        return None
    out = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        out.append(float(token))
    return out if len(out) else None


def phi_over_pi(phi):
    return float(phi) / np.pi


def make_phi_tag(phi, decimals=6):
    return f"phi{phi_over_pi(phi):.{decimals}f}pi"


def save_named_series_csv(out_path, series_map):
    """Save one or more named 1D series as a CSV, padding shorter series with NaN."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    max_len = 0
    for name, values in series_map.items():
        arr = np.asarray(values, dtype=float).reshape(-1)
        items.append((name, arr))
        if arr.size > max_len:
            max_len = arr.size

    table = np.full((max_len, len(items)), np.nan, dtype=float)
    headers = []
    for j, (name, arr) in enumerate(items):
        headers.append(name)
        table[:arr.size, j] = arr

    np.savetxt(out_path, table, delimiter=",", header=",".join(headers), comments="")
    print(f"[CSV] Saved: {out_path}")
    return out_path


def save_rows_csv(out_path, header, rows):
    """Save row-wise numeric data as CSV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = np.asarray(rows, dtype=float)
    if table.ndim == 1:
        table = table.reshape(1, -1)
    np.savetxt(out_path, table, delimiter=",", header=header, comments="")
    print(f"[CSV] Saved: {out_path}")
    return out_path


def save_figure_with_pickle(fig, out_path, dpi=240):
    """Save a matplotlib figure and optionally a same-name pickle alongside it."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    if SAVE_PICKLE_FIGS:
        pkl_path = out_path.with_suffix(".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(fig, f)
        print(f"[PKL] Saved: {pkl_path}")
    return out_path


def save_current_figure_with_pickle(out_path, dpi=240):
    return save_figure_with_pickle(plt.gcf(), out_path, dpi=dpi)


def add_time_feature_bank(t, periods, n_harmonics, exposure=None, include_linear=True):
    """Sin/cos feature dictionary over many candidate periods.

    Parameters
    ----------
    t : array-like, shape (N,) or (N,1)
        Raw times in months.
    periods : array-like, shape (P,)
        Candidate periods (months). We do NOT choose one; we include them all.
    n_harmonics : int
        Number of harmonics per period to include (h=1..n_harmonics).
        (sin^4 contains up to the 4th harmonic, so n_harmonics=4 is sufficient
        for the toy, but larger is more general.)
    exposure : float or None
        If provided and include_linear=True, the linear feature is t/exposure.
        Otherwise the linear feature is raw t.
    include_linear : bool
        Whether to prepend a linear time feature.

    Returns
    -------
    X : np.ndarray, shape (N, D)
        Feature matrix: [t/exposure] + {sin(2π h t/T_j), cos(2π h t/T_j)}.
    """
    t = np.asarray(t, dtype=float).reshape(-1, 1)  # (N,1)
    periods = np.asarray(periods, dtype=float).reshape(1, -1)  # (1,P)

    feats = []
    if include_linear:
        if exposure is None:
            feats.append(t)
        else:
            feats.append(t / float(exposure))

    # base angles for each candidate period, shape (N,P)
    base = 2.0 * np.pi * t / np.maximum(periods, 1e-12)

    # append harmonics
    for h in range(1, int(n_harmonics) + 1):
        feats.append(np.sin(h * base))
        feats.append(np.cos(h * base))

    return np.concatenate(feats, axis=1)


def prob_to_density_ratio(prob, y_train):
    """Convert p=P(y=1|x) to an odds-style score proxy for LR plotting/weighting."""
    p = np.asarray(prob, dtype=float)
    p = np.clip(p, SCORE_EPS, 1.0 - SCORE_EPS)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    pi1 = np.clip(np.mean(y_train), SCORE_EPS, 1.0 - SCORE_EPS)
    pi0 = 1.0 - pi1
    return (p / (1.0 - p)) * (pi0 / pi1)


def to_unit_interval(values, lo=5.0, hi=95.0):
    """Robust [0, 1] scaling using percentile clipping."""
    arr = np.asarray(values, dtype=float)
    lo_v, hi_v = np.percentile(arr, [lo, hi])
    if not (np.isfinite(lo_v) and np.isfinite(hi_v)) or hi_v <= lo_v:
        lo_v, hi_v = 0.0, 1.0
    return np.clip((arr - lo_v) / (hi_v - lo_v), 0.0, 1.0)


def lr_time(t, period, phi):
    """True LR(t) = sin^4(2π t/period + phi)."""
    t = np.asarray(t)
    return np.sin(2.0 * np.pi * t / float(period) + phi) ** 4


def lr_time_bg_normalized(t, period, phi):
    """Background-normalized analytic LR weight with E_bg[w]=1 for uniform bg in time."""
    return (8.0 / 3.0) * lr_time(t, period, phi)


def normalize_to_bg_mean(w, w_bg, eps=1e-12):
    """Normalize weights so the mean weight on background is unity."""
    w = np.asarray(w, dtype=float).reshape(-1)
    w_bg = np.asarray(w_bg, dtype=float).reshape(-1)
    bg_mean = np.mean(w_bg) if w_bg.size else np.nan
    if not np.isfinite(bg_mean) or bg_mean <= eps:
        return np.full_like(w, np.nan, dtype=float)
    return w / bg_mean


def build_weighted_mass_templates(sig_masses, bg_masses, w_sig, w_bg, mass_bins=MASS_BINS):
    """Build weighted signal/background mass templates for pyhf."""
    sig_masses = np.asarray(sig_masses, dtype=float).reshape(-1)
    bg_masses = np.asarray(bg_masses, dtype=float).reshape(-1)
    w_sig = np.asarray(w_sig, dtype=float).reshape(-1)
    w_bg = np.asarray(w_bg, dtype=float).reshape(-1)

    if sig_masses.size != w_sig.size:
        raise ValueError("sig_masses and w_sig must have the same length")
    if bg_masses.size != w_bg.size:
        raise ValueError("bg_masses and w_bg must have the same length")

    w_sig = np.where(np.isfinite(w_sig), w_sig, 0.0)
    w_bg = np.where(np.isfinite(w_bg), w_bg, 0.0)

    sig_hist, _ = np.histogram(sig_masses, bins=mass_bins, weights=w_sig)
    bkg_hist, _ = np.histogram(bg_masses, bins=mass_bins, weights=w_bg)
    return sig_hist.astype(float), bkg_hist.astype(float)


def pyhf_expected_mu_limit(sig_hist, bkg_hist):
    """Median expected 95% CL upper limit on mu from weighted templates."""
    sig_hist = np.asarray(sig_hist, dtype=float).reshape(-1)
    bkg_hist = np.asarray(bkg_hist, dtype=float).reshape(-1)
    if sig_hist.shape != bkg_hist.shape:
        raise ValueError("sig_hist and bkg_hist must have the same shape")

    sig_hist = np.where(np.isfinite(sig_hist), sig_hist, 0.0)
    bkg_hist = np.where(np.isfinite(bkg_hist), bkg_hist, 0.0)
    sig_hist = np.clip(sig_hist, 0.0, None)
    bkg_hist = np.clip(bkg_hist, 0.0, None)

    if np.sum(sig_hist) <= 0.0 or np.sum(bkg_hist) <= 0.0:
        return np.nan

    poi_scan = np.asarray(POI_SCAN, dtype=float)
    for _ in range(int(POI_SCAN_MAX_TRIES)):
        try:
            bkg_up = bkg_hist * (1.0 + REL_BKG_UNC)
            bkg_down = bkg_hist * max(0.0, 1.0 - REL_BKG_UNC)
            model = pyhf.simplemodels.correlated_background(
                signal=sig_hist.tolist(),
                bkg=bkg_hist.tolist(),
                bkg_up=bkg_up.tolist(),
                bkg_down=bkg_down.tolist(),
            )
            data = bkg_hist.tolist() + model.config.auxdata

            bounds = list(model.config.suggested_bounds())
            bounds[model.config.poi_index] = (0.0, max(10.0, float(poi_scan[-1]) * 2.0))
            model.config._suggested_bounds = tuple(bounds)

            _, exp_limits, _ = pyhf.infer.intervals.upper_limits.upper_limit(
                data,
                model,
                poi_scan,
                level=0.05,
                par_bounds=bounds,
                return_results=True,
            )
            mu95 = float(exp_limits[2])
        except Exception:
            mu95 = np.nan

        if np.isfinite(mu95):
            if mu95 >= 0.98 * float(poi_scan[-1]):
                poi_scan = np.linspace(float(poi_scan[0]), float(poi_scan[-1]) * 2.0, len(poi_scan))
                continue
            return mu95

        poi_scan = np.linspace(float(poi_scan[0]), float(poi_scan[-1]) * 2.0, len(poi_scan))

    return np.nan


# =============================================================================
# Diagnostics: post-hoc period scan on NN score vs raw time
# =============================================================================
def scan_period_lomb_scargle(times, scores, period_grid):
    """Scan candidate periods using Lomb-Scargle on classifier scores vs time."""
    times = np.asarray(times, dtype=float).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    period_grid = np.asarray(period_grid, dtype=float).ravel()
    if times.size == 0 or scores.size == 0 or period_grid.size == 0:
        return np.nan, np.nan, np.full_like(period_grid, np.nan, dtype=float)
    if times.size != scores.size:
        raise ValueError("times and scores must have identical lengths")

    # Remove DC offset to focus on periodic structure.
    y = scores - np.mean(scores)
    y_std = np.std(y)
    if y_std > 0:
        y = y / y_std
    else:
        return float(period_grid[0]), 0.0, np.zeros_like(period_grid, dtype=float)

    omega = 2.0 * np.pi / np.maximum(period_grid, 1e-12)
    # scipy may return a scalar when omega has length 1; keep a 1D array.
    powers = np.atleast_1d(lombscargle(times, y, omega, normalize=True)).astype(float)
    best_idx = int(np.argmax(powers))
    return float(period_grid[best_idx]), float(powers[best_idx]), powers


def estimate_global_pvalue_null_toys(
    classifier_model,
    inner_scaler,
    exposure,
    periods_feat,
    n_harmonics_feat,
    period_grid_scan,
    n_events,
    y_train_for_score,
    n_toys=200,
    rng=None,
):
    """Estimate global p-value from background-only toys (look-elsewhere)."""
    if rng is None:
        rng = np.random.default_rng(12345)

    null_max_powers = np.zeros(int(n_toys), dtype=float)
    for i in range(int(n_toys)):
        t_null = rng.uniform(0.0, float(exposure), size=int(n_events))
        x_null = add_time_feature_bank(
            t_null, periods_feat, n_harmonics_feat, exposure=exposure, include_linear=True
        )
        x_null = inner_scaler.transform(x_null)
        s_null = classifier_model.predict(x_null).ravel()
        s_null = prob_to_density_ratio(s_null, y_train_for_score)
        _, pmax, _ = scan_period_lomb_scargle(t_null, s_null, period_grid_scan)
        null_max_powers[i] = pmax
    return null_max_powers


def estimate_joint_global_pvalue_null_toys(
    model_specs,
    exposure,
    periods_feat,
    n_harmonics_feat,
    period_grid_scan,
    n_events,
    n_toys=200,
    rng=None,
):
    """
    Estimate global p-value for max( period scan power ) over a bank of models.

    model_specs : list of tuples
        Each entry is (classifier_model, inner_scaler, y_train_for_score).
    """
    if rng is None:
        rng = np.random.default_rng(12345)
    if len(model_specs) == 0:
        return np.zeros(int(n_toys), dtype=float)

    null_max_powers = np.zeros(int(n_toys), dtype=float)
    for i in range(int(n_toys)):
        max_p = -np.inf
        # Use the same toy realization across all candidate models so the
        # "max across phi" null statistic is computed on a common toy sample.
        t_null = rng.uniform(0.0, float(exposure), size=int(n_events))
        x_null_raw = add_time_feature_bank(
            t_null,
            periods_feat,
            n_harmonics_feat,
            exposure=exposure,
            include_linear=True,
        )
        for classifier_model, inner_scaler, y_train_for_score in model_specs:
            x_null = x_null_raw
            x_null = inner_scaler.transform(x_null)
            s_null = classifier_model.predict(x_null).ravel()
            s_null = prob_to_density_ratio(s_null, y_train_for_score)
            _, pmax, _ = scan_period_lomb_scargle(t_null, s_null, period_grid_scan)
            if pmax > max_p:
                max_p = pmax
        null_max_powers[i] = max_p
    return null_max_powers


def plot_period_scan(period_grid, powers, best_period, injected_period, p_global, out_path):
    """Save periodogram-style power scan plot."""
    period_grid = np.asarray(period_grid, dtype=float)
    powers = np.asarray(powers, dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(period_grid, powers, lw=2.0, color="black", label="LS power")
    ax.axvline(float(injected_period), ls="--", lw=1.8, color="royalblue",
               label=rf"Injected $T$={injected_period:.2f}")
    ax.axvline(float(best_period), ls="-.", lw=1.8, color="crimson",
               label=rf"Best-fit $T$={best_period:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("Candidate period T [months]")
    ax.set_ylabel("Lomb-Scargle power")
    ax.set_title(rf"Conditional period scan (model fixed), global p={p_global:.3g}")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save_figure_with_pickle(fig, out_path, dpi=220)
    plt.close(fig)
    save_named_series_csv(
        Path(out_path).with_suffix(".csv"),
        {
            "candidate_period_months": period_grid,
            "ls_power": powers,
            "injected_period_months": np.full_like(period_grid, float(injected_period)),
            "bestfit_period_months": np.full_like(period_grid, float(best_period)),
            "global_pvalue_conditional": np.full_like(period_grid, float(p_global)),
        },
    )


# =============================================================================
# Fig. 5/6 style plots: NN response vs time
# =============================================================================
def learned_time_response_months(
    clf,
    inner_scaler,
    exposure_months,
    period_months_for_plot,
    phi,
    y_train_for_score,
    periods_feat,
    n_harmonics_feat,
    n_grid=1200,
    smooth=False,
):
    """Return months grid and rescaled NN output g(t) in [0,1].

    This is *period-agnostic* in training. We only use period_months_for_plot to set
    a sufficiently fine plotting grid for rapid oscillations.
    """
    # adaptive grid for plotting only: >=30 points per injected cycle
    min_pts = max(int(n_grid), int(30.0 * float(exposure_months) / float(period_months_for_plot)))
    t_grid = np.linspace(0.0, float(exposure_months), min_pts)

    Xg = add_time_feature_bank(
        t_grid, periods_feat, n_harmonics_feat, exposure=exposure_months, include_linear=True
    )
    Xg = inner_scaler.transform(Xg)

    nn_raw = clf.predict(Xg).ravel()
    r_raw = prob_to_density_ratio(nn_raw, y_train_for_score)
    g = to_unit_interval(np.log(r_raw))

    if smooth and g.size >= 51:
        try:
            from scipy.signal import savgol_filter
            g = savgol_filter(g, 51, 3, mode="interp")
            g = np.clip(g, 0.0, 1.0)
        except Exception:
            pass

    return t_grid, g


def fig6_plot_nnoutput_vs_time(
    clf,
    inner_scaler,
    exposure_months,
    period_months,
    phi,
    y_train_for_score,
    periods_feat,
    n_harmonics_feat,
    out_prefix="fig6",
):
    """Overlay learned NN response with sin^4 truth in months."""
    t_grid, g = learned_time_response_months(
        clf,
        inner_scaler,
        exposure_months,
        period_months,
        phi,
        y_train_for_score,
        periods_feat=periods_feat,
        n_harmonics_feat=n_harmonics_feat,
        n_grid=1200,
    )

    omega = 2.0 * np.pi / period_months if period_months > 0 else 0.0
    lr = np.sin(omega * t_grid + phi) ** 4

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(t_grid, g, lw=2.0, label="CATHODE (NN output)", zorder=3)
    ax.plot(
        t_grid,
        lr,
        ":",
        lw=2.0,
        label=r"True $\sin^4(2\pi t/T_{\rm Period})$",
        zorder=2,
    )
    ax.set_xlim(0.0, float(exposure_months))
    ax.set_ylim(0.0, 1.2)
    ax.set_yticks(np.arange(0.0, 1.21, 0.2))
    ax.set_xlabel("t [months]")
    ax.set_ylabel("LR proxy")
    delta_label = f"{phi/np.pi:.2f}".rstrip("0").rstrip(".") or "0"
    ax.set_title(
        rf"Learned oscillation: $T_{{\rm Period}}$={period_months:.2f} months, $\delta={delta_label}$"
    )
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", top=True, right=True, direction="in")
    ax.tick_params(axis="both", which="major", length=8, width=2, direction="in")
    ax.tick_params(axis="both", which="minor", length=4, width=2, direction="in")
    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=2.0,
        borderaxespad=0.2,
        labelspacing=0.3,
    )
    fig.tight_layout(pad=0.3)
    out = f"{out_prefix}_nnoutput_T{period_months:.6g}_{make_phi_tag(phi)}.pdf"
    save_figure_with_pickle(fig, out, dpi=240)
    plt.close(fig)
    save_named_series_csv(
        Path(out).with_suffix(".csv"),
        {
            "t_months": t_grid,
            "nn_output_unit_interval": g,
            "true_sin4": lr,
            "period_months": np.full_like(t_grid, float(period_months)),
            "phi_rad": np.full_like(t_grid, float(phi)),
            "phi_over_pi": np.full_like(t_grid, phi_over_pi(phi)),
        },
    )
    print(f"[Fig6] NN-output vs time saved: {out}")
    return out


class Fig6Recorder:
    """Aggregate per-(T,phi) pyhf limits and produce Fig6 lower panel."""
    def __init__(self, T_grid_months, phi_list, exposure_months=36.0, out_prefix="fig6_cathode"):
        self.T = np.asarray(T_grid_months, dtype=float)
        self.PHI = np.asarray(phi_list, dtype=float)
        self.expo = float(exposure_months)
        self.out_prefix = out_prefix
        self.mu95_learned = np.full((len(self.PHI), len(self.T)), np.nan, dtype=float)
        self.mu95_ti = np.full((len(self.PHI), len(self.T)), np.nan, dtype=float)
        self.mu95_ratio = np.full((len(self.PHI), len(self.T)), np.nan, dtype=float)

    def record(self, T_months, phi, mu95_learned, mu95_ti):
        j = int(np.argmin(np.abs(self.T - T_months)))
        i = int(np.argmin(np.abs(self.PHI - phi)))
        self.mu95_learned[i, j] = mu95_learned
        self.mu95_ti[i, j] = mu95_ti
        if np.isfinite(mu95_learned) and np.isfinite(mu95_ti) and mu95_ti > 0.0:
            self.mu95_ratio[i, j] = mu95_learned / mu95_ti

    def save_csvs(self, time_independent_level=1.0):
        rel_lim = self.mu95_ratio
        finite_count = np.sum(np.isfinite(rel_lim), axis=0)
        with np.errstate(invalid="ignore"):
            rel_lim_avg = np.where(finite_count > 0, np.nansum(rel_lim, axis=0) / finite_count, np.nan)

        detail_rows = []
        for i, phi in enumerate(self.PHI):
            for j, period in enumerate(self.T):
                detail_rows.append(
                    [
                        float(period),
                        float(phi),
                        phi_over_pi(phi),
                        self.mu95_learned[i, j],
                        self.mu95_ti[i, j],
                        self.mu95_ratio[i, j],
                    ]
                )

        save_rows_csv(
            f"{self.out_prefix}_limits_vs_T_detail.csv",
            "T_months,phi_rad,phi_over_pi,mu95_learned,mu95_ti,mu95_ratio",
            detail_rows,
        )

        curve_map = {"T_months": self.T}
        for i, phi in enumerate(self.PHI):
            curve_map[f"ratio_phi_{phi_over_pi(phi):.6f}pi"] = rel_lim[i, :]
        curve_map["ratio_phi_avg"] = rel_lim_avg
        curve_map["time_independent"] = np.full_like(self.T, float(time_independent_level))
        save_named_series_csv(f"{self.out_prefix}_limits_vs_T_curves.csv", curve_map)

    def plot_limits(self, ylabel="Expected limit ratio mu95(learned)/mu95(TI)", time_independent_level=1.0):
        rel_lim = self.mu95_ratio
        finite_count = np.sum(np.isfinite(rel_lim), axis=0)
        with np.errstate(invalid="ignore"):
            rel_lim_avg = np.where(finite_count > 0, np.nansum(rel_lim, axis=0) / finite_count, np.nan)
        self.save_csvs(time_independent_level=time_independent_level)

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        greys = mpl.cm.Greys(np.linspace(0.35, 0.85, len(self.PHI)))
        for i, phi in enumerate(self.PHI):
            label = rf"$\delta$={phi/np.pi:.2f}$\pi$"
            ax.plot(self.T, rel_lim[i, :], color=greys[i], lw=1.8, label=label)

        ax.plot(self.T, rel_lim_avg, color="crimson", lw=2.2, ls="-.", label=r"$\delta$ ave")
        ax.plot(self.T, np.full_like(self.T, float(time_independent_level)),
                color="royalblue", lw=2.0, ls=":", label="Time independent")

        ax.set_xscale("log")
        ax.set_xlabel("Oscillation period T [months]")
        ax.set_ylabel(ylabel)
        finite_vals = rel_lim[np.isfinite(rel_lim)]
        if finite_vals.size:
            ymax = max(1.1, min(2.5, 1.15 * float(np.nanpercentile(finite_vals, 95))))
        else:
            ymax = 1.2
        ax.set_ylim(0.0, ymax)

        ax.legend(ncol=2, frameon=False, fontsize=9, loc="lower left")
        ax.grid(alpha=0.25, which="both", axis="both")
        fig.tight_layout()
        out = f"{self.out_prefix}_limits_vs_T.pdf"
        save_figure_with_pickle(fig, out, dpi=240)
        plt.close(fig)
        print(f"[Fig6] Limits-vs-T saved: {out}")
        return out


# =============================================================================
# Toy data generation
# =============================================================================
exposure = 36.0

mass_lo = 550
mass_hi = 850

_SIG_MASS_CACHE = None
_BG_MASS_CACHE = None


def load_mass_samples_cached(path, label, cache_key, retries=3, sleep_s=0.2):
    """Load a gzipped 1D mass sample file once, with basic retry/validation."""
    global _SIG_MASS_CACHE, _BG_MASS_CACHE

    if cache_key == "signal" and _SIG_MASS_CACHE is not None:
        return _SIG_MASS_CACHE
    if cache_key == "background" and _BG_MASS_CACHE is not None:
        return _BG_MASS_CACHE

    if not path.exists():
        raise FileNotFoundError(f"{label} mass file not found: {path}")

    last_exc = None
    masses = np.array([], dtype=float)
    for attempt in range(1, int(retries) + 1):
        try:
            with gzip.open(path, "rt") as f:
                masses = np.loadtxt(f, ndmin=1)
            masses = np.asarray(masses, dtype=float).reshape(-1)
            masses = masses[np.isfinite(masses)]
            if masses.size > 0:
                if cache_key == "signal":
                    _SIG_MASS_CACHE = masses
                else:
                    _BG_MASS_CACHE = masses
                print(f"[Data] Loaded {label} masses: {masses.size} entries from {path.name}")
                return masses
        except Exception as exc:
            last_exc = exc
        time.sleep(float(sleep_s) * attempt)

    if last_exc is not None:
        raise RuntimeError(
            f"Failed to load non-empty {label} mass samples from {path} "
            f"after {retries} attempts"
        ) from last_exc
    raise RuntimeError(f"Loaded zero valid {label} mass samples from {path}")


# events are (mass, time, s/b)
def gen_sig(N, op, phi):
    if int(N) <= 0:
        return np.empty((0, 3), dtype=float)

    masses_source = load_mass_samples_cached(SIGNAL_MASS_FILE, "signal", cache_key="signal")
    if masses_source.size < N:
        masses = np.tile(masses_source, int(np.ceil(N / masses_source.size)))
    else:
        masses = masses_source
    masses = masses[:N]

    # Sample times from p(t) ∝ sin^4(2πt/op + phi) on [0, exposure].
    # This avoids rejection-sampling inefficiency for very large periods.
    op = float(op)
    n_cycles = float(exposure) / max(op, 1e-12)
    n_grid = int(np.clip(np.ceil(64.0 * max(n_cycles, 1.0)), 8000, 400000))
    dt = float(exposure) / float(n_grid)
    t_grid = (np.arange(n_grid, dtype=float) + 0.5) * dt
    pdf = np.sin(2.0 * np.pi * t_grid / max(op, 1e-12) + float(phi)) ** 4
    pdf = np.clip(pdf, 0.0, None)
    pdf_sum = np.sum(pdf)
    if not np.isfinite(pdf_sum) or pdf_sum <= 0.0:
        pdf = np.full_like(t_grid, 1.0 / float(n_grid))
    else:
        pdf = pdf / pdf_sum

    idx = np.random.choice(n_grid, size=len(masses), replace=True, p=pdf)
    jitter = np.random.uniform(-0.5 * dt, 0.5 * dt, size=len(masses))
    times = np.clip(t_grid[idx] + jitter, 0.0, float(exposure))

    return np.stack((masses, times, np.ones_like(masses)), axis=-1)

def gen_bg(N):
    if int(N) <= 0:
        return np.empty((0, 3), dtype=float)

    masses_source = load_mass_samples_cached(BACKGROUND_MASS_FILE, "background", cache_key="background")
    if masses_source.size < N:
        masses = np.tile(masses_source, int(np.ceil(N / masses_source.size)))
    else:
        masses = masses_source
    masses = masses[:N]
    times = np.random.uniform(0, exposure, size=len(masses))
    return np.stack((masses, times, np.zeros_like(masses)), axis=-1)


# =============================================================================
# Scan setup
# =============================================================================
scan_output_dir = os.environ.get(
    "SCAN_OUTPUT_DIR",
    f"mc_scan_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
output_base = CURRENT_DIR / scan_output_dir
for sub in [
    "scatter",
    "hist",
    "train",
    "checkpoints",
    "roc",
    "preds",
    "residual",
    "mass_cut",
    "time_cut",
    "sanity",
    "nnoutput",
    "summary",
    "period_scan",
]:
    (output_base / sub).mkdir(parents=True, exist_ok=True)

print(f"[Output] All artifacts will be saved to: {output_base}")

SCAN_MODE = os.environ.get("SCAN_MODE", "log")  # options: "log", "piecewise", "narrow"
LOG_T_MIN = float(os.environ.get("LOG_T_MIN", "0.1"))
LOG_T_MAX = float(os.environ.get("LOG_T_MAX", "1e4"))
LOG_T_N = int(os.environ.get("LOG_T_N", "100"))
if LOG_T_N < 2:
    LOG_T_N = 2

phis_override_pi = parse_float_list_env("PHIS_PI")
if phis_override_pi is None:
    phi_mode = os.environ.get("PHI_MODE", "random").strip().lower()
    phi_n = int(os.environ.get("PHI_N", "20"))
    if phi_n < 1:
        phi_n = 1
    include_pi2 = os.environ.get("PHI_INCLUDE_PI2", "1").strip().lower() not in ("0", "false", "no")
    if phi_mode == "linspace":
        phis = np.linspace(0.0, np.pi, phi_n, endpoint=False)
        if include_pi2 and phis.size >= 1:
            idx = int(np.argmin(np.abs(phis - 0.5 * np.pi)))
            phis[idx] = 0.5 * np.pi
            phis[0] = 0.0
            phis = np.sort(np.unique(np.round(phis, 12)))
    elif phi_mode == "random":
        phi_seed = int(os.environ.get("PHI_SEED", "20260303"))
        rng_phi = np.random.default_rng(phi_seed)
        fixed = [0.0]
        if include_pi2:
            fixed.append(0.5 * np.pi)
        n_rand = max(0, phi_n - len(fixed))
        rand_part = rng_phi.uniform(0.0, np.pi, size=n_rand)
        phis = np.concatenate([np.asarray(fixed, dtype=float), rand_part]).astype(float)
        phis = np.mod(phis, np.pi)
        phis = np.sort(np.unique(np.round(phis, 12)))
        # Top up if de-duplication reduced the requested count.
        while phis.size < phi_n:
            cand = float(rng_phi.uniform(0.0, np.pi))
            if np.min(np.abs(phis - cand)) > 1e-6:
                phis = np.sort(np.append(phis, cand))
    else:
        raise ValueError(f"Unsupported PHI_MODE='{phi_mode}'. Use 'linspace' or 'random'.")
else:
    phis = np.asarray(phis_override_pi, dtype=float) * np.pi

if SCAN_MODE == "log":
    osc_periods = np.geomspace(LOG_T_MIN, LOG_T_MAX, LOG_T_N)
elif SCAN_MODE == "piecewise":
    osc_periods = np.array(
        [1.0, 1.3, 1.7, 2.2, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
         18.0, 24.0, 30.0, 36.0],
        dtype=float,
    )
else:
    osc_periods = np.linspace(10.0, 20.0, 11)

osc_periods_override = parse_float_list_env("OSC_PERIODS")
if osc_periods_override is not None:
    osc_periods = np.asarray(osc_periods_override, dtype=float)

N_SIG_EVENTS = int(os.environ.get("N_SIG_EVENTS", "10000"))
N_BG_EVENTS = int(os.environ.get("N_BG_EVENTS", "100000"))

# --- feature-bank configuration (independent of injected op) ---
# Use a fixed dictionary of candidate periods for the NN input.
# You can make this denser than osc_periods if you want.
TIME_FEATURE_MIN = float(os.environ.get("TIME_FEATURE_MIN", str(LOG_T_MIN)))
TIME_FEATURE_MAX = float(os.environ.get("TIME_FEATURE_MAX", str(LOG_T_MAX)))
TIME_FEATURE_N = int(os.environ.get("TIME_FEATURE_N", str(LOG_T_N)))
if TIME_FEATURE_N < 2:
    TIME_FEATURE_N = 2
TIME_FEATURE_PERIODS = np.geomspace(TIME_FEATURE_MIN, TIME_FEATURE_MAX, TIME_FEATURE_N)   # months
TIME_FEATURE_HARMONICS = 4                               # h=1..4
NN_LAYERS = [256, 256, 256]
NN_BATCH_SIZE = 1024
NN_LR = 1e-3

print(f"[Config] #T(injected)={len(osc_periods)}, #phi={len(phis)}, total combos={len(osc_periods) * len(phis)}")
print(f"[Config] log-scan T range [{LOG_T_MIN}, {LOG_T_MAX}] months, N={LOG_T_N}")
if phis_override_pi is None:
    print(f"[Config] phase sampling mode={phi_mode}, PHI_N={phi_n}, include_pi2={include_pi2}")
print(
    f"[Config] N_sig={N_SIG_EVENTS}, N_bg={N_BG_EVENTS}, "
    f"POI_SCAN=[{POI_SCAN_MIN}, {POI_SCAN_MAX}] with N={POI_SCAN_N}"
)
print(
    f"[TimeFeatures] periods={len(TIME_FEATURE_PERIODS)} in "
    f"[{TIME_FEATURE_MIN}, {TIME_FEATURE_MAX}] months, harmonics={TIME_FEATURE_HARMONICS}"
)
print(f"[TimeFeatures] input dimension = 1 + 2*harmonics*#periods = {1 + 2*TIME_FEATURE_HARMONICS*len(TIME_FEATURE_PERIODS)}")

colors = ["black", "red", "blue", "green", "yellow", "orange"]
caucs = [[] for _ in phis]
taucs = [[] for _ in phis]

fig6 = Fig6Recorder(osc_periods, phis, exposure_months=exposure, out_prefix="fig6_cathode_plus_T1e4")

_osc_periods_arr = np.asarray(osc_periods, dtype=float)
_T_FORCE = float(_osc_periods_arr[np.argmin(np.abs(_osc_periods_arr - 15.0))])
fig6_forced_exports = [(_T_FORCE, 0.0, "fig6_nnoutput_T15_phi0.00pi_custom")]

N_NULL_TOYS = int(os.environ.get("N_NULL_TOYS", "200"))
PLOT_TRAIN_FEATURES = False
period_scan_rows = []
period_scan_rows_joint = []
JOINT_PHI_PVALUE = os.environ.get("JOINT_PHI_PVALUE", "0") in ("1", "true", "True", "YES", "yes")
JOINT_PHASE_SEED_OFFSET = 17


# =============================================================================
# Main scan loop (injection study)
# =============================================================================
for op in osc_periods:
    print(f"[Scan] op={op:.2f} months")
    op_joint_records = []
    for idd, phi in enumerate(phis):
        phi_label = phi_over_pi(phi)
        phi_str = make_phi_tag(phi)

        print(f"====== op = {op} phi = {phi_label:.2f} π ({phi:.2f} rad)")

        # generate data
        sigs = gen_sig(N_SIG_EVENTS, op, phi)
        sigs_out = sigs[(sigs[:, 0] < mass_lo) | (sigs[:, 0] > mass_hi), :]
        sigs_in  = sigs[(sigs[:, 0] > mass_lo) & (sigs[:, 0] < mass_hi), :]
        bgs = gen_bg(N_BG_EVENTS)
        bgs_out = bgs[(bgs[:, 0] < mass_lo) | (bgs[:, 0] > mass_hi), :]
        bgs_in  = bgs[(bgs[:, 0] > mass_lo) & (bgs[:, 0] < mass_hi), :]

        # plot quick sanity scatter
        plt.scatter(bgs_in[:, 0], bgs_in[:, 1], label="bg in", s=5)
        plt.scatter(sigs_in[:, 0], sigs_in[:, 1], label="signal in", s=5)
        plt.scatter(bgs_out[:, 0], bgs_out[:, 1], label="bg out", s=5)
        plt.scatter(sigs_out[:, 0], sigs_out[:, 1], label="signal out", s=5)
        plt.xlabel("mass")
        plt.ylabel("Time [months]")
        plt.legend()
        save_current_figure_with_pickle(output_base / "scatter" / f"scatter_mass_time_mc_op{op:1.2f}_{phi_str}.pdf")
        plt.clf()

        ranges = [(450, 1200), (0, exposure), (-0.1, 1.1)]
        labels = ["mass", "Time [months]", "label"]

        for i in range(sigs_in.shape[1]):
            plt.hist(sigs_in[:, i], bins=40, range=ranges[i], label="signal", histtype="step")
            plt.hist(bgs_in[:, i],  bins=40, range=ranges[i], label="bg",     histtype="step")
            plt.xlabel(labels[i])
            plt.legend()
            save_current_figure_with_pickle(output_base / "hist" / f"hist_{i}_mc_op{op:1.2f}_{phi_str}.pdf")
            plt.clf()

        # split into train/val/test in signal mass window
        n_sig_in, n_sig_out = sigs_in.shape[0], sigs_out.shape[0]
        n_bg_in,  n_bg_out  = bgs_in.shape[0],  bgs_out.shape[0]

        outerdata_test = np.concatenate((sigs_out[int(n_sig_out * 0.75):, :],
                                         bgs_out[int(n_bg_out * 0.75):, :]))
        innerdata_train = np.concatenate((sigs_in[: int(n_sig_in * 0.5), :],
                                          bgs_in[: int(n_bg_in * 0.5), :]))
        innerdata_val = np.concatenate((sigs_in[int(n_sig_in * 0.5): int(n_sig_in * 0.75), :],
                                        bgs_in[int(n_bg_in * 0.5): int(n_bg_in * 0.75), :]))
        innerdata_test = np.concatenate((sigs_in[int(n_sig_in * 0.75):, :],
                                         bgs_in[int(n_bg_in * 0.75):, :]))

        # KDE for mass distribution in the signal window (for background sampling)
        m_scaler = LogitScaler(epsilon=1e-8)
        m_train = m_scaler.fit_transform(innerdata_train[:, 0:1])
        kde_model = KernelDensity(bandwidth=0.01, kernel="gaussian")
        kde_model.fit(m_train)

        # fixed oversampling so training conditions are identical across op
        sample_mult = 4
        m_samples = kde_model.sample(sample_mult * len(m_train)).astype(np.float32)
        m_samples = m_scaler.inverse_transform(m_samples)

        # sample time uniformly for background model (toy assumption)
        X_samples = np.random.uniform(0, exposure, size=len(m_samples)).reshape(-1, 1)
        samples = np.hstack([m_samples, X_samples, np.zeros((m_samples.shape[0], 1))])

        # sanity check: compare sampled vs background in the signal region
        for i in range(innerdata_test[:, :-1].shape[1]):
            _, binning, _ = plt.hist(innerdata_test[innerdata_test[:, -1] == 0, i],
                                     bins=100, label="data background",
                                     density=True, histtype="step")
            _ = plt.hist(samples[:, i],
                         bins=binning, label="sampled background",
                         density=True, histtype="step")
            plt.legend()
            plt.ylim(0, plt.gca().get_ylim()[1] * 1.2)
            plt.xlabel(f"feature {i}")
            plt.ylabel("counts (norm.)")
            save_current_figure_with_pickle(output_base / "sanity" / f"sanity{i}_mc_op{op:1.2f}_{phi_str}.pdf")
            plt.clf()

        # build classifier train/val sets: data labeled 1, samples labeled 0
        clsf_train_data = innerdata_train.copy()
        clsf_train_data[:, -1] = 1.0
        clsf_val_data = innerdata_val.copy()
        clsf_val_data[:, -1] = 1.0

        n_train = len(clsf_train_data)
        n_val = len(clsf_val_data)
        n_samples_train = int(n_train / (n_train + n_val) * len(samples))
        samples_train = samples[:n_samples_train]
        samples_val = samples[n_samples_train:]

        clsf_train_set = np.vstack([clsf_train_data, samples_train])
        clsf_val_set = np.vstack([clsf_val_data, samples_val])
        clsf_train_set = shuffle(clsf_train_set, random_state=42)
        clsf_val_set = shuffle(clsf_val_set, random_state=42)

        # --- period-agnostic features here ---
        inner_scaler = StandardScaler()
        inner_scaler.fit(
            add_time_feature_bank(
                clsf_train_set[:, 1], TIME_FEATURE_PERIODS, TIME_FEATURE_HARMONICS,
                exposure=exposure, include_linear=True
            )
        )

        X_train = inner_scaler.transform(
            add_time_feature_bank(
                clsf_train_set[:, 1], TIME_FEATURE_PERIODS, TIME_FEATURE_HARMONICS,
                exposure=exposure, include_linear=True
            )
        )
        y_train = clsf_train_set[:, -1]
        X_val = inner_scaler.transform(
            add_time_feature_bank(
                clsf_val_set[:, 1], TIME_FEATURE_PERIODS, TIME_FEATURE_HARMONICS,
                exposure=exposure, include_linear=True
            )
        )
        y_val = clsf_val_set[:, -1]

        print("Xtrain shape:", X_train.shape)

        if PLOT_TRAIN_FEATURES:
            for i in range(X_train.shape[1]):
                _, bins, _ = plt.hist(X_train[y_train[:] == 0, i], bins=50, range=(-3, 3),
                                      label="train bg", density=True, histtype="step")
                _ = plt.hist(X_train[y_train[:] == 1, i], bins=bins,
                             label="train data", density=True, histtype="step")
                plt.legend()
                plt.xlabel(f"feature {i}")
                plt.ylabel("counts (norm.)")
                save_current_figure_with_pickle(output_base / "train" / f"train{i}_mc_op{op:1.2f}_{phi_str}.pdf")
                plt.clf()

        classifier_savedir = str(output_base / "checkpoints" / f"clf_op{op:1.2f}_{phi_str}")
        if exists(classifier_savedir):
            shutil.rmtree(classifier_savedir)

        patience = 15
        classifier_model = NeuralNetworkClassifier(
            save_path=classifier_savedir,
            n_inputs=X_train.shape[1],
            layers=NN_LAYERS,
            batch_size=NN_BATCH_SIZE,
            lr=NN_LR,
            use_class_weights=False,
            early_stopping=True,
            patience=patience,
            epochs=None,
            verbose=True,
        )
        classifier_model.fit(X_train, y_train, X_val, y_val)

        # --- Fig6 expected limits: learned weights -> weighted mass templates -> pyhf mu95 ---
        sig_times_all = sigs[:, 1]
        bg_times_all = bgs[:, 1]
        if sig_times_all.size and bg_times_all.size:
            X_sig = inner_scaler.transform(
                add_time_feature_bank(
                    sig_times_all,
                    TIME_FEATURE_PERIODS,
                    TIME_FEATURE_HARMONICS,
                    exposure=exposure,
                    include_linear=True,
                )
            )
            X_bg = inner_scaler.transform(
                add_time_feature_bank(
                    bg_times_all,
                    TIME_FEATURE_PERIODS,
                    TIME_FEATURE_HARMONICS,
                    exposure=exposure,
                    include_linear=True,
                )
            )

            w_sig_raw = prob_to_density_ratio(classifier_model.predict(X_sig).ravel(), y_train)
            w_bg_raw = prob_to_density_ratio(classifier_model.predict(X_bg).ravel(), y_train)

            w_sig = normalize_to_bg_mean(w_sig_raw, w_bg_raw)
            w_bg = normalize_to_bg_mean(w_bg_raw, w_bg_raw)

            sig_hist_learned, bkg_hist_learned = build_weighted_mass_templates(
                sig_masses=sigs[:, 0],
                bg_masses=bgs[:, 0],
                w_sig=w_sig,
                w_bg=w_bg,
                mass_bins=MASS_BINS,
            )
            sig_hist_ti, bkg_hist_ti = build_weighted_mass_templates(
                sig_masses=sigs[:, 0],
                bg_masses=bgs[:, 0],
                w_sig=np.ones_like(sigs[:, 0], dtype=float),
                w_bg=np.ones_like(bgs[:, 0], dtype=float),
                mass_bins=MASS_BINS,
            )
            sig_hist_true, bkg_hist_true = build_weighted_mass_templates(
                sig_masses=sigs[:, 0],
                bg_masses=bgs[:, 0],
                w_sig=lr_time_bg_normalized(sigs[:, 1], float(op), float(phi)),
                w_bg=lr_time_bg_normalized(bgs[:, 1], float(op), float(phi)),
                mass_bins=MASS_BINS,
            )

            mu95_learned = pyhf_expected_mu_limit(sig_hist_learned, bkg_hist_learned)
            mu95_ti = pyhf_expected_mu_limit(sig_hist_ti, bkg_hist_ti)
            mu95_true = pyhf_expected_mu_limit(sig_hist_true, bkg_hist_true)
            ratio = (
                mu95_learned / mu95_ti
                if np.isfinite(mu95_learned) and np.isfinite(mu95_ti) and mu95_ti > 0.0
                else np.nan
            )
            ratio_true = (
                mu95_true / mu95_ti
                if np.isfinite(mu95_true) and np.isfinite(mu95_ti) and mu95_ti > 0.0
                else np.nan
            )
            order_ok = (
                np.isfinite(mu95_true)
                and np.isfinite(mu95_learned)
                and np.isfinite(mu95_ti)
                and (mu95_true < mu95_learned < mu95_ti)
            )
            print(
                f"[Fig6] op={op:.3f}, phi={phi/np.pi:.2f}pi: "
                f"mu95_true={mu95_true:.5g}, mu95_learned={mu95_learned:.5g}, "
                f"mu95_ti={mu95_ti:.5g}, ratio_true={ratio_true:.5g}, ratio={ratio:.5g}, "
                f"order_ok={order_ok}"
            )
            if not order_ok:
                print(
                    f"[Fig6][SanityWarn] Expected ordering violated at "
                    f"op={op:.3f}, phi={phi/np.pi:.2f}pi: "
                    f"mu95_true={mu95_true:.5g}, mu95_learned={mu95_learned:.5g}, mu95_ti={mu95_ti:.5g}"
                )
            fig6.record(op, phi, mu95_learned=mu95_learned, mu95_ti=mu95_ti)

        # plot NN output vs time for every point (can be heavy; keep for now)
        fig6_plot_nnoutput_vs_time(
            classifier_model,
            inner_scaler,
            exposure,
            float(op),
            phi,
            y_train_for_score=y_train,
            periods_feat=TIME_FEATURE_PERIODS,
            n_harmonics_feat=TIME_FEATURE_HARMONICS,
            out_prefix=str(output_base / "nnoutput" / "nnoutput"),
        )

        for T_sel, phi_sel, prefix in fig6_forced_exports:
            if np.isclose(float(op), float(T_sel), rtol=0.0, atol=1e-9) and np.isclose(
                float(phi), float(phi_sel), rtol=0.0, atol=1e-12
            ):
                fig6_plot_nnoutput_vs_time(
                    classifier_model,
                    inner_scaler,
                    exposure,
                    float(op),
                    phi,
                    y_train_for_score=y_train,
                    periods_feat=TIME_FEATURE_PERIODS,
                    n_harmonics_feat=TIME_FEATURE_HARMONICS,
                    out_prefix=prefix,
                )

        # evaluate ROC/AUC vs true LR(t)
        X_test_raw = add_time_feature_bank(
            innerdata_test[:, 1], TIME_FEATURE_PERIODS, TIME_FEATURE_HARMONICS,
            exposure=exposure, include_linear=True
        )
        X_test = inner_scaler.transform(X_test_raw)
        y_test = innerdata_test[:, -1]
        Xo_test = inner_scaler.transform(
            add_time_feature_bank(outerdata_test[:, 1], TIME_FEATURE_PERIODS, TIME_FEATURE_HARMONICS,
                                  exposure=exposure, include_linear=True)
        )
        yo_test = outerdata_test[:, -1]

        preds_test = classifier_model.predict(X_test).ravel()
        predso_test = classifier_model.predict(Xo_test).ravel()
        r_test = prob_to_density_ratio(preds_test, y_train)
        r_o_test = prob_to_density_ratio(predso_test, y_train)

        # Period-agnostic discovery *diagnostic*: scan periods from NN score vs raw time
        best_T, best_power, ls_powers = scan_period_lomb_scargle(innerdata_test[:, 1], r_test, osc_periods)
        null_max_powers = estimate_global_pvalue_null_toys(
            classifier_model=classifier_model,
            inner_scaler=inner_scaler,
            exposure=exposure,
            periods_feat=TIME_FEATURE_PERIODS,
            n_harmonics_feat=TIME_FEATURE_HARMONICS,
            period_grid_scan=osc_periods,
            n_events=len(innerdata_test),
            y_train_for_score=y_train,
            n_toys=N_NULL_TOYS,
            rng=np.random.default_rng(20260217 + int(1000 * op) + int(1000 * phi)),
        )
        p_global = (1.0 + np.sum(null_max_powers >= best_power)) / (len(null_max_powers) + 1.0)
        print(
            f"[PeriodScan] injected T={op:.3f}, phi={phi_label:.2f}pi -> "
            f"T_hat={best_T:.3f}, max_power={best_power:.4f}, global_p={p_global:.4g} "
            f"(N_toys={len(null_max_powers)})"
        )
        period_scan_rows.append([float(op), float(phi), float(best_T), float(best_power), float(p_global), int(len(null_max_powers))])
        if JOINT_PHI_PVALUE:
            op_joint_records.append(
                {
                    "phi": float(phi),
                    "phi_label": float(phi_label),
                    "inner_times": innerdata_test[:, 1].copy(),
                    "inner_times_feature_raw": X_test_raw,
                    "n_events": int(len(innerdata_test)),
                    "best_T_conditional": float(best_T),
                    "best_power_conditional": float(best_power),
                    "classifier_model": classifier_model,
                    "inner_scaler": inner_scaler,
                    "y_train": y_train.copy(),
                }
            )
        plot_period_scan(
            period_grid=osc_periods,
            powers=ls_powers,
            best_period=best_T,
            injected_period=op,
            p_global=p_global,
            out_path=output_base / "period_scan" / f"period_scan_mc_op{op:1.2f}_{phi_str}.pdf",
        )

        # hist of NN output
        pred_signal = np.concatenate((preds_test[y_test[:] == 1], predso_test[yo_test[:] == 1]))
        pred_bg = np.concatenate((preds_test[y_test[:] == 0], predso_test[yo_test[:] == 0]))
        pred_density_signal, pred_edges = np.histogram(pred_signal, bins=500, range=(0, 1), density=True)
        pred_density_bg, _ = np.histogram(pred_bg, bins=pred_edges, density=True)
        plt.hist(pred_signal, bins=pred_edges, label="signal", histtype="step", density=True)
        plt.hist(pred_bg, bins=pred_edges, label="bg", histtype="step", density=True)
        plt.legend()
        plt.xlabel("NN output")
        preds_out = output_base / "preds" / f"preds_mc_op{op:1.2f}_{phi_str}.pdf"
        save_current_figure_with_pickle(preds_out)
        plt.clf()
        save_named_series_csv(
            preds_out.with_suffix(".csv"),
            {
                "bin_left": pred_edges[:-1],
                "bin_right": pred_edges[1:],
                "signal_density": pred_density_signal,
                "bg_density": pred_density_bg,
            },
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            fpr, tpr, _ = roc_curve(y_test, preds_test)
        croc_auc = auc(fpr, tpr)

        preds_lrt = lr_time(innerdata_test[:, 1], op, phi)
        with np.errstate(divide="ignore", invalid="ignore"):
            fpr_t, tpr_t, _ = roc_curve(y_test, preds_lrt)
        troc_auc = auc(fpr_t, tpr_t)

        print("c AUC =", croc_auc)
        print("t AUC =", troc_auc)

        taucs[idd].append(troc_auc)
        caucs[idd].append(croc_auc)

        plt.plot(tpr, 1.0 - fpr, label="CATHODE")
        plt.plot(tpr_t, 1.0 - fpr_t, label="LR(t)")
        plt.xlabel("True Positive Rate")
        plt.ylabel("1 - False positive rate")
        plt.legend(loc="upper right")
        roc_out = output_base / "roc" / f"roc_mc_op{op:1.2f}_{phi_str}.pdf"
        save_current_figure_with_pickle(roc_out)
        plt.clf()
        save_named_series_csv(
            roc_out.with_suffix(".csv"),
            {
                "fpr_cathode": fpr,
                "tpr_cathode": tpr,
                "fpr_true": fpr_t,
                "tpr_true": tpr_t,
            },
        )

        # significance improvement curve
        with np.errstate(divide="ignore", invalid="ignore"):
            sic = tpr / np.sqrt(fpr)
            sic = np.where(np.isfinite(sic), sic, 0.0)
            random_tpr = np.linspace(0, 1, 300)
            random_sic = random_tpr / np.sqrt(random_tpr)
            random_sic = np.where(np.isfinite(random_sic), random_sic, 0.0)

        plt.plot(tpr, sic, label="CATHODE")
        plt.plot(random_tpr, random_sic, ":", label="random")
        plt.xlabel("True Positive Rate")
        plt.ylabel("Significance Improvement")
        plt.legend(loc="upper right")
        sic_out = output_base / "residual" / f"res_mc_op{op:1.2f}_{phi_str}.pdf"
        save_current_figure_with_pickle(sic_out)
        plt.clf()
        save_named_series_csv(
            sic_out.with_suffix(".csv"),
            {
                "tpr_cathode": tpr,
                "sic_cathode": sic,
                "random_tpr": random_tpr,
                "random_sic": random_sic,
            },
        )

        # money plots (mass/time weighted)
        r_test_np = np.array(r_test).reshape(-1)
        r_o_test_np = np.array(r_o_test).reshape(-1)

        mass_edges = np.linspace(MASS_RANGE[0], MASS_RANGE[1], N_MASS_BINS + 1)
        mass_bg = np.concatenate((outerdata_test[int(n_sig_out * 0.25):, 0], innerdata_test[int(n_sig_in * 0.25):, 0]))
        mass_sig = np.concatenate((outerdata_test[:int(n_sig_out * 0.25), 0], innerdata_test[:int(n_sig_in * 0.25), 0]))
        mass_bg_w = np.concatenate((r_o_test_np[int(n_sig_out * 0.25):], r_test_np[int(n_sig_in * 0.25):]))
        mass_sig_w = np.concatenate((r_o_test_np[:int(n_sig_out * 0.25)], r_test_np[:int(n_sig_in * 0.25)]))
        mass_bg_counts, _ = np.histogram(mass_bg, bins=mass_edges)
        mass_sig_counts, _ = np.histogram(mass_sig, bins=mass_edges)
        mass_bg_weighted, _ = np.histogram(mass_bg, bins=mass_edges, weights=mass_bg_w)
        mass_sig_weighted, _ = np.histogram(mass_sig, bins=mass_edges, weights=mass_sig_w)
        plt.hist(mass_bg, bins=mass_edges, label="Bg", histtype="step", color="black")
        plt.hist(mass_sig, bins=mass_edges, label="Sig", histtype="step", color="red")
        plt.hist(mass_bg, weights=mass_bg_w, bins=mass_edges, label="Bg NN weighted", histtype="step", color="black", linestyle=":")
        plt.hist(mass_sig, weights=mass_sig_w, bins=mass_edges, label="Sig NN weighted", histtype="step", color="red", linestyle=":")
        plt.xlabel("Mass")
        plt.legend()
        mass_out = output_base / "mass_cut" / f"mass_cut_mc_op{op:1.2f}_{phi_str}.pdf"
        save_current_figure_with_pickle(mass_out)
        plt.clf()
        save_named_series_csv(
            mass_out.with_suffix(".csv"),
            {
                "bin_left": mass_edges[:-1],
                "bin_right": mass_edges[1:],
                "bg_counts": mass_bg_counts,
                "sig_counts": mass_sig_counts,
                "bg_nn_weighted": mass_bg_weighted,
                "sig_nn_weighted": mass_sig_weighted,
            },
        )

        time_edges = np.linspace(0.0, exposure, 101)
        time_all = np.concatenate((outerdata_test[:, 1], innerdata_test[:, 1]))
        time_weights = np.concatenate((r_o_test_np[:], r_test_np[:]))
        time_counts, _ = np.histogram(time_all, bins=time_edges)
        time_weighted, _ = np.histogram(time_all, bins=time_edges, weights=time_weights)
        plt.hist(time_all, bins=time_edges, label="Data", histtype="step", color="black")
        _ = plt.hist(time_all, weights=time_weights, bins=time_edges, label="Data weighted", histtype="step", color="black", linestyle=":")
        plt.xlabel("Time [months]")
        plt.legend()
        time_out = output_base / "time_cut" / f"time_cut_mc_op{op:1.2f}_{phi_str}.pdf"
        save_current_figure_with_pickle(time_out)
        plt.clf()
        save_named_series_csv(
            time_out.with_suffix(".csv"),
            {
                "bin_left": time_edges[:-1],
                "bin_right": time_edges[1:],
                "data_counts": time_counts,
                "data_weighted": time_weighted,
            },
        )

    if JOINT_PHI_PVALUE and len(op_joint_records) > 0:
        if len(op_joint_records) != len(phis):
            print(
                f"[Warn] op={op:.3f} has incomplete phi scan "
                f"({len(op_joint_records)}/{len(phis)} records); skipping joint-phi p-value."
            )
        else:
            model_specs = [
                (entry["classifier_model"], entry["inner_scaler"], entry["y_train"])
                for entry in op_joint_records
            ]
            for rec in op_joint_records:
                observed_joint_power = -np.inf
                observed_joint_T = rec["best_T_conditional"]
                observed_joint_phi = rec["phi"]

                for model_entry in op_joint_records:
                    s_model = prob_to_density_ratio(
                        model_entry["classifier_model"].predict(
                            model_entry["inner_scaler"].transform(rec["inner_times_feature_raw"])
                        ).ravel(),
                        model_entry["y_train"],
                    )
                    cand_T, pmax, _ = scan_period_lomb_scargle(rec["inner_times"], s_model, osc_periods)
                    if pmax > observed_joint_power:
                        observed_joint_power = float(pmax)
                        observed_joint_phi = model_entry["phi"]
                        observed_joint_T = float(cand_T)

                null_max_powers_joint = estimate_joint_global_pvalue_null_toys(
                    model_specs=model_specs,
                    exposure=exposure,
                    periods_feat=TIME_FEATURE_PERIODS,
                    n_harmonics_feat=TIME_FEATURE_HARMONICS,
                    period_grid_scan=osc_periods,
                    n_events=rec["n_events"],
                    n_toys=N_NULL_TOYS,
                    rng=np.random.default_rng(
                        20260217 + int(1000 * op) + JOINT_PHASE_SEED_OFFSET + int(1000 * rec["phi"])
                    ),
                )
                p_global_joint = (1.0 + np.sum(null_max_powers_joint >= observed_joint_power)) / (
                    len(null_max_powers_joint) + 1.0
                )
                period_scan_rows_joint.append(
                    [
                        float(op),
                        float(rec["phi"]),
                        float(observed_joint_phi),
                        float(observed_joint_T),
                        float(observed_joint_power),
                        float(p_global_joint),
                        int(len(null_max_powers_joint)),
                    ]
                )
                print(
                    f"[PeriodScan-JointPhi] injected T={op:.3f}, injected phi={rec['phi_label']:.2f}pi -> "
                    f"T_hat={observed_joint_T:.3f}, phi_hat={observed_joint_phi/np.pi:.2f}pi, "
                    f"max_power={observed_joint_power:.4f}, global_p={p_global_joint:.4g} "
                    f"(N_toys={len(null_max_powers_joint)})"
                )


# =============================================================================
# Summary plots
# =============================================================================
for i, phi in enumerate(phis):
    phi_label = phi / np.pi
    plt.plot(osc_periods, taucs[i], "-", label=rf"True $\phi$={phi_label:.2f}$\pi$", color=colors[i % len(colors)])
    plt.plot(osc_periods, caucs[i], ":", label=rf"CATHODE $\phi$={phi_label:.2f}$\pi$", color=colors[i % len(colors)])

plt.xlabel("Oscillation period T [months]")
plt.ylabel("AUC")
plt.xscale("log")
plt.legend().get_frame().set_linewidth(0)
plt.tight_layout()
save_current_figure_with_pickle(output_base / "summary" / "auc_cathode.pdf")
plt.clf()

plt.figure(figsize=(8, 6))
for idx, phi in enumerate(phis):
    phi_label = phi / np.pi
    periods_array = np.array(osc_periods)
    auc_true = np.array(taucs[idx])
    auc_nn = np.array(caucs[idx])
    plt.plot(periods_array, auc_true, "--", lw=2.0, label=rf"True LR $\phi$={phi_label:.2f}$\pi$")
    plt.plot(periods_array, auc_nn, "-o", lw=2.0, label=rf"NN $\phi$={phi_label:.2f}$\pi$")
plt.xscale("log")
plt.xlabel("Oscillation period T [months]")
plt.ylabel("AUC")
plt.ylim(0.45, 0.9)
plt.legend(frameon=False)
plt.tight_layout()
save_current_figure_with_pickle(output_base / "summary" / "auc_vs_period.pdf")
plt.close()

auc_rows = []
for i, phi in enumerate(phis):
    for period, auc_true, auc_nn in zip(osc_periods, taucs[i], caucs[i]):
        auc_rows.append([float(period), float(phi), phi_over_pi(phi), float(auc_true), float(auc_nn)])
save_rows_csv(
    output_base / "summary" / "auc_curves.csv",
    "T_months,phi_rad,phi_over_pi,auc_true,auc_nn",
    auc_rows,
)

taucs_arr = np.asarray(taucs, dtype=float)
caucs_arr = np.asarray(caucs, dtype=float)
save_named_series_csv(
    output_base / "summary" / "auc_average.csv",
    {
        "T_months": np.asarray(osc_periods, dtype=float),
        "auc_true_avg": np.nanmean(taucs_arr, axis=0),
        "auc_nn_avg": np.nanmean(caucs_arr, axis=0),
    },
)

if len(period_scan_rows) > 0:
    period_scan_rows = np.asarray(period_scan_rows, dtype=float)
    out_csv = output_base / "summary" / "period_scan_conditional_global_pvalues.csv"
    header = "injected_period_months,phi_rad,bestfit_period_months,max_ls_power,global_pvalue_conditional,n_null_toys"
    np.savetxt(out_csv, period_scan_rows, delimiter=",", header=header, comments="")
    print(f"[Summary] Saved period-scan conditional-global p-values: {out_csv}")

if len(period_scan_rows_joint) > 0:
    period_scan_rows_joint = np.asarray(period_scan_rows_joint, dtype=float)
    out_csv = output_base / "summary" / "period_scan_joint_phi_global_pvalues.csv"
    header = (
        "injected_period_months,injected_phi_rad,"
        "bestfit_phi_rad,bestfit_period_months,max_ls_power_joint,"
        "global_pvalue_joint_across_phi,n_null_toys"
    )
    np.savetxt(out_csv, period_scan_rows_joint, delimiter=",", header=header, comments="")
    print(f"[Summary] Saved period-scan joint-phi global p-values: {out_csv}")

fig6.plot_limits(
    ylabel="Expected limit ratio",
    time_independent_level=1.0,
)
