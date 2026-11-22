"""NeuroReader Extreme-Value Theory & Tail-Risk Engine (NeuroRisk Integration).

Adapts the NeuroRisk mathematical formulation natively to Medtronic Percept JSON
exports (multi-amplitude & multi-frequency BrainSenseTimeDomain sweeps, plus fallback to
LfpMontageTimeDomain survey contact arms when a session has constant amplitude).

Core concepts (matching neurorisk.py):
  - Causal RBJ 4th-order bandpass + causal moving-RMS envelope ratio
        z(t) = e(t) / median(e_0mA)
    sampled every 20 ms (50 Hz, DEC=5 at FS=250 Hz) after 0.5 s settling (SETTLE=125).
  - Targets:
      * beta: patient-specific 1/f-corrected beta peak ± 3 Hz (bursts)
      * ftg:  dynamic half-stimulation-rate band f_stim / 2 ± ftg_hw (FTG surges)
  - VaR_α & CVaR_α (Expected Shortfall) per arm (α = 0.95 or 0.99)
  - POT + GPD above clinical red line u = P90(0 mA), runs-declustered (100 ms / RUN=5):
      shape ξ, scale σ, cluster rate λ_0, extremal index θ, return levels + bootstrap 90% CI
  - Breach-rate Poisson GLM: log λ(a) = r_0 + r_1·a → P(≥1 breach in H) = 1 − exp(−λ(a)·H)
  - Block maxima (1-s blocks, BLOCK=50) + shared-shape GEV MLE → H-second return level
  - EVT-VaR_99.9 & EVT-ES_99.9 (POT above arm's own P90 with contact's ξ)
  - 5 Risk-Based Selection Rules + Kupiec VaR Calibration Backtest & MRL/ξ Diagnostics
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.signal import sosfilt, welch

FS = 250.0
SETTLE = 125        # 0.5 s settle after amplitude transition
DEC = 5             # 20 ms envelope step (50 Hz feature rate)
BLOCK = 50          # 1-s block maxima (50 steps @ 50 Hz)
RUN = 5             # runs declustering: clusters separated by ≥ 100 ms below u
U_Q = 90.0          # red line u = P90 of 0 mA envelope ratio
FTG_ENV_N = 25      # moving-RMS samples for FTG envelope (100 ms @ 250 Hz)
BW_Q = (0.5411961, 1.3065630)
T_GRID = [round(float(v), 2) for v in np.logspace(0.0, math.log10(600.0), 32)]

DEFAULT_RISK_PARAMS: dict[str, Any] = {
    "hemisphere": "Left",
    "stim_rate_filter": "ALL",
    "target": "beta",
    "alpha": 0.99,
    "horizon": 60.0,
    "kappa": 0.6,
    "eps": 0.10,
    "metric": "CVaR",
}


# ── Causal Filter & Envelope Primitives (exact mirror of neurorisk.py / neurorisk.js) ──
def _rbj(kind: str, f: float, q: float, fs: float = FS) -> np.ndarray:
    f = max(0.5, min(float(f), 0.48 * fs))
    w0 = 2.0 * math.pi * f / fs
    c, al = math.cos(w0), math.sin(w0) / (2.0 * q)
    if kind == "lp":
        b = [(1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0]
    else:
        b = [(1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0]
    a0 = 1.0 + al
    return np.array([b[0] / a0, b[1] / a0, b[2] / a0, 1.0, -2.0 * c / a0, (1.0 - al) / a0], dtype=float)


def _bandpass(lo: float, hi: float, fs: float = FS) -> np.ndarray:
    lo = max(1.0, min(float(lo), 0.45 * fs))
    hi = max(lo + 1.0, min(float(hi), 0.48 * fs))
    return np.vstack([_rbj("hp", lo, q, fs) for q in BW_Q] + [_rbj("lp", hi, q, fs) for q in BW_Q])


def _mov_rms(y: np.ndarray, n: int) -> np.ndarray:
    n = max(2, int(n))
    return np.sqrt(2.0 * np.maximum(np.convolve(y**2, np.ones(n, dtype=float))[: y.size] / n, 0.0))


def _beta_peak(x: np.ndarray, fs: float = FS) -> float:
    if len(x) < 500:
        return 20.0
    f, p = welch(x, fs=fs, nperseg=min(len(x), 500))
    p = np.maximum(p, 1e-12)
    fit = (f >= 3.0) & (f <= 45.0) & ~((f >= 12.0) & (f <= 32.0))
    if np.sum(fit) < 4:
        b = (f >= 13.0) & (f <= 30.0)
        return float(f[b][np.argmax(p[b])]) if np.any(b) else 20.0
    c = np.polyfit(np.log10(f[fit]), np.log10(p[fit]), 1)
    res = np.log10(p) - np.polyval(c, np.log10(np.maximum(f, 1e-3)))
    b = (f >= 13.0) & (f <= 30.0)
    return float(f[b][np.argmax(res[b])]) if np.any(b) else 20.0


def _segment_amplitude_runs(
    n_samples: int,
    fs: float,
    amp_t: np.ndarray,
    amp_ma: np.ndarray,
) -> list[dict[str, Any]]:
    if len(amp_t) == 0 or len(amp_ma) == 0:
        return [{"start": 0, "n": n_samples, "amp": 0.0}]
    q_amp = np.round(amp_ma.astype(float), 1)
    segs = []
    cur_amp = float(q_amp[0])
    cur_start_s = 0.0
    for i in range(1, len(q_amp)):
        if abs(float(q_amp[i]) - cur_amp) >= 0.05:
            end_s = float(amp_t[i])
            s0 = max(0, int(round(cur_start_s * fs)))
            s1 = min(n_samples, int(round(end_s * fs)))
            if s1 > s0:
                segs.append({"start": s0, "n": s1 - s0, "amp": cur_amp})
            cur_amp = float(q_amp[i])
            cur_start_s = end_s
    s0 = max(0, int(round(cur_start_s * fs)))
    if n_samples > s0:
        segs.append({"start": s0, "n": n_samples - s0, "amp": cur_amp})
    return segs or [{"start": 0, "n": n_samples, "amp": float(q_amp[0])}]


# ── EVT Toolkit (exact mirror of neurorisk.py) ──
def decluster(z: np.ndarray, u: float, run: int = RUN) -> tuple[np.ndarray, int, np.ndarray]:
    """Runs declustering: cluster maxima, total exceedances, and peak indices."""
    ex = np.flatnonzero(z > u)
    if ex.size == 0:
        return np.array([], dtype=float), 0, np.array([], dtype=int)
    brk = np.flatnonzero(np.diff(ex) > run)
    starts = np.r_[0, brk + 1]
    ends = np.r_[brk, ex.size - 1]
    peaks = []
    peak_idx = []
    for a, b in zip(starts, ends):
        sl = z[ex[a] : ex[b] + 1]
        k = int(np.argmax(sl))
        peaks.append(float(sl[k]))
        peak_idx.append(int(ex[a] + k))
    return np.asarray(peaks, dtype=float), int(ex.size), np.asarray(peak_idx, dtype=int)


def _gpd_pwm(excess: np.ndarray) -> tuple[float, float]:
    """Fast closed-form Probability Weighted Moments (Hosking & Wallis) GPD fit."""
    ex = np.sort(np.asarray(excess, dtype=float))
    ex = ex[ex > 0]
    k = ex.size
    if k < 3:
        m = float(np.mean(ex)) if k > 0 else 0.1
        return 0.0, max(m, 1e-4)
    w = 1.0 - (np.arange(1, k + 1, dtype=float) - 0.35) / k
    a0 = float(np.mean(ex))
    a1 = float(np.mean(w * ex))
    denom = a0 - 2.0 * a1
    if abs(denom) < 1e-9 or a0 <= 0:
        return 0.0, max(a0, 1e-4)
    xi = float(np.clip(2.0 - a0 / denom, -0.55, 0.55))
    sigma = float(max(2.0 * a0 * a1 / denom, 1e-4))
    return xi, sigma


def gpd_fit(excess: np.ndarray) -> tuple[float, float]:
    """MLE GPD fit (loc=0) with PWM fallback for sparse tails."""
    ex = np.asarray(excess, dtype=float)
    ex = ex[np.isfinite(ex) & (ex > 0)]
    if ex.size < 5:
        return _gpd_pwm(ex)
    try:
        c, _, sc = stats.genpareto.fit(ex, floc=0)
        if np.isfinite(c) and np.isfinite(sc) and sc > 0 and -0.8 < c < 0.8:
            return float(c), float(sc)
    except Exception:
        pass
    return _gpd_pwm(ex)


def pot_return_level(u: float, xi: float, sigma: float, lam: np.ndarray | float, T: np.ndarray | float) -> np.ndarray:
    m = np.asarray(lam, dtype=float) * np.asarray(T, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if abs(xi) > 1e-6:
            rl = u + (sigma / xi) * (np.power(np.maximum(m, 1e-12), xi) - 1.0)
        else:
            rl = u + sigma * np.log(np.maximum(m, 1e-12))
    return np.where(m > 1.0, rl, np.nan)


def block_maxima(seqs_z: list[np.ndarray], block: int = BLOCK) -> np.ndarray:
    out = []
    for z in seqs_z:
        nb = len(z) // block
        for i in range(nb):
            out.append(float(np.max(z[i * block : (i + 1) * block])))
    return np.asarray(out, dtype=float)


def _gev_logpdf_sum(x: np.ndarray, mu: float, sg: float, xi: float) -> float:
    """Fast exact GEV log-likelihood sum (Coles sign convention, matching scipy.stats.genextreme(-xi))."""
    z = (x - mu) / sg
    if abs(xi) > 1e-6:
        t = 1.0 + xi * z
        if np.any(t <= 1e-12):
            return -1e12
        inv_xi = 1.0 / xi
        return float(-x.size * math.log(sg) - (1.0 + inv_xi) * np.sum(np.log(t)) - np.sum(np.power(t, -inv_xi)))
    return float(-x.size * math.log(sg) - np.sum(z) - np.sum(np.exp(-np.clip(z, -40.0, 40.0))))


def gev_shared_fit(groups: list[np.ndarray]) -> dict[str, Any]:
    """Joint GEV MLE: arm-specific (μ, log σ), one shape ξ shared by all arms of a series."""
    g = [np.asarray(x, dtype=float) for x in groups if len(x) >= 2]
    k = len(g)
    if k == 0:
        return {"xi": 0.0, "params": [], "ok": False}
    x0 = np.r_[
        np.concatenate([[float(x.mean() - 0.45 * x.std()), math.log(max(float(x.std()) * 0.78, 1e-3))] for x in g]),
        0.0,
    ]

    def nll(p: np.ndarray) -> float:
        xi = float(p[-1])
        tot = 0.0
        for i, x in enumerate(g):
            mu, sg = float(p[2 * i]), math.exp(float(np.clip(p[2 * i + 1], -8.0, 5.0)))
            val = _gev_logpdf_sum(x, mu, sg, xi)
            if val <= -1e11 or not math.isfinite(val):
                return 1e12
            tot -= val
        return tot if math.isfinite(tot) else 1e12

    try:
        r = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=[(None, None)] * (2 * k) + [(-0.55, 0.55)])
        p = r.x
        return {
            "xi": float(p[-1]),
            "params": [(float(p[2 * i]), float(math.exp(np.clip(p[2 * i + 1], -8.0, 5.0)))) for i in range(k)],
            "ok": bool(r.success),
        }
    except Exception:
        return {
            "xi": 0.0,
            "params": [(float(x.mean() - 0.45 * x.std()), max(float(x.std()) * 0.78, 1e-3)) for x in g],
            "ok": False,
        }


def gev_return_level(mu: float, sigma: float, xi: float, T: float | np.ndarray) -> float | np.ndarray:
    T_arr = np.maximum(np.asarray(T, dtype=float), 1.001)
    y = -np.log(1.0 - 1.0 / T_arr)
    if abs(xi) < 1e-6:
        val = mu - sigma * np.log(y)
    else:
        val = mu + (sigma / xi) * (np.power(y, -xi) - 1.0)
    return float(val) if np.ndim(T) == 0 else val


def poisson_rate_fit(amps: np.ndarray, counts: np.ndarray, exposure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """log λ(a) = r0 + r1·a, Poisson MLE with exposure; returns params and covariance."""
    amps = np.asarray(amps, dtype=float)
    counts = np.asarray(counts, dtype=float)
    exposure = np.maximum(np.asarray(exposure, dtype=float), 1e-6)
    base_rate = max(float(counts.sum()) / max(float(exposure.sum()), 1e-6), 1e-3)

    if len(np.unique(amps)) < 2 or counts.sum() == 0:
        return np.array([math.log(base_rate), 0.0]), np.diag([0.25, 0.05])

    def nll(p: np.ndarray) -> float:
        eta = np.clip(p[0] + p[1] * amps, -15.0, 8.0)
        lam = np.exp(eta) * exposure
        # tiny L2 regularization on slope to avoid divergence when high-amp counts are all 0
        reg = 1e-3 * (p[1] ** 2)
        return float(np.sum(lam - counts * np.log(lam + 1e-300)) + reg)

    try:
        r = optimize.minimize(nll, [math.log(base_rate), -0.4], method="BFGS")
        p = np.array([float(np.clip(r.x[0], -10.0, 5.0)), float(np.clip(r.x[1], -8.0, 5.0))])
        cov = np.atleast_2d(r.hess_inv)
        if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
            cov = np.diag([0.15, 0.15])
        # Ensure positive semi-definite covariance
        eigvals, eigvecs = np.linalg.eigh(0.5 * (cov + cov.T))
        cov = eigvecs @ np.diag(np.clip(eigvals, 1e-6, 4.0)) @ eigvecs.T
        return p, cov
    except Exception:
        return np.array([math.log(base_rate), 0.0]), np.diag([0.25, 0.05])


def evt_var(z: np.ndarray, p: float, xi: float, uq: float = 90.0) -> tuple[float, float]:
    """POT quantile above the arm's own P90 with the series's ξ (σ by mean excess)."""
    if len(z) < 20 or xi >= 0.95:
        return float("nan"), float("nan")
    u = float(np.percentile(z, uq))
    ex = z[z > u] - u
    if ex.size < 3:
        return float("nan"), float("nan")
    sigma = float(ex.mean()) * (1.0 - xi)
    if sigma <= 0:
        return float("nan"), float("nan")
    zeta = ex.size / z.size
    if abs(xi) > 1e-6:
        val = u + (sigma / xi) * (((1.0 - p) / max(zeta, 1e-9)) ** (-xi) - 1.0)
    else:
        val = u - sigma * math.log((1.0 - p) / max(zeta, 1e-9))
    es = (val + sigma - xi * u) / (1.0 - xi)
    return float(val), float(es)


def _clean_num(v: Any, nd: int = 4) -> float | None:
    try:
        fv = float(v)
        return round(fv, nd) if math.isfinite(fv) else None
    except Exception:
        return None


def _downsample_envelope_trace(
    seqs: list[dict[str, Any]],
    tg: str,
    u: float,
    max_pts: int = 200,
) -> dict[str, Any]:
    """Build compact envelope ratio trace + amplitude staircase + declustered breach peaks."""
    t_all, z_all, a_all = [], [], []
    breach_t, breach_z, breach_amp = [], [], []
    for q in seqs:
        z = q[f"z_{tg}"]
        if len(z) == 0:
            continue
        t = q["t"] / FS
        t_all.append(t)
        z_all.append(z)
        a_all.append(np.full(len(z), q["amp"], dtype=float))
        pks, _, pidx = decluster(z, u)
        for pv, pi in zip(pks, pidx):
            breach_t.append(round(float(t[pi]), 2))
            breach_z.append(round(float(pv), 3))
            breach_amp.append(round(float(q["amp"]), 2))

    if not t_all:
        return {"t": [], "z": [], "z_max": [], "amp": [], "breaches": []}

    t_cat = np.concatenate(t_all)
    z_cat = np.concatenate(z_all)
    a_cat = np.concatenate(a_all)
    n = len(t_cat)

    if n > max_pts:
        edges = np.linspace(0, n, max_pts + 1, dtype=int)
        t_out, z_med, z_hi, a_out = [], [], [], []
        for i in range(max_pts):
            i0, i1 = edges[i], max(edges[i] + 1, edges[i + 1])
            t_out.append(round(float(0.5 * (t_cat[i0] + t_cat[i1 - 1])), 2))
            z_med.append(round(float(np.mean(z_cat[i0:i1])), 3))
            z_hi.append(round(float(np.max(z_cat[i0:i1])), 3))
            a_out.append(round(float(a_cat[i0]), 2))
    else:
        t_out = [round(float(v), 2) for v in t_cat]
        z_med = [round(float(v), 3) for v in z_cat]
        z_hi = z_med
        a_out = [round(float(v), 2) for v in a_cat]

    # Cap stored breach markers to keep payload compact
    breaches = [
        {"t": bt, "z": bz, "amp": ba}
        for bt, bz, ba in zip(breach_t, breach_z, breach_amp)
    ]
    if len(breaches) > 100:
        # Keep top peaks + uniform stride
        idx = np.linspace(0, len(breaches) - 1, 100, dtype=int)
        breaches = [breaches[i] for i in idx]

    return {
        "t": t_out,
        "z": z_med,
        "z_max": z_hi,
        "amp": a_out,
        "dur_s": round(float(t_cat[-1]), 1),
        "breaches": breaches,
    }


def _tail_survival_points(z: np.ndarray, n_pts: int = 30) -> dict[str, list[float]]:
    """Compact empirical survival points (z, P(Z > z)) above median for log-survival plot."""
    if len(z) < 10:
        return {"z": [], "p": []}
    s = np.sort(z)
    n = len(s)
    start = n // 2
    # Log-spaced indices toward the upper tail so extreme points are crisp
    rem = n - start
    if rem <= n_pts:
        idxs = np.arange(start, n)
    else:
        # Mix linear in bulk + dense near top 10%
        p_lin = np.linspace(0.50, 0.90, n_pts // 2, endpoint=False)
        p_top = 1.0 - np.logspace(-1.0, math.log10(max(1.0 / n, 1e-4)), n_pts - n_pts // 2)
        idxs = np.unique(np.clip((np.r_[p_lin, p_top] * (n - 1)).astype(int), start, n - 1))
    return {
        "z": [round(float(s[k]), 3) for k in idxs],
        "p": [round(float((n - k) / n), 5) for k in idxs],
    }


def _threshold_diagnostics(seqs: list[dict[str, Any]], tg: str, base: np.ndarray) -> dict[str, list[float | None]]:
    """Mean residual life & GPD shape ξ stability across P70..P96 of baseline."""
    if len(base) < 60:
        return {"u": [], "mrl": [], "xi": []}
    qs = np.arange(70.0, 97.0, 2.0)
    us = np.percentile(base, qs)
    mrl_out, xi_out = [], []
    base_seqs = [q[f"z_{tg}"] for q in seqs if q["amp"] <= 0.01] or [base]
    for u_val in us:
        pks_list = [decluster(zz, float(u_val))[0] for zz in base_seqs]
        pks = np.concatenate(pks_list) if pks_list else np.array([])
        if pks.size >= 4:
            ex = pks - float(u_val)
            mrl_out.append(_clean_num(float(np.mean(ex)), 3))
            xi_v, _ = _gpd_pwm(ex) if pks.size >= 6 else (float("nan"), 0.1)
            xi_out.append(_clean_num(xi_v, 3))
        else:
            mrl_out.append(None)
            xi_out.append(None)
    return {
        "u": [round(float(v), 3) for v in us],
        "mrl": mrl_out,
        "xi": xi_out,
    }


def _kupiec_backtest(
    seqs: list[dict[str, Any]],
    tg: str,
    xi0: float,
    series_label: str,
    pair: str,
    stim_hz: float,
    contacts: list[str],
    hemisphere: str,
) -> list[dict[str, Any]]:
    """Kupiec proportion-of-failures backtest: train first half of each segment, test second half."""
    rows = []
    for p in (0.95, 0.99):
        for meth in ("Empirical", "EVT (POT)"):
            x_fail = 0
            n_tot = 0
            for q in seqs:
                z = q[f"z_{tg}"]
                if len(z) < 120:
                    continue
                half = len(z) // 2
                tr, te = z[:half], z[half:]
                v = float(np.percentile(tr, 100.0 * p)) if meth == "Empirical" else evt_var(tr, p, xi0)[0]
                if not math.isfinite(v):
                    continue
                x_fail += int(np.sum(te > v))
                n_tot += int(te.size)
            if n_tot == 0:
                continue
            pi_hat = x_fail / n_tot
            p_tail = 1.0 - p
            ll0 = (n_tot - x_fail) * math.log(max(1.0 - p_tail, 1e-12)) + x_fail * math.log(max(p_tail, 1e-12))
            ll1 = (n_tot - x_fail) * math.log(max(1.0 - pi_hat, 1e-12)) + x_fail * math.log(max(pi_hat, 1e-12))
            lr = max(0.0, -2.0 * (ll0 - ll1))
            kp = float(1.0 - stats.chi2.cdf(lr, 1)) if math.isfinite(lr) else float("nan")
            rows.append(
                {
                    "series": series_label,
                    "hemisphere": hemisphere,
                    "pair": pair,
                    "contacts": contacts,
                    "stim_hz": round(float(stim_hz), 1),
                    "target": tg,
                    "alpha": p,
                    "method": meth,
                    "test_steps": n_tot,
                    "breaches": x_fail,
                    "expected": round(n_tot * p_tail, 1),
                    "breach_rate": round(pi_hat, 4),
                    "kupiec_p": _clean_num(kp, 4),
                }
            )
    return rows


def extract_risk_bundle(
    full_store: dict[str, Any],
    ftg_hw: float = 2.0,
    n_boot: int = 80,
) -> dict[str, Any]:
    """Extract causal envelopes, POT/GPD fits, GEV block maxima, Poisson breach-rate models,
    diagnostics, and Kupiec backtests directly from parsed Percept JSON recordings."""
    payload = full_store["payload"]
    raw_signals = full_store["raw_signals"]
    recordings = payload.get("recordings") or []
    rng = np.random.default_rng(7)
    Tg = np.asarray(T_GRID, dtype=float)

    hemis_out: dict[str, Any] = {}
    for hemi in ("Left", "Right"):
        stream_recs = [r for r in recordings if r["source"] == "BrainSenseTimeDomain" and r["hemisphere"] == hemi]
        montage_recs = [r for r in recordings if r["source"] == "LfpMontageTimeDomain" and r["hemisphere"] == hemi]

        distinct_stream_amps = set()
        for r in stream_recs:
            for st in r.get("amp_steps") or []:
                distinct_stream_amps.add(round(float(st["amp_ma"]), 1))
        has_amp_sweeps = len(distinct_stream_amps) >= 2

        source_recs = stream_recs if has_amp_sweeps else (stream_recs + montage_recs)
        mode_kind = "amplitude_sweep" if has_amp_sweeps else "contact_survey_and_stream"

        # Group recordings by (pair, stim_hz) so multiple segments of the same pair/rate pool cleanly
        grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
        for r in source_recs:
            raw = raw_signals.get(r["id"])
            if not raw or raw["time_domain_uv"] is None or len(raw["time_domain_uv"]) < SETTLE + 50:
                continue
            fs = float(raw.get("fs_hz") or FS)
            if abs(fs - FS) > 1.0:
                continue
            shz = round(float(r.get("stim_rate_hz") or payload["config"]["default_stim_rate_hz"] or 145.0), 1)
            grouped.setdefault((str(r["pair"]), shz), []).append(r)

        # Pass 1: Compute raw causal envelopes for every series
        series_raw: list[dict[str, Any]] = []
        hemi_base_pool: dict[str, list[np.ndarray]] = {"beta": [], "ftg": []}

        for (pair, shz), rec_list in sorted(grouped.items()):
            # Concatenate segments across recordings of the same (pair, shz)
            all_segs = []
            offset_t = 0
            x_parts = []
            contacts_union: set[str] = set()
            rec_id0 = rec_list[0]["id"]

            for r in rec_list:
                raw = raw_signals[r["id"]]
                x = np.asarray(raw["time_domain_uv"], dtype=float)
                x = np.where(np.isfinite(x), x, 0.0)
                amp_t = raw.get("amp_t")
                amp_ma = raw.get("amp_ma")
                segs = _segment_amplitude_runs(
                    len(x),
                    FS,
                    amp_t if amp_t is not None else np.array([]),
                    amp_ma if amp_ma is not None else np.array([]),
                )
                for s in segs:
                    all_segs.append({"start": offset_t + s["start"], "n": s["n"], "amp": s["amp"]})
                x_parts.append(x)
                offset_t += len(x)
                for c in (r.get("contacts") or []) + (r.get("stim_contacts") or []):
                    if c and c != "Case":
                        contacts_union.add(c)

            x_full = np.concatenate(x_parts)
            # Find baseline 0 mA segments for beta peak
            base_chunks = [x_full[s["start"] : s["start"] + s["n"]] for s in all_segs if s["amp"] <= 0.01 and s["n"] >= 500]
            base_sig = np.concatenate(base_chunks) if base_chunks else x_full[: min(len(x_full), 2000)]
            b_pk = _beta_peak(base_sig, FS)
            b_band = (max(8.0, b_pk - 3.0), min(35.0, b_pk + 3.0))
            env_nb = max(4, int(round(1.5 * FS / max(b_pk, 8.0))))

            ftg_center = shz / 2.0
            ftg_band = (max(35.0, ftg_center - max(1.5, float(ftg_hw))), min(120.0, ftg_center + max(1.5, float(ftg_hw))))

            sos_b = _bandpass(*b_band, FS)
            sos_f = _bandpass(*ftg_band, FS)

            seqs = []
            for s in all_segs:
                if s["n"] <= SETTLE + DEC * 4:
                    continue
                xs = x_full[s["start"] : s["start"] + s["n"]]
                xs = xs - float(np.mean(xs))
                eb = _mov_rms(sosfilt(sos_b, xs), env_nb)
                ef = _mov_rms(sosfilt(sos_f, xs), FTG_ENV_N)
                j = np.arange(SETTLE, s["n"], DEC)
                if j.size < 4:
                    continue
                seqs.append({"amp": round(float(s["amp"]), 2), "t": s["start"] + j, "beta": eb[j], "ftg": ef[j]})

            if not seqs:
                continue

            for tg in ("beta", "ftg"):
                zero_arrs = [q[tg] for q in seqs if q["amp"] <= 0.01]
                if zero_arrs:
                    hemi_base_pool[tg].append(np.concatenate(zero_arrs))

            series_raw.append(
                {
                    "id": f"{hemi}|{pair}|{int(round(shz))}Hz",
                    "hemisphere": hemi,
                    "pair": pair,
                    "contacts": sorted(contacts_union),
                    "stim_hz": shz,
                    "ftg_center_hz": round(ftg_center, 2),
                    "beta_peak_hz": round(b_pk, 2),
                    "beta_band": [round(b_band[0], 2), round(b_band[1], 2)],
                    "ftg_band": [round(ftg_band[0], 2), round(ftg_band[1], 2)],
                    "rec_id": rec_id0,
                    "seqs": seqs,
                }
            )

        if not series_raw:
            hemis_out[hemi] = {
                "hemisphere": hemi,
                "mode": mode_kind,
                "amax": 1.0,
                "stim_rates": [],
                "series": [],
                "backtest": [],
            }
            continue

        hemi_fallback_med = {
            tg: float(np.median(np.concatenate(hemi_base_pool[tg])))
            if hemi_base_pool[tg]
            else float(np.median(np.concatenate([q[tg] for sr in series_raw for q in sr["seqs"]])))
            for tg in ("beta", "ftg")
        }

        # Pass 2: Normalize envelopes by 0 mA median & fit EVT models per series × target
        series_out: list[dict[str, Any]] = []
        backtest_all: list[dict[str, Any]] = []
        all_amps_hemi: list[float] = []
        all_rates_hemi: set[float] = set()

        for sr in series_raw:
            all_rates_hemi.add(float(sr["stim_hz"]))
            target_models: dict[str, Any] = {}

            for tg in ("beta", "ftg"):
                seqs = sr["seqs"]
                zero_raw = [q[tg] for q in seqs if q["amp"] <= 0.01]
                med_base = float(np.median(np.concatenate(zero_raw))) if zero_raw else hemi_fallback_med[tg]
                med_base = max(med_base, 1e-9)

                for q in seqs:
                    q[f"z_{tg}"] = q[tg] / med_base

                base_seqs = [q[f"z_{tg}"] for q in seqs if q["amp"] <= 0.01]
                base = np.concatenate(base_seqs) if base_seqs else np.concatenate([q[f"z_{tg}"] for q in seqs])
                u = float(np.percentile(base, U_Q))

                # Baseline POT + GPD
                if base_seqs:
                    b_res = [decluster(zz, u) for zz in base_seqs]
                else:
                    b_res = [decluster(q[f"z_{tg}"], u) for q in seqs]
                bpk = np.concatenate([x[0] for x in b_res if len(x[0])]) if any(len(x[0]) for x in b_res) else np.array([])
                bnex = int(sum(x[1] for x in b_res))
                T0 = max(base.size / 50.0, 0.2)
                lam0 = float(bpk.size / T0)
                theta0 = float(bpk.size / bnex) if bnex > 0 else 1.0
                xi0, sg0 = gpd_fit(bpk - u) if bpk.size >= 3 else (0.0, 0.2)
                rl0 = pot_return_level(u, xi0, sg0, lam0, Tg)

                # Fast PWM bootstrap for POT return-level 90% CI
                if bpk.size >= 5:
                    boots = []
                    for _ in range(n_boot):
                        pk_b = rng.choice(bpk, bpk.size, replace=True)
                        xi_b, sg_b = _gpd_pwm(pk_b - u)
                        lam_b = max(rng.poisson(bpk.size), 1) / T0
                        boots.append(pot_return_level(u, xi_b, sg_b, lam_b, Tg))
                    boots_arr = np.asarray(boots, dtype=float)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        rl0_lo = np.nanpercentile(boots_arr, 5, axis=0)
                        rl0_hi = np.nanpercentile(boots_arr, 95, axis=0)
                else:
                    rl0_lo = rl0 * 0.9
                    rl0_hi = rl0 * 1.1

                # Per-arm metrics
                amps = sorted({q["amp"] for q in seqs})
                all_amps_hemi.extend(amps)
                arm_rows = []
                bm_groups = []
                for a in amps:
                    zs = [q[f"z_{tg}"] for q in seqs if q["amp"] == a]
                    z = np.concatenate(zs)
                    if z.size < 5:
                        continue
                    pk_list = [decluster(zz, u) for zz in zs]
                    ncl = int(sum(len(p_[0]) for p_ in pk_list))
                    nex = int(sum(p_[1] for p_ in pk_list))
                    bm = block_maxima(zs)
                    bm_groups.append(bm)
                    v95 = float(np.percentile(z, 95))
                    v99 = float(np.percentile(z, 99))
                    cv95 = float(z[z >= v95].mean()) if np.any(z >= v95) else v95
                    cv99 = float(z[z >= v99].mean()) if np.any(z >= v99) else v99
                    ev, es = evt_var(z, 0.999, xi0)
                    sec = z.size / 50.0
                    arm_rows.append(
                        {
                            "arm_id": f"{hemi}|{sr['pair']}|{int(round(sr['stim_hz']))}Hz|{a:g}mA",
                            "hemisphere": hemi,
                            "pair": sr["pair"],
                            "contacts": sr["contacts"],
                            "stim_hz": sr["stim_hz"],
                            "ftg_center_hz": sr["ftg_center_hz"],
                            "beta_peak_hz": sr["beta_peak_hz"],
                            "rec_id": sr["rec_id"],
                            "label": f"{sr['pair']} · {int(round(sr['stim_hz']))}Hz @ {a:g} mA",
                            "short_label": f"{sr['pair']}@{a:g}mA",
                            "amp": round(float(a), 2),
                            "seconds": round(sec, 2),
                            "n": int(z.size),
                            "mean": round(float(z.mean()), 4),
                            "VaR95": round(v95, 4),
                            "VaR99": round(v99, 4),
                            "CVaR95": round(cv95, 4),
                            "CVaR99": round(cv99, 4),
                            "EVT_VaR999": _clean_num(ev, 4),
                            "EVT_ES999": _clean_num(es, 4),
                            "exceed": nex,
                            "clusters": ncl,
                            "theta": _clean_num(ncl / nex, 3) if nex > 0 else None,
                            "rate": round(ncl / max(sec, 0.05), 4),
                            "blocks": int(bm.size),
                            "max": round(float(z.max()), 4),
                            "tail_pts": _tail_survival_points(z),
                        }
                    )

                if not arm_rows:
                    continue

                # Poisson breach-rate model λ(a) = exp(r0 + r1·a)
                amp_arr = np.array([a["amp"] for a in arm_rows], dtype=float)
                cnt_arr = np.array([a["clusters"] for a in arm_rows], dtype=float)
                exp_arr = np.array([a["seconds"] for a in arm_rows], dtype=float)
                (r0, r1), cov = poisson_rate_fit(amp_arr, cnt_arr, exp_arr)
                a_hi = max(float(amp_arr.max()), 0.5)
                ag = np.round(np.linspace(0.0, a_hi, 36), 3)
                draws = rng.multivariate_normal([r0, r1], cov, 240)
                lam_g = np.exp(np.clip(r0 + r1 * ag, -12.0, 6.0))
                lam_d = np.exp(np.clip(draws[:, :1] + draws[:, 1:] * ag, -12.0, 6.0))
                lam_lo = np.percentile(lam_d, 5, axis=0)
                lam_hi = np.percentile(lam_d, 95, axis=0)

                # Shared-shape GEV block maxima fit
                ok_idx = [i for i, g in enumerate(bm_groups) if len(g) >= 2]
                gev = gev_shared_fit([bm_groups[i] for i in ok_idx])
                gev_xi = float(gev["xi"])
                for a_row in arm_rows:
                    a_row["lam_model"] = round(float(math.exp(np.clip(r0 + r1 * a_row["amp"], -12.0, 6.0))), 5)
                    a_row["gev_mu"] = None
                    a_row["gev_sigma"] = None
                    a_row["RL60_gev"] = None
                    rl60p = pot_return_level(u, xi0, sg0, a_row["lam_model"], 60.0)
                    a_row["RL60_pot"] = _clean_num(float(rl60p), 4) if np.isfinite(rl60p) else None

                for j, i in enumerate(ok_idx):
                    mu_j, sg_j = gev["params"][j]
                    arm_rows[i]["gev_mu"] = round(mu_j, 4)
                    arm_rows[i]["gev_sigma"] = round(sg_j, 4)
                    arm_rows[i]["RL60_gev"] = _clean_num(gev_return_level(mu_j, sg_j, gev_xi, 60.0), 4)

                # Compact envelope trace + threshold diagnostics + Kupiec backtest
                env_trace = _downsample_envelope_trace(seqs, tg, u)
                diag = _threshold_diagnostics(seqs, tg, base)
                bt_rows = _kupiec_backtest(
                    seqs,
                    tg,
                    xi0,
                    f"{sr['pair']} · {int(round(sr['stim_hz']))}Hz",
                    sr["pair"],
                    sr["stim_hz"],
                    sr["contacts"],
                    hemi,
                )
                backtest_all.extend(bt_rows)

                target_models[tg] = {
                    "u": round(u, 4),
                    "med_base": round(med_base, 5),
                    "xi0": round(xi0, 4),
                    "sigma0": round(sg0, 4),
                    "lam0": round(lam0, 4),
                    "theta0": round(theta0, 4),
                    "n_clusters0": int(bpk.size),
                    "gev_xi": round(gev_xi, 4),
                    "gev_ok": bool(gev["ok"]),
                    "rate": [round(float(r0), 4), round(float(r1), 4)],
                    "rl0": [_clean_num(v, 4) for v in rl0],
                    "rl0_lo": [_clean_num(v, 4) for v in rl0_lo],
                    "rl0_hi": [_clean_num(v, 4) for v in rl0_hi],
                    "lam_grid": [round(float(v), 3) for v in ag],
                    "lam_mid": [_clean_num(v, 4) for v in lam_g],
                    "lam_lo": [_clean_num(v, 4) for v in lam_lo],
                    "lam_hi": [_clean_num(v, 4) for v in lam_hi],
                    "arms": arm_rows,
                    "env_trace": env_trace,
                    "diagnostics": diag,
                }

            if target_models:
                series_out.append(
                    {
                        "id": sr["id"],
                        "hemisphere": hemi,
                        "pair": sr["pair"],
                        "contacts": sr["contacts"],
                        "stim_hz": sr["stim_hz"],
                        "ftg_center_hz": sr["ftg_center_hz"],
                        "beta_peak_hz": sr["beta_peak_hz"],
                        "beta_band": sr["beta_band"],
                        "ftg_band": sr["ftg_band"],
                        "rec_id": sr["rec_id"],
                        "label": f"{sr['pair']} · {int(round(sr['stim_hz']))}Hz",
                        "targets": target_models,
                    }
                )

        hemis_out[hemi] = {
            "hemisphere": hemi,
            "mode": mode_kind,
            "amax": round(max(all_amps_hemi) if all_amps_hemi else 1.0, 2),
            "stim_rates": sorted(all_rates_hemi),
            "series": series_out,
            "backtest": backtest_all,
        }

    return {
        "spec": {
            "fs_hz": FS,
            "settle_samples": SETTLE,
            "dec": DEC,
            "block_steps": BLOCK,
            "run_steps": RUN,
            "u_quantile": U_Q,
            "ftg_hw_hz": ftg_hw,
            "T_grid": T_GRID,
        },
        "default_params": DEFAULT_RISK_PARAMS,
        "hemispheres": hemis_out,
    }


def solve_risk_exact(
    risk_bundle: dict[str, Any],
    hemisphere: str = "Left",
    stim_rate_filter: str | float | None = "ALL",
    params: dict[str, Any] | None = None,
    selected_contacts: list[str] | None = None,
) -> dict[str, Any]:
    """Compute exact 5-rule NeuroRisk arm selection, arms table, series summary, and Kupiec backtest,
    dynamically scoped to the contacts selected on the DBS Lead Model."""
    P = dict(DEFAULT_RISK_PARAMS)
    if params:
        for k in ("target", "alpha", "horizon", "kappa", "eps", "metric"):
            if k in params and params[k] is not None:
                P[k] = params[k]
    P["hemisphere"] = hemisphere
    P["stim_rate_filter"] = stim_rate_filter
    tg = str(P.get("target") or "beta")
    alpha = float(P.get("alpha") or 0.99)
    lvl = "99" if alpha >= 0.985 else "95"
    H = max(2.0, float(P.get("horizon") or 60.0))
    kappa = max(0.0, float(P.get("kappa") if P.get("kappa") is not None else 0.6))
    eps = max(0.005, min(0.99, float(P.get("eps") if P.get("eps") is not None else 0.10)))

    h_data = (risk_bundle.get("hemispheres") or {}).get(hemisphere) or {}
    all_series = h_data.get("series") or []
    sel_set = set(selected_contacts) if selected_contacts is not None else None

    # Contact & stim-rate filtering
    series = []
    for s in all_series:
        if sel_set is not None:
            if not any(f"{s['hemisphere']}:{c}" in sel_set for c in (s.get("contacts") or [])):
                continue
        if stim_rate_filter not in (None, "", "ALL", "All"):
            try:
                if abs(float(s["stim_hz"]) - float(stim_rate_filter)) >= 1.0:
                    continue
            except Exception:
                pass
        if tg in (s.get("targets") or {}):
            series.append(s)

    if not series:
        return {
            "hemisphere": hemisphere,
            "stim_rate_filter": stim_rate_filter,
            "params": P,
            "amax": float(h_data.get("amax") or 1.0),
            "risk_arms_df": pd.DataFrame(),
            "risk_selection_df": pd.DataFrame(),
            "risk_series_df": pd.DataFrame(),
            "risk_backtest_df": pd.DataFrame(),
            "majority_arm": None,
            "majority_votes": "0 of 0",
            "picks": {},
        }

    # Gather all scoped arms and recompute horizon-dependent breach_P, GEV RL(H), energy, score
    arm_rows = []
    series_rows = []
    for s in series:
        tm = s["targets"][tg]
        series_rows.append(
            {
                "series": s["label"],
                "pair": s["pair"],
                "stim_hz": s["stim_hz"],
                "target": tg,
                "u_red_line": tm["u"],
                "baseline_clusters": tm["n_clusters0"],
                "lam0_per_s": tm["lam0"],
                "theta0": tm["theta0"],
                "GPD_xi": tm["xi0"],
                "GPD_sigma": tm["sigma0"],
                "GEV_xi": tm["gev_xi"],
                "rate_slope_per_mA": tm["rate"][1],
            }
        )
        r0, r1 = tm["rate"]
        gev_xi = float(tm["gev_xi"])
        for a in tm.get("arms") or []:
            lam_a = math.exp(np.clip(r0 + r1 * float(a["amp"]), -12.0, 6.0))
            breach_p = 1.0 - math.exp(-lam_a * H)
            rl_h = (
                gev_return_level(float(a["gev_mu"]), float(a["gev_sigma"]), gev_xi, H)
                if a.get("gev_mu") is not None and a.get("gev_sigma") is not None
                else float("nan")
            )
            arm_rows.append(
                {
                    "arm_id": a["arm_id"],
                    "label": a["label"],
                    "pair": a["pair"],
                    "stim_hz": a["stim_hz"],
                    "amp_ma": float(a["amp"]),
                    "seconds": float(a["seconds"]),
                    "n_steps": int(a["n"]),
                    "VaR": float(a[f"VaR{lvl}"]),
                    "CVaR": float(a[f"CVaR{lvl}"]),
                    "EVT_VaR999": float(a["EVT_VaR999"]) if a.get("EVT_VaR999") is not None else float("nan"),
                    "RL_H_gev": float(rl_h) if math.isfinite(rl_h) else float("nan"),
                    "clusters": int(a["clusters"]),
                    "rate_obs": float(a["rate"]),
                    "lam_model": round(lam_a, 4),
                    "breach_P": round(breach_p, 4),
                    "max_z": float(a["max"]),
                }
            )

    arms_df = pd.DataFrame(arm_rows)
    if arms_df.empty:
        return {
            "hemisphere": hemisphere,
            "stim_rate_filter": stim_rate_filter,
            "params": P,
            "amax": float(h_data.get("amax") or 1.0),
            "risk_arms_df": pd.DataFrame(),
            "risk_selection_df": pd.DataFrame(),
            "risk_series_df": pd.DataFrame(series_rows),
            "risk_backtest_df": pd.DataFrame(),
            "majority_arm": None,
            "majority_votes": "0 of 0",
            "picks": {},
        }

    amax = max(float(arms_df["amp_ma"].max()), 0.5)
    arms_df["energy"] = np.round((arms_df["amp_ma"] / amax) ** 2, 4)
    cv = arms_df["CVaR"].values
    cv_lo, cv_hi = float(np.nanmin(cv)), float(np.nanmax(cv))
    arms_df["risk_score"] = np.round((cv - cv_lo) / (cv_hi - cv_lo + 1e-9) + kappa * arms_df["energy"].values, 4)

    def _argmin_row(col: str):
        sub = arms_df[np.isfinite(arms_df[col].values)]
        if sub.empty:
            return None
        return sub.loc[sub[col].idxmin()]

    r_cvar = _argmin_row("CVaR")
    r_evt = _argmin_row("EVT_VaR999")
    r_rl = _argmin_row("RL_H_gev")
    r_score = _argmin_row("risk_score")
    safe_df = arms_df[(arms_df["breach_P"] <= eps) & (arms_df["amp_ma"] > 0)].sort_values(["energy", "CVaR"])
    r_safe = safe_df.iloc[0] if len(safe_df) else None

    sel_rows = [
        {
            "rule": f"1. min CVaR{lvl} (Expected Shortfall)",
            "pick": str(r_cvar["label"]) if r_cvar is not None else "—",
            "value": round(float(r_cvar["CVaR"]), 3) if r_cvar is not None else None,
            "profile": f"CVaR {r_cvar['CVaR']:.2f} · P(br) {100*r_cvar['breach_P']:.0f}% · E {r_cvar['energy']:.2f}" if r_cvar is not None else "—",
        },
        {
            "rule": "2. min EVT-VaR99.9 (POT Extrapolated)",
            "pick": str(r_evt["label"]) if r_evt is not None else "—",
            "value": round(float(r_evt["EVT_VaR999"]), 3) if r_evt is not None else None,
            "profile": f"EVT {r_evt['EVT_VaR999']:.2f} · CVaR {r_evt['CVaR']:.2f}" if r_evt is not None else "—",
        },
        {
            "rule": f"3. min {H:.0f}-s Return Level (GEV)",
            "pick": str(r_rl["label"]) if r_rl is not None else "—",
            "value": round(float(r_rl["RL_H_gev"]), 3) if r_rl is not None else None,
            "profile": f"RL({H:.0f}s) {r_rl['RL_H_gev']:.2f} · CVaR {r_rl['CVaR']:.2f}" if r_rl is not None else "—",
        },
        {
            "rule": f"4. Risk-Adjusted (CVaR + {kappa:.2f}·E)",
            "pick": str(r_score["label"]) if r_score is not None else "—",
            "value": round(float(r_score["risk_score"]), 3) if r_score is not None else None,
            "profile": f"CVaR {r_score['CVaR']:.2f} · E {r_score['energy']:.2f}" if r_score is not None else "—",
        },
        {
            "rule": f"5. Safe Constrained (P(br {H:.0f}s) ≤ {100*eps:.0f}%)",
            "pick": str(r_safe["label"]) if r_safe is not None else "none feasible",
            "value": round(float(r_safe["breach_P"]), 3) if r_safe is not None else None,
            "profile": f"P(br) {100*r_safe['breach_P']:.1f}% · E {r_safe['energy']:.2f}" if r_safe is not None else "relax ε or shorten H",
        },
    ]
    sel_df = pd.DataFrame(sel_rows)

    valid_picks = [r["pick"] for r in sel_rows if r["pick"] not in ("—", "none feasible")]
    majority_arm, majority_count = None, 0
    if valid_picks:
        vc = pd.Series(valid_picks).value_counts()
        majority_arm = str(vc.index[0])
        majority_count = int(vc.iloc[0])

    # Filter Kupiec backtest rows to scoped contacts, rate, target
    bt_filtered = []
    for row in h_data.get("backtest") or []:
        if row.get("target") != tg:
            continue
        if sel_set is not None and not any(f"{row['hemisphere']}:{c}" in sel_set for c in (row.get("contacts") or [])):
            continue
        if stim_rate_filter not in (None, "", "ALL", "All"):
            try:
                if abs(float(row["stim_hz"]) - float(stim_rate_filter)) >= 1.0:
                    continue
            except Exception:
                pass
        bt_filtered.append({k: v for k, v in row.items() if k != "contacts"})

    return {
        "hemisphere": hemisphere,
        "stim_rate_filter": stim_rate_filter,
        "params": P,
        "amax": round(amax, 2),
        "risk_arms_df": arms_df,
        "risk_selection_df": sel_df,
        "risk_series_df": pd.DataFrame(series_rows),
        "risk_backtest_df": pd.DataFrame(bt_filtered),
        "majority_arm": majority_arm,
        "majority_votes": f"{majority_count} of {len(valid_picks)}",
        "picks": {
            "cvar": str(r_cvar["label"]) if r_cvar is not None else None,
            "evt": str(r_evt["label"]) if r_evt is not None else None,
            "rl": str(r_rl["label"]) if r_rl is not None else None,
            "score": str(r_score["label"]) if r_score is not None else None,
            "safe": str(r_safe["label"]) if r_safe is not None else None,
        },
    }
