"""NeuroReader Game-Theory Engine (Arms Extraction + Exact & Live Equilibrium Solvers).

Follows the NeuroGame mathematical formulation and adapts it natively to Medtronic Percept JSON
exports (multi-amplitude & multi-frequency BrainSenseTimeDomain sweeps, plus fallback to
LfpMontageTimeDomain survey contact arms when a session has constant amplitude).

Utility (all terms in dB-equivalents):
    U = w_beta * delta_beta_dB - w_F * max(0, FTG_dB - tau) - w_E * 6 * (a / a_max)^2

Solution concepts:
    1. Multi-Armed Bandits (Thompson, UCB1, ε-greedy, Safe-UCB, LinUCB, Exhaustive)
    2. Pure Minimax (Maximin over 4 clinical priority scenarios)
    3. Mixed Zero-Sum Nash Equilibrium (exact LP via scipy.optimize.linprog + live MWU)
    4. Cooperative Nash Bargaining Solution on the Efficacy vs. Safety Pareto frontier
    5. Stackelberg Leader–Follower (fitted β sigmoid + FTG hinge, continuous 0.05 mA grid)
    6. Robust Stackelberg (pessimistic 10th-pct β / 90th-pct FTG bootstrap bands)
    7. 7-Method Majority Consensus Board
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, linprog
from scipy.signal import welch

FS = 250.0
WIN = 500        # 2.0 s Hann window
HOP = 250        # 1.0 s hop (50% overlap)
SETTLE = 125     # 0.5 s settle after amplitude transition
FTG_HW = 1.5     # FTG narrowband half-width around f_stim / 2

SCENARIOS: dict[str, dict[str, float]] = {
    "Efficacy-first": {"wb": 1.0, "wf": 0.2, "we": 0.2},
    "Dyskinesia-averse": {"wb": 1.0, "wf": 1.5, "we": 0.2},
    "Battery-saver": {"wb": 1.0, "wf": 0.4, "we": 1.5},
    "Balanced": {"wb": 1.0, "wf": 0.7, "we": 0.6},
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "wb": 1.0,
    "wf": 0.6,
    "we": 0.5,
    "tau": 2.0,
    "ftg_safe": 4.0,
}


def utility_fn(
    beta_supp: np.ndarray | float,
    ftg_db: np.ndarray | float,
    amp: np.ndarray | float,
    amax: float,
    wb: float = 1.0,
    wf: float = 0.6,
    we: float = 0.5,
    tau: float = 2.0,
    **_: Any,
) -> np.ndarray | float:
    """U = wβ·Δβ − wF·max(0, FTG − τ) − wE·6·(a/amax)² (dB-equivalents)."""
    a_norm = np.asarray(amp, dtype=float) / max(float(amax), 0.1)
    return wb * np.asarray(beta_supp, dtype=float) - wf * np.maximum(0.0, np.asarray(ftg_db, dtype=float) - tau) - we * 6.0 * (a_norm**2)


def _find_beta_peak(x: np.ndarray, fs: float = FS) -> float:
    """1/f-corrected beta peak (13–30 Hz) from a baseline LFP segment."""
    if len(x) < WIN:
        return 20.0
    f, p = welch(x, fs=fs, nperseg=min(len(x), WIN))
    p = np.maximum(p, 1e-12)
    fit_mask = (f >= 3.0) & (f <= 45.0) & ~((f >= 12.0) & (f <= 32.0))
    if np.sum(fit_mask) < 4:
        b_mask = (f >= 13.0) & (f <= 30.0)
        return float(f[b_mask][np.argmax(p[b_mask])]) if np.any(b_mask) else 20.0
    coef = np.polyfit(np.log10(f[fit_mask]), np.log10(p[fit_mask]), 1)
    res = np.log10(p) - np.polyval(coef, np.log10(np.maximum(f, 1e-3)))
    b_mask = (f >= 13.0) & (f <= 30.0)
    return float(f[b_mask][np.argmax(res[b_mask])]) if np.any(b_mask) else 20.0


def _sig(a: np.ndarray, smax: float, a50: float, k: float) -> np.ndarray:
    return smax / (1.0 + np.exp(-np.clip((a - a50) / max(k, 1e-3), -40.0, 40.0)))


def _hinge(a: np.ndarray, f0: float, g: float, aon: float) -> np.ndarray:
    return f0 + g * np.maximum(0.0, a - aon)


def _fit_beta_curve(a: np.ndarray, y: np.ndarray):
    a = np.asarray(a, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(np.unique(a)) < 3:
        c = np.polyfit(a, y, min(1, max(0, len(np.unique(a)) - 1)))
        return lambda q: np.polyval(c, q)
    try:
        p, _ = curve_fit(
            _sig,
            a,
            y,
            p0=[max(float(np.max(y)), 1.0), float(np.median(a)) + 0.1, 0.5],
            bounds=([0.0, 0.0, 0.05], [25.0, max(10.0, float(np.max(a)) * 2.0), 5.0]),
            maxfev=2500,
        )
        return lambda q: _sig(np.asarray(q, dtype=float), *p)
    except Exception:
        deg = min(2, max(1, len(np.unique(a)) - 1))
        c = np.polyfit(a, y, deg)
        return lambda q: np.polyval(c, q)


def _fit_ftg_curve(a: np.ndarray, y: np.ndarray):
    a = np.asarray(a, dtype=float)
    y = np.asarray(y, dtype=float)
    amax = float(np.max(a)) if len(a) else 1.0
    if len(np.unique(a)) < 2 or amax <= 0.0:
        m = float(np.mean(y)) if len(y) else 0.0
        return lambda q: np.full_like(np.asarray(q, dtype=float), m)
    best = None
    for aon in np.linspace(0.0, amax, 21):
        X = np.c_[np.ones_like(a), np.maximum(0.0, a - aon)]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        coef[1] = max(float(coef[1]), 0.0)
        sse = float(np.sum((X @ coef - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, coef, aon)
    _, coef, aon = best
    return lambda q: _hinge(np.asarray(q, dtype=float), float(coef[0]), float(coef[1]), float(aon))


def _segment_amplitude_runs(
    n_samples: int,
    fs: float,
    amp_t: np.ndarray,
    amp_ma: np.ndarray,
) -> list[dict[str, Any]]:
    """Split a time-domain recording into contiguous constant-amplitude segments."""
    if len(amp_t) == 0 or len(amp_ma) == 0:
        return [{"start": 0, "n": n_samples, "amp": 0.0}]

    # Quantize amplitudes to 0.1 mA steps so minor telemetry jitter stays grouped
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


def extract_game_theory_bundle(
    full_store: dict[str, Any],
    ftg_hw: float = FTG_HW,
    n_boot: int = 80,
) -> dict[str, Any]:
    """Extract window-level NeuroGame biomarkers, arms, and Stackelberg dose–response fits
    directly from parsed Percept JSON recordings (both hemispheres, all stim frequencies)."""
    payload = full_store["payload"]
    raw_signals = full_store["raw_signals"]
    recordings = payload.get("recordings") or []

    hann = np.hanning(WIN)
    freqs = np.fft.rfftfreq(WIN, 1.0 / FS)
    s2 = float(np.sum(hann**2))
    rng = np.random.default_rng(42)

    hemis_out: dict[str, Any] = {}
    for hemi in ("Left", "Right"):
        stream_recs = [r for r in recordings if r["source"] == "BrainSenseTimeDomain" and r["hemisphere"] == hemi]
        montage_recs = [r for r in recordings if r["source"] == "LfpMontageTimeDomain" and r["hemisphere"] == hemi]

        # Check if we have multi-amplitude sweeps in streaming recs
        distinct_stream_amps = set()
        for r in stream_recs:
            for st in r.get("amp_steps") or []:
                distinct_stream_amps.add(round(float(st["amp_ma"]), 1))
        has_amp_sweeps = len(distinct_stream_amps) >= 2

        source_recs = stream_recs if has_amp_sweeps else (stream_recs + montage_recs)
        mode_kind = "amplitude_sweep" if has_amp_sweeps else "contact_survey_and_stream"

        window_rows: list[dict[str, Any]] = []
        series_meta: list[dict[str, Any]] = []

        for r in source_recs:
            raw = raw_signals.get(r["id"])
            if not raw or raw["time_domain_uv"] is None or len(raw["time_domain_uv"]) < WIN:
                continue
            x = np.asarray(raw["time_domain_uv"], dtype=float)
            fs = float(raw.get("fs_hz") or FS)
            if abs(fs - FS) > 1.0:
                continue

            amp_t = raw.get("amp_t")
            amp_ma = raw.get("amp_ma")
            segs = _segment_amplitude_runs(len(x), fs, amp_t if amp_t is not None else np.array([]), amp_ma if amp_ma is not None else np.array([]))

            # Find baseline 0 mA segments for patient/channel-specific beta peak
            base_chunks = [x[s["start"] : s["start"] + s["n"]] for s in segs if s["amp"] <= 0.01 and s["n"] >= WIN]
            base_sig = np.concatenate(base_chunks) if base_chunks else x[: min(len(x), WIN * 6)]
            b_peak = _find_beta_peak(base_sig, fs)
            b_band = (max(8.0, b_peak - 3.0), min(35.0, b_peak + 3.0))

            stim_hz = float(r.get("stim_rate_hz") or payload["config"]["default_stim_rate_hz"] or 145.0)
            ftg_tgt = stim_hz / 2.0
            bb_mask = (freqs >= b_band[0]) & (freqs <= b_band[1])
            fb_mask = np.abs(freqs - ftg_tgt) <= max(1.0, float(ftg_hw))
            nb_mask = (
                ((freqs >= ftg_tgt - 6.0) & (freqs <= ftg_tgt - 2.5))
                | ((freqs >= ftg_tgt + 2.5) & (freqs <= ftg_tgt + 6.0))
            )

            rec_wins = []
            for s in segs:
                start_off = SETTLE if s["n"] >= WIN + SETTLE else 0
                for pos in range(start_off, s["n"] - WIN + 1, HOP):
                    chunk = x[s["start"] + pos : s["start"] + pos + WIN]
                    P = (np.abs(np.fft.rfft((chunk - np.mean(chunk)) * hann)) ** 2) / (fs * s2)
                    b_db = float(10.0 * np.log10(max(float(np.mean(P[bb_mask])), 1e-12)))
                    f_num = max(float(np.mean(P[fb_mask])), 1e-12) if np.any(fb_mask) else 1e-12
                    f_den = max(float(np.median(P[nb_mask])), 1e-12) if np.any(nb_mask) else 1e-12
                    f_db = float(10.0 * np.log10(f_num / f_den))
                    rec_wins.append(
                        {
                            "rec_id": r["id"],
                            "pair": r["pair"],
                            "source": r["source"],
                            "stim_hz": round(stim_hz, 1),
                            "ftg_center_hz": round(ftg_tgt, 2),
                            "amp": round(float(s["amp"]), 2),
                            "beta_db": b_db,
                            "ftg_db": f_db,
                        }
                    )

            if not rec_wins:
                continue

            # Compute within-recording beta_supp vs 0 mA baseline (or fallback later)
            zero_vals = [w["beta_db"] for w in rec_wins if w["amp"] <= 0.01]
            base_beta_db = float(np.mean(zero_vals)) if zero_vals else None
            for w in rec_wins:
                w["base_beta_db"] = base_beta_db
            window_rows.extend(rec_wins)
            series_meta.append(
                {
                    "rec_id": r["id"],
                    "pair": r["pair"],
                    "stim_hz": round(stim_hz, 1),
                    "ftg_center_hz": round(ftg_tgt, 2),
                    "beta_peak_hz": round(b_peak, 2),
                    "beta_band": [round(b_band[0], 2), round(b_band[1], 2)],
                    "n_windows": len(rec_wins),
                }
            )

        if not window_rows:
            hemis_out[hemi] = {
                "hemisphere": hemi,
                "mode": mode_kind,
                "amax": 1.0,
                "stim_rates": [],
                "arms": [],
                "stackelberg_fits": [],
                "series_meta": [],
            }
            continue

        df_w = pd.DataFrame(window_rows)
        # Fill missing baselines (e.g. off-stim survey pairs or constant-stim recordings) with hemisphere median beta_db
        hemi_med_beta = float(df_w["beta_db"].median())
        df_w["base_beta_db"] = df_w["base_beta_db"].fillna(hemi_med_beta)
        df_w["beta_supp"] = df_w["base_beta_db"] - df_w["beta_db"]

        amax = max(float(df_w["amp"].max()), 0.5)
        stim_rates = sorted(set(float(v) for v in df_w["stim_hz"].unique()))

        # Group into Arms: (pair, stim_hz, amp)
        rec_meta_by_id = {r["id"]: r for r in source_recs}
        series_peak_map = {(m["pair"], m["stim_hz"]): m["beta_peak_hz"] for m in series_meta}
        arms_list: list[dict[str, Any]] = []
        for (pair, shz, amp_val), g in df_w.groupby(["pair", "stim_hz", "amp"], sort=True):
            arm_id = f"{hemi}|{pair}|{int(round(shz))}Hz|{amp_val:g}mA"
            bs_arr = np.round(g["beta_supp"].values, 3)
            ft_arr = np.round(g["ftg_db"].values, 3)
            rid0 = str(g["rec_id"].iloc[0])
            r_meta = rec_meta_by_id.get(rid0, {})
            arm_contacts = sorted(set((r_meta.get("contacts") or []) + (r_meta.get("stim_contacts") or [])))
            arms_list.append(
                {
                    "arm_id": arm_id,
                    "hemisphere": hemi,
                    "pair": str(pair),
                    "contacts": arm_contacts,
                    "stim_hz": float(shz),
                    "ftg_center_hz": round(float(shz) / 2.0, 2),
                    "beta_peak_hz": series_peak_map.get((str(pair), round(float(shz), 1)), 20.0),
                    "amp": round(float(amp_val), 2),
                    "energy": round(float((float(amp_val) / amax) ** 2), 4),
                    "n": int(len(g)),
                    "rec_id": rid0,
                    "label": f"{pair} · {int(round(shz))}Hz @ {amp_val:g} mA",
                    "short_label": f"{pair}@{amp_val:g}mA",
                    "beta_supp_mean": round(float(np.mean(bs_arr)), 3),
                    "beta_supp_sd": round(float(np.std(bs_arr)), 3),
                    "ftg_db_mean": round(float(np.mean(ft_arr)), 3),
                    "ftg_db_sd": round(float(np.std(ft_arr)), 3),
                    "win_beta_supp": bs_arr.tolist(),
                    "win_ftg_db": ft_arr.tolist(),
                }
            )

        # Fit Stackelberg Follower Dose–Response Curves per (pair, stim_hz) with bootstrap bands
        stack_fits: list[dict[str, Any]] = []
        for (pair, shz), g in df_w.groupby(["pair", "stim_hz"], sort=True):
            a_vals = g["amp"].values.astype(float)
            if len(np.unique(a_vals)) < 2:
                continue
            rid0 = str(g["rec_id"].iloc[0])
            r_meta = rec_meta_by_id.get(rid0, {})
            fit_contacts = sorted(set((r_meta.get("contacts") or []) + (r_meta.get("stim_contacts") or [])))
            yb = g["beta_supp"].values.astype(float)
            yf = g["ftg_db"].values.astype(float)
            a_hi = float(np.max(a_vals))
            grid = np.round(np.arange(0.0, a_hi + 1e-9, 0.05), 2)
            if len(grid) < 2:
                grid = np.linspace(0.0, max(a_hi, 0.5), 11)

            mid_b = _fit_beta_curve(a_vals, yb)(grid)
            mid_f = _fit_ftg_curve(a_vals, yf)(grid)

            boots_b = []
            boots_f = []
            n_pts = len(a_vals)
            for _ in range(n_boot):
                ix = rng.integers(0, n_pts, n_pts)
                if len(np.unique(a_vals[ix])) < 2:
                    continue
                boots_b.append(_fit_beta_curve(a_vals[ix], yb[ix])(grid))
                boots_f.append(_fit_ftg_curve(a_vals[ix], yf[ix])(grid))

            if boots_b:
                bb_mat = np.asarray(boots_b)
                bf_mat = np.asarray(boots_f)
                b_lo = np.percentile(bb_mat, 10, axis=0)
                b_hi = np.percentile(bb_mat, 90, axis=0)
                f_lo = np.percentile(bf_mat, 10, axis=0)
                f_hi = np.percentile(bf_mat, 90, axis=0)
            else:
                b_lo, b_hi = mid_b - 0.5, mid_b + 0.5
                f_lo, f_hi = mid_f - 0.5, mid_f + 0.5

            stack_fits.append(
                {
                    "id": f"{hemi}|{pair}|{int(round(shz))}Hz",
                    "hemisphere": hemi,
                    "pair": str(pair),
                    "contacts": fit_contacts,
                    "stim_hz": float(shz),
                    "ftg_center_hz": round(float(shz) / 2.0, 2),
                    "beta_peak_hz": series_peak_map.get((str(pair), round(float(shz), 1)), 20.0),
                    "amax_series": round(a_hi, 2),
                    "grid": [round(float(v), 2) for v in grid],
                    "beta": [round(float(v), 3) for v in mid_b],
                    "beta_lo": [round(float(v), 3) for v in b_lo],
                    "beta_hi": [round(float(v), 3) for v in b_hi],
                    "ftg": [round(float(v), 3) for v in mid_f],
                    "ftg_lo": [round(float(v), 3) for v in f_lo],
                    "ftg_hi": [round(float(v), 3) for v in f_hi],
                }
            )

        hemis_out[hemi] = {
            "hemisphere": hemi,
            "mode": mode_kind,
            "amax": round(amax, 2),
            "stim_rates": stim_rates,
            "n_windows_total": int(len(df_w)),
            "arms": arms_list,
            "stackelberg_fits": stack_fits,
            "series_meta": series_meta,
        }

    return {
        "window_spec": {"fs_hz": FS, "win_samples": WIN, "hop_samples": HOP, "settle_samples": SETTLE, "ftg_hw_hz": ftg_hw},
        "scenarios": SCENARIOS,
        "default_weights": DEFAULT_WEIGHTS,
        "hemispheres": hemis_out,
    }


def solve_game_theory_exact(
    gt_bundle: dict[str, Any],
    hemisphere: str = "Left",
    stim_rate_filter: str | float | None = "ALL",
    weights: dict[str, float] | None = None,
    bandit_live: dict[str, Any] | None = None,
    selected_contacts: list[str] | None = None,
) -> dict[str, Any]:
    """Solve all exact game-theoretic equilibria (LP Zero-Sum Nash, Maximin, Nash Bargaining,
    Pareto Frontier, Stackelberg & Robust Stackelberg, and 7-Method Consensus) in Python,
    dynamically filtered by the contacts selected on the DBS Lead Model."""
    W = dict(DEFAULT_WEIGHTS)
    if weights:
        for k in ("wb", "wf", "we", "tau", "ftg_safe"):
            if k in weights and weights[k] is not None:
                W[k] = float(weights[k])

    h_data = (gt_bundle.get("hemispheres") or {}).get(hemisphere) or {}
    all_arms = h_data.get("arms") or []
    amax = float(h_data.get("amax") or 1.0)

    # Filter strictly by contacts selected on the main DBS Lead Model
    sel_set = set(selected_contacts) if selected_contacts is not None else None
    if sel_set is not None:
        contact_arms = [
            a for a in all_arms
            if any(f"{a['hemisphere']}:{c}" in sel_set for c in (a.get("contacts") or []))
        ]
    else:
        contact_arms = list(all_arms)

    # Optional stimulation rate filter (e.g., 125, 145, 165, or "ALL")
    if stim_rate_filter not in (None, "", "ALL", "All"):
        try:
            rf = float(stim_rate_filter)
            arms = [a for a in contact_arms if abs(float(a["stim_hz"]) - rf) < 1.0]
        except Exception:
            arms = list(contact_arms)
    else:
        arms = list(contact_arms)

    if not arms:
        return {
            "hemisphere": hemisphere,
            "stim_rate_filter": stim_rate_filter,
            "mode": h_data.get("mode", ""),
            "weights": W,
            "amax": amax,
            "arms_df": pd.DataFrame(),
            "consensus_df": pd.DataFrame(),
            "majority_arm": None,
            "majority_votes": "0 of 0",
            "game_value": 0.0,
            "payoff_matrix": [],
            "scenario_names": list(SCENARIOS.keys()),
            "x_star": [],
            "nature_mix": {},
            "maximin_arm": None,
            "nash_bargaining_arm": None,
            "disagreement_point": [0.0, 0.0],
            "stackelberg": [],
            "bandit_live": bandit_live or {},
        }

    # Evaluate utility per arm (both mean and sd from real windows)
    rows = []
    for a in arms:
        bs = np.asarray(a["win_beta_supp"], dtype=float)
        ft = np.asarray(a["win_ftg_db"], dtype=float)
        u_vals = utility_fn(bs, ft, a["amp"], amax, **W)
        u1 = float(a["beta_supp_mean"])
        u2 = float(-(W["wf"] * max(0.0, float(a["ftg_db_mean"]) - W["tau"]) + W["we"] * 6.0 * float(a["energy"])))
        rows.append(
            {
                "arm_id": a["arm_id"],
                "label": a["label"],
                "pair": a["pair"],
                "stim_hz": a["stim_hz"],
                "ftg_center_hz": a["ftg_center_hz"],
                "amp_ma": a["amp"],
                "n_windows": a["n"],
                "beta_supp_db": a["beta_supp_mean"],
                "ftg_db": a["ftg_db_mean"],
                "energy": a["energy"],
                "U_mean": round(float(np.mean(u_vals)), 3),
                "U_sd": round(float(np.std(u_vals)), 3),
                "u1_efficacy": round(u1, 3),
                "u2_safety": round(u2, 3),
            }
        )
    arms_df = pd.DataFrame(rows)

    # 1. Exact Zero-Sum Nash Equilibrium via Linear Programming (Clinician vs Nature's 4 scenarios)
    scen_names = list(SCENARIOS.keys())
    A = np.column_stack(
        [
            utility_fn(
                arms_df["beta_supp_db"].values,
                arms_df["ftg_db"].values,
                arms_df["amp_ma"].values,
                amax,
                tau=W["tau"],
                **sc,
            )
            for sc in SCENARIOS.values()
        ]
    )
    m, k = A.shape
    x_star = np.zeros(m)
    y_star = np.full(k, 1.0 / k)
    game_value = float(np.max(A.min(axis=1)))

    try:
        r1 = linprog(
            c=np.r_[np.zeros(m), -1.0],
            A_ub=np.c_[-A.T, np.ones(k)],
            b_ub=np.zeros(k),
            A_eq=np.r_[np.ones(m), 0.0][None, :],
            b_eq=[1.0],
            bounds=[(0.0, None)] * m + [(None, None)],
            method="highs",
        )
        if r1.success and r1.x is not None:
            x_star = np.maximum(0.0, r1.x[:m])
            x_star /= max(float(np.sum(x_star)), 1e-9)
            game_value = float(r1.x[-1])

        r2 = linprog(
            c=np.r_[np.zeros(k), 1.0],
            A_ub=np.c_[A, -np.ones(m)],
            b_ub=np.zeros(m),
            A_eq=np.r_[np.ones(k), 0.0][None, :],
            b_eq=[1.0],
            bounds=[(0.0, None)] * k + [(None, None)],
            method="highs",
        )
        if r2.success and r2.x is not None:
            y_star = np.maximum(0.0, r2.x[:k])
            y_star /= max(float(np.sum(y_star)), 1e-9)
    except Exception:
        x_star[int(np.argmax(A.min(axis=1)))] = 1.0

    maximin_i = int(np.argmax(A.min(axis=1)))
    best_i = int(np.argmax(arms_df["U_mean"].values))

    # 2. Exact Cooperative Nash Bargaining Solution & Pareto Frontier
    u1_v = arms_df["u1_efficacy"].values
    u2_v = arms_df["u2_safety"].values
    d1, d2 = float(np.min(u1_v)), float(np.min(u2_v))
    nash_prod = (u1_v - d1) * (u2_v - d2)
    nbs_i = int(np.argmax(nash_prod))
    pareto = np.array(
        [not np.any((u1_v >= u1_v[i]) & (u2_v >= u2_v[i]) & ((u1_v > u1_v[i]) | (u2_v > u2_v[i]))) for i in range(m)]
    )

    arms_df["pareto"] = pareto
    arms_df["nash_product"] = np.round(nash_prod, 3)
    arms_df["p_nash_mixed"] = np.round(x_star, 4)

    # 3. Exact Stackelberg & Robust Stackelberg on Continuous Fitted Curves (filtered by selected contacts)
    fits = h_data.get("stackelberg_fits") or []
    if sel_set is not None:
        fits = [f for f in fits if any(f"{f['hemisphere']}:{c}" in sel_set for c in (f.get("contacts") or []))]
    if stim_rate_filter not in (None, "", "ALL", "All"):
        try:
            rf = float(stim_rate_filter)
            fits = [f for f in fits if abs(float(f["stim_hz"]) - rf) < 1.0]
        except Exception:
            pass

    stack_out = []
    for f in fits:
        g = np.asarray(f["grid"], dtype=float)
        U_std = utility_fn(np.asarray(f["beta"]), np.asarray(f["ftg"]), g, amax, **W)
        U_rob = utility_fn(np.asarray(f["beta_lo"]), np.asarray(f["ftg_hi"]), g, amax, **W)
        i_s = int(np.argmax(U_std))
        i_r = int(np.argmax(U_rob))
        stack_out.append(
            {
                "id": f["id"],
                "pair": f["pair"],
                "stim_hz": f["stim_hz"],
                "ftg_center_hz": f["ftg_center_hz"],
                "a_star": round(float(g[i_s]), 2),
                "U_star": round(float(U_std[i_s]), 3),
                "a_robust": round(float(g[i_r]), 2),
                "U_robust": round(float(U_rob[i_r]), 3),
            }
        )

    # 4. 7-Method Consensus Table & Majority Vote
    def _nearest_arm_label(pair: str, stim_hz: float, amp_target: float) -> str:
        sub = arms_df[(arms_df["pair"] == pair) & (np.abs(arms_df["stim_hz"] - stim_hz) < 1.0)]
        if sub.empty:
            sub = arms_df
        idx = (sub["amp_ma"] - amp_target).abs().idxmin()
        return str(arms_df.loc[idx, "label"])

    top_mix = np.argsort(-x_star)[:3]
    mix_str = ", ".join(f"{arms_df.iloc[i]['label']} ({x_star[i]:.0%})" for i in top_mix if x_star[i] > 0.01)
    nat_str = ", ".join(f"{n} {p:.0%}" for n, p in zip(scen_names, y_star) if p > 0.01)

    cons_rows = [
        {
            "method": "1. Mean Utility Oracle",
            "recommendation": str(arms_df.iloc[best_i]["label"]),
            "voted_arm": str(arms_df.iloc[best_i]["label"]),
            "U_dB_eq": round(float(arms_df.iloc[best_i]["U_mean"]), 2),
            "note": f"Max mean U across {int(arms_df.iloc[best_i]['n_windows'])} windows",
        },
        {
            "method": "2. Pure Minimax (Maximin)",
            "recommendation": str(arms_df.iloc[maximin_i]["label"]),
            "voted_arm": str(arms_df.iloc[maximin_i]["label"]),
            "U_dB_eq": round(float(A[maximin_i].min()), 2),
            "note": "Best worst-case across 4 clinical scenarios",
        },
        {
            "method": "3. Zero-Sum Nash (Mixed LP)",
            "recommendation": mix_str or str(arms_df.iloc[maximin_i]["label"]),
            "voted_arm": str(arms_df.iloc[int(np.argmax(x_star))]["label"]),
            "U_dB_eq": round(game_value, 2),
            "note": f"Nature mix y*: {nat_str}",
        },
        {
            "method": "4. Nash Bargaining (Pareto)",
            "recommendation": str(arms_df.iloc[nbs_i]["label"]),
            "voted_arm": str(arms_df.iloc[nbs_i]["label"]),
            "U_dB_eq": round(float(arms_df.iloc[nbs_i]["U_mean"]), 2),
            "note": f"Max (u₁−d₁)(u₂−d₂) = {nash_prod[nbs_i]:.2f}",
        },
    ]

    if stack_out:
        sb = max(stack_out, key=lambda s: s["U_star"])
        sr = max(stack_out, key=lambda s: s["U_robust"])
        cons_rows.append(
            {
                "method": "5. Stackelberg (Fitted Curve)",
                "recommendation": f"{sb['pair']} · {int(round(sb['stim_hz']))}Hz @ {sb['a_star']:.2f} mA",
                "voted_arm": _nearest_arm_label(sb["pair"], sb["stim_hz"], sb["a_star"]),
                "U_dB_eq": round(sb["U_star"], 2),
                "note": "Continuous leader optimum on β-sigmoid & FTG-hinge",
            }
        )
        cons_rows.append(
            {
                "method": "6. Robust Stackelberg (Pessimistic)",
                "recommendation": f"{sr['pair']} · {int(round(sr['stim_hz']))}Hz @ {sr['a_robust']:.2f} mA",
                "voted_arm": _nearest_arm_label(sr["pair"], sr["stim_hz"], sr["a_robust"]),
                "U_dB_eq": round(sr["U_robust"], 2),
                "note": "10th-pct Δβ & 90th-pct FTG bootstrap bands",
            }
        )

    if bandit_live and bandit_live.get("recommended_label"):
        cons_rows.append(
            {
                "method": f"7. Live Bandit ({bandit_live.get('algo', 'thompson')})",
                "recommendation": str(bandit_live["recommended_label"]),
                "voted_arm": str(bandit_live["recommended_label"]),
                "U_dB_eq": bandit_live.get("recommended_U"),
                "note": f"{bandit_live.get('pulls', 0)} pulls · regret {bandit_live.get('regret', 0):.1f}",
            }
        )
    else:
        cons_rows.append(
            {
                "method": "7. Live Bandit (DAG Node)",
                "recommendation": "Run Bandit node to vote",
                "voted_arm": "",
                "U_dB_eq": None,
                "note": "Insert or run Live Bandit node on DAG canvas",
            }
        )

    consensus_df = pd.DataFrame(cons_rows)
    valid_votes = [v for v in consensus_df["voted_arm"].tolist() if v]
    majority_arm, majority_count = None, 0
    if valid_votes:
        counts = pd.Series(valid_votes).value_counts()
        majority_arm = str(counts.index[0])
        majority_count = int(counts.iloc[0])

    return {
        "hemisphere": hemisphere,
        "stim_rate_filter": stim_rate_filter,
        "mode": h_data.get("mode", ""),
        "weights": W,
        "amax": amax,
        "arms_df": arms_df,
        "consensus_df": consensus_df,
        "majority_arm": majority_arm,
        "majority_votes": f"{majority_count} of {len(valid_votes)}",
        "game_value": round(game_value, 3),
        "payoff_matrix": np.round(A, 3).tolist(),
        "scenario_names": scen_names,
        "x_star": [round(float(v), 4) for v in x_star],
        "nature_mix": {n: round(float(p), 4) for n, p in zip(scen_names, y_star)},
        "maximin_arm": str(arms_df.iloc[maximin_i]["label"]),
        "nash_bargaining_arm": str(arms_df.iloc[nbs_i]["label"]),
        "disagreement_point": [round(d1, 3), round(d2, 3)],
        "stackelberg": stack_out,
        "bandit_live": bandit_live or {},
    }
