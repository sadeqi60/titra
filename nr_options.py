"""NeuroReader Option-Pricing Engine (NeuroOptions Integration).

Adapts the NeuroOptions mathematical formulation natively to Medtronic Percept JSON
exports (multi-amplitude & multi-frequency BrainSenseTimeDomain sweeps, plus fallback to
LfpMontageTimeDomain survey contact arms when a session has constant amplitude).

Core concepts (matching neurooptions.py):
  - Causal RBJ 4th-order bandpass + causal moving-RMS envelope ratio
        S_t = e(t) / median(e_0mA)
    sampled every 20 ms (DT = 0.02 s, FS = 250 Hz, SETTLE = 125, DEC = 5).
  - Log-OU dynamics per series × amplitude via AR(1) on log S_t:
        θ (log level), κ (reversion speed 1/s), t_{1/2} = 1000·ln(2)/κ (ms),
        σ = s·√(2κ) (burstiness), s (stationary log-sd).
  - Exact Black-76 (r = 0, physical measure) expected clinical payoffs:
        call (burst load), strangle (corridor), straddle (setpoint), butterfly (target prob),
        bullspread (capped load), calendar (persistence), ftgcall (side-effect load).
  - Nelson–Ramaswamy mean-reverting binomial lattice (European + American early exercise):
        early-exercise boundary S*(t) = optimal aDBS trigger; Am − Eu = value of adaptivity.
  - Realized market prices & implied-volatility smile IV(K) vs flat log-normal s.
  - Weighted quadratic θ(a), s(a) fits → Alpha = ∂Π/∂a and continuous optimum a*.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.signal import sosfilt, welch
from scipy.stats import norm

FS = 250.0
SETTLE = 125        # 0.5 s settle after amplitude transition
DEC = 5             # 20 ms envelope step (50 Hz)
DT = 0.02
FTG_ENV_N = 25
BW_Q = (0.5411961, 1.3065630)
STRATS = ("call", "strangle", "straddle", "butterfly", "bullspread")
SMILE_KS = [round(float(k), 2) for k in np.arange(0.5, 2.21, 0.1)]
CHAIN_KS = [round(float(0.4 + 0.15 * i), 2) for i in range(12)]

DEFAULT_OPT_PARAMS: dict[str, Any] = {
    "hemisphere": "Left",
    "stim_rate_filter": "ALL",
    "strategy": "call",
    "T": 0.25,
    "k_burst_mult": 1.0,
    "k_low": 0.5,
    "k_set": 0.7,
    "wing": 0.25,
    "w_ftg": 0.5,
    "w_energy": 0.02,
    "steps": 10,
}


# ── Causal Envelope Primitives ──
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


# ── Option Pricing Library (exact mirror of neurooptions.py) ──
def ou_fit(seqs: list[np.ndarray]) -> dict[str, float]:
    """AR(1) on log S across sequences → θ (level), κ (1/s), σ (log-vol/√s), s (stationary log-sd)."""
    L = [np.log(np.maximum(s, 1e-4)) for s in seqs if len(s) > 5]
    if not L:
        flat = np.concatenate([np.log(np.maximum(s, 1e-4)) for s in seqs if len(s) > 0])
        th = float(flat.mean()) if flat.size else 0.0
        s = max(float(flat.std()), 0.05) if flat.size > 1 else 0.25
        kap = 6.0
        return {
            "theta": th,
            "kappa": kap,
            "sigma": float(s * math.sqrt(2.0 * kap)),
            "s": s,
            "n": int(flat.size),
            "halflife_ms": float(1000.0 * math.log(2.0) / kap),
        }
    x = np.concatenate(L)
    xs = np.concatenate([l[:-1] for l in L])
    ys = np.concatenate([l[1:] for l in L])
    mx, my = float(xs.mean()), float(ys.mean())
    denom = float(np.sum((xs - mx) ** 2))
    phi = float(np.clip(np.sum((xs - mx) * (ys - my)) / max(denom, 1e-12), 1e-3, 0.999))
    kap = -math.log(phi) / DT
    th = float(x.mean())
    s2 = max(float(x.var()), 1e-4)
    return {
        "theta": th,
        "kappa": float(kap),
        "sigma": float(math.sqrt(s2 * 2.0 * kap)),
        "s": float(math.sqrt(s2)),
        "n": int(x.size),
        "halflife_ms": float(1000.0 * math.log(2.0) / kap),
    }


def ou_moments(x0: float, p: dict[str, float], T: float) -> tuple[float, float]:
    e = math.exp(-p["kappa"] * T)
    return p["theta"] + (x0 - p["theta"]) * e, (p["s"] ** 2) * (1.0 - e * e)


def black76(F: float, K: float, v: float, kind: str = "call") -> float:
    """Black-76 with r = 0: expected payoff under log-normal(mean=F, log-var=v)."""
    if K <= 0:
        return max(F, 0.0) if kind == "call" else 0.0
    v = max(float(v), 1e-12)
    sv = math.sqrt(v)
    d1 = (math.log(max(F, 1e-12) / K) + 0.5 * v) / sv
    d2 = d1 - sv
    if kind == "call":
        return float(F * norm.cdf(d1) - K * norm.cdf(d2))
    return float(K * norm.cdf(-d2) - F * norm.cdf(-d1))


def price_from_logn(m: float, v: float, legs: list[tuple[float, str, float]]) -> float:
    F = math.exp(m + 0.5 * max(v, 0.0))
    return float(sum(sg * black76(F, K, v, kd) for sg, kd, K in legs if K > 0))


def strategy_legs(name: str, K: dict[str, float]) -> list[tuple[float, str, float]]:
    kb = float(K["burst"])
    kl = float(K["low"])
    ks = float(K["set"])
    kc = kb + float(K.get("ceiling_add", 0.6))
    w = max(0.05, float(K.get("wing", 0.25)))
    kf = float(K["ftg"])
    mapping = {
        "call": [(1.0, "call", kb)],
        "strangle": [(1.0, "put", kl), (1.0, "call", kb)],
        "straddle": [(1.0, "put", ks), (1.0, "call", ks)],
        "butterfly": [(1.0 / w, "call", max(0.05, ks - w)), (-2.0 / w, "call", ks), (1.0 / w, "call", ks + w)],
        "bullspread": [(1.0, "call", kb), (-1.0, "call", kc)],
        "calendar": [(1.0, "call", kb)],
        "ftgcall": [(1.0, "call", kf)],
    }
    return mapping.get(name, mapping["call"])


def payoff_arr(S: np.ndarray, legs: list[tuple[float, str, float]]) -> np.ndarray:
    tot = np.zeros_like(S, dtype=float)
    for sg, kd, K in legs:
        tot += sg * (np.maximum(S - K, 0.0) if kd == "call" else np.maximum(K - S, 0.0))
    return tot


def binomial_ou(
    x0: float,
    p: dict[str, float],
    T: float,
    K: float,
    steps: int = 80,
    kind: str = "call",
    american: bool = True,
) -> tuple[float, float]:
    """Mean-reverting recombining lattice on x = log S (Nelson–Ramaswamy)."""
    steps = max(4, int(steps))
    dt = max(T, 1e-3) / steps
    sig = max(float(p["sigma"]), 1e-4)
    dx = sig * math.sqrt(dt)
    idx_end = np.arange(steps + 1, dtype=float)
    x_end = x0 + dx * (2.0 * idx_end - steps)
    S_end = np.exp(np.clip(x_end, -15.0, 15.0))
    V = np.maximum(S_end - K, 0.0) if kind == "call" else np.maximum(K - S_end, 0.0)
    Veu = V.copy()
    kap = float(p["kappa"])
    th = float(p["theta"])
    c_pu = kap * math.sqrt(dt) / (2.0 * sig)
    for n in range(steps - 1, -1, -1):
        idx = np.arange(n + 1, dtype=float)
        x = x0 + dx * (2.0 * idx - n)
        pu = np.clip(0.5 + c_pu * (th - x), 0.0, 1.0)
        cont = pu * V[1:] + (1.0 - pu) * V[:-1]
        Veu = pu * Veu[1:] + (1.0 - pu) * Veu[:-1]
        if american:
            S_n = np.exp(np.clip(x, -15.0, 15.0))
            ex = np.maximum(S_n - K, 0.0) if kind == "call" else np.maximum(K - S_n, 0.0)
            V = np.maximum(cont, ex)
        else:
            V = cont
    return float(V[0]), float(Veu[0])


def implied_vol(price: float, F: float, K: float, kind: str = "call") -> float:
    intrinsic = max(F - K, 0.0) if kind == "call" else max(K - F, 0.0)
    upper = F if kind == "call" else K
    if not (intrinsic + 1e-7 < price < upper - 1e-7):
        return float("nan")
    try:
        return float(brentq(lambda s: black76(F, K, s * s, kind) - price, 1e-4, 5.0))
    except Exception:
        return float("nan")


def _clean_num(v: Any, nd: int = 5) -> float | None:
    try:
        fv = float(v)
        return round(fv, nd) if math.isfinite(fv) else None
    except Exception:
        return None


def _compact_hist(S: np.ndarray, xmax: float = 2.6, n_bins: int = 28) -> list[float]:
    if len(S) == 0:
        return [0.0] * n_bins
    h, _ = np.histogram(S, bins=n_bins, range=(0.0, xmax), density=True)
    mx = float(np.max(h))
    if mx <= 0:
        return [0.0] * n_bins
    return [round(float(v / mx), 3) for v in h]


def _fit_param_curves(arm_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Weighted quadratic (or linear) fits of θ(a), s(a) per series for Alpha = ∂Π/∂a."""
    amps = np.array([a["amp"] for a in arm_rows], dtype=float)
    w = np.sqrt(np.maximum(np.array([a["b_n"] for a in arm_rows], dtype=float), 1.0))
    amax = float(np.max(amps)) if len(amps) else 0.5
    n_uniq = len(np.unique(amps))
    grid = [round(float(v), 2) for v in np.arange(0.0, amax + 1e-9, 0.05)] if amax > 0 else [0.0]
    if len(grid) < 2 and amax > 0:
        grid = [0.0, round(amax, 2)]

    out: dict[str, Any] = {"amax": round(amax, 2), "grid": grid}
    for col in ("b_theta", "b_s", "f_theta", "f_s"):
        y = np.array([a[col] for a in arm_rows], dtype=float)
        if n_uniq >= 4:
            coef = np.polyfit(amps, y, 2, w=w)
        elif n_uniq >= 2:
            coef = np.polyfit(amps, y, 1, w=w)
        else:
            coef = np.array([float(np.mean(y))])
        out[col] = [round(float(c), 5) for c in coef]
    return out


def extract_options_bundle(
    full_store: dict[str, Any],
    ftg_hw: float = 2.0,
) -> dict[str, Any]:
    """Extract log-OU dynamics, Black-76 & realized payoffs, binomial American/European values,
    IV smiles, histograms, and smooth θ(a)/s(a) Alpha curves from parsed Percept JSON recordings."""
    payload = full_store["payload"]
    raw_signals = full_store["raw_signals"]
    recordings = payload.get("recordings") or []

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

        # Pass 1: Causal envelopes per series
        series_raw: list[dict[str, Any]] = []
        hemi_base_pool: dict[str, list[np.ndarray]] = {"beta": [], "ftg": []}

        for (pair, shz), rec_list in sorted(grouped.items()):
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
                if s["n"] <= SETTLE + DEC * 6:
                    continue
                xs = x_full[s["start"] : s["start"] + s["n"]]
                xs = xs - float(np.mean(xs))
                eb = _mov_rms(sosfilt(sos_b, xs), env_nb)
                ef = _mov_rms(sosfilt(sos_f, xs), FTG_ENV_N)
                j = np.arange(SETTLE, s["n"], DEC)
                if j.size < 6:
                    continue
                seqs.append({"amp": round(float(s["amp"]), 2), "beta": eb[j], "ftg": ef[j]})

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
            }
            continue

        hemi_fallback_med = {
            tg: float(np.median(np.concatenate(hemi_base_pool[tg])))
            if hemi_base_pool[tg]
            else float(np.median(np.concatenate([q[tg] for sr in series_raw for q in sr["seqs"]])))
            for tg in ("beta", "ftg")
        }

        # Pass 2: Normalize envelopes, fit log-OU, compute premiums/smiles/curves
        series_out: list[dict[str, Any]] = []
        all_amps_hemi: list[float] = []
        all_rates_hemi: set[float] = set()

        for sr in series_raw:
            all_rates_hemi.add(float(sr["stim_hz"]))
            seqs = sr["seqs"]
            for tg in ("beta", "ftg"):
                z_raw = [q[tg] for q in seqs if q["amp"] <= 0.01]
                med = float(np.median(np.concatenate(z_raw))) if z_raw else hemi_fallback_med[tg]
                med = max(med, 1e-9)
                for q in seqs:
                    q[f"S_{tg}"] = q[tg] / med

            base_b_list = [q["S_beta"] for q in seqs if q["amp"] <= 0.01]
            base_f_list = [q["S_ftg"] for q in seqs if q["amp"] <= 0.01]
            base_b = np.concatenate(base_b_list) if base_b_list else np.concatenate([q["S_beta"] for q in seqs])
            base_f = np.concatenate(base_f_list) if base_f_list else np.concatenate([q["S_ftg"] for q in seqs])
            K = {
                "burst": round(float(np.percentile(base_b, 75)), 4),
                "low": float(DEFAULT_OPT_PARAMS["k_low"]),
                "set": float(DEFAULT_OPT_PARAMS["k_set"]),
                "ceiling_add": 0.6,
                "wing": float(DEFAULT_OPT_PARAMS["wing"]),
                "ftg": round(float(np.percentile(base_f, 90)), 4),
            }

            amps = sorted({q["amp"] for q in seqs})
            all_amps_hemi.extend(amps)
            arm_rows: list[dict[str, Any]] = []
            smiles: dict[str, dict[str, Any]] = {}

            for a in amps:
                sb_list = [q["S_beta"] for q in seqs if q["amp"] == a]
                sf_list = [q["S_ftg"] for q in seqs if q["amp"] == a]
                Sb = np.concatenate(sb_list)
                Sf = np.concatenate(sf_list)
                if Sb.size < 6:
                    continue
                pb = ou_fit(sb_list)
                pf = ou_fit(sf_list)

                row: dict[str, Any] = {
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
                    "seconds": round(float(Sb.size * DT), 2),
                    "b_theta": round(pb["theta"], 4),
                    "b_kappa": round(pb["kappa"], 3),
                    "b_sigma": round(pb["sigma"], 4),
                    "b_s": round(pb["s"], 4),
                    "b_n": int(pb["n"]),
                    "b_halflife_ms": round(pb["halflife_ms"], 1),
                    "f_theta": round(pf["theta"], 4),
                    "f_kappa": round(pf["kappa"], 3),
                    "f_sigma": round(pf["sigma"], 4),
                    "f_s": round(pf["s"], 4),
                    "f_n": int(pf["n"]),
                    "f_halflife_ms": round(pf["halflife_ms"], 1),
                    "hist_b": _compact_hist(Sb),
                    "hist_f": _compact_hist(Sf),
                }

                for st in STRATS:
                    legs = strategy_legs(st, K)
                    row[f"prem_{st}"] = round(price_from_logn(pb["theta"], pb["s"] ** 2, legs), 5)
                    row[f"emp_{st}"] = round(float(np.mean(payoff_arr(Sb, legs))), 5)
                row["prem_ftg"] = round(price_from_logn(pf["theta"], pf["s"] ** 2, [(1.0, "call", K["ftg"])]), 5)
                row["emp_ftg"] = round(float(np.mean(np.maximum(Sf - K["ftg"], 0.0))), 5)

                # Calendar & American intervention value at burst onset S0 = K_burst
                x0 = math.log(max(K["burst"], 0.05))
                T0 = float(DEFAULT_OPT_PARAMS["T"])
                m1, v1 = ou_moments(x0, pb, T0)
                m2, v2 = ou_moments(x0, pb, 2.0 * T0)
                row["calendar"] = round(
                    price_from_logn(m2, v2, [(1.0, "call", K["burst"])]) - price_from_logn(m1, v1, [(1.0, "call", K["burst"])]),
                    5,
                )
                am, eu = binomial_ou(x0, pb, T0, K["burst"], steps=60, american=True)
                row["american"] = round(am, 5)
                row["european_tree"] = round(eu, 5)
                row["eu_black"] = round(price_from_logn(m1, v1, [(1.0, "call", K["burst"])]), 5)
                row["adaptivity_premium"] = round(max(0.0, am - eu), 5)

                # Implied-vol smile from realized payoffs
                F_emp = float(Sb.mean())
                ivs = []
                for k_val in SMILE_KS:
                    c_emp = float(np.mean(np.maximum(Sb - k_val, 0.0)))
                    ivs.append(_clean_num(implied_vol(c_emp, F_emp, k_val), 4))
                smiles[f"{round(float(a), 2):g}"] = {"K": SMILE_KS, "iv": ivs, "model": round(pb["s"], 4)}

                arm_rows.append(row)

            if not arm_rows:
                continue

            curves = _fit_param_curves(arm_rows)
            series_out.append(
                {
                    "id": sr["id"],
                    "hemisphere": hemi,
                    "pair": sr["pair"],
                    "contacts": sr["contacts"],
                    "stim_hz": sr["stim_hz"],
                    "ftg_center_hz": sr["ftg_center_hz"],
                    "beta_peak_hz": sr["beta_peak_hz"],
                    "rec_id": sr["rec_id"],
                    "label": f"{sr['pair']} · {int(round(sr['stim_hz']))}Hz",
                    "K": K,
                    "arms": arm_rows,
                    "smiles": smiles,
                    "curves": curves,
                }
            )

        hemis_out[hemi] = {
            "hemisphere": hemi,
            "mode": mode_kind,
            "amax": round(max(all_amps_hemi) if all_amps_hemi else 1.0, 2),
            "stim_rates": sorted(all_rates_hemi),
            "series": series_out,
        }

    return {
        "spec": {
            "fs_hz": FS,
            "settle_samples": SETTLE,
            "dec": DEC,
            "dt_s": DT,
            "ftg_hw_hz": ftg_hw,
            "hist_xmax": 2.6,
            "hist_bins": 28,
            "chain_ks": CHAIN_KS,
        },
        "default_params": DEFAULT_OPT_PARAMS,
        "hemispheres": hemis_out,
    }


def _polyval_list(coefs: list[float], x: float) -> float:
    acc = 0.0
    for c in coefs:
        acc = acc * x + c
    return acc


def solve_options_exact(
    opt_bundle: dict[str, Any],
    hemisphere: str = "Left",
    stim_rate_filter: str | float | None = "ALL",
    params: dict[str, Any] | None = None,
    selected_contacts: list[str] | None = None,
) -> dict[str, Any]:
    """Compute exact NeuroOptions strategy premiums, portfolio Π, Alpha = ∂Π/∂a optimum,
    5-rule selection board, and option chain Greeks dynamically scoped by selected contacts."""
    P = dict(DEFAULT_OPT_PARAMS)
    if params:
        for k in ("strategy", "T", "k_burst_mult", "k_low", "k_set", "wing", "w_ftg", "w_energy", "steps"):
            if k in params and params[k] is not None:
                P[k] = params[k]
    P["hemisphere"] = hemisphere
    P["stim_rate_filter"] = stim_rate_filter

    strat = str(P.get("strategy") or "call")
    T = max(0.05, float(P.get("T") or 0.25))
    kb_mult = max(0.4, float(P.get("k_burst_mult") or 1.0))
    k_low = max(0.1, float(P.get("k_low") or 0.5))
    k_set = max(0.2, float(P.get("k_set") or 0.7))
    wing = max(0.05, float(P.get("wing") or 0.25))
    w_ftg = max(0.0, float(P.get("w_ftg") if P.get("w_ftg") is not None else 0.5))
    w_energy = max(0.0, float(P.get("w_energy") if P.get("w_energy") is not None else 0.02))
    sign = -1.0 if strat == "butterfly" else 1.0
    is_ftg = strat == "ftgcall"

    h_data = (opt_bundle.get("hemispheres") or {}).get(hemisphere) or {}
    all_series = h_data.get("series") or []
    sel_set = set(selected_contacts) if selected_contacts is not None else None

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
        series.append(s)

    if not series:
        return {
            "hemisphere": hemisphere,
            "stim_rate_filter": stim_rate_filter,
            "params": P,
            "amax": float(h_data.get("amax") or 1.0),
            "opt_arms_df": pd.DataFrame(),
            "opt_selection_df": pd.DataFrame(),
            "opt_chain_df": pd.DataFrame(),
            "majority_arm": None,
            "majority_votes": "0 of 0",
            "picks": {},
        }

    def _stat_prem(st_name: str, p_ou: dict[str, float], K_d: dict[str, float]) -> float:
        legs = strategy_legs(st_name, K_d)
        if st_name == "calendar":
            x0 = math.log(max(K_d["burst"], 0.05))
            m1, v1 = ou_moments(x0, p_ou, T)
            m2, v2 = ou_moments(x0, p_ou, 2.0 * T)
            return price_from_logn(m2, v2, legs) - price_from_logn(m1, v1, legs)
        return price_from_logn(p_ou["theta"], p_ou["s"] ** 2, legs)

    arm_rows = []
    alpha_best_overall = None

    for s in series:
        K_s = {
            "burst": float(s["K"]["burst"]) * kb_mult,
            "low": k_low,
            "set": k_set,
            "ceiling_add": 0.6,
            "wing": wing,
            "ftg": float(s["K"]["ftg"]),
        }
        s_arms = s.get("arms") or []
        kb_mean = float(np.mean([a["b_kappa"] for a in s_arms])) if s_arms else 6.0
        kf_mean = float(np.mean([a["f_kappa"] for a in s_arms])) if s_arms else 6.0

        for a in s_arms:
            pb = {"theta": a["b_theta"], "kappa": a["b_kappa"], "sigma": a["b_sigma"], "s": a["b_s"]}
            pf = {"theta": a["f_theta"], "kappa": a["f_kappa"], "sigma": a["f_sigma"], "s": a["f_s"]}
            sp = _stat_prem(strat, pf if is_ftg else pb, K_s)
            fp = price_from_logn(pf["theta"], pf["s"] ** 2, [(1.0, "call", K_s["ftg"])])
            port = sign * sp + (0.0 if is_ftg else w_ftg * fp) + w_energy * (float(a["amp"]) ** 2)
            # Recompute adaptivity at current T & K_burst
            x0 = math.log(max(K_s["burst"], 0.05))
            am, eu = binomial_ou(x0, pb, T, K_s["burst"], steps=40, american=True)
            adapt = max(0.0, am - eu)
            arm_rows.append(
                {
                    "arm_id": a["arm_id"],
                    "label": a["label"],
                    "pair": a["pair"],
                    "stim_hz": a["stim_hz"],
                    "amp_ma": float(a["amp"]),
                    "seconds": float(a["seconds"]),
                    "theta_b": a["b_theta"],
                    "kappa_b": a["b_kappa"],
                    "halflife_ms": a["b_halflife_ms"],
                    "sigma_b": a["b_sigma"],
                    "s_b": a["b_s"],
                    "strat_prem": round(sp, 5),
                    "ftg_call": round(fp, 5),
                    "portfolio_Pi": round(port, 5),
                    "american": round(am, 5),
                    "european": round(eu, 5),
                    "adaptivity": round(adapt, 5),
                    "emp_call": a.get("emp_call"),
                }
            )

        # Evaluate Alpha = ∂Π/∂a continuous curve minimum for this series
        cv = s.get("curves") or {}
        grid = cv.get("grid") or []
        if len(grid) >= 2:
            best_pt = None
            for a_val in grid:
                pb_g = {"theta": _polyval_list(cv["b_theta"], a_val), "s": max(0.05, _polyval_list(cv["b_s"], a_val)), "kappa": kb_mean}
                pf_g = {"theta": _polyval_list(cv["f_theta"], a_val), "s": max(0.05, _polyval_list(cv["f_s"], a_val)), "kappa": kf_mean}
                sp_g = _stat_prem(strat, pf_g if is_ftg else pb_g, K_s)
                fp_g = price_from_logn(pf_g["theta"], pf_g["s"] ** 2, [(1.0, "call", K_s["ftg"])])
                pi_g = sign * sp_g + (0.0 if is_ftg else w_ftg * fp_g) + w_energy * (a_val**2)
                if best_pt is None or pi_g < best_pt["val"]:
                    best_pt = {"series": s["label"], "pair": s["pair"], "stim_hz": s["stim_hz"], "amp": a_val, "val": pi_g}
            if best_pt and (alpha_best_overall is None or best_pt["val"] < alpha_best_overall["val"]):
                alpha_best_overall = best_pt

    arms_df = pd.DataFrame(arm_rows)
    if arms_df.empty:
        return {
            "hemisphere": hemisphere,
            "stim_rate_filter": stim_rate_filter,
            "params": P,
            "amax": float(h_data.get("amax") or 1.0),
            "opt_arms_df": pd.DataFrame(),
            "opt_selection_df": pd.DataFrame(),
            "opt_chain_df": pd.DataFrame(),
            "majority_arm": None,
            "majority_votes": "0 of 0",
            "picks": {},
        }

    r_port = arms_df.loc[arms_df["portfolio_Pi"].idxmin()]
    r_strat = arms_df.loc[(sign * arms_df["strat_prem"]).idxmin()]
    pos_df = arms_df[arms_df["amp_ma"] > 0]
    r_ftg = pos_df.loc[pos_df["ftg_call"].idxmin()] if len(pos_df) else arms_df.loc[arms_df["ftg_call"].idxmin()]
    r_adapt = pos_df.loc[pos_df["adaptivity"].idxmax()] if len(pos_df) else arms_df.loc[arms_df["adaptivity"].idxmax()]

    # Nearest recorded arm for Alpha=0 vote
    alpha_vote_label = None
    alpha_disp = "—"
    alpha_val = None
    if alpha_best_overall is not None:
        alpha_disp = f"{alpha_best_overall['pair']} · {int(round(alpha_best_overall['stim_hz']))}Hz @ {alpha_best_overall['amp']:.2f} mA"
        alpha_val = round(float(alpha_best_overall["val"]), 4)
        sub = arms_df[(arms_df["pair"] == alpha_best_overall["pair"]) & (np.abs(arms_df["stim_hz"] - alpha_best_overall["stim_hz"]) < 1.0)]
        if sub.empty:
            sub = arms_df
        alpha_vote_label = str(arms_df.loc[(sub["amp_ma"] - alpha_best_overall["amp"]).abs().idxmin(), "label"])

    sel_rows = [
        {
            "rule": "1. min Portfolio Π (±strat + wF·FTG + wE·a²)",
            "pick": str(r_port["label"]),
            "voted_arm": str(r_port["label"]),
            "value": round(float(r_port["portfolio_Pi"]), 4),
            "profile": f"strat {r_port['strat_prem']:.4f} · FTG {r_port['ftg_call']:.4f}",
        },
        {
            "rule": f"2. {'max' if sign < 0 else 'min'} {strat} premium",
            "pick": str(r_strat["label"]),
            "voted_arm": str(r_strat["label"]),
            "value": round(float(r_strat["strat_prem"]), 4),
            "profile": f"Π {r_strat['portfolio_Pi']:.4f} · t½ {r_strat['halflife_ms']:.0f}ms",
        },
        {
            "rule": "3. min FTG Call (side-effect load, a > 0)",
            "pick": str(r_ftg["label"]),
            "voted_arm": str(r_ftg["label"]),
            "value": round(float(r_ftg["ftg_call"]), 4),
            "profile": f"Π {r_ftg['portfolio_Pi']:.4f} · strat {r_ftg['strat_prem']:.4f}",
        },
        {
            "rule": "4. max Value of Adaptivity (Am − Eu, aDBS)",
            "pick": str(r_adapt["label"]),
            "voted_arm": str(r_adapt["label"]),
            "value": round(float(r_adapt["adaptivity"]), 4),
            "profile": f"Am {r_adapt['american']:.4f} · Eu {r_adapt['european']:.4f}",
        },
        {
            "rule": "5. Alpha = ∂Π/∂a = 0 (fitted curve)",
            "pick": alpha_disp,
            "voted_arm": alpha_vote_label or "",
            "value": alpha_val,
            "profile": "Continuous quadratic θ(a), s(a) optimum" if alpha_best_overall else "Single amplitude only",
        },
    ]
    sel_df = pd.DataFrame(sel_rows)

    valid_votes = [r["voted_arm"] for r in sel_rows if r["voted_arm"]]
    majority_arm, majority_count = None, 0
    if valid_votes:
        vc = pd.Series(valid_votes).value_counts()
        majority_arm = str(vc.index[0])
        majority_count = int(vc.iloc[0])

    # Option Chain & Greeks around the portfolio-optimal arm (at S0 = exp(theta + s^2/2))
    p_ref = {"theta": float(r_port["theta_b"]), "kappa": float(r_port["kappa_b"]), "sigma": float(r_port["sigma_b"]), "s": float(r_port["s_b"])}
    S0_ref = math.exp(p_ref["theta"] + 0.5 * p_ref["s"] ** 2)
    x0_ref = math.log(max(S0_ref, 0.05))
    chain_rows = []
    hS = 0.03
    for k_val in CHAIN_KS:
        m0, v0 = ou_moments(x0_ref, p_ref, T)
        c0 = price_from_logn(m0, v0, [(1.0, "call", k_val)])
        p0 = price_from_logn(m0, v0, [(1.0, "put", k_val)])
        m_u, v_u = ou_moments(math.log(S0_ref * (1.0 + hS)), p_ref, T)
        m_d, v_d = ou_moments(math.log(S0_ref * (1.0 - hS)), p_ref, T)
        cu = price_from_logn(m_u, v_u, [(1.0, "call", k_val)])
        cd = price_from_logn(m_d, v_d, [(1.0, "call", k_val)])
        delta = (cu - cd) / (2.0 * S0_ref * hS)
        gamma = (cu - 2.0 * c0 + cd) / ((S0_ref * hS) ** 2)
        p_up = dict(p_ref, s=p_ref["s"] + 0.01)
        p_dn = dict(p_ref, s=max(0.01, p_ref["s"] - 0.01))
        m_vu, v_vu = ou_moments(x0_ref, p_up, T)
        m_vd, v_vd = ou_moments(x0_ref, p_dn, T)
        vega = (price_from_logn(m_vu, v_vu, [(1.0, "call", k_val)]) - price_from_logn(m_vd, v_vd, [(1.0, "call", k_val)])) / 0.02
        t_hi, t_lo = T + 0.05, max(0.02, T - 0.05)
        m_th, v_th = ou_moments(x0_ref, p_ref, t_hi)
        m_tl, v_tl = ou_moments(x0_ref, p_ref, t_lo)
        theta_g = -(price_from_logn(m_th, v_th, [(1.0, "call", k_val)]) - price_from_logn(m_tl, v_tl, [(1.0, "call", k_val)])) / (t_hi - t_lo)
        chain_rows.append(
            {
                "strike_K": k_val,
                "call": round(c0, 4),
                "delta": round(delta, 4),
                "gamma": round(gamma, 4),
                "vega": round(vega, 4),
                "theta_per_s": round(theta_g, 4),
                "put": round(p0, 4),
            }
        )

    return {
        "hemisphere": hemisphere,
        "stim_rate_filter": stim_rate_filter,
        "params": P,
        "amax": round(max(float(arms_df["amp_ma"].max()), 0.5), 2),
        "opt_arms_df": arms_df,
        "opt_selection_df": sel_df,
        "opt_chain_df": pd.DataFrame(chain_rows),
        "majority_arm": majority_arm,
        "majority_votes": f"{majority_count} of {len(valid_votes)}",
        "picks": {
            "portfolio": str(r_port["label"]),
            "strategy": str(r_strat["label"]),
            "ftg": str(r_ftg["label"]),
            "adapt": str(r_adapt["label"]),
            "alpha": alpha_disp if alpha_best_overall else None,
        },
    }
