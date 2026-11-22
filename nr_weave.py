"""NeuroReader Graph Signal Processing Engine (NeuroWeave 5.0 Integration).

Adapts the NeuroWeave 5.0 mathematical formulation (weave_core.py / weave_widget.py)
natively to Medtronic Percept JSON exports:
  - Multi-amplitude BrainSenseTimeDomain sweeps (amplitude-state graph where
    nodes = stimulation amplitude steps 0, 0.5, 1.0, ... mA and signal X_i(f) = log10 Welch PSD).
  - Graceful fallback to LfpMontageTimeDomain directional contact-pair graphs
    when a hemisphere has only constant-amplitude streams (nodes = 15+ bipolar pairs).

Pipeline (exact mirror of neuroweave5/weave_core.py and nr_dag.js):
  1. Welch PSD per node   256-pt periodic Hann, hop 128, density scaling, pooled across
                          segments at each amplitude (skipping 1 s settle when long enough).
  2. log10 PSD            X_i(f) = log10 P_i(f), f in [0.5, 124.0] Hz (126 bins).
  3. Squared distance     d²_ij = Σ_f (X_i(f) - X_j(f))².
  4. RBF Adjacency        W_ij = exp(-d²_ij / (2σ²)), W_ii = 0  (default σ = 2.0).
  5. Graph Laplacians     L = D - W,  L_sym = D^(-1/2) L D^(-1/2).
  6. Eigendecomposition   L = U Λ Uᵀ,  L_sym = U_sym M U_symᵀ  (Fiedler value λ₂, vector u₂).
  7. Spectral GSP metrics GTV(f), LGFE(f), GSE(f), amplitude-step coherence, Fiedler breakpoint,
                          local node variation, and kernel σ-sensitivity sweep.
  8. Oscillatory Phase    Hilbert circular resultant R(a) & mean phase per node, analytic
                          phase-plane loop γ(t), and Takens phase-delay embedding Φ_τ(t).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert

FS = 250.0
NPERSEG = 256
HOP = 128
SETTLE = 250
MIN_WIN = 3
F_LO, F_HI = 0.5, 124.0
BETA_BAND = (13.0, 30.0)
SIGMA_DEFAULT = 2.0
SIGMA_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]

HANN = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(NPERSEG) / NPERSEG)
SCALE = 1.0 / (FS * float((HANN**2).sum()))
FREQS_ALL = np.arange(NPERSEG // 2 + 1) * FS / NPERSEG
FMASK = (FREQS_ALL >= F_LO) & (FREQS_ALL <= F_HI)
FREQS = FREQS_ALL[FMASK]

MEASURES = [
    "fiedler_value",
    "gtv_beta",
    "gtv_ftg",
    "gtv_total",
    "lgfe_beta",
    "lgfe_ftg",
    "lgfe_all",
    "gse_beta",
    "gse_ftg",
    "gse_all",
    "coherence",
    "lambda_max",
    "spectral_gap_ratio",
    "mean_w",
    "fiedler_breakpoint",
    "gtv_peak_freq",
    "gtv_beta_peak_freq",
    "gtv_beta_peak_power",
    "gtv_ftg_peak_freq",
    "gtv_ftg_peak_power",
]

CORE_MEASURES = [
    "fiedler_value",
    "gtv_beta",
    "gtv_ftg",
    "lgfe_beta",
    "lgfe_ftg",
    "gse_all",
    "coherence",
]

DEFAULT_GSP_PARAMS: dict[str, Any] = {
    "hemisphere": "Left",
    "stim_rate_filter": "ALL",
    "sigma": SIGMA_DEFAULT,
    "series_id": "",
    "phase_band": "ftg",
    "node_size_metric": "local_ftg",
}


def _rlist(arr: Any, nd: int = 4) -> list[Any]:
    a = np.asarray(arr)
    if a.ndim == 0:
        v = float(a)
        return round(v, nd) if np.isfinite(v) else None
    if a.ndim == 1:
        return [round(float(v), nd) if np.isfinite(v) else None for v in a]
    return [_rlist(row, nd) for row in a]


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


def _settle_offset(seg_n: int) -> int:
    return SETTLE if seg_n - SETTLE >= NPERSEG else 0


def _window_periodograms(xs: np.ndarray) -> np.ndarray:
    if len(xs) < NPERSEG:
        return np.zeros((0, NPERSEG // 2 + 1), dtype=float)
    fr = np.lib.stride_tricks.sliding_window_view(xs, NPERSEG)[0::HOP]
    fr = fr - fr.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(fr * HANN, axis=1)) ** 2 * SCALE
    P[:, 1:-1] *= 2.0
    return P


def _bandpass_hilbert(x: np.ndarray, fs: float, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    lo_n = max(0.005, min(0.95, float(lo) / nyq))
    hi_n = max(lo_n + 0.01, min(0.98, float(hi) / nyq))
    try:
        b, a = butter(3, [lo_n, hi_n], btype="band")
        xf = filtfilt(b, a, x.astype(float))
    except Exception:
        xf = x.astype(float) - float(np.mean(x))
    z = hilbert(xf)
    return xf, z


def _sqdist(X: np.ndarray) -> np.ndarray:
    return ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)


def _build_W(d2: np.ndarray, sigma: float) -> np.ndarray:
    sig = max(0.1, float(sigma))
    W = np.exp(-d2 / (2.0 * sig * sig))
    np.fill_diagonal(W, 0.0)
    return W


def _build_L(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = W.sum(1)
    L = np.diag(d) - W
    dis = np.where(d > 0.0, 1.0 / np.sqrt(d + 1e-12), 0.0)
    Lsym = dis[:, None] * L * dis[None, :]
    return L, Lsym


def _gft_power(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    return (U.T @ Xc) ** 2 + 1e-12


def _band_mask(lo_hi: tuple[float, float], freqs: np.ndarray = FREQS) -> np.ndarray:
    m = (freqs >= lo_hi[0]) & (freqs <= lo_hi[1])
    if not np.any(m):
        idx = int(np.argmin(np.abs(freqs - 0.5 * (lo_hi[0] + lo_hi[1]))))
        m[idx] = True
    return m


def analyse_graph(
    X: np.ndarray,
    d2: np.ndarray,
    amps: list[float] | np.ndarray,
    sigma: float = SIGMA_DEFAULT,
    ftg_band: tuple[float, float] = (60.0, 65.0),
    freqs: np.ndarray = FREQS,
) -> dict[str, Any]:
    """Exact NeuroWeave 5.0 GSP analysis given node log10-PSD matrix X (K x F) and d2 (K x K)."""
    K = int(X.shape[0])
    amps_arr = np.asarray(amps, dtype=float)
    out: dict[str, Any] = {"n_nodes": K, "sigma": float(sigma), "ftg_band": [round(ftg_band[0], 2), round(ftg_band[1], 2)]}
    if K < 2:
        out["measures"] = {m: None for m in MEASURES}
        out["measures"]["n_nodes"] = K
        return out

    W = _build_W(d2, sigma)
    L, Ls = _build_L(W)
    lam, U = np.linalg.eigh(L)
    mu, Us = np.linalg.eigh(Ls)
    lam = np.maximum(lam, 0.0)
    mu = np.maximum(mu, 0.0)

    D = X[:, None, :] - X[None, :, :]
    gtv = (W[:, :, None] * (D**2)).sum((0, 1))

    p_all = _gft_power(X, Us)
    half = max(1, K // 2)
    lg_f = p_all[:half].sum(0) / p_all.sum(0)
    p_norm = p_all / p_all.sum(0, keepdims=True)
    ge_f = -(p_norm * np.log(p_norm)).sum(0) / math.log(K + 1e-12)

    Z = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    coh_f = (Z[:-1] * Z[1:]).mean(0)
    coherence = float(np.mean((Z[:-1] * Z[1:]).sum(1)))

    fv = U[:, 1].copy()
    if abs(fv[0]) > 1e-12:
        fv *= 1.0 if fv[0] >= 0.0 else -1.0
    bp = next((float(amps_arr[k]) for k in range(1, K) if fv[k] < 0.0), float("nan"))

    mb = _band_mask(BETA_BAND, freqs)
    mf = _band_mask(ftg_band, freqs)
    ib = int(np.argmax(np.where(mb, gtv, -np.inf)))
    jf = int(np.argmax(np.where(mf, gtv, -np.inf)))
    ipk = int(np.argmax(gtv))

    local_beta = (W * ((X[:, None, mb] - X[None, :, mb]) ** 2).mean(-1)).sum(1)
    local_ftg = (W * ((X[:, None, mf] - X[None, :, mf]) ** 2).mean(-1)).sum(1)
    triu_i, triu_j = np.triu_indices(K, 1)

    measures = {
        "n_nodes": K,
        "fiedler_value": float(lam[1]),
        "gtv_beta": float(gtv[mb].mean()),
        "gtv_ftg": float(gtv[mf].mean()),
        "gtv_total": float(gtv.mean()),
        "lgfe_beta": float(lg_f[mb].mean()),
        "lgfe_ftg": float(lg_f[mf].mean()),
        "lgfe_all": float(lg_f.mean()),
        "gse_beta": float(ge_f[mb].mean()),
        "gse_ftg": float(ge_f[mf].mean()),
        "gse_all": float(ge_f.mean()),
        "coherence": coherence,
        "lambda_max": float(lam[-1]),
        "spectral_gap_ratio": float(lam[1] / lam[-1]) if lam[-1] > 1e-12 else float("nan"),
        "mean_w": float(W[triu_i, triu_j].mean()) if len(triu_i) else 0.0,
        "fiedler_breakpoint": bp,
        "gtv_peak_freq": float(freqs[ipk]),
        "gtv_beta_peak_freq": float(freqs[ib]),
        "gtv_beta_peak_power": float(gtv[ib]),
        "gtv_ftg_peak_freq": float(freqs[jf]),
        "gtv_ftg_peak_power": float(gtv[jf]),
    }

    out.update(
        W=W,
        L=L,
        Lsym=Ls,
        lam=lam,
        mu=mu,
        U=U,
        Usym=Us,
        fiedler_vec=fv,
        gtv_f=gtv,
        lgfe_f=lg_f,
        gse_f=ge_f,
        coh_f=coh_f,
        E_beta=p_all[:, mb].mean(1),
        E_ftg=p_all[:, mf].mean(1),
        E_all=p_all.mean(1),
        local_beta=local_beta,
        local_ftg=local_ftg,
        mean_log_beta=X[:, mb].mean(1),
        mean_log_ftg=X[:, mf].mean(1),
        measures={
            k: (round(float(v), 5) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else (v if isinstance(v, int) else None))
            for k, v in measures.items()
        },
    )
    return out


def _compute_phase_bundle(
    x_full: np.ndarray,
    segs: list[dict[str, Any]],
    amps: list[float],
    ftg_band: tuple[float, float],
    fs: float = FS,
) -> dict[str, Any]:
    """Compute Hilbert phase clock R(a), analytic phase-plane loop, and Takens delay embedding."""
    n_tot = len(x_full)
    phase: dict[str, Any] = {}
    if n_tot < 64 or not segs:
        return phase

    best_seg = max(segs, key=lambda s: (float(s["amp"]) > 0.0, float(s["amp"]), int(s["n"])))
    s0 = best_seg["start"] + _settle_offset(best_seg["n"])
    s1 = min(best_seg["start"] + best_seg["n"], s0 + int(1.2 * fs))
    if s1 - s0 < 64:
        s0, s1 = 0, min(n_tot, 300)

    for bname, band, tau_samp in (("beta", BETA_BAND, 3), ("ftg", ftg_band, 1)):
        xf, z = _bandpass_hilbert(x_full, fs, band[0], band[1])
        env = np.abs(z)
        ang = np.angle(z)

        clock_rows = []
        for a_val in amps:
            mask = np.zeros(n_tot, dtype=bool)
            for s in segs:
                if abs(float(s["amp"]) - float(a_val)) < 1e-4:
                    a0 = s["start"] + _settle_offset(s["n"])
                    a1 = s["start"] + s["n"]
                    mask[a0:a1] = True
            if mask.sum() >= 32:
                za = np.exp(1j * ang[mask])
                R = float(np.abs(np.mean(za)))
                mu_ang = float(np.angle(np.mean(za)))
                mu_env = float(np.mean(env[mask]))
            else:
                R, mu_ang, mu_env = 0.0, 0.0, 0.0
            clock_rows.append(
                {
                    "label": f"{a_val:g}",
                    "amp": round(float(a_val), 3),
                    "R": round(R, 4),
                    "angle": round(mu_ang, 4),
                    "env": round(mu_env, 3),
                }
            )

        re_seg = np.real(z[s0:s1])
        im_seg = np.imag(z[s0:s1])
        scale_p = float(np.percentile(np.hypot(re_seg, im_seg), 95)) or 1.0
        idx_sub = np.linspace(0, len(re_seg) - 1, min(180, len(re_seg)), dtype=int)
        loop_x = [round(float(re_seg[i] / scale_p), 3) for i in idx_sub]
        loop_y = [round(float(im_seg[i] / scale_p), 3) for i in idx_sub]

        x_d = xf[s0:s1]
        scale_d = float(np.percentile(np.abs(x_d), 95)) or 1.0
        n_d = max(1, len(x_d) - tau_samp)
        idx_d = np.linspace(0, n_d - 1, min(180, n_d), dtype=int)
        delay_x = [round(float(x_d[i] / scale_d), 3) for i in idx_d]
        delay_y = [round(float(x_d[i + tau_samp] / scale_d), 3) for i in idx_d]

        phase[bname] = {
            "band": [round(float(band[0]), 2), round(float(band[1]), 2)],
            "seg_label": f"{float(best_seg['amp']):g} mA",
            "tau_ms": round(1000.0 * tau_samp / fs, 1),
            "clock": clock_rows,
            "loop_x": loop_x,
            "loop_y": loop_y,
            "delay_x": delay_x,
            "delay_y": delay_y,
        }
    return phase


def _compute_montage_phase_bundle(
    pair_signals: list[tuple[str, np.ndarray]],
    ftg_band: tuple[float, float],
    fs: float = FS,
) -> dict[str, Any]:
    """Phase clock / loop / delay for a montage-pair survey graph (when no amplitude sweep exists)."""
    phase: dict[str, Any] = {}
    if not pair_signals:
        return phase
    rep_label, rep_x = max(pair_signals, key=lambda kv: len(kv[1]))
    s0 = 0
    s1 = min(len(rep_x), int(1.2 * fs))
    for bname, band, tau_samp in (("beta", BETA_BAND, 3), ("ftg", ftg_band, 1)):
        clock_rows = []
        for idx, (lbl, x) in enumerate(pair_signals):
            if len(x) < 64:
                clock_rows.append({"label": lbl, "amp": float(idx), "R": 0.0, "angle": 0.0, "env": 0.0})
                continue
            _, z = _bandpass_hilbert(x, fs, band[0], band[1])
            ang = np.angle(z)
            za = np.exp(1j * ang)
            clock_rows.append(
                {
                    "label": lbl,
                    "amp": float(idx),
                    "R": round(float(np.abs(np.mean(za))), 4),
                    "angle": round(float(np.angle(np.mean(za))), 4),
                    "env": round(float(np.mean(np.abs(z))), 3),
                }
            )
        xf, z = _bandpass_hilbert(rep_x, fs, band[0], band[1])
        re_seg = np.real(z[s0:s1])
        im_seg = np.imag(z[s0:s1])
        scale_p = float(np.percentile(np.hypot(re_seg, im_seg), 95)) or 1.0
        idx_sub = np.linspace(0, len(re_seg) - 1, min(180, len(re_seg)), dtype=int)
        x_d = xf[s0:s1]
        scale_d = float(np.percentile(np.abs(x_d), 95)) or 1.0
        n_d = max(1, len(x_d) - tau_samp)
        idx_d = np.linspace(0, n_d - 1, min(180, n_d), dtype=int)
        phase[bname] = {
            "band": [round(float(band[0]), 2), round(float(band[1]), 2)],
            "seg_label": f"Pair {rep_label}",
            "tau_ms": round(1000.0 * tau_samp / fs, 1),
            "clock": clock_rows,
            "loop_x": [round(float(re_seg[i] / scale_p), 3) for i in idx_sub],
            "loop_y": [round(float(im_seg[i] / scale_p), 3) for i in idx_sub],
            "delay_x": [round(float(x_d[i] / scale_d), 3) for i in idx_d],
            "delay_y": [round(float(x_d[i + tau_samp] / scale_d), 3) for i in idx_d],
        }
    return phase


def extract_weave_bundle(full_store: dict[str, Any], ftg_hw: float = 2.5) -> dict[str, Any]:
    """Extract compact per-hemisphere NeuroWeave 5.0 GSP series bundle from a parsed Percept JSON."""
    payload = full_store["payload"]
    raw_signals = full_store["raw_signals"]
    recordings = payload.get("recordings") or []
    default_stim = float(payload.get("config", {}).get("default_stim_rate_hz") or 145.0)

    hemis_out: dict[str, Any] = {}

    for hemi in ("Left", "Right"):
        stream_recs = [r for r in recordings if r["source"] == "BrainSenseTimeDomain" and r["hemisphere"] == hemi]
        montage_recs = [r for r in recordings if r["source"] == "LfpMontageTimeDomain" and r["hemisphere"] == hemi]

        distinct_stream_amps = set()
        for r in stream_recs:
            for st in r.get("amp_steps") or []:
                distinct_stream_amps.add(round(float(st["amp_ma"]), 1))
        has_amp_sweeps = len(distinct_stream_amps) >= 2

        series_out: list[dict[str, Any]] = []
        stim_rates_hemi: set[float] = set()

        if has_amp_sweeps:
            mode_kind = "amplitude_sweep"
            grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
            for r in stream_recs:
                raw = raw_signals.get(r["id"])
                if not raw or raw.get("time_domain_uv") is None or len(raw["time_domain_uv"]) < NPERSEG:
                    continue
                fs = float(raw.get("fs_hz") or FS)
                if abs(fs - FS) > 1.0:
                    continue
                shz = round(float(r.get("stim_rate_hz") or default_stim), 1)
                grouped.setdefault((str(r["pair"]), shz), []).append(r)

            for (pair, shz), rec_list in sorted(grouped.items()):
                all_segs: list[dict[str, Any]] = []
                x_parts: list[np.ndarray] = []
                offset = 0
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
                        all_segs.append({"start": offset + s["start"], "n": s["n"], "amp": s["amp"]})
                    x_parts.append(x)
                    offset += len(x)
                    for c in (r.get("contacts") or []) + (r.get("stim_contacts") or []):
                        if c and c != "Case":
                            contacts_union.add(c)

                if not x_parts:
                    continue
                x_full = np.concatenate(x_parts)

                # Pooled Welch PSD per amplitude node (exact weave_core.node_psds logic)
                acc: dict[float, tuple[np.ndarray, int]] = {}
                for s in all_segs:
                    a = s["start"] + _settle_offset(s["n"])
                    b = s["start"] + s["n"]
                    if b - a < NPERSEG:
                        continue
                    P_win = _window_periodograms(x_full[a:b])
                    if len(P_win) == 0:
                        continue
                    prev = acc.get(s["amp"])
                    if prev is None:
                        acc[s["amp"]] = (P_win.sum(0), len(P_win))
                    else:
                        acc[s["amp"]] = (prev[0] + P_win.sum(0), prev[1] + len(P_win))

                amps = sorted(a for a, (_, nw) in acc.items() if nw >= MIN_WIN)
                if len(amps) < 2:
                    amps = sorted(a for a, (_, nw) in acc.items() if nw >= 1)
                if len(amps) < 2:
                    continue

                P_mat = np.vstack([acc[a][0] / acc[a][1] for a in amps])[:, FMASK]
                X = np.log10(np.clip(P_mat, 1e-12, None))
                d2 = _sqdist(X)
                nwin = [int(acc[a][1]) for a in amps]
                node_labels = [f"{a:g}" for a in amps]

                ftg_center = shz / 2.0
                ftg_b = (max(35.0, ftg_center - float(ftg_hw)), min(122.0, ftg_center + float(ftg_hw)))
                phase = _compute_phase_bundle(x_full, all_segs, amps, ftg_b, FS)
                g_def = analyse_graph(X, d2, amps, sigma=SIGMA_DEFAULT, ftg_band=ftg_b)

                stim_rates_hemi.add(shz)
                series_out.append(
                    {
                        "id": f"{hemi}|{pair}|{int(round(shz))}Hz",
                        "label": f"{pair} · {int(round(shz))} Hz ({len(amps)} amps)",
                        "kind": "amplitude_sweep",
                        "hemisphere": hemi,
                        "pair": pair,
                        "contacts": sorted(contacts_union),
                        "stim_hz": shz,
                        "ftg_center_hz": round(ftg_center, 2),
                        "rec_id": rec_id0,
                        "duration_s": round(len(x_full) / FS, 1),
                        "amps": [round(float(a), 3) for a in amps],
                        "node_labels": node_labels,
                        "nwin": nwin,
                        "X": _rlist(X, 3),
                        "d2": _rlist(d2, 3),
                        "phase": phase,
                        "default_measures": g_def["measures"],
                    }
                )
        else:
            mode_kind = "montage_survey_graph"
            # Build a directional contact-pair graph across all LfpMontageTimeDomain pairs in this hemisphere
            pair_rows = []
            pair_sigs: list[tuple[str, np.ndarray]] = []
            contacts_union_all: set[str] = set()
            shz = default_stim
            for r in montage_recs:
                raw = raw_signals.get(r["id"])
                if not raw or raw.get("time_domain_uv") is None or len(raw["time_domain_uv"]) < NPERSEG:
                    continue
                x = np.asarray(raw["time_domain_uv"], dtype=float)
                x = np.where(np.isfinite(x), x, 0.0)
                P_win = _window_periodograms(x)
                if len(P_win) == 0:
                    continue
                shz = round(float(r.get("stim_rate_hz") or default_stim), 1)
                pair_rows.append((str(r["pair"]), P_win.mean(0)[FMASK], len(P_win), r["id"]))
                pair_sigs.append((str(r["pair"]), x))
                for c in r.get("contacts") or []:
                    if c and c != "Case":
                        contacts_union_all.add(c)

            if len(pair_rows) >= 2:
                P_mat = np.vstack([pr[1] for pr in pair_rows])
                X = np.log10(np.clip(P_mat, 1e-12, None))
                d2 = _sqdist(X)
                amps = [float(i) for i in range(len(pair_rows))]
                node_labels = [pr[0] for pr in pair_rows]
                nwin = [int(pr[2]) for pr in pair_rows]
                ftg_center = shz / 2.0
                ftg_b = (max(35.0, ftg_center - float(ftg_hw)), min(122.0, ftg_center + float(ftg_hw)))
                phase = _compute_montage_phase_bundle(pair_sigs, ftg_b, FS)
                g_def = analyse_graph(X, d2, amps, sigma=SIGMA_DEFAULT, ftg_band=ftg_b)
                stim_rates_hemi.add(shz)
                series_out.append(
                    {
                        "id": f"{hemi}|Survey|{int(round(shz))}Hz",
                        "label": f"{hemi} Montage Graph ({len(pair_rows)} pairs)",
                        "kind": "montage_survey",
                        "hemisphere": hemi,
                        "pair": "Survey",
                        "contacts": sorted(contacts_union_all),
                        "stim_hz": shz,
                        "ftg_center_hz": round(ftg_center, 2),
                        "rec_id": pair_rows[0][3],
                        "duration_s": round(sum(len(ps[1]) for ps in pair_sigs) / FS, 1),
                        "amps": amps,
                        "node_labels": node_labels,
                        "nwin": nwin,
                        "X": _rlist(X, 3),
                        "d2": _rlist(d2, 3),
                        "phase": phase,
                        "default_measures": g_def["measures"],
                    }
                )

        hemis_out[hemi] = {
            "hemisphere": hemi,
            "mode": mode_kind,
            "stim_rates": sorted(stim_rates_hemi),
            "series": series_out,
        }

    return {
        "spec": {
            "fs_hz": FS,
            "nperseg": NPERSEG,
            "hop": HOP,
            "settle_samples": SETTLE,
            "beta_band": [BETA_BAND[0], BETA_BAND[1]],
            "ftg_hw_hz": float(ftg_hw),
            "sigma_default": SIGMA_DEFAULT,
            "sigma_grid": SIGMA_GRID,
        },
        "freqs": _rlist(FREQS, 2),
        "default_params": DEFAULT_GSP_PARAMS,
        "hemispheres": hemis_out,
    }


def solve_weave_exact(
    weave_bundle: dict[str, Any],
    hemisphere: str = "Left",
    stim_rate_filter: str | float | None = "ALL",
    params: dict[str, Any] | None = None,
    selected_contacts: list[str] | None = None,
    ftg_half_bw: float | None = None,
) -> dict[str, Any]:
    """Compute exact NeuroWeave 5.0 GSP biomarkers, node table, and σ-sensitivity sweep
    for all contact-scoped series in the active hemisphere."""
    P = dict(DEFAULT_GSP_PARAMS)
    if params:
        for k in ("sigma", "series_id", "phase_band", "node_size_metric"):
            if k in params and params[k] is not None:
                P[k] = params[k]
    P["hemisphere"] = hemisphere
    P["stim_rate_filter"] = stim_rate_filter

    sigma = max(0.25, min(12.0, float(P.get("sigma") or SIGMA_DEFAULT)))
    P["sigma"] = round(sigma, 3)

    freqs = np.asarray(weave_bundle.get("freqs") or FREQS, dtype=float)
    hw = float(ftg_half_bw) if ftg_half_bw is not None else float((weave_bundle.get("spec") or {}).get("ftg_hw_hz") or 2.5)
    hw = max(0.5, min(15.0, hw))

    h_data = (weave_bundle.get("hemispheres") or {}).get(hemisphere) or {}
    all_series = h_data.get("series") or []
    sel_set = set(selected_contacts) if selected_contacts is not None else None

    series = []
    for s in all_series:
        if sel_set is not None and s.get("contacts"):
            if not any(f"{s['hemisphere']}:{c}" in sel_set for c in s["contacts"]):
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
            "active_series": None,
            "active_measures": {},
            "gsp_series_df": pd.DataFrame(),
            "gsp_nodes_df": pd.DataFrame(),
            "gsp_sigma_df": pd.DataFrame(),
        }

    series_rows = []
    analyses: dict[str, dict[str, Any]] = {}
    for s in series:
        X = np.asarray(s["X"], dtype=float)
        d2 = np.asarray(s["d2"], dtype=float)
        amps = np.asarray(s["amps"], dtype=float)
        ftg_c = float(s.get("ftg_center_hz") or (float(s["stim_hz"]) / 2.0))
        ftg_b = (max(35.0, ftg_c - hw), min(122.0, ftg_c + hw))
        g = analyse_graph(X, d2, amps, sigma=sigma, ftg_band=ftg_b, freqs=freqs)
        analyses[s["id"]] = g
        m = g["measures"]
        series_rows.append(
            {
                "series_id": s["id"],
                "label": s["label"],
                "pair": s["pair"],
                "stim_hz": float(s["stim_hz"]),
                "n_nodes": int(g["n_nodes"]),
                "fiedler_value": m.get("fiedler_value"),
                "gtv_beta": m.get("gtv_beta"),
                "gtv_ftg": m.get("gtv_ftg"),
                "gtv_total": m.get("gtv_total"),
                "lgfe_beta": m.get("lgfe_beta"),
                "lgfe_ftg": m.get("lgfe_ftg"),
                "gse_all": m.get("gse_all"),
                "coherence": m.get("coherence"),
                "mean_w": m.get("mean_w"),
                "fiedler_breakpoint": m.get("fiedler_breakpoint"),
                "gtv_peak_freq": m.get("gtv_peak_freq"),
            }
        )

    active = next((s for s in series if s["id"] == P.get("series_id")), None) or max(series, key=lambda s: len(s["amps"]))
    P["series_id"] = active["id"]
    g_act = analyses[active["id"]]

    node_rows = []
    if g_act["n_nodes"] >= 2:
        W_act = g_act["W"]
        deg = W_act.sum(1)
        for k, a in enumerate(active["amps"]):
            node_rows.append(
                {
                    "node": k,
                    "label": active["node_labels"][k],
                    "amp_ma": float(a),
                    "n_windows": int(active["nwin"][k]),
                    "degree": round(float(deg[k]), 4),
                    "fiedler_vector": round(float(g_act["fiedler_vec"][k]), 4),
                    "fiedler_side": int(g_act["fiedler_vec"][k] >= 0.0),
                    "local_beta": round(float(g_act["local_beta"][k]), 4),
                    "local_ftg": round(float(g_act["local_ftg"][k]), 4),
                    "mean_log10_beta": round(float(g_act["mean_log_beta"][k]), 4),
                    "mean_log10_ftg": round(float(g_act["mean_log_ftg"][k]), 4),
                }
            )

    sigma_rows = []
    X_act = np.asarray(active["X"], dtype=float)
    d2_act = np.asarray(active["d2"], dtype=float)
    amps_act = np.asarray(active["amps"], dtype=float)
    ftg_c_act = float(active.get("ftg_center_hz") or (float(active["stim_hz"]) / 2.0))
    ftg_b_act = (max(35.0, ftg_c_act - hw), min(122.0, ftg_c_act + hw))
    for sg in SIGMA_GRID:
        gs = analyse_graph(X_act, d2_act, amps_act, sigma=sg, ftg_band=ftg_b_act, freqs=freqs)
        ms = gs["measures"]
        sigma_rows.append(
            {
                "sigma": sg,
                "fiedler_value": ms.get("fiedler_value"),
                "gtv_beta": ms.get("gtv_beta"),
                "gtv_ftg": ms.get("gtv_ftg"),
                "lgfe_beta": ms.get("lgfe_beta"),
                "lgfe_ftg": ms.get("lgfe_ftg"),
                "gse_all": ms.get("gse_all"),
                "mean_w": ms.get("mean_w"),
            }
        )

    return {
        "hemisphere": hemisphere,
        "stim_rate_filter": stim_rate_filter,
        "params": P,
        "active_series": active["label"],
        "active_series_id": active["id"],
        "active_measures": g_act["measures"],
        "gsp_series_df": pd.DataFrame(series_rows),
        "gsp_nodes_df": pd.DataFrame(node_rows),
        "gsp_sigma_df": pd.DataFrame(sigma_rows),
    }
