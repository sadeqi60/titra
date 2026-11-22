"""NeuroReader Percept JSON Parser & Signal Analysis Engine.

Parses Medtronic Percept PC/RC JSON session exports (e.g., clinical1.json, clinical2.json)
into a clean, rich, analysis-ready structure for both the NeuroReader DAG AnyWidget and
reactive downstream Python workflows.

Highlights:
- Full SenSight directional (8-contact: 0, 1a/1b/1c, 2a/2b/2c, 3) & 4-ring lead parsing
- Beta (13–30 Hz) and FTG (neighborhood of f_stim / 2, e.g. 62.5 / 72.5 / 82.5 Hz) metrics
- Automatic alignment of BrainSenseTimeDomain (250 Hz) with BrainSenseLfp (mA staircase)
- Full 30-channel LFPMontage + LfpMontageTimeDomain survey spectra, Welch PSD, STFT spectrograms,
  and Hilbert analytic phase portraits for Beta and FTG
- Reactive Python selection hydration (`build_selection_output`) returning structured dicts,
  numpy signal arrays, and pandas DataFrames.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

BETA_BAND = (13.0, 30.0)
DEFAULT_FTG_HALF_BW = 4.0

CONTACTS_ORDER = ["0", "1a", "1b", "1c", "2a", "2b", "2c", "3"]
RING_TO_CONTACTS = {
    "0": ["0"],
    "1": ["1a", "1b", "1c"],
    "2": ["2a", "2b", "2c"],
    "3": ["3"],
    "1a": ["1a"],
    "1b": ["1b"],
    "1c": ["1c"],
    "2a": ["2a"],
    "2b": ["2b"],
    "2c": ["2c"],
}

_TOKEN_MAP = {
    "ZERO": "0",
    "ONE": "1",
    "TWO": "2",
    "THREE": "3",
    "ONE_A": "1a",
    "ONE_B": "1b",
    "ONE_C": "1c",
    "TWO_A": "2a",
    "TWO_B": "2b",
    "TWO_C": "2c",
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "1A": "1a",
    "1B": "1b",
    "1C": "1c",
    "2A": "2a",
    "2B": "2b",
    "2C": "2c",
}


def _strip_enum(val: Any) -> str:
    if val is None:
        return ""
    s = str(val)
    return s.split(".")[-1] if "." in s else s


def _normalize_electrode(name: Any) -> str:
    s = _strip_enum(name)
    if not s:
        return ""
    low = s.lower()
    if "case" in low:
        return "Case"
    m = re.search(r"(?:sensight_|electrode_)?([0-3][abc]?)$", low)
    if m:
        return m.group(1)
    # handle ZERO_RING etc.
    for k, v in _TOKEN_MAP.items():
        if low == k.lower() or low.endswith("_" + k.lower()):
            return v
    return s


def _parse_pair_string(raw: str) -> tuple[str, str, str, str]:
    """Parse Medtronic channel/sensing strings into (hemi, e1, e2, pair_key).

    Examples:
      'ZERO_TWO_LEFT' -> ('Left', '0', '2', '0-2')
      'ONE_A_AND_TWO_A_LEFT_SEGMENT' -> ('Left', '1a', '2a', '1a-2a')
      'TWO_A_AND_TWO_B' (with external hemi) -> ('', '2a', '2b', '2a-2b')
    """
    s = _strip_enum(raw).upper()
    hemi = ""
    if "LEFT" in s:
        hemi = "Left"
        s = re.sub(r"_?LEFT(_RING|_SEGMENT)?", "", s)
    elif "RIGHT" in s:
        hemi = "Right"
        s = re.sub(r"_?RIGHT(_RING|_SEGMENT)?", "", s)
    s = re.sub(r"_(RING|SEGMENT)$", "", s).strip("_")

    e1, e2 = "", ""
    if "_AND_" in s:
        parts = s.split("_AND_", 1)
        e1 = _TOKEN_MAP.get(parts[0], parts[0].lower())
        e2 = _TOKEN_MAP.get(parts[1], parts[1].lower())
    else:
        # e.g. ZERO_TWO, ONE_THREE, ONE_A_TWO_A
        tokens = list(_TOKEN_MAP.keys())
        # sort longer tokens first (ONE_A before ONE)
        tokens.sort(key=len, reverse=True)
        for t1 in tokens:
            if s.startswith(t1 + "_"):
                rest = s[len(t1) + 1 :]
                if rest in _TOKEN_MAP:
                    e1 = _TOKEN_MAP[t1]
                    e2 = _TOKEN_MAP[rest]
                    break
    if not e1:
        e1 = s.lower()
    pair_key = f"{e1}-{e2}" if e2 else e1
    return hemi, e1, e2, pair_key


def _contacts_for_pair(e1: str, e2: str) -> list[str]:
    out = []
    for e in (e1, e2):
        for c in RING_TO_CONTACTS.get(e, [e] if e else []):
            if c and c not in out:
                out.append(c)
    return out


def _clean_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _round_list(arr: np.ndarray | list[float], ndigits: int = 3) -> list[float]:
    a = np.asarray(arr, dtype=float)
    a = np.where(np.isfinite(a), a, 0.0)
    return [round(float(v), ndigits) for v in a]


def _band_metrics_from_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    beta_band: tuple[float, float],
    ftg_band: tuple[float, float],
) -> dict[str, float]:
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    eps = 1e-12

    b_mask = (freqs >= beta_band[0]) & (freqs <= beta_band[1])
    f_mask = (freqs >= ftg_band[0]) & (freqs <= ftg_band[1])
    tot_mask = (freqs >= 4.0) & (freqs <= 100.0)

    def _stats(mask: np.ndarray) -> tuple[float, float, float]:
        if not np.any(mask):
            return 0.0, 0.0, 0.0
        ff = freqs[mask]
        pp = np.maximum(psd[mask], 0.0)
        mean_p = float(np.mean(pp))
        idx = int(np.argmax(pp))
        peak_f = float(ff[idx])
        peak_p = float(pp[idx])
        return mean_p, peak_f, peak_p

    b_mean, b_peak_f, b_peak_p = _stats(b_mask)
    f_mean, f_peak_f, f_peak_p = _stats(f_mask)
    tot_mean = float(np.mean(np.maximum(psd[tot_mask], 0.0))) if np.any(tot_mask) else 1.0

    b_db = 10.0 * math.log10(max(b_mean, eps))
    f_db = 10.0 * math.log10(max(f_mean, eps))
    ratio = f_mean / max(b_mean, eps)
    ratio_db = f_db - b_db

    return {
        "beta_power": round(b_mean, 5),
        "beta_db": round(b_db, 2),
        "beta_peak_hz": round(b_peak_f, 2),
        "beta_peak_power": round(b_peak_p, 5),
        "beta_rel": round(b_mean / max(tot_mean, eps), 4),
        "ftg_power": round(f_mean, 5),
        "ftg_db": round(f_db, 2),
        "ftg_peak_hz": round(f_peak_f, 2),
        "ftg_peak_power": round(f_peak_p, 5),
        "ftg_rel": round(f_mean / max(tot_mean, eps), 4),
        "ftg_beta_ratio": round(ratio, 4),
        "ftg_beta_db_diff": round(ratio_db, 2),
    }


def _compute_hilbert_phase(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float],
    max_pts: int = 240,
) -> dict[str, Any]:
    """Compute bandpass Hilbert analytic phasor loop & burst metrics."""
    n = len(x)
    nyq = 0.5 * fs
    lo = max(1.0, min(band[0], nyq - 2.0))
    hi = max(lo + 1.0, min(band[1], nyq - 1.0))
    if n < 128 or hi <= lo:
        return {"re": [], "im": [], "rms": 0.0, "burst_pct": 0.0, "roundness": 0.0, "inst_hz": 0.0}

    # Use a clean representative segment (up to 12 seconds centered on highest signal energy or middle)
    seg_len = min(n, int(12 * fs))
    start = max(0, (n - seg_len) // 2)
    seg = x[start : start + seg_len] - np.mean(x[start : start + seg_len])
    try:
        sos = signal.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
        filt = signal.sosfiltfilt(sos, seg)
        z = signal.hilbert(filt)
    except Exception:
        return {"re": [], "im": [], "rms": 0.0, "burst_pct": 0.0, "roundness": 0.0, "inst_hz": 0.0}

    env = np.abs(z)
    rms = float(np.sqrt(np.mean(filt**2))) + 1e-12
    burst_thr = 1.25 * rms
    burst_pct = float(100.0 * np.mean(env > burst_thr))

    # Instantaneous frequency
    phase = np.unwrap(np.angle(z))
    inst_f = np.diff(phase) * fs / (2.0 * np.pi)
    inst_f = inst_f[(inst_f >= lo * 0.5) & (inst_f <= hi * 1.5)]
    med_inst_f = float(np.median(inst_f)) if len(inst_f) else 0.5 * (lo + hi)

    # Covariance roundness of phasor (re, im)
    cov = np.cov(np.vstack([np.real(z), np.imag(z)]))
    eig = np.sort(np.maximum(np.linalg.eigvalsh(cov), 1e-12))
    roundness = float(eig[0] / eig[1])

    # Extract 2.0 s crisp loop for SVG rendering
    loop_n = min(len(z), int(2.0 * fs))
    # pick subwindow around peak burst
    if len(z) > loop_n:
        k_peak = int(np.argmax(env))
        s0 = max(0, min(len(z) - loop_n, k_peak - loop_n // 2))
        z_loop = z[s0 : s0 + loop_n]
    else:
        z_loop = z

    step = max(1, len(z_loop) // max_pts)
    z_sub = z_loop[::step][:max_pts]
    scale = max(float(np.percentile(np.abs(z_sub), 98)), 1e-9)

    return {
        "re": _round_list(np.real(z_sub), 3),
        "im": _round_list(np.imag(z_sub), 3),
        "rms": round(rms, 3),
        "burst_thr": round(burst_thr, 3),
        "scale": round(scale, 3),
        "burst_pct": round(burst_pct, 1),
        "roundness": round(roundness, 3),
        "inst_hz": round(med_inst_f, 2),
    }


def _analyze_time_domain(
    raw: list[float] | np.ndarray,
    fs: float,
    stim_rate_hz: float,
    ftg_half_bw: float = DEFAULT_FTG_HALF_BW,
    amp_timeline: tuple[list[float], list[float]] | None = None,
) -> dict[str, Any]:
    """Compute compact LFP trace, Welch PSD, STFT spectrogram, Beta/FTG envelopes,
    amplitude-step dose-response, and Hilbert phase portraits for one recording."""
    x = np.asarray(raw, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    n = len(x)
    dur_s = n / max(fs, 1.0)
    if n < 32:
        return {}

    x_dc = x - np.mean(x)
    rms_uv = float(np.sqrt(np.mean(x_dc**2)))

    ftg_center = stim_rate_hz / 2.0 if stim_rate_hz > 0 else 72.5
    ftg_band = (max(35.0, ftg_center - ftg_half_bw), min(fs / 2.0 - 2.0, ftg_center + ftg_half_bw))

    # 1) Downsampled min-max LFP trace for crisp SVG waveform (up to 360 time bins -> 720 pts or 320 bins)
    n_bins = min(320, n)
    if n > n_bins:
        edges = np.linspace(0, n, n_bins + 1, dtype=int)
        t_trace = []
        y_min = []
        y_max = []
        y_mean = []
        for i in range(n_bins):
            seg = x_dc[edges[i] : max(edges[i] + 1, edges[i + 1])]
            t_trace.append(0.5 * (edges[i] + edges[i + 1]) / fs)
            y_min.append(float(np.min(seg)))
            y_max.append(float(np.max(seg)))
            y_mean.append(float(np.mean(seg)))
    else:
        t_trace = (np.arange(n) / fs).tolist()
        y_min = x_dc.tolist()
        y_max = x_dc.tolist()
        y_mean = x_dc.tolist()

    # 2) Welch PSD (0..120 Hz at ~0.5 Hz resolution)
    nperseg = min(n, int(fs * 2))  # 500 samples -> 0.5 Hz bins
    if nperseg < 32:
        nperseg = n
    freqs, psd = signal.welch(x_dc, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    f_mask = freqs <= min(120.0, fs / 2.0)
    freqs_c = freqs[f_mask]
    psd_c = psd[f_mask]
    psd_db = 10.0 * np.log10(np.maximum(psd_c, 1e-12))

    band_stats = _band_metrics_from_psd(freqs_c, psd_c, BETA_BAND, ftg_band)

    # 3) STFT Spectrogram (binned to <= 56 time columns x 46 freq rows in 0..115 Hz)
    stft_nperseg = min(n, max(64, int(fs * 1.0)))
    stft_noverlap = int(stft_nperseg * 0.75)
    sf, st, sxx = signal.spectrogram(
        x_dc, fs=fs, nperseg=stft_nperseg, noverlap=stft_noverlap, mode="psd"
    )
    sf_mask = sf <= min(115.0, fs / 2.0)
    sf = sf[sf_mask]
    sxx = sxx[sf_mask, :]
    sxx_db = 10.0 * np.log10(np.maximum(sxx, 1e-12))

    # Beta & FTG envelopes over time from STFT
    b_rows = (sf >= BETA_BAND[0]) & (sf <= BETA_BAND[1])
    f_rows = (sf >= ftg_band[0]) & (sf <= ftg_band[1])
    beta_env = np.mean(sxx[b_rows, :], axis=0) if np.any(b_rows) else np.zeros(len(st))
    ftg_env = np.mean(sxx[f_rows, :], axis=0) if np.any(f_rows) else np.zeros(len(st))
    beta_env_db = 10.0 * np.log10(np.maximum(beta_env, 1e-12))
    ftg_env_db = 10.0 * np.log10(np.maximum(ftg_env, 1e-12))

    # Downsample spectrogram grid cleanly for fast pure-SVG vector rendering
    target_t = min(56, len(st))
    target_f = min(44, len(sf))
    t_idx = np.linspace(0, len(st), target_t + 1, dtype=int)
    f_idx = np.linspace(0, len(sf), target_f + 1, dtype=int)
    spec_t = []
    spec_f = []
    spec_grid = np.zeros((target_f, target_t), dtype=float)
    for fi in range(target_f):
        f0, f1 = f_idx[fi], max(f_idx[fi] + 1, f_idx[fi + 1])
        spec_f.append(float(np.mean(sf[f0:f1])))
        for ti in range(target_t):
            t0, t1 = t_idx[ti], max(t_idx[ti] + 1, t_idx[ti + 1])
            spec_grid[fi, ti] = float(np.mean(sxx_db[f0:f1, t0:t1]))
    for ti in range(target_t):
        t0, t1 = t_idx[ti], max(t_idx[ti] + 1, t_idx[ti + 1])
        spec_t.append(float(np.mean(st[t0:t1])))

    env_step = max(1, len(st) // 90)
    env_t = st[::env_step]
    env_beta_db = beta_env_db[::env_step]
    env_ftg_db = ftg_env_db[::env_step]

    # 4) Amplitude-step dose-response (if amp_timeline provided and has > 1 distinct amp)
    amp_steps = []
    amp_curve_t = []
    amp_curve_ma = []
    if amp_timeline and len(amp_timeline[0]) > 0:
        at_arr = np.asarray(amp_timeline[0], dtype=float)
        am_arr = np.asarray(amp_timeline[1], dtype=float)
        # downsample amp curve cleanly
        a_step = max(1, len(at_arr) // 240)
        amp_curve_t = _round_list(at_arr[::a_step], 2)
        amp_curve_ma = _round_list(am_arr[::a_step], 2)

        # Group STFT columns by rounded amplitude
        st_amps = np.interp(st, at_arr, am_arr)
        rounded_amps = np.round(st_amps, 1)
        unique_amps = sorted(set(float(a) for a in rounded_amps))
        for amp_val in unique_amps:
            cols = np.where(np.abs(rounded_amps - amp_val) < 0.051)[0]
            if len(cols) < 2:
                continue
            step_psd = np.mean(sxx[:, cols], axis=1)
            step_stats = _band_metrics_from_psd(sf, step_psd, BETA_BAND, ftg_band)
            # downsample step PSD to match 0..115 Hz (~60 bins)
            p_idx = np.linspace(0, len(sf) - 1, min(60, len(sf)), dtype=int)
            amp_steps.append(
                {
                    "amp_ma": round(amp_val, 2),
                    "n_win": int(len(cols)),
                    "dur_s": round(float(len(cols) * (st[1] - st[0]) if len(st) > 1 else 1.0), 1),
                    "beta_db": step_stats["beta_db"],
                    "ftg_db": step_stats["ftg_db"],
                    "beta_power": step_stats["beta_power"],
                    "ftg_power": step_stats["ftg_power"],
                    "ftg_beta_ratio": step_stats["ftg_beta_ratio"],
                    "ftg_beta_db_diff": step_stats["ftg_beta_db_diff"],
                    "psd_freqs": _round_list(sf[p_idx], 2),
                    "psd_db": _round_list(10.0 * np.log10(np.maximum(step_psd[p_idx], 1e-12)), 2),
                }
            )

    # 5) Hilbert Phase Portraits for Beta & FTG
    phase_beta = _compute_hilbert_phase(x_dc, fs, BETA_BAND, max_pts=200)
    phase_ftg = _compute_hilbert_phase(x_dc, fs, ftg_band, max_pts=200)

    vmin = float(np.percentile(spec_grid, 5))
    vmax = float(np.percentile(spec_grid, 98))
    if vmax <= vmin + 1e-6:
        vmax = vmin + 1.0

    return {
        "n_samples": int(n),
        "fs_hz": float(fs),
        "duration_s": round(dur_s, 2),
        "rms_uv": round(rms_uv, 3),
        "stim_rate_hz": round(stim_rate_hz, 2),
        "beta_band": [BETA_BAND[0], BETA_BAND[1]],
        "ftg_center_hz": round(ftg_center, 2),
        "ftg_band": [round(ftg_band[0], 2), round(ftg_band[1], 2)],
        "band_stats": band_stats,
        "trace": {
            "t": _round_list(t_trace, 2),
            "min": _round_list(y_min, 2),
            "max": _round_list(y_max, 2),
            "mean": _round_list(y_mean, 2),
        },
        "amp_curve": {
            "t": amp_curve_t,
            "ma": amp_curve_ma,
        },
        "psd": {
            "freqs": _round_list(freqs_c, 2),
            "db": _round_list(psd_db, 2),
            "power": _round_list(psd_c, 5),
        },
        "spectrogram": {
            "t": _round_list(spec_t, 2),
            "f": _round_list(spec_f, 2),
            "db": [[round(float(v), 1) for v in row] for row in spec_grid],
            "vmin": round(vmin, 1),
            "vmax": round(vmax, 1),
        },
        "envelope": {
            "t": _round_list(env_t, 2),
            "beta_db": _round_list(env_beta_db, 2),
            "ftg_db": _round_list(env_ftg_db, 2),
        },
        "amp_steps": amp_steps,
        "phase": {
            "beta": phase_beta,
            "ftg": phase_ftg,
        },
    }


def _parse_iso_seconds(ts: str | None) -> float:
    if not ts:
        return 0.0
    try:
        from datetime import datetime

        t = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(t).timestamp()
    except Exception:
        return 0.0


def parse_percept_json(
    json_path: str | pathlib.Path,
    ftg_half_bw: float = DEFAULT_FTG_HALF_BW,
    raw_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse a Medtronic Percept JSON file into (widget_payload, full_store)."""
    path = pathlib.Path(json_path) if json_path else pathlib.Path("uploaded.json")
    if raw_dict is None:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    else:
        d = raw_dict

    # --- 1. Session, Patient & Device Metadata ---
    pat_f = (d.get("PatientInformation") or {}).get("Final") or (d.get("PatientInformation") or {}).get("Initial") or {}
    dev_f = (d.get("DeviceInformation") or {}).get("Final") or (d.get("DeviceInformation") or {}).get("Initial") or {}
    bat = d.get("BatteryInformation") or {}
    stim = d.get("Stimulation") or {}

    metadata = {
        "file_name": path.name,
        "data_version": d.get("DataVersion", ""),
        "session_date": d.get("SessionDate", ""),
        "session_end_date": d.get("SessionEndDate", ""),
        "timezone": d.get("ProgrammerTimezone", ""),
        "programmer_version": d.get("ProgrammerVersion", ""),
        "diagnosis": _strip_enum(pat_f.get("Diagnosis", "")),
        "patient_gender": _strip_enum(pat_f.get("PatientGender", "")),
        "patient_label": f"{pat_f.get('PatientFirstName', '')[:1]}.{pat_f.get('PatientLastName', '')[:1]}.".strip(".") or "De-ID",
        "device_name": dev_f.get("Neurostimulator", "Percept"),
        "device_model": dev_f.get("NeurostimulatorModel", ""),
        "device_location": _strip_enum(dev_f.get("NeurostimulatorLocation", "")),
        "implant_date": dev_f.get("ImplantDate", ""),
        "battery_pct": bat.get("BatteryPercentage"),
        "battery_status": _strip_enum(bat.get("BatteryStatus", "")),
        "eri_date": bat.get("ERIDate", ""),
        "initial_stim_status": _strip_enum(stim.get("InitialStimStatus", "")),
        "final_stim_status": _strip_enum(stim.get("FinalStimStatus", "")),
        "top_level_keys": list(d.keys()),
    }

    # --- 2. Lead Configuration ---
    leads_raw = (d.get("LeadConfiguration") or {}).get("Final") or (d.get("LeadConfiguration") or {}).get("Initial") or []
    leads: dict[str, dict[str, Any]] = {}
    for ld in leads_raw:
        hemi = _strip_enum(ld.get("Hemisphere", ""))
        if not hemi:
            continue
        model = _strip_enum(ld.get("Model", ""))
        is_directional = "33005" in model or "33015" in model or "SENSIGHT" in model.upper()
        leads[hemi] = {
            "hemisphere": hemi,
            "model": model,
            "directional": is_directional,
            "target": _strip_enum(ld.get("LeadLocation", "")).upper(),
            "electrode_ports": _strip_enum(ld.get("ElectrodeNumber", "")),
            "tip_offset": _strip_enum(ld.get("TipOffset", "")),
            "orientation_deg": _clean_float(ld.get("OrientationInDegrees", 0)),
            "contacts": list(CONTACTS_ORDER if is_directional else ["0", "1", "2", "3"]),
        }

    # --- 3. Therapy Groups (Final & Initial) ---
    groups_out: list[dict[str, Any]] = []
    default_stim_rate = 145.0
    for stage in ("Final", "Initial"):
        stage_groups = (d.get("Groups") or {}).get(stage) or []
        for g in stage_groups:
            gid = _strip_enum(g.get("GroupId", ""))
            active = bool(g.get("ActiveGroup", False))
            ps = g.get("ProgramSettings") or {}
            rate = _clean_float(ps.get("RateInHertz"), 0.0)
            hemi_progs: dict[str, dict[str, Any]] = {}

            for sc in ps.get("SensingChannel") or []:
                h = _strip_enum(sc.get("HemisphereLocation", ""))
                r_hz = _clean_float(sc.get("RateInHertz"), rate)
                if active and stage == "Final" and r_hz > 0:
                    default_stim_rate = r_hz
                ss = sc.get("SensingSetup") or {}
                ch_raw = (ss.get("ChannelSignalResult") or {}).get("Channel") or ""
                _, e1, e2, pkey = _parse_pair_string(ch_raw) if ch_raw else ("", "", "", "")
                elecs = []
                total_ma = 0.0
                for es in sc.get("ElectrodeState") or []:
                    cn = _normalize_electrode(es.get("Electrode"))
                    pol = _strip_enum(es.get("ElectrodeStateResult"))
                    ma = _clean_float(es.get("ElectrodeAmplitudeInMilliAmps"), 0.0)
                    frac = es.get("ElectrodeFractionOf64")
                    if pol == "Negative":
                        total_ma += ma
                    elecs.append({"contact": cn, "polarity": pol, "ma": round(ma, 2), "fraction64": frac})
                hemi_progs[h] = {
                    "hemisphere": h,
                    "rate_hz": round(r_hz, 1),
                    "ftg_center_hz": round(r_hz / 2.0, 2) if r_hz > 0 else 0.0,
                    "pw_us": _clean_float(sc.get("PulseWidthInMicroSecond"), 0.0),
                    "lower_ma": _clean_float(sc.get("LowerLimitInMilliAmps"), 0.0),
                    "upper_ma": _clean_float(sc.get("UpperLimitInMilliAmps"), 0.0),
                    "total_ma": round(total_ma, 2),
                    "sensing_pair": pkey,
                    "tracked_freq_hz": round(_clean_float(ss.get("FrequencyInHertz"), 0.0), 2),
                    "electrodes": elecs,
                }

            for side_key, h in (("LeftHemisphere", "Left"), ("RightHemisphere", "Right")):
                if side_key in ps and h not in hemi_progs:
                    progs = (ps[side_key] or {}).get("Programs") or []
                    if progs:
                        pr = progs[0]
                        r_hz = _clean_float(pr.get("RateInHertz"), rate)
                        elecs = []
                        for es in pr.get("ElectrodeState") or []:
                            cn = _normalize_electrode(es.get("Electrode"))
                            pol = _strip_enum(es.get("ElectrodeStateResult"))
                            ma = _clean_float(es.get("ElectrodeAmplitudeInMilliAmps"), 0.0)
                            elecs.append({"contact": cn, "polarity": pol, "ma": round(ma, 2), "fraction64": es.get("ElectrodeFractionOf64")})
                        hemi_progs[h] = {
                            "hemisphere": h,
                            "rate_hz": round(r_hz, 1),
                            "ftg_center_hz": round(r_hz / 2.0, 2) if r_hz > 0 else 0.0,
                            "pw_us": _clean_float(pr.get("PulseWidthInMicroSecond"), 0.0),
                            "lower_ma": 0.0,
                            "upper_ma": _clean_float(pr.get("AmplitudeInMilliAmps"), 0.0),
                            "total_ma": round(_clean_float(pr.get("AmplitudeInMilliAmps"), 0.0), 2),
                            "sensing_pair": "",
                            "tracked_freq_hz": 0.0,
                            "electrodes": elecs,
                        }

            groups_out.append(
                {
                    "id": f"{stage}_{gid}",
                    "stage": stage,
                    "group_id": gid,
                    "active": active,
                    "rate_hz": round(rate, 1),
                    "ftg_center_hz": round(rate / 2.0, 2) if rate > 0 else 0.0,
                    "hemispheres": hemi_progs,
                }
            )

    # --- 4. Impedances ---
    impedances: dict[str, dict[str, Any]] = {"Left": {"monopolar": {}, "bipolar": []}, "Right": {"monopolar": {}, "bipolar": []}}
    imp_list = d.get("Impedance") or []
    if imp_list:
        latest_imp = imp_list[-1]
        for h_block in latest_imp.get("Hemisphere") or []:
            h = _strip_enum(h_block.get("Hemisphere", ""))
            if h not in impedances:
                impedances[h] = {"monopolar": {}, "bipolar": []}
            si = h_block.get("SessionImpedance") or {}
            for m in si.get("Monopolar") or []:
                e1 = _normalize_electrode(m.get("Electrode1"))
                e2 = _normalize_electrode(m.get("Electrode2"))
                c = e2 if e1 == "Case" else e1
                impedances[h]["monopolar"][c] = int(round(_clean_float(m.get("ResultValue"), 0)))
            for b in si.get("Bipolar") or []:
                e1 = _normalize_electrode(b.get("Electrode1"))
                e2 = _normalize_electrode(b.get("Electrode2"))
                impedances[h]["bipolar"].append(
                    {"e1": e1, "e2": e2, "pair": f"{e1}-{e2}", "ohm": int(round(_clean_float(b.get("ResultValue"), 0)))}
                )

    # --- 5. LFPMontage (Survey PSDs across all 15 bipolar pairs per hemisphere) ---
    montage_pairs: list[dict[str, Any]] = []
    ftg_default_band = (
        max(35.0, default_stim_rate / 2.0 - ftg_half_bw),
        default_stim_rate / 2.0 + ftg_half_bw,
    )
    for item in d.get("LFPMontage") or []:
        h = _strip_enum(item.get("Hemisphere", ""))
        _, e1, e2, pkey = _parse_pair_string(item.get("SensingElectrodes", ""))
        f_arr = np.asarray(item.get("LFPFrequency") or [], dtype=float)
        m_arr = np.asarray(item.get("LFPMagnitude") or [], dtype=float)
        # Magnitude is uV peak -> power proxy is magnitude^2
        p_arr = m_arr**2
        stats = _band_metrics_from_psd(f_arr, p_arr, BETA_BAND, ftg_default_band)
        is_seg = any(ch in pkey for ch in ("a", "b", "c"))
        montage_pairs.append(
            {
                "id": f"{h}:{pkey}",
                "hemisphere": h,
                "e1": e1,
                "e2": e2,
                "pair": pkey,
                "contacts": _contacts_for_pair(e1, e2),
                "kind": "segment" if is_seg else "ring",
                "peak_hz": round(_clean_float(item.get("PeakFrequencyInHertz")), 2),
                "peak_uv": round(_clean_float(item.get("PeakMagnitudeInMicroVolt")), 4),
                "artifact": _strip_enum(item.get("ArtifactStatus", "")),
                "freqs": _round_list(f_arr, 2),
                "mag_uv": _round_list(m_arr, 4),
                "band_stats": stats,
            }
        )

    # --- 6. Recordings: BrainSenseTimeDomain (aligned with BrainSenseLfp) + LfpMontageTimeDomain ---
    recordings: list[dict[str, Any]] = []
    raw_signals_store: dict[str, dict[str, Any]] = {}

    # Index BrainSenseLfp entries by (timestamp, side)
    bsl_entries = d.get("BrainSenseLfp") or []
    bsl_parsed = []
    for idx_b, b in enumerate(bsl_entries):
        ts_sec = _parse_iso_seconds(b.get("FirstPacketDateTime"))
        ch_str = b.get("Channel", "")
        fs_lfp = _clean_float(b.get("SampleRateInHz"), 2.0)
        snap = b.get("TherapySnapshot") or {}
        lfp_rows = b.get("LfpData") or []
        bsl_parsed.append((idx_b, ts_sec, ch_str, fs_lfp, snap, lfp_rows))

    # 6a. On-stim BrainSenseTimeDomain streaming runs
    for idx_td, td in enumerate(d.get("BrainSenseTimeDomain") or []):
        samples = td.get("TimeDomainData") or []
        if len(samples) < 32:
            continue
        fs = _clean_float(td.get("SampleRateInHz"), 250.0)
        ch_raw = td.get("Channel", "")
        hemi, e1, e2, pkey = _parse_pair_string(ch_raw)
        dt_str = td.get("FirstPacketDateTime", "")
        ts_td = _parse_iso_seconds(dt_str)

        # Match best BrainSenseLfp by closest timestamp + channel overlap
        best_bsl = None
        best_diff = 1e9
        for b in bsl_parsed:
            _, ts_b, ch_b, fs_lfp, snap, lfp_rows = b
            if ch_raw in ch_b or (hemi and hemi.upper() in ch_b.upper()):
                diff = abs(ts_b - ts_td)
                if diff < best_diff and diff < 15.0:
                    best_diff = diff
                    best_bsl = b

        stim_rate = default_stim_rate
        pw_us = 0.0
        tracked_freq = 0.0
        stim_contacts: list[str] = []
        amp_timeline = None
        amp_min, amp_max = 0.0, 0.0
        adaptive_status = ""

        if best_bsl is not None:
            _, _, _, fs_lfp, snap, lfp_rows = best_bsl
            side_snap = snap.get(hemi) or snap.get("Left") or snap.get("Right") or {}
            stim_rate = _clean_float(side_snap.get("RateInHertz"), default_stim_rate)
            pw_us = _clean_float(side_snap.get("PulseWidthInMicroSecond"), 0.0)
            tracked_freq = _clean_float(side_snap.get("FrequencyInHertz"), 0.0)
            adaptive_status = _strip_enum(side_snap.get("AdaptiveTherapyStatus", ""))
            for es in side_snap.get("ElectrodeState") or []:
                if _strip_enum(es.get("ElectrodeStateResult")) == "Negative":
                    stim_contacts.append(_normalize_electrode(es.get("Electrode")))
            if lfp_rows:
                t_a, v_a = [], []
                t0_ms = _clean_float(lfp_rows[0].get("TicksInMs"), 0.0)
                for row_i, row in enumerate(lfp_rows):
                    side_obj = row.get(hemi) or row.get("Left") or row.get("Right") or {}
                    if "mA" in side_obj and side_obj["mA"] is not None:
                        tk = _clean_float(row.get("TicksInMs"), t0_ms + row_i * (1000.0 / max(fs_lfp, 1.0)))
                        t_rel = (tk - t0_ms) / 1000.0
                        if t_rel < 0 or t_rel > (len(samples) / fs) + 5.0:
                            t_rel = row_i / max(fs_lfp, 1.0)
                        t_a.append(t_rel)
                        v_a.append(_clean_float(side_obj["mA"], 0.0))
                if t_a:
                    amp_timeline = (t_a, v_a)
                    amp_min = round(float(min(v_a)), 2)
                    amp_max = round(float(max(v_a)), 2)

        analysis = _analyze_time_domain(samples, fs, stim_rate, ftg_half_bw, amp_timeline)
        rec_id = f"stream_{idx_td}_{hemi}_{pkey}_{int(round(stim_rate))}Hz"
        label = f"{hemi} {pkey} · {int(round(stim_rate))} Hz"
        if amp_max > amp_min:
            label += f" ({amp_min:g}–{amp_max:g} mA)"
        elif amp_max > 0:
            label += f" ({amp_max:g} mA)"

        rec_obj = {
            "id": rec_id,
            "source": "BrainSenseTimeDomain",
            "label": label,
            "hemisphere": hemi,
            "e1": e1,
            "e2": e2,
            "pair": pkey,
            "contacts": _contacts_for_pair(e1, e2),
            "stim_contacts": stim_contacts,
            "datetime": dt_str,
            "stim_rate_hz": round(stim_rate, 1),
            "pw_us": round(pw_us, 1),
            "tracked_freq_hz": round(tracked_freq, 2),
            "adaptive_status": adaptive_status,
            "amp_range_ma": [amp_min, amp_max],
            **analysis,
        }
        recordings.append(rec_obj)
        raw_signals_store[rec_id] = {
            "time_domain_uv": np.asarray(samples, dtype=np.float32),
            "fs_hz": fs,
            "amp_t": np.asarray(amp_timeline[0], dtype=np.float32) if amp_timeline else np.array([], dtype=np.float32),
            "amp_ma": np.asarray(amp_timeline[1], dtype=np.float32) if amp_timeline else np.array([], dtype=np.float32),
        }

    # 6b. Off-stim Survey LfpMontageTimeDomain recordings (30 pairs)
    for idx_m, td in enumerate(d.get("LfpMontageTimeDomain") or []):
        samples = td.get("TimeDomainData") or []
        if len(samples) < 32:
            continue
        fs = _clean_float(td.get("SampleRateInHz"), 250.0)
        ch_raw = td.get("Channel", "")
        hemi, e1, e2, pkey = _parse_pair_string(ch_raw)
        dt_str = td.get("FirstPacketDateTime", "")
        analysis = _analyze_time_domain(samples, fs, default_stim_rate, ftg_half_bw, None)
        rec_id = f"montage_{idx_m}_{hemi}_{pkey}"
        rec_obj = {
            "id": rec_id,
            "source": "LfpMontageTimeDomain",
            "label": f"{hemi} {pkey} · Survey Off-Stim",
            "hemisphere": hemi,
            "e1": e1,
            "e2": e2,
            "pair": pkey,
            "contacts": _contacts_for_pair(e1, e2),
            "stim_contacts": [],
            "datetime": dt_str,
            "stim_rate_hz": round(default_stim_rate, 1),
            "pw_us": 0.0,
            "tracked_freq_hz": 0.0,
            "adaptive_status": "OFF_STIM_SURVEY",
            "amp_range_ma": [0.0, 0.0],
            **analysis,
        }
        recordings.append(rec_obj)
        raw_signals_store[rec_id] = {
            "time_domain_uv": np.asarray(samples, dtype=np.float32),
            "fs_hz": fs,
            "amp_t": np.array([], dtype=np.float32),
            "amp_ma": np.array([], dtype=np.float32),
        }

    # --- 7. Chronic LFP Trend Logs & EventSummary ---
    trends: dict[str, dict[str, Any]] = {}
    dd = d.get("DiagnosticData") or {}
    for hemi_key, days_dict in (dd.get("LFPTrendLogs") or {}).items():
        h = _strip_enum(hemi_key)
        pts = []
        for day_k in sorted(days_dict.keys()):
            for pt in days_dict[day_k] or []:
                pts.append(
                    (
                        _parse_iso_seconds(pt.get("DateTime")),
                        pt.get("DateTime", ""),
                        _clean_float(pt.get("LFP"), 0.0),
                        _clean_float(pt.get("AmplitudeInMilliAmps"), 0.0),
                    )
                )
        pts.sort(key=lambda r: r[0])
        # Downsample cleanly to <= 260 points for smooth SVG rendering
        if len(pts) > 260:
            idx_s = np.linspace(0, len(pts) - 1, 260, dtype=int)
            pts_sub = [pts[i] for i in idx_s]
        else:
            pts_sub = pts
        trends[h] = {
            "hemisphere": h,
            "n_days": len(days_dict),
            "n_total_points": len(pts),
            "dates": [p[1][:10] for p in pts_sub],
            "timestamps": [p[1] for p in pts_sub],
            "lfp": [round(p[2], 1) for p in pts_sub],
            "ma": [round(p[3], 2) for p in pts_sub],
        }

    event_summary = []
    for es in (d.get("EventSummary") or {}).get("LfpAndAmplitudeSummary") or []:
        event_summary.append(
            {
                "group": _strip_enum(es.get("GroupId")),
                "hemisphere": _strip_enum(es.get("Hemisphere")),
                "target": _strip_enum(es.get("LeadLocation")).upper(),
                "above_pct": _clean_float(es.get("AboveThresholdPercent")),
                "between_pct": _clean_float(es.get("BetweenThresholdPercent")),
                "below_pct": _clean_float(es.get("BelowThresholdPercent")),
                "avg_ma": _clean_float(es.get("AverageAmplitude")),
                "adaptive_pct": _clean_float(es.get("TimeAdaptiveRunningPercent")),
            }
        )

    # --- 8. Selectable JSON Inspector Tree ---
    json_sections = []
    for k, v in d.items():
        if isinstance(v, dict):
            sub_items = []
            for sk, sv in list(v.items())[:25]:
                if isinstance(sv, (dict, list)):
                    preview = f"{type(sv).__name__} ({len(sv)})"
                else:
                    preview = str(sv)[:64]
                sub_items.append({"path": f"{k}.{sk}", "key": sk, "value": preview, "type": type(sv).__name__})
            json_sections.append({"path": k, "key": k, "type": "dict", "count": len(v), "children": sub_items})
        elif isinstance(v, list):
            json_sections.append(
                {
                    "path": k,
                    "key": k,
                    "type": "list",
                    "count": len(v),
                    "children": [
                        {"path": f"{k}[{i}]", "key": f"[{i}]", "value": str(item)[:64] if not isinstance(item, dict) else ", ".join(list(item.keys())[:4]), "type": type(item).__name__}
                        for i, item in enumerate(v[:12])
                    ],
                }
            )
        else:
            json_sections.append({"path": k, "key": k, "type": type(v).__name__, "count": 1, "value": str(v), "children": []})

    # Available distinct stimulation frequencies in streaming recordings
    stim_rates_avail = sorted(
        set(r["stim_rate_hz"] for r in recordings if r["source"] == "BrainSenseTimeDomain" and r["stim_rate_hz"] > 0)
    )
    if not stim_rates_avail:
        stim_rates_avail = [round(default_stim_rate, 1)]

    payload = {
        "metadata": metadata,
        "config": {
            "beta_band": [BETA_BAND[0], BETA_BAND[1]],
            "ftg_half_bw_hz": float(ftg_half_bw),
            "default_stim_rate_hz": round(default_stim_rate, 1),
            "stim_rates_available": stim_rates_avail,
        },
        "leads": leads,
        "groups": groups_out,
        "impedances": impedances,
        "montage": montage_pairs,
        "recordings": recordings,
        "trends": trends,
        "event_summary": event_summary,
        "json_sections": json_sections,
    }

    full_store = {
        "payload": payload,
        "raw_signals": raw_signals_store,
        "raw_json": d,
    }

    # --- 9. NeuroGame Arms & Dose-Response Bundle ---
    import nr_game as _ng
    import nr_options as _no
    import nr_risk as _nr
    import nr_weave as _nw

    payload["game_theory"] = _ng.extract_game_theory_bundle(
        full_store,
        ftg_hw=max(1.0, min(2.5, float(ftg_half_bw) * 0.5)),
        n_boot=48,
    )

    # --- 10. NeuroRisk Extreme-Value Theory & Tail-Risk Bundle ---
    payload["risk"] = _nr.extract_risk_bundle(
        full_store,
        ftg_hw=max(1.5, min(4.0, float(ftg_half_bw) * 0.5)),
        n_boot=64,
    )

    # --- 11. NeuroOptions Log-OU & Option-Pricing Bundle ---
    payload["options"] = _no.extract_options_bundle(
        full_store,
        ftg_hw=max(1.5, min(4.0, float(ftg_half_bw) * 0.5)),
    )

    # --- 12. NeuroWeave 5.0 Graph Signal Processing (GSP) Bundle ---
    payload["weave"] = _nw.extract_weave_bundle(
        full_store,
        ftg_hw=max(1.5, min(4.0, float(ftg_half_bw))),
    )
    return full_store


def _resolve_json_path(obj: Any, path_str: str) -> Any:
    """Resolve a dot/bracket JSON path like 'DeviceInformation.Final' or 'Groups[0]'."""
    cur = obj
    for part in re.split(r"\.(?![^\[]*\])", path_str):
        if cur is None:
            return None
        m = re.match(r"^([^\[\]]+)\[(\d+)\]$", part)
        if m:
            key, idx = m.group(1), int(m.group(2))
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return None
            if isinstance(cur, list) and idx < len(cur):
                cur = cur[idx]
            else:
                return None
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def all_contacts_with_data(payload: dict[str, Any]) -> list[str]:
    """Return every 'Hemisphere:contact' that has recorded LFP, montage, or impedance data,
    ordered cleanly by hemisphere and physical contact order."""
    found: set[str] = set()
    for r in payload.get("recordings") or []:
        h = r.get("hemisphere", "")
        if not h:
            continue
        for c in (r.get("contacts") or []) + (r.get("stim_contacts") or []):
            if c and c != "Case":
                found.add(f"{h}:{c}")
    for m in payload.get("montage") or []:
        h = m.get("hemisphere", "")
        if not h:
            continue
        for c in m.get("contacts") or []:
            if c and c != "Case":
                found.add(f"{h}:{c}")
    for h, imp in (payload.get("impedances") or {}).items():
        for c, v in (imp.get("monopolar") or {}).items():
            if c and c != "Case" and v:
                found.add(f"{h}:{c}")

    ordered: list[str] = []
    for h in ("Left", "Right"):
        for c in CONTACTS_ORDER:
            cid = f"{h}:{c}"
            if cid in found:
                ordered.append(cid)
    for cid in sorted(found):
        if cid not in ordered:
            ordered.append(cid)
    return ordered


def _matches_selected_contacts(hemi: str, contacts: list[str], sel_set: set[str]) -> bool:
    if not sel_set:
        return False
    return any(f"{hemi}:{c}" in sel_set for c in contacts if c and c != "Case")


def build_selection_output(
    full_store: dict[str, Any],
    selection: dict[str, Any] | None,
    dag_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Hydrate the interactive widget selection + DAG state into a rich Python data structure.

    Enforces dynamic contact-driven scoping: only recordings, bipolar pairs, impedances, and
    game-theory arms belonging to the contacts selected on the DBS Lead Model are exposed
    downstream, alongside general DBS device/session/lead/therapy metadata.
    """
    payload = full_store["payload"]
    raw_signals = full_store["raw_signals"]
    raw_json = full_store["raw_json"]
    sel = selection or {}
    dag = dag_state or {}

    active_hemi = sel.get("hemisphere", "Left")
    if "contacts" in sel and sel["contacts"] is not None:
        sel_contacts = list(sel["contacts"])
    else:
        sel_contacts = all_contacts_with_data(payload)
    sel_contact_set = set(sel_contacts)

    sel_json_paths = sel.get("json_paths") or []
    stim_hz = _clean_float(sel.get("stim_rate_hz"), payload["config"]["default_stim_rate_hz"])
    ftg_bw = _clean_float(sel.get("ftg_half_bw_hz"), payload["config"]["ftg_half_bw_hz"])
    ftg_center = round(stim_hz / 2.0, 2)
    live_ftg_band = (max(35.0, ftg_center - ftg_bw), min(120.0, ftg_center + ftg_bw))

    # Active final group for electrical overlay lookup
    active_group = next((g for g in payload["groups"] if g["stage"] == "Final" and g["active"]), None)
    if active_group is None and payload["groups"]:
        active_group = payload["groups"][0]

    # 1. Hydrate ONLY selected contacts
    contacts_list = []
    for cid in sel_contacts:
        if ":" in cid:
            h, c = cid.split(":", 1)
        else:
            h, c = active_hemi, cid
        mono_ohm = (payload["impedances"].get(h) or {}).get("monopolar", {}).get(c)
        stim_pol, stim_ma = "Off", 0.0
        if active_group and h in active_group["hemispheres"]:
            for es in active_group["hemispheres"][h]["electrodes"]:
                if es["contact"] == c:
                    stim_pol = es["polarity"]
                    stim_ma = es["ma"]
        contacts_list.append(
            {
                "id": f"{h}:{c}",
                "hemisphere": h,
                "contact": c,
                "lead_model": (payload["leads"].get(h) or {}).get("model", ""),
                "target": (payload["leads"].get(h) or {}).get("target", ""),
                "impedance_ohm": mono_ohm,
                "active_polarity": stim_pol,
                "active_ma": stim_ma,
            }
        )

    # 2. Hydrate bipolar pairs dynamically filtered by selected contacts (recomputed with live ftg_bw!)
    pairs_list = []
    for m in payload["montage"]:
        if not _matches_selected_contacts(m["hemisphere"], m.get("contacts") or [], sel_contact_set):
            continue
        f_arr = np.asarray(m.get("freqs") or [], dtype=float)
        p_arr = np.asarray(m.get("mag_uv") or [], dtype=float) ** 2
        bs = _band_metrics_from_psd(f_arr, p_arr, BETA_BAND, live_ftg_band) if len(f_arr) else m["band_stats"]
        pairs_list.append(
            {
                "id": m["id"],
                "hemisphere": m["hemisphere"],
                "pair": m["pair"],
                "kind": m["kind"],
                "contacts": m["contacts"],
                "peak_hz": m["peak_hz"],
                "peak_uv": m["peak_uv"],
                "artifact": m["artifact"],
                "beta_db": bs["beta_db"],
                "beta_peak_hz": bs["beta_peak_hz"],
                "ftg_db": bs["ftg_db"],
                "ftg_peak_hz": bs["ftg_peak_hz"],
                "ftg_beta_ratio": bs["ftg_beta_ratio"],
            }
        )

    # 3. Hydrate ONLY recordings related to selected contacts (with live ftg_bw metrics!)
    recs_out: dict[str, dict[str, Any]] = {}
    recs_rows = []
    for r in payload["recordings"]:
        r_contacts = (r.get("contacts") or []) + (r.get("stim_contacts") or [])
        if not _matches_selected_contacts(r["hemisphere"], r_contacts, sel_contact_set):
            continue
        rid = r["id"]
        raw_entry = raw_signals.get(rid, {})
        r_stim = float(r.get("stim_rate_hz") or stim_hz)
        r_ftg_c = round(r_stim / 2.0, 2)
        r_ftg_band = (max(35.0, r_ftg_c - ftg_bw), min(120.0, r_ftg_c + ftg_bw))
        psd_obj = r.get("psd") or {}
        if psd_obj.get("freqs") and psd_obj.get("power"):
            bs = _band_metrics_from_psd(np.asarray(psd_obj["freqs"]), np.asarray(psd_obj["power"]), BETA_BAND, r_ftg_band)
        else:
            bs = r.get("band_stats", {})
        recs_out[rid] = {
            "id": rid,
            "label": r["label"],
            "source": r["source"],
            "hemisphere": r["hemisphere"],
            "pair": r["pair"],
            "contacts": r.get("contacts") or [],
            "stim_contacts": r.get("stim_contacts") or [],
            "stim_rate_hz": r_stim,
            "ftg_center_hz": r_ftg_c,
            "ftg_band_hz": [round(r_ftg_band[0], 2), round(r_ftg_band[1], 2)],
            "beta_band_hz": r["beta_band"],
            "fs_hz": r["fs_hz"],
            "duration_s": r["duration_s"],
            "amp_range_ma": r["amp_range_ma"],
            "band_stats": bs,
            "amp_steps": r.get("amp_steps", []),
            "time_domain_uv": raw_entry.get("time_domain_uv"),
            "amp_t_s": raw_entry.get("amp_t"),
            "amp_ma": raw_entry.get("amp_ma"),
        }
        recs_rows.append(
            {
                "id": rid,
                "label": r["label"],
                "source": r["source"],
                "hemisphere": r["hemisphere"],
                "pair": r["pair"],
                "stim_rate_hz": r_stim,
                "ftg_center_hz": r_ftg_c,
                "duration_s": r["duration_s"],
                "beta_db": bs.get("beta_db"),
                "beta_peak_hz": bs.get("beta_peak_hz"),
                "ftg_db": bs.get("ftg_db"),
                "ftg_peak_hz": bs.get("ftg_peak_hz"),
                "ftg_beta_ratio": bs.get("ftg_beta_ratio"),
            }
        )

    # 4. DAG topology
    dag_nodes = dag.get("nodes") or []
    dag_edges = dag.get("edges") or []
    dag_nodes_rows = [
        {
            "id": n.get("id"),
            "type": n.get("type"),
            "title": n.get("title"),
            "hemisphere": n.get("hemisphere"),
            "pair": n.get("pair"),
            "recording_id": n.get("recordingId"),
            "x": round(_clean_float(n.get("x")), 1),
            "y": round(_clean_float(n.get("y")), 1),
            "w": round(_clean_float(n.get("w")), 1),
            "h": round(_clean_float(n.get("h")), 1),
            "collapsed": bool(n.get("collapsed", False)),
        }
        for n in dag_nodes
    ]

    # 5. Selected JSON paths
    json_extracts = {}
    for p in sel_json_paths:
        json_extracts[p] = _resolve_json_path(raw_json, p)

    # 6. Exact Game-Theory Equilibrium Solutions (reactive to widget weights/filter/bandit)
    import nr_game as _ng
    import nr_options as _no
    import nr_risk as _nr
    import nr_weave as _nw

    gt_bundle = payload.get("game_theory") or {}
    gt_params = sel.get("gt_params") or {}
    gt_hemi = gt_params.get("hemisphere") or active_hemi
    gt_rate = gt_params.get("stim_rate_filter", "ALL")
    gt_weights = {
        "wb": _clean_float(gt_params.get("wb"), 1.0),
        "wf": _clean_float(gt_params.get("wf"), 0.6),
        "we": _clean_float(gt_params.get("we"), 0.5),
        "tau": _clean_float(gt_params.get("tau"), 2.0),
        "ftg_safe": _clean_float(gt_params.get("ftg_safe"), 4.0),
    }
    gt_exact = _ng.solve_game_theory_exact(
        gt_bundle,
        hemisphere=gt_hemi,
        stim_rate_filter=gt_rate,
        weights=gt_weights,
        bandit_live=sel.get("gt_bandit_live"),
        selected_contacts=sel_contacts,
    )
    gt_selected_arms = sel.get("gt_selected_arms") or []

    # 7. Exact NeuroRisk Extreme-Value Selection & Backtest (reactive to widget risk_params)
    risk_bundle = payload.get("risk") or {}
    risk_params_raw = sel.get("risk_params") or {}
    risk_hemi = risk_params_raw.get("hemisphere") or active_hemi
    risk_rate = risk_params_raw.get("stim_rate_filter", "ALL")
    risk_params = {
        "target": risk_params_raw.get("target") or "beta",
        "alpha": _clean_float(risk_params_raw.get("alpha"), 0.99),
        "horizon": _clean_float(risk_params_raw.get("horizon"), 60.0),
        "kappa": _clean_float(risk_params_raw.get("kappa"), 0.6),
        "eps": _clean_float(risk_params_raw.get("eps"), 0.10),
        "metric": risk_params_raw.get("metric") or "CVaR",
    }
    risk_exact = _nr.solve_risk_exact(
        risk_bundle,
        hemisphere=risk_hemi,
        stim_rate_filter=risk_rate,
        params=risk_params,
        selected_contacts=sel_contacts,
    )
    risk_selected_arms = sel.get("risk_selected_arms") or []

    # 8. Exact NeuroOptions Option-Pricing Selection & Greeks (reactive to widget opt_params)
    opt_bundle = payload.get("options") or {}
    opt_params_raw = sel.get("opt_params") or {}
    opt_hemi = opt_params_raw.get("hemisphere") or active_hemi
    opt_rate = opt_params_raw.get("stim_rate_filter", "ALL")
    opt_params = {
        "strategy": opt_params_raw.get("strategy") or "call",
        "T": _clean_float(opt_params_raw.get("T"), 0.25),
        "k_burst_mult": _clean_float(opt_params_raw.get("k_burst_mult"), 1.0),
        "k_low": _clean_float(opt_params_raw.get("k_low"), 0.5),
        "k_set": _clean_float(opt_params_raw.get("k_set"), 0.7),
        "wing": _clean_float(opt_params_raw.get("wing"), 0.25),
        "w_ftg": _clean_float(opt_params_raw.get("w_ftg"), 0.5),
        "w_energy": _clean_float(opt_params_raw.get("w_energy"), 0.02),
        "steps": int(_clean_float(opt_params_raw.get("steps"), 10)),
    }
    opt_exact = _no.solve_options_exact(
        opt_bundle,
        hemisphere=opt_hemi,
        stim_rate_filter=opt_rate,
        params=opt_params,
        selected_contacts=sel_contacts,
    )
    opt_selected_arms = sel.get("opt_selected_arms") or []

    # 9. Exact NeuroWeave 5.0 GSP Biomarkers, Node Table & σ-Sweep (reactive to widget gsp_params)
    weave_bundle = payload.get("weave") or {}
    gsp_params_raw = sel.get("gsp_params") or {}
    gsp_hemi = gsp_params_raw.get("hemisphere") or active_hemi
    gsp_rate = gsp_params_raw.get("stim_rate_filter", "ALL")
    gsp_params = {
        "sigma": _clean_float(gsp_params_raw.get("sigma"), 2.0),
        "series_id": gsp_params_raw.get("series_id") or "",
        "phase_band": gsp_params_raw.get("phase_band") or "ftg",
        "node_size_metric": gsp_params_raw.get("node_size_metric") or "local_ftg",
    }
    weave_exact = _nw.solve_weave_exact(
        weave_bundle,
        hemisphere=gsp_hemi,
        stim_rate_filter=gsp_rate,
        params=gsp_params,
        selected_contacts=sel_contacts,
        ftg_half_bw=ftg_bw,
    )
    gsp_selected_node = sel.get("gsp_selected_node_idx")

    general_dbs = {
        "metadata": payload["metadata"],
        "leads": payload["leads"],
        "groups": payload["groups"],
        "event_summary": payload.get("event_summary") or [],
    }

    return {
        "summary": {
            "file_name": payload["metadata"]["file_name"],
            "device": f"{payload['metadata']['device_name']} ({payload['metadata']['device_model']})",
            "diagnosis": payload["metadata"]["diagnosis"],
            "session_date": payload["metadata"]["session_date"],
            "active_hemisphere": active_hemi,
            "stim_rate_hz": stim_hz,
            "beta_band_hz": [BETA_BAND[0], BETA_BAND[1]],
            "ftg_center_hz": ftg_center,
            "ftg_half_bw_hz": ftg_bw,
            "ftg_band_hz": [round(live_ftg_band[0], 2), round(live_ftg_band[1], 2)],
            "n_selected_contacts": len(contacts_list),
            "n_selected_pairs": len(pairs_list),
            "n_selected_recordings": len(recs_out),
            "n_dag_nodes": len(dag_nodes),
            "n_dag_edges": len(dag_edges),
            "gt_majority_arm": gt_exact.get("majority_arm"),
            "gt_majority_votes": gt_exact.get("majority_votes"),
            "gt_game_value": gt_exact.get("game_value"),
            "risk_target": risk_params["target"],
            "risk_majority_arm": risk_exact.get("majority_arm"),
            "risk_majority_votes": risk_exact.get("majority_votes"),
            "opt_strategy": opt_params["strategy"],
            "opt_majority_arm": opt_exact.get("majority_arm"),
            "opt_majority_votes": opt_exact.get("majority_votes"),
            "gsp_sigma": weave_exact["params"]["sigma"],
            "gsp_active_series": weave_exact.get("active_series"),
            "gsp_fiedler": (weave_exact.get("active_measures") or {}).get("fiedler_value"),
            "gsp_gtv_ftg": (weave_exact.get("active_measures") or {}).get("gtv_ftg"),
            "gsp_gtv_beta": (weave_exact.get("active_measures") or {}).get("gtv_beta"),
        },
        "general_dbs": general_dbs,
        "selection_raw": sel,
        "contacts": contacts_list,
        "pairs": pairs_list,
        "recordings": recs_out,
        "bands": {
            "beta_hz": [BETA_BAND[0], BETA_BAND[1]],
            "ftg_center_hz": ftg_center,
            "ftg_hz": [round(ftg_center - ftg_bw, 2), round(ftg_center + ftg_bw, 2)],
            "selected_band": sel.get("selected_band", "both"),
            "freq_window_hz": sel.get("freq_window_hz"),
            "time_window_s": sel.get("time_window_s"),
        },
        "dag": {
            "nodes": dag_nodes_rows,
            "edges": dag_edges,
            "viewport": dag.get("viewport", {}),
        },
        "game_theory": {
            "hemisphere": gt_hemi,
            "stim_rate_filter": gt_rate,
            "weights": gt_exact["weights"],
            "selected_arm_ids": gt_selected_arms,
            "majority_arm": gt_exact.get("majority_arm"),
            "majority_votes": gt_exact.get("majority_votes"),
            "game_value": gt_exact.get("game_value"),
            "maximin_arm": gt_exact.get("maximin_arm"),
            "nash_bargaining_arm": gt_exact.get("nash_bargaining_arm"),
            "nature_mix": gt_exact.get("nature_mix"),
            "x_star": gt_exact.get("x_star"),
            "disagreement_point": gt_exact.get("disagreement_point"),
            "stackelberg": gt_exact.get("stackelberg"),
            "bandit_live": gt_exact.get("bandit_live"),
        },
        "risk": {
            "hemisphere": risk_hemi,
            "stim_rate_filter": risk_rate,
            "params": risk_exact["params"],
            "amax": risk_exact.get("amax"),
            "selected_arm_ids": risk_selected_arms,
            "majority_arm": risk_exact.get("majority_arm"),
            "majority_votes": risk_exact.get("majority_votes"),
            "picks": risk_exact.get("picks"),
        },
        "options": {
            "hemisphere": opt_hemi,
            "stim_rate_filter": opt_rate,
            "params": opt_exact["params"],
            "amax": opt_exact.get("amax"),
            "selected_arm_ids": opt_selected_arms,
            "majority_arm": opt_exact.get("majority_arm"),
            "majority_votes": opt_exact.get("majority_votes"),
            "picks": opt_exact.get("picks"),
        },
        "weave": {
            "hemisphere": gsp_hemi,
            "stim_rate_filter": gsp_rate,
            "params": weave_exact["params"],
            "active_series": weave_exact.get("active_series"),
            "active_series_id": weave_exact.get("active_series_id"),
            "active_measures": weave_exact.get("active_measures"),
            "selected_node_idx": gsp_selected_node,
        },
        "json_extracts": json_extracts,
        "dataframes": {
            "contacts_df": pd.DataFrame(contacts_list),
            "pairs_df": pd.DataFrame(pairs_list),
            "recordings_df": pd.DataFrame(recs_rows),
            "dag_nodes_df": pd.DataFrame(dag_nodes_rows),
            "gt_arms_df": gt_exact["arms_df"],
            "gt_consensus_df": gt_exact["consensus_df"],
            "risk_arms_df": risk_exact["risk_arms_df"],
            "risk_selection_df": risk_exact["risk_selection_df"],
            "risk_series_df": risk_exact["risk_series_df"],
            "risk_backtest_df": risk_exact["risk_backtest_df"],
            "opt_arms_df": opt_exact["opt_arms_df"],
            "opt_selection_df": opt_exact["opt_selection_df"],
            "opt_chain_df": opt_exact["opt_chain_df"],
            "gsp_series_df": weave_exact["gsp_series_df"],
            "gsp_nodes_df": weave_exact["gsp_nodes_df"],
            "gsp_sigma_df": weave_exact["gsp_sigma_df"],
        },
    }
