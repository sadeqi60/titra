"""neuroReader data input — detect, verify, normalise and load ANY JSON for the DAG pipeline.

Supported families (auto-detected from content, not from the file name):

* Medtronic Percept session report ("Report_Json_Session_Report", clinical exports, MATLAB mocks).
  Shape quirks are repaired before parsing: sections that MATLAB's jsonencode collapses from a
  one-element array into a single object (BrainSenseTimeDomain, BrainSenseLfp, Impedance,
  Groups.Final, ElectrodeState …) are wrapped back into lists. Every time-domain section is used:
  BrainSenseTimeDomain (on-stim streaming), IndefiniteStreaming (off-stim multichannel), and —
  when no streaming exists — SenseChannelTests / CalibrationTests snippets.
* Bravo-style amplitude titration exports (one file per stimulated contact: ParticipantId, Contact,
  TherapyParameters, data[].Timeseries_Data[] segments per Stim_Amplitude). Converted to a
  Percept-shaped session (streams + BrainSenseLfp mA staircase + therapy group + 4-ring lead), so
  the full analysis stack runs unchanged. Several files of one participant/date merge into one
  bilateral session.
* GroupHistory-only exports and any other JSON are recognised and rejected with a clear reason.

Public API
  scan_folder(folder)                  → list of JSON files (path, name, size, folder)
  inspect_path(path) / inspect_bytes() → verification report (dict) — fast, never raises
  load_path(path) / load_bytes()       → (store, report) — store["payload"] feeds the widget
  load_paths([paths])                  → merged Bravo session from several files
"""

from __future__ import annotations

import copy
import json
import math
import pathlib
import re
import time
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent

MIN_TD_SAMPLES = 1000          # ≥ 4 s at 250 Hz for spectral analysis
ASSUMED_FS_BRAVO = 250.0       # Bravo exports do not store the sampling rate

TD_SECTIONS = ("BrainSenseTimeDomain", "IndefiniteStreaming", "SenseChannelTests", "CalibrationTests")
PERCEPT_MARKERS = (
    "DeviceInformation", "LeadConfiguration", "Groups", "BrainSenseTimeDomain", "BrainSenseLfp",
    "IndefiniteStreaming", "LFPMontage", "LfpMontageTimeDomain", "DiagnosticData", "Impedance",
    "SessionDate", "PatientInformation", "BatteryInformation", "SenseChannelTests",
)
# Keys whose value must be a list of objects (MATLAB may emit a single object instead)
LIST_KEYS = {
    "BrainSenseTimeDomain", "BrainSenseLfp", "IndefiniteStreaming", "LfpMontageTimeDomain",
    "LFPMontage", "BrainSenseSurveys", "BrainSenseSurveysTimeDomain", "SenseChannelTests",
    "CalibrationTests", "Impedance", "GroupHistory", "Monopolar", "Bipolar", "SensingChannel",
    "ElectrodeState", "Programs", "LfpData", "Annotations", "GroupUsagePercentage", "RechargeCount",
    "MostRecentInSessionSignalCheck",
}
# Keys that are lists only below these parents (elsewhere "Final"/"Hemisphere" are objects/strings)
PARENT_LIST_KEYS = {
    "Groups": {"Final", "Initial"},
    "LeadConfiguration": {"Final", "Initial"},
    "Impedance": {"Hemisphere"},
    # events carry a nested hemisphere-keyed "LfpFrequencySnapshotEvents" object: only the
    # DiagnosticData-level key is a list
    "DiagnosticData": {"EventLogs", "LfpFrequencySnapshotEvents"},
}
NUM_TOKENS = {"0": "ZERO", "1": "ONE", "2": "TWO", "3": "THREE"}


# ── small helpers ────────────────────────────────────────────────────────
def _fmt_size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} kB"


def _as_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, dict):
        return [v]
    return []


def _enum(v: Any) -> str:
    s = "" if v is None else str(v)
    return s.split(".")[-1] if "." in s else s


def _hemi_of_channel(ch: str) -> str:
    u = str(ch).upper()
    return "Left" if "LEFT" in u else ("Right" if "RIGHT" in u else "")


def _td_len(item: dict) -> int:
    td = item.get("TimeDomainData")
    if isinstance(td, list):
        return len(td)
    return 1 if isinstance(td, (int, float)) else 0


def _numeric_ok(arr: Any, probe: int = 200) -> bool:
    if not isinstance(arr, list) or not arr:
        return False
    sample = arr[:probe] + arr[-probe:]
    good = sum(1 for x in sample if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x))
    return good >= 0.9 * len(sample)


def _report(fmt: str, label: str, name: str, size: int) -> dict:
    return {
        "ok": False, "status": "error", "format": fmt, "format_label": label, "file": name,
        "size": _fmt_size(size) if size else "", "summary": "", "checks": [], "capabilities": [],
        "warnings": [], "errors": [], "fixes": [], "elapsed_s": 0.0,
    }


def _check(rep: dict, item: str, status: str, detail: str) -> None:
    rep["checks"].append({"item": item, "status": status, "detail": detail})


# ── Percept normalisation ───────────────────────────────────────────────
_MATLAB_ENUM_KEY = re.compile(r"^(HemisphereLocationDef)_(Left|Right)$")

def normalize_percept(d: dict, fixes: list[str] | None = None, include_tests: bool | None = None) -> dict:
    """Return a repaired copy: list-shaped sections, unified time-domain list with origin tags."""
    fixes = fixes if fixes is not None else []
    out = shape_fix(d, fixes)
    return _unify_td(out, fixes, include_tests)


def shape_fix(d: dict, fixes: list[str] | None = None) -> dict:
    """List-shaped sections and restored MATLAB field names (copy; the input is not modified)."""
    fixes = fixes if fixes is not None else []
    out = copy.copy(d)

    def walk(o: Any, parent: str = "", path: str = "") -> Any:
        if isinstance(o, dict):
            new = {}
            for k, v in o.items():
                # MATLAB exports cannot use dots in field names: HemisphereLocationDef_Left → HemisphereLocationDef.Left
                mk = _MATLAB_ENUM_KEY.match(k) if isinstance(k, str) else None
                if mk:
                    k2 = f"{mk.group(1)}.{mk.group(2)}"
                    if k2 not in o:
                        fixes.append(f"{path + k}: MATLAB field name restored to {k2}")
                        k = k2
                must_list = k in LIST_KEYS or k in PARENT_LIST_KEYS.get(parent, ())
                if must_list and isinstance(v, dict):
                    fixes.append(f"{path + k}: single object wrapped into a list")
                    v = [v]
                if k == "TimeDomainData" and isinstance(v, (int, float)):
                    v = [v]
                new[k] = walk(v, k, f"{path}{k}.")
            return new
        if isinstance(o, list):
            # GroupHistory entries carry a "Groups" list
            if parent == "GroupHistory":
                return [
                    {kk: (_as_list(vv) if kk == "Groups" else vv) for kk, vv in it.items()} if isinstance(it, dict) else it
                    for it in (walk(x, "", path) for x in o)
                ]
            return [walk(x, parent, path) for x in o]
        return o

    for key in list(out.keys()):
        out[key] = walk({key: out[key]}, "", "")[key]
    return out


def _unify_td(out: dict, fixes: list[str], include_tests: bool | None) -> dict:
    # Unified time-domain list (the parser's streaming loop reads BrainSenseTimeDomain)
    bstd = [x for x in _as_list(out.get("BrainSenseTimeDomain")) if isinstance(x, dict)]
    origins = ["BrainSense streaming"] * len(bstd)
    extra = [x for x in _as_list(out.get("IndefiniteStreaming")) if isinstance(x, dict) and _td_len(x) >= MIN_TD_SAMPLES]
    if extra:
        fixes.append(f"IndefiniteStreaming: {len(extra)} off-stim channel(s) added as streams")
    bstd = bstd + extra
    origins += ["Indefinite streaming"] * len(extra)
    usable_stream = any(_td_len(x) >= MIN_TD_SAMPLES for x in bstd)
    if include_tests or (include_tests is None and not usable_stream):
        tests = []
        for sec, lab in (("SenseChannelTests", "Sense channel test"), ("CalibrationTests", "Calibration test")):
            for x in _as_list(out.get(sec)):
                if isinstance(x, dict) and _td_len(x) >= MIN_TD_SAMPLES:
                    tests.append((x, lab))
        if tests:
            fixes.append(f"No streaming recordings — {len(tests)} sense/calibration test snippet(s) used as signals")
            bstd += [t[0] for t in tests]
            origins += [t[1] for t in tests]
    out["BrainSenseTimeDomain"] = bstd
    out["_nr_td_origins"] = origins
    return out


# ── Bravo conversion ─────────────────────────────────────────────────────
_BRAVO_CH = re.compile(r"(Left|Right)\s+(\w+)?\s*E(\d{2})\s*-\s*E(\d{2})", re.I)


def is_bravo(d: Any) -> bool:
    return isinstance(d, dict) and "data" in d and ("Contact" in d or "TherapyParameters" in d) and isinstance(d.get("data"), list)


def _bravo_contact(e: str) -> str:
    n = int(e)
    return str(n - 8 if n >= 8 else n)  # E00–E03 (left IPG port) / E08–E11 (right port) → 0–3


def bravo_to_percept(docs: list[tuple[str, dict]], fixes: list[str] | None = None) -> dict:
    """Merge one or more Bravo titration exports into a Percept-shaped session dict."""
    fixes = fixes if fixes is not None else []
    bstd, bsl, leads, sens = [], [], {}, []
    rate_all, pw_all, dates = set(), set(), set()
    t_base = 0
    for name, d in docs:
        contact = str(d.get("Contact", ""))
        m_hemi = re.match(r"\s*(Left|Right)\s+(\w+)?\s*E(\d{2})", contact, re.I)
        hemi = (m_hemi.group(1).capitalize() if m_hemi else "") or "Left"
        target = (m_hemi.group(2) or "").upper() if m_hemi else ""
        stim_c = _bravo_contact(m_hemi.group(3)) if m_hemi else ""
        tp = str(d.get("TherapyParameters", ""))
        rate = float(re.search(r"([\d.]+)\s*Hz", tp, re.I).group(1)) if re.search(r"([\d.]+)\s*Hz", tp, re.I) else 0.0
        pw = float(re.search(r"([\d.]+)\s*u?Sec", tp, re.I).group(1)) if re.search(r"([\d.]+)\s*u?Sec", tp, re.I) else 0.0
        rate_all.add(rate)
        pw_all.add(pw)
        date = str(d.get("Date", "2000-01-01"))[:10]
        dates.add(date)
        leads[hemi] = target or leads.get(hemi, "")
        for blk in _as_list(d.get("data")):
            segs = [s for s in _as_list(blk.get("Timeseries_Data")) if isinstance(s, dict)]
            by_ch: dict[str, list[dict]] = {}
            for s in segs:
                by_ch.setdefault(str(s.get("Channel", "")), []).append(s)
            for ch, items in by_ch.items():
                mc = _BRAVO_CH.search(ch)
                if not mc:
                    continue
                ch_hemi = mc.group(1).capitalize()
                e1, e2 = _bravo_contact(mc.group(3)), _bravo_contact(mc.group(4))
                chan = f"{NUM_TOKENS.get(e1, e1)}_{NUM_TOKENS.get(e2, e2)}_{ch_hemi.upper()}"
                samples, steps = [], []
                for s in items:
                    x = s.get("Timeseries")
                    if isinstance(x, list) and x and _numeric_ok(x):
                        steps.append((float(s.get("Stim_Amplitude") or 0.0), len(samples), len(x)))
                        samples.extend(float(v) for v in x)
                if len(samples) < MIN_TD_SAMPLES:
                    continue
                fs = ASSUMED_FS_BRAVO
                t0 = f"{date}T08:{(t_base // 60) % 60:02d}:{t_base % 60:02d}.000Z"
                t_base += 7
                bstd.append({"Channel": chan, "SampleRateInHz": fs, "FirstPacketDateTime": t0, "TimeDomainData": samples,
                             "_nr_origin": f"Bravo titration · stim {stim_c or '?'}"})
                # 2 Hz mA staircase aligned to the concatenated samples
                rows, n_tot = [], len(samples)
                for k in range(int(n_tot / fs * 2) + 1):
                    idx = k * fs / 2
                    amp = next((a for a, st, ln in steps if st <= idx < st + ln), steps[-1][0])
                    rows.append({"TicksInMs": int(k * 500), ch_hemi: {"mA": amp, "LFP": 0}})
                bsl.append({
                    "Channel": chan, "SampleRateInHz": 2, "FirstPacketDateTime": t0,
                    "TherapySnapshot": {ch_hemi: {
                        "RateInHertz": rate, "PulseWidthInMicroSecond": pw,
                        "ElectrodeState": [{"Electrode": f"ElectrodeDef.ELECTRODE_{stim_c}", "ElectrodeStateResult": "ElectrodeStateDef.Negative"}] if stim_c else [],
                    }},
                    "LfpData": rows,
                })
                sens.append({"hemi": ch_hemi, "rate": rate, "pw": pw, "stim": stim_c, "chan": chan, "amax": max(a for a, _, _ in steps)})
    if len(dates) > 1:
        fixes.append(f"Merged recordings from {len(dates)} dates: {', '.join(sorted(dates))}")
    rate0 = max(rate_all) if rate_all else 125.0
    # one therapy group per stimulation contact (E01 → GROUP_A, E02 → GROUP_B …);
    # each group carries every hemisphere that was titrated on that contact,
    # with the highest amplitude reached and the sensing pair that recorded it.
    groups = []
    if sens:
        by_c: dict[str, dict[str, dict]] = {}
        for s in sens:
            slot = by_c.setdefault(s["stim"] or "?", {})
            cur = slot.get(s["hemi"])
            if cur is None or s["amax"] > cur["amax"]:
                slot[s["hemi"]] = s
        letters = "ABCDEFGHIJKLMNOP"
        for gi, (c, by_h) in enumerate(sorted(by_c.items())):
            groups.append({
                "GroupId": f"GroupIdDef.GROUP_{letters[gi % 16]}", "ActiveGroup": gi == 0,
                "GroupName": f"Stim contact {c}",
                "ProgramSettings": {"RateInHertz": max(x["rate"] for x in by_h.values()), "SensingChannel": [
                    {"HemisphereLocation": f"HemisphereLocationDef.{h}", "RateInHertz": x["rate"], "PulseWidthInMicroSecond": x["pw"],
                     "ElectrodeState": [{"Electrode": f"ElectrodeDef.ELECTRODE_{x['stim']}", "ElectrodeStateResult": "ElectrodeStateDef.Negative",
                                         "ElectrodeAmplitudeInMilliAmps": x["amax"]}] if x["stim"] else [],
                     "SensingSetup": {"ChannelSignalResult": {"Channel": f"SensingElectrodeConfigDef.{x['chan']}"}}}
                    for h, x in sorted(by_h.items())]},
            })
    pid = next((str(d.get("ParticipantId", ""))[:8] for _, d in docs if d.get("ParticipantId")), "")
    return {
        "SessionDate": f"{min(dates) if dates else '2000-01-01'}T08:00:00Z",
        "SessionEndDate": f"{max(dates) if dates else '2000-01-01'}T09:00:00Z",
        "DataVersion": "bravo-titration",
        "PatientInformation": {"Final": {"Diagnosis": "DiagnosisTypeDef.Unknown"}},
        "DeviceInformation": {"Final": {"Neurostimulator": "Bravo export", "NeurostimulatorModel": f"participant {pid}" if pid else "titration"}},
        "LeadConfiguration": {"Final": [
            {"Hemisphere": f"HemisphereLocationDef.{h}", "Model": "LeadModelDef.RING_4_CONTACT", "LeadLocation": f"LeadLocationDef.{t or 'UNKNOWN'}"}
            for h, t in leads.items()]},
        "Groups": {"Final": groups, "Initial": []},
        "BrainSenseTimeDomain": bstd,
        "BrainSenseLfp": bsl,
        "_nr_td_origins": [x.pop("_nr_origin") for x in bstd],
    }


# ── detection & verification ─────────────────────────────────────────────
def _capabilities(n_streams: int, n_sweep: int, max_steps: int, montage: int, trends: bool, imp: bool, groups: int) -> list[dict]:
    def cap(name, ok, why):
        return {"analysis": name, "ok": bool(ok), "why": why}
    return [
        cap("LFP trace · PSD · spectrogram · β/FTG · phase", n_streams > 0, f"{n_streams} time-domain signal(s)" if n_streams else "needs ≥ 1 time-domain signal (≥ 4 s)"),
        cap("Dose–response · game theory · risk · options", n_sweep > 0, f"{n_sweep} amplitude sweep(s), up to {max_steps} steps" if n_sweep else "needs a stream with ≥ 2 stimulation amplitudes"),
        cap("Graph signal processing (GSP)", max_steps >= 3, f"{max_steps} amplitude nodes" if max_steps >= 3 else "needs ≥ 3 amplitude steps in one stream"),
        cap("Directional montage survey", montage > 0, f"{montage} bipolar pair spectra" if montage else "no LFPMontage section"),
        cap("Chronic LFP trend", trends, "LFPTrendLogs present" if trends else "no LFPTrendLogs"),
        cap("Impedance matrix", imp, "impedance results present" if imp else "no impedance results"),
        cap("Therapy groups", groups > 0, f"{groups} group(s)" if groups else "no Groups section"),
    ]


def _verify_percept(d: dict, rep: dict) -> None:
    fixes: list[str] = []
    nd = normalize_percept(d, fixes)
    rep["fixes"] = list(dict.fromkeys(fixes))[:12]
    origins = nd.get("_nr_td_origins", [])
    tds = nd.get("BrainSenseTimeDomain", [])
    good, bad, short, fs_set, hemis, durs, by_origin = [], 0, 0, set(), set(), [], {}
    for x, org in zip(tds, origins):
        n = _td_len(x)
        if not _numeric_ok(x.get("TimeDomainData")):
            bad += 1
            continue
        if n < MIN_TD_SAMPLES:
            short += n >= 32  # the parser still loads ≥ 32-sample entries
            continue
        fs = float(x.get("SampleRateInHz") or 250)
        good.append(x)
        fs_set.add(fs)
        h = _hemi_of_channel(x.get("Channel", ""))
        if h:
            hemis.add(h)
        durs.append(n / max(fs, 1))
        by_origin[org] = by_origin.get(org, 0) + 1
    # amplitude sweeps from BrainSenseLfp
    sweeps, max_steps = 0, 0
    for b in _as_list(nd.get("BrainSenseLfp")):
        amps = set()
        for row in _as_list(b.get("LfpData")):
            for side in ("Left", "Right"):
                s = row.get(side) if isinstance(row, dict) else None
                if isinstance(s, dict) and isinstance(s.get("mA"), (int, float)):
                    amps.add(round(float(s["mA"]), 1))
        if len(amps) >= 2:
            sweeps += 1
            max_steps = max(max_steps, len(amps))
    if good:
        src = " + ".join(f"{v} {k}" for k, v in by_origin.items())
        _check(rep, "Time-domain LFP", "ok", f"{len(good)} signal(s): {src} · {'/'.join(sorted(hemis)) or '?'} · {', '.join(f'{f:g}' for f in sorted(fs_set))} Hz · {min(durs):.0f}–{max(durs):.0f} s")
    else:
        _check(rep, "Time-domain LFP", "missing", "no BrainSense / indefinite streaming or test signals ≥ 4 s")
    if short:
        rep["warnings"].append(f"{short} short time-domain entr{'y' if short == 1 else 'ies'} (< 4 s) will also load but give unreliable spectra")
    if bad:
        rep["warnings"].append(f"{bad} time-domain entr{'y' if bad == 1 else 'ies'} skipped (empty or non-numeric samples)")
    _check(rep, "Stimulation amplitude sweeps", "ok" if sweeps else ("warn" if good else "missing"),
           f"{sweeps} stream(s) with ≥ 2 mA steps (max {max_steps})" if sweeps else ("streams are single-amplitude / off-stim" if good else "no BrainSenseLfp"))
    montage = len([m for m in _as_list(nd.get("LFPMontage")) if isinstance(m, dict) and m.get("LFPMagnitude")])
    mtd = len([m for m in _as_list(nd.get("LfpMontageTimeDomain")) if isinstance(m, dict) and _td_len(m) >= 100])
    _check(rep, "Directional montage survey", "ok" if montage else "missing", f"{montage} pair spectra · {mtd} survey time series" if montage else "no LFPMontage")
    trends = bool(((nd.get("DiagnosticData") or {}).get("LFPTrendLogs")))
    snaps = len(_as_list((nd.get("DiagnosticData") or {}).get("LfpFrequencySnapshotEvents")))
    _check(rep, "Chronic trends & snapshots", "ok" if trends or snaps else "missing",
           f"{'LFPTrendLogs' if trends else 'no trend logs'} · {snaps} event snapshot(s)")
    imp_n = 0
    for it in _as_list(nd.get("Impedance")):
        for hb in _as_list(it.get("Hemisphere") if isinstance(it, dict) else None):
            imp_n += len(_as_list(((hb or {}).get("SessionImpedance") or {}).get("Monopolar")))
    _check(rep, "Impedance", "ok" if imp_n else "missing", f"{imp_n} monopolar values" if imp_n else "no impedance results")
    groups = len(_as_list((nd.get("Groups") or {}).get("Final"))) + len(_as_list((nd.get("Groups") or {}).get("Initial")))
    leads = len(_as_list((nd.get("LeadConfiguration") or {}).get("Final")) or _as_list((nd.get("LeadConfiguration") or {}).get("Initial")))
    _check(rep, "Leads & therapy groups", "ok" if groups or leads else "missing", f"{leads} lead(s) · {groups} group(s)")
    rep["capabilities"] = _capabilities(len(good), sweeps, max_steps, montage, trends, imp_n > 0, groups)
    if good:
        rep["status"], rep["ok"] = "ready", True
        rep["summary"] = f"{len(good)} LFP signal(s) ready for the full pipeline"
    elif montage or trends or snaps:
        rep["status"], rep["ok"] = "partial", True
        rep["summary"] = "No time-domain LFP — survey / trend objects only"
        rep["warnings"].append("Signal objects (LFP, PSD, spectrogram, game theory, risk, options, GSP) will be empty")
    else:
        rep["errors"].append("This Percept file contains no LFP signals, survey spectra or trend logs that the pipeline can analyse.")
        rep["summary"] = "Nothing to analyse"


def _verify_bravo(docs: list[tuple[str, dict]], rep: dict) -> None:
    fixes: list[str] = []
    try:
        nd = bravo_to_percept(docs, fixes)
    except Exception as e:  # malformed Bravo structure
        rep["errors"].append(f"Bravo structure could not be read: {e}")
        rep["summary"] = "Unreadable titration export"
        return
    rep["fixes"] = fixes + [f"Sampling rate not stored in the file — assumed {ASSUMED_FS_BRAVO:g} Hz (Percept)"]
    tds = nd["BrainSenseTimeDomain"]
    contacts = sorted({str(d.get("Contact", "")) for _, d in docs})
    if tds:
        durs = [len(x["TimeDomainData"]) / ASSUMED_FS_BRAVO for x in tds]
        hemis = sorted({_hemi_of_channel(x["Channel"]) for x in tds})
        _check(rep, "Time-domain LFP", "ok", f"{len(tds)} channel(s) · {'/'.join(hemis)} · {min(durs):.0f}–{max(durs):.0f} s · stim contact(s) {', '.join(contacts)}")
    else:
        _check(rep, "Time-domain LFP", "fail", "all Timeseries segments empty or non-numeric")
    steps = []
    for b in nd["BrainSenseLfp"]:
        side = next(iter(b["TherapySnapshot"]))
        steps.append(len({r[side]["mA"] for r in b["LfpData"]}))
    mx = max(steps) if steps else 0
    _check(rep, "Stimulation amplitude sweeps", "ok" if mx >= 2 else "missing", f"{len(steps)} titration(s), up to {mx} amplitude steps" if steps else "no amplitude segments with data")
    tp = sorted({str(d.get("TherapyParameters", "")) for _, d in docs})
    _check(rep, "Leads & therapy groups", "ok", f"4-contact ring lead (E00–E03) · {', '.join(tp)}")
    _check(rep, "Directional montage survey", "missing", "not part of titration exports")
    _check(rep, "Chronic trends & snapshots", "missing", "not part of titration exports")
    rep["capabilities"] = _capabilities(len(tds), sum(1 for s in steps if s >= 2), mx, 0, False, False, 1 if tds else 0)
    if tds:
        rep["status"], rep["ok"] = "ready", True
        rep["summary"] = f"Amplitude titration · {len(tds)} channel(s) ready for the pipeline"
    else:
        rep["errors"].append("The titration file has no non-empty numeric time series.")
        rep["summary"] = "No usable samples"


def detect_family(d: Any) -> str:
    if isinstance(d, list) and d and all(is_bravo(x) for x in d if isinstance(x, dict)):
        return "bravo"
    if is_bravo(d):
        return "bravo"
    if isinstance(d, dict):
        hits = sum(1 for k in PERCEPT_MARKERS if k in d)
        if hits >= 2 or any(k in d for k in TD_SECTIONS):
            return "percept"
        if set(d.keys()) <= {"GroupHistory"} and "GroupHistory" in d:
            return "group_history"
    return "unknown"


LABELS = {
    "percept": "Medtronic Percept session report",
    "bravo": "Amplitude titration export (Bravo-style)",
    "group_history": "Percept GroupHistory export",
    "unknown": "Unrecognised JSON",
    "invalid": "Not valid JSON",
}


def inspect_obj(d: Any, name: str = "data.json", size: int = 0) -> dict:
    t0 = time.perf_counter()
    fam = detect_family(d)
    rep = _report(fam, LABELS[fam], name, size)
    if fam == "percept":
        _verify_percept(d, rep)
    elif fam == "bravo":
        docs = [(name, x) for x in d] if isinstance(d, list) else [(name, d)]
        _verify_bravo(docs, rep)
    elif fam == "group_history":
        n = len(_as_list(d.get("GroupHistory")))
        _check(rep, "Therapy group history", "ok", f"{n} session(s) of group settings")
        _check(rep, "Time-domain LFP", "missing", "history-only export (no recordings)")
        rep["errors"].append("This is a GroupHistory-only export: it lists past therapy settings but contains no LFP data. "
                             "Load the matching Report_Json_Session_Report file instead.")
        rep["summary"] = "Therapy history only — no signals"
    else:
        keys = list(d.keys())[:8] if isinstance(d, dict) else []
        what = f"top-level keys: {', '.join(keys)}" if keys else f"top-level {type(d).__name__}"
        rep["errors"].append(f"No known DBS/LFP structure was found ({what}). Expected a Percept session report "
                             "(DeviceInformation, BrainSenseTimeDomain, LFPMontage …) or a titration export (Contact, data[].Timeseries_Data).")
        rep["summary"] = "Unrecognised structure"
    rep["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return rep


def _read_json_bytes(raw: bytes, name: str) -> tuple[Any, dict | None]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except Exception:
            rep = _report("invalid", LABELS["invalid"], name, len(raw))
            rep["errors"].append("The file is not text (binary data) — export it as JSON.")
            rep["summary"] = "Binary file"
            return None, rep
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        rep = _report("invalid", LABELS["invalid"], name, len(raw))
        rep["errors"].append(f"JSON syntax error at line {e.lineno}, column {e.colno}: {e.msg}.")
        rep["summary"] = "Invalid JSON"
        return None, rep


_CACHE: dict[str, Any] = {}  # path|mtime → parsed JSON (one entry: inspect then load reads the file once)


def _read_path(path: str | pathlib.Path) -> tuple[Any, dict | None, int]:
    p = pathlib.Path(path)
    if not p.exists():
        rep = _report("invalid", "File not found", p.name, 0)
        rep["errors"].append(f"File not found: {p}")
        rep["summary"] = "Missing file"
        return None, rep, 0
    key = f"{p.resolve()}|{p.stat().st_mtime_ns}"
    size = p.stat().st_size
    if key in _CACHE:
        return _CACHE[key], None, size
    d, err = _read_json_bytes(p.read_bytes(), p.name)
    if err is None:
        _CACHE.clear()
        _CACHE[key] = d
    return d, err, size


def inspect_path(path: str | pathlib.Path) -> dict:
    d, err, size = _read_path(path)
    if err:
        return err
    return inspect_obj(d, pathlib.Path(path).name, size)


def inspect_bytes(raw: bytes, name: str = "upload.json") -> dict:
    d, err = _read_json_bytes(raw, name)
    return err if err else inspect_obj(d, name, len(raw))


# ── loading ──────────────────────────────────────────────────────────────
def _parse(nd: dict, name: str, ftg_half_bw: float) -> dict:
    import nr_parser

    origins = nd.pop("_nr_td_origins", [])
    store = nr_parser.parse_percept_json(name, ftg_half_bw=ftg_half_bw, raw_dict=nd)
    payload = store["payload"]
    # tag recordings with their origin (stream_<idx>_… ids follow the BrainSenseTimeDomain order)
    for r in payload.get("recordings") or []:
        m = re.match(r"stream_(\d+)_", str(r.get("id", "")))
        if m and int(m.group(1)) < len(origins):
            org = origins[int(m.group(1))]
            r["origin"] = org
            if org != "BrainSense streaming":
                short = {"Indefinite streaming": "indefinite", "Sense channel test": "sense test", "Calibration test": "calibration"}.get(org, org.replace("Bravo titration · ", ""))
                r["label"] = f'{r.get("label", "")} · {short}'
    payload.setdefault("metadata", {})["input_format"] = nd.get("DataVersion", "")
    _clean_impedances(payload)
    _ring_lead_fix(store)
    return store


_VALID_CONTACTS = ("0", "1", "1a", "1b", "1c", "2", "2a", "2b", "2c", "3")


def _clean_impedances(payload: dict) -> None:
    """Drop impedance entries whose electrode names are not lead contacts (mock exports carry junk names)."""
    for h, imp in list((payload.get("impedances") or {}).items()):
        if not isinstance(imp, dict):
            continue
        mono = imp.get("monopolar")
        if isinstance(mono, dict):
            imp["monopolar"] = {c: v for c, v in mono.items() if str(c) in _VALID_CONTACTS or str(c) == "Case"}
        bip = imp.get("bipolar")
        if isinstance(bip, list):
            imp["bipolar"] = [b for b in bip if isinstance(b, dict) and str(b.get("e1")) in _VALID_CONTACTS and str(b.get("e2")) in _VALID_CONTACTS]


_SEG_RE = re.compile(r"^([12])[abc]$")


def _ring_lead_fix(store: dict) -> None:
    """4-ring (non-directional) leads have whole rings at levels 1 and 2.

    The Percept parser expands pair ends to SenSight segments (1 → 1a/1b/1c); for ring
    leads collapse every contact list back to ring tokens so selection, lead drawing,
    arms and impedance all speak the same contact names.
    """
    payload = store.get("payload") or {}
    leads = payload.get("leads") or {}
    ring = {h for h, ld in leads.items()
            if isinstance(ld, dict) and (ld.get("directional") is False
                                          or (ld.get("contacts") and not any(_SEG_RE.match(str(c)) for c in ld["contacts"])))}
    if not ring:
        return

    def collapse(cs: list) -> list:
        out: list = []
        for c in cs:
            c2 = _SEG_RE.sub(r"\1", str(c)) if isinstance(c, str) else c
            if c2 not in out:
                out.append(c2)
        return out

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            if o.get("hemisphere") in ring:
                for k in ("contacts", "stim_contacts"):
                    if isinstance(o.get(k), list):
                        o[k] = collapse(o[k])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for key, val in payload.items():
        if key not in ("raw_json", "raw", "json_tree"):
            walk(val)
    for h in ring:
        imp = (payload.get("impedances") or {}).get(h)
        if isinstance(imp, dict) and isinstance(imp.get("monopolar"), dict):
            mono: dict = {}
            for c, v in imp["monopolar"].items():
                c2 = _SEG_RE.sub(r"\1", str(c))
                if v and not mono.get(c2):
                    mono[c2] = v
                else:
                    mono.setdefault(c2, v)
            imp["monopolar"] = mono
        if isinstance(imp, dict) and isinstance(imp.get("bipolar"), list):
            seen, bip = set(), []
            for b in imp["bipolar"]:
                if not isinstance(b, dict):
                    continue
                e1, e2 = _SEG_RE.sub(r"\1", str(b.get("e1"))), _SEG_RE.sub(r"\1", str(b.get("e2")))
                if e1 == e2 or (e1, e2) in seen:
                    continue
                seen.add((e1, e2))
                bip.append({**b, "e1": e1, "e2": e2, "pair": f"{e1}-{e2}"})
            imp["bipolar"] = bip
    sel = store.get("selection")
    if isinstance(sel, dict) and isinstance(sel.get("contacts"), list):
        out = []
        for cid in sel["contacts"]:
            h, _, c = str(cid).partition(":")
            cid2 = f"{h}:{_SEG_RE.sub(r'\1', c)}" if h in ring else cid
            if cid2 not in out:
                out.append(cid2)
        order = {c: i for i, c in enumerate(_VALID_CONTACTS)}
        out.sort(key=lambda cid: (0 if str(cid).startswith("Left:") else 1, order.get(str(cid).partition(":")[2], 99)))
        sel["contacts"] = out


def load_obj(d: Any, name: str = "data.json", ftg_half_bw: float = 4.0, size: int = 0) -> tuple[dict, dict]:
    rep = inspect_obj(d, name, size)
    if not rep["ok"]:
        raise ValueError(" ".join(rep["errors"]) or rep["summary"])
    if rep["format"] == "bravo":
        docs = [(name, x) for x in d] if isinstance(d, list) else [(name, d)]
        nd = bravo_to_percept(docs)
    else:
        nd = normalize_percept(d)
    store = _parse(nd, name, ftg_half_bw)
    store["payload"].setdefault("metadata", {})["input_report"] = {k: rep[k] for k in ("format_label", "status", "summary")}
    return store, rep


def load_path(path: str | pathlib.Path, ftg_half_bw: float = 4.0) -> tuple[dict, dict]:
    d, err, size = _read_path(path)
    if err:
        raise ValueError(" ".join(err["errors"]))
    return load_obj(d, pathlib.Path(path).name, ftg_half_bw, size)


def load_bytes(raw: bytes, name: str = "upload.json", ftg_half_bw: float = 4.0) -> tuple[dict, dict]:
    d, err = _read_json_bytes(raw, name)
    if err:
        raise ValueError(" ".join(err["errors"]))
    return load_obj(d, name, ftg_half_bw, len(raw))


def bravo_group_key(path: str | pathlib.Path) -> str | None:
    """Participant|date key for a Bravo file (None if the file is not a Bravo export)."""
    d, err, _ = _read_path(path)
    if err or not is_bravo(d):
        return None
    return f'{d.get("ParticipantId", "")}|{str(d.get("Date", ""))[:10]}'


def load_paths(paths: list[str | pathlib.Path], ftg_half_bw: float = 4.0) -> tuple[dict, dict]:
    """Load several Bravo files of one participant/date as one bilateral session."""
    docs = []
    for p in paths:
        d, err, _ = _read_path(p)
        if err:
            raise ValueError(" ".join(err["errors"]))
        if not is_bravo(d):
            raise ValueError(f"{pathlib.Path(p).name} is not a titration export — only titration files can be combined.")
        docs.append(d)
    name = f"{pathlib.Path(paths[0]).parent.name} ({len(paths)} files)"
    return load_obj(docs, name, ftg_half_bw, sum(pathlib.Path(p).stat().st_size for p in paths))


def inspect_paths(paths: list[str | pathlib.Path]) -> dict:
    docs = []
    for p in paths:
        d, err, _ = _read_path(p)
        if err:
            return err
        docs.append(d)
    name = f"{pathlib.Path(paths[0]).parent.name} ({len(paths)} files)"
    return inspect_obj(docs, name, sum(pathlib.Path(p).stat().st_size for p in paths))


# ── sessions: one or more files of the same structure merged into one dataset ──
class MergeConflict(ValueError):
    """The added file contradicts data that is already loaded (nothing was changed)."""

    def __init__(self, name: str, conflicts: list[str]):
        self.name = name
        self.conflicts = conflicts
        super().__init__(f"{name} conflicts with the loaded data: " + "; ".join(conflicts[:3]) + (" …" if len(conflicts) > 3 else ""))


_KEY_FIELDS = ("FirstPacketDateTime", "DateTime", "Channel", "SensingElectrodes", "Hemisphere", "HemisphereLocation",
               "EventID", "EventName", "SessionDate")
_DATA_SECTIONS = ("BrainSenseTimeDomain", "BrainSenseLfp", "IndefiniteStreaming", "LfpMontageTimeDomain", "LFPMontage",
                  "BrainSenseSurveys", "BrainSenseSurveysTimeDomain", "SenseChannelTests", "CalibrationTests",
                  "MostRecentInSessionSignalCheck", "GroupHistory")
_NICE = {"BrainSenseTimeDomain": "streaming recording", "BrainSenseLfp": "streaming power trace", "IndefiniteStreaming": "indefinite stream",
         "LfpMontageTimeDomain": "survey time series", "LFPMontage": "montage pair", "SenseChannelTests": "sense test",
         "CalibrationTests": "calibration test", "GroupHistory": "group-history session", "EventLogs": "event",
         "LfpFrequencySnapshotEvents": "snapshot event", "MostRecentInSessionSignalCheck": "signal check",
         "BrainSenseSurveys": "survey", "BrainSenseSurveysTimeDomain": "survey time series"}


def _digest(o: Any) -> str:
    import hashlib
    return hashlib.md5(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def _item_key(it: Any) -> str:
    if isinstance(it, dict):
        k = [(f, str(it[f])) for f in _KEY_FIELDS if f in it and not isinstance(it[f], (dict, list))]
        if k:
            return json.dumps(k)
    return "#" + _digest(it)


def _describe(it: Any) -> str:
    if not isinstance(it, dict):
        return "item"
    bits = [str(it[f]).replace("HemisphereLocationDef.", "") for f in ("FirstPacketDateTime", "DateTime", "Channel", "SensingElectrodes", "Hemisphere") if f in it and not isinstance(it[f], (dict, list))]
    return " · ".join(bits[:3]) or "item"


def _merge_list(base: list, new: list, section: str, conflicts: list[str], added: dict, dup: list[int]) -> list:
    """Union of two entry lists. Identical entries are skipped; an entry whose identity (time stamp, channel …)
    matches exactly one existing entry but whose values differ is a conflict."""
    from collections import Counter

    digests = {_digest(x) for x in base}
    kb = Counter(_item_key(x) for x in base)
    kn = Counter(_item_key(x) for x in new)
    by_key = {_item_key(x): x for x in base}
    out = list(base)
    for it in new:
        dg = _digest(it)
        if dg in digests:
            dup[0] += 1
            continue
        k = _item_key(it)
        if kb.get(k) == 1 and kn.get(k) == 1 and not k.startswith("#"):
            conflicts.append(f"{section}: {_NICE.get(section, 'entry')} {_describe(by_key[k])} has different values in the two files")
            continue
        digests.add(dg)
        out.append(it)
        added[section] = added.get(section, 0) + 1
    return out


def _first(d: Any, *path: str) -> Any:
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def _serial(d: dict) -> str:
    for stage in ("Final", "Initial"):
        v = _first(d, "DeviceInformation", stage, "NeurostimulatorSerialNumber")
        if v:
            return str(v)
    return ""


def _lead_map(d: dict) -> dict:
    out = {}
    for ld in _as_list(_first(d, "LeadConfiguration", "Final")):
        if isinstance(ld, dict):
            out[_enum(ld.get("Hemisphere"))] = (_enum(ld.get("Model")), _enum(ld.get("LeadLocation")))
    return out


def _group_map(d: dict) -> dict:
    out = {}
    for g in _as_list(_first(d, "Groups", "Final")):
        if isinstance(g, dict):
            out[_enum(g.get("GroupId"))] = g
    return out


def _group_core(g: dict) -> str:
    return _digest({k: v for k, v in g.items() if k not in ("ActiveGroup", "GroupName", "GroupSettings")})


def _rate(g: dict) -> str:
    r = _first(g, "ProgramSettings", "RateInHertz")
    if r is None:
        r = _first(_as_list(_first(g, "ProgramSettings", "SensingChannel"))[0] if _as_list(_first(g, "ProgramSettings", "SensingChannel")) else {}, "RateInHertz")
    return f"{r} Hz" if r is not None else "settings"


def merge_percept(base: dict, new: dict, name: str) -> tuple[dict, dict]:
    """Merge a second Percept export into `base` (both shape-fixed). Raises MergeConflict on contradictions."""
    conflicts: list[str] = []
    warnings: list[str] = []
    added: dict[str, int] = {}
    dup = [0]
    sa, sb = _serial(base), _serial(new)
    if sa and sb and sa != sb:
        conflicts.append(f"different neurostimulator (serial {sa} vs {sb})")
    pa, pb = _first(base, "PatientInformation", "Final") or {}, _first(new, "PatientInformation", "Final") or {}
    for f in ("PatientId", "PatientFirstName", "PatientLastName", "PatientDateOfBirth"):
        va, vb = str(pa.get(f) or "").strip() if isinstance(pa, dict) else "", str(pb.get(f) or "").strip() if isinstance(pb, dict) else ""
        if va and vb and va != vb:
            conflicts.append(f"different patient ({f})")
            break
    la, lb = _lead_map(base), _lead_map(new)
    for h in sorted(set(la) & set(lb)):
        if la[h] != lb[h]:
            conflicts.append(f"{h} lead differs ({'/'.join(la[h])} vs {'/'.join(lb[h])})")
    ga, gb = _group_map(base), _group_map(new)
    for gid in sorted(set(ga) & set(gb)):
        if _group_core(ga[gid]) != _group_core(gb[gid]):
            ra, rb = _rate(ga[gid]), _rate(gb[gid])
            what = f"{ra} vs {rb}" if ra != rb else "amplitude, pulse width or contacts differ"
            conflicts.append(f"therapy {gid.replace('GROUP_', 'Group ')} is programmed differently ({what})")

    out = dict(base)
    for key, val in new.items():
        if key in _DATA_SECTIONS or (isinstance(val, list) and isinstance(out.get(key), list) and key not in ("Impedance",)):
            out[key] = _merge_list(_as_list(out.get(key)), _as_list(val), key, conflicts, added, dup)
        elif key == "DiagnosticData" and isinstance(val, dict):
            dd = dict(out.get(key) or {})
            for k2, v2 in val.items():
                if k2 == "LFPTrendLogs" and isinstance(v2, dict):
                    tl = {h: dict(days) for h, days in (dd.get(k2) or {}).items() if isinstance(days, dict)}
                    for h, days in v2.items():
                        if not isinstance(days, dict):
                            continue
                        tgt = tl.setdefault(h, {})
                        for day, pts in days.items():
                            before = len(_as_list(tgt.get(day)))
                            tgt[day] = _merge_list(_as_list(tgt.get(day)), _as_list(pts), "LFPTrendLogs", conflicts, {}, dup)
                            added["LFPTrendLogs"] = added.get("LFPTrendLogs", 0) + len(tgt[day]) - before
                    dd[k2] = tl
                elif isinstance(v2, list):
                    dd[k2] = _merge_list(_as_list(dd.get(k2)), v2, k2, conflicts, added, dup)
                elif k2 not in dd:
                    dd[k2] = v2
                    added[k2] = added.get(k2, 0) + 1
            out[key] = dd
        elif key == "Groups" and isinstance(val, dict):
            g = dict(out.get(key) or {})
            fin = _as_list(g.get("Final"))
            have = {_enum(x.get("GroupId")) for x in fin if isinstance(x, dict)}
            for x in _as_list(val.get("Final")):
                if isinstance(x, dict) and _enum(x.get("GroupId")) not in have:
                    fin.append(dict(x, ActiveGroup=False) if fin else x)
                    added["Groups"] = added.get("Groups", 0) + 1
            g["Final"] = fin
            if not g.get("Initial") and val.get("Initial"):
                g["Initial"] = val["Initial"]
            out[key] = g
        elif key == "Impedance":
            if not out.get(key) and val:
                out[key] = val
                added["Impedance"] = 1
            elif val and _digest(out.get(key)) != _digest(val):
                warnings.append(f"{name}: impedance was measured again with different values — the first file's impedance is kept")
        elif key == "LeadConfiguration" and isinstance(val, dict):
            lc = dict(out.get(key) or {})
            fin = _as_list(lc.get("Final"))
            for x in _as_list(val.get("Final")):
                if isinstance(x, dict) and _enum(x.get("Hemisphere")) not in la:
                    fin.append(x)
                    added["LeadConfiguration"] = added.get("LeadConfiguration", 0) + 1
            lc["Final"] = fin
            out[key] = lc
        elif key == "SessionDate" and val and out.get(key):
            out[key] = min(str(out[key]), str(val))
        elif key == "SessionEndDate" and val and out.get(key):
            out[key] = max(str(out[key]), str(val))
        elif key not in out or out.get(key) in (None, "", [], {}):
            if val not in (None, "", [], {}):
                out[key] = val
                if isinstance(val, (list, dict)):
                    added[key] = added.get(key, 0) + 1
    if conflicts:
        raise MergeConflict(name, conflicts)
    return out, {"added": added, "duplicates": dup[0], "warnings": warnings}


def _bravo_slot(d: dict) -> str:
    return f'{d.get("Contact", "")}|{d.get("TherapyParameters", "")}|{str(d.get("Date", ""))[:10]}'


def merge_bravo(base_docs: list[tuple[str, dict]], new_docs: list[tuple[str, dict]], name: str):
    conflicts: list[str] = []
    pids = {str(d.get("ParticipantId", "")) for _, d in base_docs if d.get("ParticipantId")}
    slots = {_bravo_slot(d): _digest(d) for _, d in base_docs}
    out, n_add, dup = list(base_docs), 0, 0
    for nm, d in new_docs:
        pid = str(d.get("ParticipantId", ""))
        if pid and pids and pid not in pids:
            conflicts.append(f"{nm}: different participant ({pid[:8]} vs {next(iter(pids))[:8]})")
            continue
        sl = _bravo_slot(d)
        if sl in slots:
            if slots[sl] == _digest(d):
                dup += 1
            else:
                conflicts.append(f"{nm}: {d.get('Contact', '')} · {d.get('TherapyParameters', '')} on {str(d.get('Date', ''))[:10]} is already loaded with different data")
            continue
        slots[sl] = _digest(d)
        out.append((nm, d))
        n_add += 1
    if conflicts:
        raise MergeConflict(name, conflicts)
    return out, {"added": {"titration file": n_add} if n_add else {}, "duplicates": dup, "warnings": []}


def _expand(name: str, d: Any) -> list[tuple[str, dict]]:
    return [(name, x) for x in d] if isinstance(d, list) else [(name, d)]


def new_session(items: list[tuple[str, Any, int]]) -> dict:
    """items: (name, parsed JSON, size). One Percept file, or one/several titration files."""
    fam = detect_family(items[0][1])
    docs = [x for nm, d, _ in items for x in _expand(nm, d)]
    sources = [{"name": nm, "size": _fmt_size(sz) if sz else "", "added": "initial file"} for nm, _, sz in items]
    if fam == "percept":
        return {"family": fam, "doc": items[0][1], "fixed": False, "sources": sources, "size": items[0][2]}
    return {"family": fam, "docs": docs, "sources": sources, "size": sum(sz for _, _, sz in items)}


def session_label(sess: dict) -> str:
    src = sess.get("sources") or []
    if not src:
        return "data.json"
    if len(src) == 1:
        return src[0]["name"]
    return f'{src[0]["name"]} + {len(src) - 1} more'


def session_obj(sess: dict) -> Any:
    return sess["doc"] if sess["family"] == "percept" else [d for _, d in sess["docs"]]


def inspect_session(sess: dict) -> dict:
    rep = inspect_obj(session_obj(sess), session_label(sess), sess.get("size", 0))
    rep["sources"] = sess.get("sources", [])
    return rep


def load_session(sess: dict, ftg_half_bw: float = 4.0) -> tuple[dict, dict]:
    store, rep = load_obj(session_obj(sess), session_label(sess), ftg_half_bw, sess.get("size", 0))
    rep["sources"] = sess.get("sources", [])
    store["payload"].setdefault("metadata", {})["sources"] = [s["name"] for s in sess.get("sources", [])]
    return store, rep


def _summarise_added(info: dict) -> str:
    parts = []
    for k, n in (info.get("added") or {}).items():
        if not n:
            continue
        if k == "LFPTrendLogs":
            parts.append(f"+{n} trend point(s)")
        else:
            parts.append(f"+{n} {_NICE.get(k, k)}{'s' if n > 1 and not _NICE.get(k, k).endswith('s') else ''}")
    return ", ".join(parts)


def add_to_session(sess: dict, items: list[tuple[str, Any, int]]) -> tuple[dict, dict]:
    """Merge more files into a session. Raises MergeConflict (session unchanged) on contradictions.

    Returns (new session, info) where info = {added, duplicates, warnings, summary, nothing_new}.
    """
    fam0 = sess["family"]
    new = {k: (list(v) if isinstance(v, list) else v) for k, v in sess.items()}
    new["sources"] = [dict(x) for x in sess.get("sources", [])]
    total = {"added": {}, "duplicates": 0, "warnings": []}
    for nm, d, sz in items:
        fam = detect_family(d)
        if fam == "invalid" or fam == "unknown":
            raise MergeConflict(nm, ["not a recognised DBS/LFP JSON structure"])
        if fam0 == "percept" and fam == "group_history":
            d = {"GroupHistory": d.get("GroupHistory")}
            fam = "percept"
        if fam != fam0:
            art = lambda t: ("an " if t[:1].lower() in "aeiou" else "a ") + t
            raise MergeConflict(nm, [f"different structure: the loaded data is {art(LABELS[fam0])}, this file is {art(LABELS[fam])}"])
        if fam0 == "percept":
            # a file without signals may still add montage / trends / groups / history
            base = new["doc"] if new.get("fixed") else shape_fix(new["doc"])
            merged, info = merge_percept(base, shape_fix(d), nm)
            new["doc"], new["fixed"] = merged, True
        else:
            merged, info = merge_bravo(new["docs"], _expand(nm, d), nm)
            new["docs"] = merged
        for k, n in info["added"].items():
            total["added"][k] = total["added"].get(k, 0) + n
        total["duplicates"] += info["duplicates"]
        total["warnings"] += info["warnings"]
        summ = _summarise_added(info) or "nothing new (all entries already loaded)"
        new["sources"].append({"name": nm, "size": _fmt_size(sz) if sz else "", "added": summ})
        new["size"] = new.get("size", 0) + (sz or 0)
    total["summary"] = _summarise_added(total)
    total["nothing_new"] = not total["summary"]
    return new, total


def read_items_paths(paths: list[str | pathlib.Path]) -> list[tuple[str, Any, int]]:
    items = []
    for p in paths:
        d, err, size = _read_path(p)
        if err:
            raise ValueError(" ".join(err["errors"]))
        items.append((pathlib.Path(p).name, d, size))
    return items


def read_item_bytes(raw: bytes, name: str) -> tuple[str, Any, int]:
    d, err = _read_json_bytes(raw, name)
    if err:
        raise ValueError(" ".join(err["errors"]))
    return (name, d, len(raw))


# ── folder scan ──────────────────────────────────────────────────────────
def scan_folder(folder: str | pathlib.Path, max_files: int = 400) -> list[dict]:
    """JSON files below `folder` (recursive) + one 'combine' entry per folder of ≥ 2 titration files."""
    root = pathlib.Path(folder)
    if not root.exists():
        return []
    files = []
    for p in sorted(root.rglob("*.json")):
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        rel = p.relative_to(root).as_posix()
        files.append({"path": str(p), "name": p.name, "rel": rel, "folder": str(p.parent.relative_to(root).as_posix()) if p.parent != root else "",
                      "size": _fmt_size(sz), "bytes": sz})
        if len(files) >= max_files:
            break
    # titration folders → offer "all files in folder" entries (detected by a cheap name/size heuristic, verified on inspect)
    by_folder: dict[str, list[dict]] = {}
    for f in files:
        by_folder.setdefault(f["folder"], []).append(f)
    combos = []
    for fol, fs in by_folder.items():
        cand = [f for f in fs if re.search(r"(left|right).*e\d{2}", f["name"], re.I)]
        if len(cand) >= 2:
            combos.append({"path": "|".join(f["path"] for f in cand), "name": f"All {len(cand)} titration files", "rel": f"{fol}/ (combined)",
                           "folder": fol, "size": _fmt_size(sum(f["bytes"] for f in cand)), "bytes": sum(f["bytes"] for f in cand), "combined": True})
    return combos + files
