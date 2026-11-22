"""Titra report builder — PDF (Typst), DOCX (python-docx), PNG (Typst raster) and ZIP.

Pure-pip toolchain, no LaTeX / Pandoc / Quarto / Cairo required:
    pip install typst python-docx

* PDF  — `nr_report.typ` compiled by the embedded Typst compiler (vector figures, Computer Modern).
* PNG  — the same compiler rasterises the DAG canvas SVG / the one-page infographic.
* DOCX — python-docx document with the infographic, canvas and every object as high-res PNGs
         plus native Word tables.

Entry point: `build_report(payload, request, out_dir=...)`. The `request` dict is produced by the
Titra DAG widget (Export ▾ menu) and holds the canvas SVG, per-object SVGs, the live analysis
picks, the selection and the DAG topology. It can also be assembled by hand (only `canvas_svg`
is needed for a canvas-only report).
"""

from __future__ import annotations

import datetime as _dt
import io
import json
import math
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
TEMPLATE = HERE / "nr_report.typ"

FORMATS = ("pdf", "docx", "png_canvas", "png_infographic", "svg", "zip", "obj_png", "obj_pdf")
MIN_TYPST = (0, 11)
MIME = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "png": "image/png",
    "svg": "image/svg+xml",
    "zip": "application/zip",
    "tex": "application/x-tex",
    "txt": "text/plain",
}

CRIMSON = (0xB8, 0x1D, 0x33)
INK = (0x11, 0x13, 0x17)
GRAY = (0x4B, 0x55, 0x63)

_PX_TO_PT = 0.75  # CSS px → pt (96 dpi)
_MAX_RASTER_PX = 7000


@dataclass
class ReportResult:
    files: list[tuple[str, str, bytes]] = field(default_factory=list)  # (name, mime, bytes)
    saved: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)  # format → error message (partial failures)


# ── helpers ──────────────────────────────────────────────────────────────
def pip_command(*pkgs: str) -> str:
    """The exact pip command for THIS kernel's Python (avoids installing into the wrong environment)."""
    pkgs = pkgs or ("typst", "python-docx")
    return f'"{sys.executable}" -m pip install -U {" ".join(pkgs)}'


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", str(v))[:3]) or (0,)


def _require_typst():
    import importlib

    importlib.invalidate_caches()  # see packages installed while the kernel is running
    try:
        import typst
    except ImportError as e:
        raise RuntimeError(
            "The Typst compiler is not installed in the Python that runs this notebook "
            f"({sys.executable}). Use Export ▾ → 'Install / update export tools', or run:\n{pip_command()}"
        ) from e
    ver = getattr(typst, "__version__", "0")
    if _version_tuple(ver) < MIN_TYPST:
        raise RuntimeError(
            f"typst {ver} is too old (need ≥ {'.'.join(map(str, MIN_TYPST))}; it lacks PNG output and "
            "template inputs). Use Export ▾ → 'Install / update export tools' and restart the kernel, "
            f"or run:\n{pip_command()}"
        )
    return typst


def _clean_json(o: Any) -> Any:
    """Replace NaN/inf (invalid JSON for Typst) with None, recursively."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_json(v) for v in o]
    return o


def _fmt(v: Any, nd: int = 1, unit: str = "") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    if isinstance(v, (int, float)):
        s = f"{v:.{nd}f}" if isinstance(v, float) or nd else str(v)
        return f"{s} {unit}".strip()
    return str(v)


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _frac(v: Any, lo: float, span: float) -> float:
    return max(0.0, min(1.0, (v - lo) / span)) if _finite(v) else 0.0


def _trim(s: Any, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _date(s: Any) -> str:
    if not s:
        return "—"
    try:
        return _dt.datetime.fromisoformat(str(s).replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(s)


def _svg_size(svg: str) -> tuple[float, float]:
    head = svg[: svg.find(">", svg.find("<svg")) + 1] if "<svg" in svg else svg[:600]
    w = re.search(r'\bwidth="([\d.]+)', head)
    h = re.search(r'\bheight="([\d.]+)', head)
    if w and h:
        return float(w.group(1)), float(h.group(1))
    vb = re.search(r'viewBox="[\d.\-]+\s+[\d.\-]+\s+([\d.]+)\s+([\d.]+)', head)
    if vb:
        return float(vb.group(1)), float(vb.group(2))
    return 1200.0, 800.0


def _png_size(png: bytes) -> tuple[int, int]:
    """Width/height from a PNG IHDR chunk (no Pillow needed)."""
    if png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) >= 24:
        return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    return 2000, 1000


def _prep_svg(svg: str) -> str:
    """Make widget SVG self-sufficient for Typst: Computer Modern first, no XML prolog issues."""
    svg = svg.replace("KaTeX_Main", "New Computer Modern")
    if "xmlns=" not in svg[:400]:
        svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    return svg


def _safe_stem(s: str) -> str:
    s = re.sub(r"\.json$", "", str(s or "session"), flags=re.I)
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")[:60] or "session"


# ── data model for the template ──────────────────────────────────────────
def _report_data(payload: dict, request: dict, fig_files: list[dict], canvas_file: str | None) -> dict:
    meta = payload.get("metadata") or {}
    cfg = payload.get("config") or {}
    sel = request.get("selection") or {}
    dag = request.get("dag") or {}
    recs = payload.get("recordings") or []

    stim = sel.get("stim_rate_hz") or cfg.get("default_stim_rate_hz") or 145.0
    hbw = sel.get("ftg_half_bw_hz") or cfg.get("ftg_half_bw_hz") or 4.0
    ftg_c = float(stim) / 2.0
    beta = cfg.get("beta_band") or [13, 30]

    # Session duration
    dur = "—"
    try:
        t0 = _dt.datetime.fromisoformat(str(meta.get("session_date")).replace("Z", "+00:00"))
        t1 = _dt.datetime.fromisoformat(str(meta.get("session_end_date")).replace("Z", "+00:00"))
        dur = f"{(t1 - t0).total_seconds() / 60:.0f} min"
    except Exception:
        pass

    # Streams (BrainSense time-domain)
    streams_raw = [r for r in recs if r.get("source") == "BrainSenseTimeDomain"][:14]
    dbs = [v for r in streams_raw for v in ((r.get("band_stats") or {}).get("beta_db"), (r.get("band_stats") or {}).get("ftg_db")) if _finite(v)]
    lo = (min(dbs) - 2.0) if dbs else -30.0
    hi = (max(dbs) + 1.0) if dbs else 0.0
    span = max(1e-6, hi - lo)
    streams = []
    _times = {str(r.get("datetime") or "")[11:16] for r in streams_raw}

    def _compact(cs: list) -> str:
        cs = [str(c) for c in (cs or []) if c and c != "Case"]
        out: list[str] = []
        for c in cs:
            m = re.match(r"^([12])[abc]$", c)
            if m and all(m.group(1) + x in cs for x in "abc"):
                if m.group(1) not in out:
                    out.append(m.group(1))
            elif c not in out:
                out.append(c)
        return "+".join(out)

    for r in streams_raw:
        bs = r.get("band_stats") or {}
        amp = r.get("amp_range_ma") or []
        amp_s = (f"{amp[-1]:g} mA" if len(amp) == 1 or (len(amp) == 2 and amp[0] == amp[1]) else f"{amp[0]:g}–{amp[-1]:g} mA") if amp else ""
        tm_full = str(r.get("datetime") or "")
        tm = tm_full[11:16] if len(tm_full) >= 16 else tm_full
        stim_c = _compact(r.get("stim_contacts") or [])
        # row tag: stim contact when streams differ by contact (titration), time when they differ by time
        tag = f"stim {stim_c}" if stim_c and (len(_times) <= 1 or str(r.get("origin", "")).startswith("Bravo")) else (tm or "—")
        streams.append({
            "label": r.get("label", r.get("id", "")),
            "time": tm or "—",
            "tag": tag,
            "stim": stim_c or "—",
            "hemisphere": r.get("hemisphere", ""),
            "pair": r.get("pair", ""),
            "stim_rate": _fmt(r.get("stim_rate_hz"), 0, "Hz"),
            "amp": amp_s,
            "beta_db": _fmt(bs.get("beta_db"), 1),
            "ftg_db": _fmt(bs.get("ftg_db"), 1),
            "beta_peak": _fmt(bs.get("beta_peak_hz"), 1, "Hz"),
            "ftg_peak": _fmt(bs.get("ftg_peak_hz"), 1, "Hz"),
            "ratio_db": _fmt(bs.get("ftg_beta_db_diff"), 1, "dB"),
            "rms_uv": _fmt(r.get("rms_uv"), 2),
            "duration": _fmt(r.get("duration_s"), 1),
            "beta_frac": _frac(bs.get("beta_db"), lo, span),
            "ftg_frac": _frac(bs.get("ftg_db"), lo, span),
        })

    # Dose response (titration recordings with ≥ 2 amplitude steps): β / FTG change from lowest to highest amplitude
    titration = []
    for r in streams_raw:
        st = [x for x in (r.get("amp_steps") or []) if _finite(x.get("beta_db")) and _finite(x.get("amp_ma"))]
        if len(st) < 2:
            continue
        st.sort(key=lambda x: x["amp_ma"])
        d_beta = st[-1]["beta_db"] - st[0]["beta_db"]
        d_ftg = (st[-1]["ftg_db"] - st[0]["ftg_db"]) if _finite(st[-1].get("ftg_db")) and _finite(st[0].get("ftg_db")) else None
        stim_c = _compact(r.get("stim_contacts") or [])
        titration.append({
            "row": f'{str(r.get("hemisphere", ""))[:1]} {r.get("pair", "")}' + (f" · stim {stim_c}" if stim_c else ""),
            "range": f'≤{st[-1]["amp_ma"]:g} mA',
            "range_long": f'{st[0]["amp_ma"]:g}–{st[-1]["amp_ma"]:g} mA · {len(st)} steps',
            "d_beta": f"{d_beta:+.1f}", "d_ftg": f"{d_ftg:+.1f}" if d_ftg is not None else "—",
            "_db": d_beta, "_df": d_ftg or 0.0,
        })
    mx_t = max([abs(t["_db"]) for t in titration] + [abs(t["_df"]) for t in titration] + [1e-9])
    for t in titration:
        t["beta_frac"] = min(1.0, abs(t.pop("_db")) / mx_t)
        t["ftg_frac"] = min(1.0, abs(t.pop("_df")) / mx_t)
    titration = titration[:8]

    # Directional survey
    montage = payload.get("montage") or []
    survey = [
        {
            "hemisphere": m.get("hemisphere", ""),
            "pair": m.get("pair", ""),
            "kind": m.get("kind", ""),
            "peak_hz": _fmt(m.get("peak_hz"), 2),
            "peak_uv": _fmt(m.get("peak_uv"), 3),
            "artifact": str(m.get("artifact", "")).replace("ARTIFACT_", "").replace("_", " ").lower(),
            "_uv": float(m.get("peak_uv")) if _finite(m.get("peak_uv")) else 0.0,
        }
        for m in montage
    ]
    survey.sort(key=lambda r: (r["hemisphere"], -r["_uv"]))
    survey_top = []
    for hemi in sorted({r["hemisphere"] for r in survey}):
        rows = [r for r in survey if r["hemisphere"] == hemi][:6]
        mx = max([r["_uv"] for r in rows] + [1e-9])
        survey_top.append({
            "hemisphere": hemi,
            "rows": [{"pair": r["pair"], "peak_uv": f'{r["_uv"]:.2f}', "frac": r["_uv"] / mx, "rank": i + 1} for i, r in enumerate(rows)],
        })
    for r in survey:
        r.pop("_uv", None)

    # Therapy groups (final)
    groups = []
    for g in [g for g in (payload.get("groups") or []) if g.get("stage") == "Final"]:
        lines = []
        for h, hp in (g.get("hemispheres") or {}).items():
            cath = ", ".join(f'{e["contact"]}({e["ma"]})' for e in hp.get("electrodes", []) if e.get("polarity") == "Negative")
            lines.append(f'{h}: −[{cath or "none"}] · {_fmt(hp.get("pw_us"), 0, "µs")} · {_fmt(hp.get("total_ma"), 1, "mA")} · sense {hp.get("sensing_pair") or "—"}')
        short = []
        for h, hp in (g.get("hemispheres") or {}).items():
            cath = " ".join(e["contact"] for e in hp.get("electrodes", []) if e.get("polarity") == "Negative")
            short.append(f'{h[0]}: −[{cath or "none"}] · {_fmt(hp.get("total_ma"), 1, "mA")} · {_fmt(hp.get("pw_us"), 0, "µs")}')
        groups.append({
            "short_lines": short,
            "group_id": str(g.get("group_id", "")).replace("GROUP_", "Group "),
            "active": bool(g.get("active")),
            "rate": _fmt(g.get("rate_hz"), 0, "Hz"),
            "ftg": _fmt(g.get("ftg_center_hz"), 1, "Hz"),
            "lines": lines,
        })

    # Impedance (monopolar) matrix: contact × hemisphere
    imp = payload.get("impedances") or {}
    hemis = [h for h in ("Left", "Right") if h in imp] + [h for h in imp if h not in ("Left", "Right")]
    contacts: list[str] = []
    for h in hemis:
        for c in (imp[h].get("monopolar") or {}):
            if c not in contacts:
                contacts.append(c)
    impedance = [[c] + [_fmt((imp[h].get("monopolar") or {}).get(c), 0) for h in hemis] for c in contacts]
    impedance_header = ["Contact"] + hemis

    events = [
        {
            "group": str(e.get("group", "")).replace("GROUP_", "Group "),
            "hemisphere": e.get("hemisphere", ""),
            "target": e.get("target", ""),
            "above": _fmt(e.get("above_pct"), 0),
            "between": _fmt(e.get("between_pct"), 0),
            "below": _fmt(e.get("below_pct"), 0),
            "avg_ma": _fmt(e.get("avg_ma"), 1),
        }
        for e in (payload.get("event_summary") or [])
    ]

    # KPI tiles
    leads = payload.get("leads") or {}
    def _is_ring(v: dict) -> bool:
        cs = v.get("contacts") or []
        return v.get("directional") is False or (bool(cs) and not any(re.match(r"^[12][abc]$", str(c)) for c in cs))

    def _lead_name(v: dict) -> str:
        m = str(v.get("model", "")).replace("LEAD_", "").replace("LeadModelDef.", "")
        if _is_ring(v):
            return "4-ring" if not m or m == "RING_4_CONTACT" else f"{m} (4-ring)"
        return f"SenSight {m}" if m else "SenSight"

    lead_models = sorted({_lead_name(v) for v in leads.values()})
    lead_note = " · ".join(
        f'{h[0]} {v.get("target", "")}' + ("" if _is_ring(v) else f' {v.get("orientation_deg", "")}°') for h, v in sorted(leads.items()))
    n_streams = len([r for r in recs if r.get("source") == "BrainSenseTimeDomain"])
    best = None
    for r in streams_raw:
        b = (r.get("band_stats") or {}).get("beta_db")
        if _finite(b) and (best is None or b > best[0]):
            best = (b, r)
    kpis = [
        {"label": "Device", "value": meta.get("device_name", "—"), "note": f'{meta.get("device_model", "")} · battery {meta.get("battery_pct", "—")}%'},
        {"label": "Diagnosis", "value": re.sub(r"(?<!^)(?=[A-Z])", " ", str(meta.get("diagnosis", "—"))).replace("Parkinsons", "Parkinson's"), "note": f'Implant {_date(meta.get("implant_date"))[:10]}'},
        {"label": "Session", "value": _date(meta.get("session_date"))[:10], "note": f"{dur} · programmer {meta.get('programmer_version', '')}"},
        {"label": "Leads", "value": f'{len(leads)} × {"/".join(lead_models) or "—"}', "note": lead_note},
        {"label": "Stimulation", "value": f"{float(stim):.0f}", "unit": "Hz", "note": f"FTG {ftg_c:.1f} ± {float(hbw):.1f} Hz", "accent": True},
        {"label": "Recordings", "value": str(len(recs)), "note": f"{n_streams} streams · {len(recs) - n_streams} survey"},
        {
            "label": "Strongest β stream",
            "value": f'{best[1].get("hemisphere", "")} {best[1].get("pair", "")}' if best else "—",
            "note": f'{best[0]:.1f} dB @ {(best[1].get("band_stats") or {}).get("beta_peak_hz", "—")} Hz' if best else "",
            "accent": bool(best),
        },
        {"label": "Analysis DAG", "value": f'{dag.get("n_nodes", len(dag.get("nodes") or []))} objects', "note": f'{dag.get("n_edges", len(dag.get("edges") or []))} edges · {dag.get("preset", "")}'},
    ]
    for k in kpis:
        k["label"] = k["label"].upper().replace("Β", "β")
        k["note"] = _trim(k.get("note", ""), 30)

    # Analysis (live picks computed in the widget)
    an = request.get("analysis") or {}
    analysis_tiles, analysis_tables = [], []
    gt = an.get("gt") or {}
    if gt.get("picks"):
        p0 = gt["picks"][0]
        analysis_tiles.append({"label": f'GAME THEORY · {str(gt.get("hemi", "")).upper()}', "value": p0.get("arm", "—"), "note": f'{p0.get("method", "")} {p0.get("value", "")}'.strip(), "accent": True})
        analysis_tables.append({
            "title": f'Game-theory consensus ({gt.get("hemi", "")})',
            "note": gt.get("note", ""),
            "columns": [1, 1, 1],
            "header": ["Method", "Recommended arm", "Value"],
            "rows": [[p.get("method", ""), p.get("arm", ""), p.get("value", "")] for p in gt["picks"]],
        })
    for key, label in (("risk", "Extreme-value risk"), ("options", "Option pricing")):
        blk = an.get(key) or {}
        if blk.get("rules"):
            analysis_tiles.append({"label": f'{label.upper()} · {str(blk.get("hemi", "")).upper()}', "value": blk.get("majority", "—"), "note": blk.get("note", "")})
            analysis_tables.append({
                "title": f'{label} — rule-based selection ({blk.get("hemi", "")})',
                "note": blk.get("note", ""),
                "columns": [1, 1, 1],
                "header": ["Rule", "Selected arm", "Value"],
                "rows": [[r.get("name", ""), r.get("arm", "—"), r.get("val", "")] for r in blk["rules"]],
            })
    gsp = an.get("gsp") or {}
    if gsp.get("rows"):
        analysis_tables.append({"title": "Graph signal processing KPIs", "note": gsp.get("note", ""), "columns": [1, 1], "header": ["KPI", "Value"], "rows": gsp["rows"]})

    # Selection rows
    def _lst(v, n=24):
        v = list(v or [])
        return ", ".join(map(str, v[:n])) + (f" … (+{len(v) - n})" if len(v) > n else "") if v else "—"

    def _params(p):
        return " · ".join(f"{k}={v}" for k, v in (p or {}).items() if k not in ("hemisphere",)) or "—"

    selection_rows = [
        ["Hemisphere", str(sel.get("hemisphere", "—"))],
        ["Contacts", f'{len(sel.get("contacts") or [])}: {_lst(sel.get("contacts"))}'],
        ["Bipolar pairs", f'{len(sel.get("pairs") or [])}: {_lst(sel.get("pairs"), 12)}'],
        ["Recordings", str(len(sel.get("recordings") or []))],
        ["Stimulation rate / FTG Δf", f"{float(stim):.0f} Hz / ±{float(hbw):.1f} Hz"],
        ["Selected objects", _lst(sel.get("selected_node_titles") or sel.get("selected_nodes"))],
        ["Game-theory parameters", _params(sel.get("gt_params"))],
        ["Risk parameters", _params(sel.get("risk_params"))],
        ["Option parameters", _params(sel.get("opt_params"))],
        ["GSP parameters", _params(sel.get("gsp_params"))],
    ]

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return {
        "meta": {
            "title": request.get("title") or "DBS Sensing & Programming Report",
            "subtitle": "DBS LFP analysis report",
            "file_name": meta.get("file_name", "session"),
            "sources": ", ".join(str(x) for x in (meta.get("sources") or [])) if len(meta.get("sources") or []) > 1 else "",
            "device": meta.get("device_name", "Percept"),
            "device_model": meta.get("device_model", ""),
            "diagnosis": re.sub(r"(?<!^)(?=[A-Z])", " ", str(meta.get("diagnosis", ""))).replace("Parkinsons", "Parkinson's"),
            "session_date": _date(meta.get("session_date")),
            "generated_at": now,
            "footnote": (
                f"β band {beta[0]:g}–{beta[1]:g} Hz. Finely-tuned gamma (FTG) band = f_stim/2 ± Δf "
                f"({ftg_c - float(hbw):.1f}–{ftg_c + float(hbw):.1f} Hz at {float(stim):.0f} Hz). "
                "Band power from Welch PSD of 250 Hz BrainSense streams; survey peaks from the directional "
                "LFP montage. Decision-analysis picks reflect the live widget parameters at export time and "
                "are research outputs, not clinical recommendations."
            ),
        },
        "config": {
            "beta_band": f"{beta[0]:g}–{beta[1]:g} Hz",
            "ftg_band": f"{ftg_c:.1f} ± {float(hbw):.1f} Hz",
        },
        "scales": {"db_lo": f"{lo:.0f}", "db_hi": f"{hi:.0f}"},
        "kpis": kpis,
        "streams": streams,
        "survey": survey,
        "survey_top": survey_top,
        "titration": titration,
        "groups": groups,
        "impedance": impedance,
        "impedance_header": impedance_header,
        "events": events,
        "analysis_tiles": analysis_tiles[:4],
        "analysis_tables": analysis_tables,
        "selection_rows": selection_rows,
        "dag": {
            "preset": dag.get("preset", ""),
            "n_nodes": dag.get("n_nodes", len(dag.get("nodes") or [])),
            "n_edges": dag.get("n_edges", len(dag.get("edges") or [])),
            "edges": [{"source": e.get("source", ""), "target": e.get("target", ""), "label": e.get("label", "")} for e in (dag.get("edges") or [])],
        },
        "figures": fig_files,
        "canvas": {"file": canvas_file} if canvas_file else None,
    }


# ── compile steps ────────────────────────────────────────────────────────
def _write_assets(work: pathlib.Path, request: dict) -> tuple[list[dict], str | None, list[dict]]:
    """Write SVGs into the build dir. Returns (figure entries, canvas filename, png raster items)."""
    fig_entries, png_items = [], []
    canvas_file = None
    if request.get("canvas_svg"):
        svg = _prep_svg(request["canvas_svg"])
        (work / "canvas.svg").write_text(svg, encoding="utf-8")
        w, h = _svg_size(svg)
        canvas_file = "canvas.svg"
        scale = min(2.0, _MAX_RASTER_PX / max(w, h, 1.0))
        png_items.append({"key": "canvas", "file": canvas_file, "w_pt": w * _PX_TO_PT * scale / 2.0})
    for i, ob in enumerate(request.get("objects") or []):
        if not ob.get("svg"):
            continue
        svg = _prep_svg(ob["svg"])
        fn = f"fig_{i:02d}.svg"
        (work / fn).write_text(svg, encoding="utf-8")
        w, h = _svg_size(svg)
        avail_w, max_h = 180.0, 118.0  # mm on A4 portrait
        disp_w = min(avail_w, max_h * w / max(h, 1.0))
        fig_entries.append({"file": fn, "title": ob.get("title", ""), "badge": ob.get("badge", ""), "desc": ob.get("desc", ""), "disp_w_mm": round(disp_w, 1), "id": ob.get("id", "")})
        png_items.append({"key": f"fig_{i:02d}", "file": fn, "w_pt": w * _PX_TO_PT, "title": ob.get("title", ""), "desc": ob.get("desc", "")})
    return fig_entries, canvas_file, png_items


def _raster(typst, work: pathlib.Path, items: list[dict], ppi: float = 192.0) -> dict[str, bytes]:
    """Rasterise several SVGs in ONE Typst compile (one page per SVG) → {key: png}."""
    if not items:
        return {}
    (work / "raster.json").write_text(json.dumps(_clean_json(items)), encoding="utf-8")
    (work / "raster.typ").write_text(
        '#set page(width: auto, height: auto, margin: 0pt, fill: white)\n'
        '#let items = json("raster.json")\n'
        '#for (i, it) in items.enumerate() {\n'
        '  if i > 0 { pagebreak() }\n'
        '  image(it.file, width: it.w_pt * 1pt)\n'
        '}\n',
        encoding="utf-8",
    )
    out = typst.compile(str(work / "raster.typ"), root=str(work), format="png", ppi=ppi)
    pages = out if isinstance(out, list) else [out]
    return {it["key"]: pg for it, pg in zip(items, pages)}


def _write_data(work: pathlib.Path, data: dict, mode: str) -> None:
    payload = dict(data, mode=mode)
    (work / "data.json").write_text(json.dumps(_clean_json(payload), ensure_ascii=False, default=str, allow_nan=False), encoding="utf-8")


# Static Titra brand mark for reports (the studio shows the animated version)
TITRA_ICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32">'
    '<circle cx="16" cy="18.2" r="9" fill="none" stroke="#B81D33" stroke-width="1.4" opacity="0.35"/>'
    '<circle cx="16" cy="18.2" r="6" fill="none" stroke="#B81D33" stroke-width="1.6" opacity="0.7"/>'
    '<rect x="12.2" y="1.5" width="7.6" height="28.5" rx="3.8" fill="#1F242C"/>'
    '<rect x="11.4" y="22.6" width="9.2" height="3.2" rx="1.1" fill="#D4D8DF"/>'
    '<rect x="11.4" y="16.6" width="9.2" height="3.2" rx="1.1" fill="#B81D33"/>'
    '<rect x="11.4" y="10.6" width="9.2" height="3.2" rx="1.1" fill="#D4D8DF"/>'
    '<rect x="11.4" y="4.6" width="9.2" height="3.2" rx="1.1" fill="#D4D8DF"/>'
    '</svg>'
)


def _compile_report(typst, work: pathlib.Path, data: dict, mode: str, fmt: str = "pdf", ppi: float | None = None):
    # The template mode travels inside data.json (no sys_inputs → works on every typst-py ≥ 0.11)
    _write_data(work, data, mode)
    (work / "titra_icon.svg").write_text(TITRA_ICON_SVG, encoding="utf-8")
    kw: dict[str, Any] = {"root": str(work)}
    if fmt == "png":
        kw.update(format="png", ppi=ppi or 200.0)
    try:
        return typst.compile(str(work / "report.typ"), **kw)
    except Exception as e:  # surface the Typst diagnostic text
        raise RuntimeError(f"Typst failed to compile the {mode} template: {e}") from e


# ── DOCX ─────────────────────────────────────────────────────────────────
def _build_docx(data: dict, infographic_png: bytes | None, rasters: dict[str, bytes], png_items: list[dict]) -> bytes:
    try:
        from docx import Document
        from docx.enum.section import WD_ORIENT, WD_SECTION
        from docx.enum.table import WD_TABLE_ALIGNMENT
        from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.shared import Cm, Pt, RGBColor
    except ImportError as e:
        raise RuntimeError(
            f"Word export needs python-docx in {sys.executable}. Use Export ▾ → 'Install / update export tools', "
            f"or run:\n{pip_command('python-docx')}"
        ) from e

    crimson, ink, gray = RGBColor(*CRIMSON), RGBColor(*INK), RGBColor(*GRAY)
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Cambria"
    st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Cambria")
    for lvl, size in ((1, 15), (2, 12)):
        hs = doc.styles[f"Heading {lvl}"]
        hs.font.name = "Cambria"
        hs.font.size = Pt(size)
        hs.font.bold = True
        hs.font.color.rgb = crimson if lvl == 1 else ink

    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(1.6)
    sec.top_margin = sec.bottom_margin = Cm(1.5)
    full_w = Cm(21.0 - 3.2)

    meta = data["meta"]
    # Footer
    fp = sec.footer.paragraphs[0]
    fr = fp.add_run(f'Titra · {meta["file_name"]} · generated {meta["generated_at"]}')
    fr.font.size, fr.font.color.rgb = Pt(8), gray

    def shade(cell, hex_fill):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_fill)
        tcPr.append(shd)

    def add_table(header, rows, widths=None):
        t = doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(header):
            c = t.rows[0].cells[i]
            c.text = ""
            r = c.paragraphs[0].add_run(str(h))
            r.bold, r.font.size, r.font.color.rgb = True, Pt(8.5), RGBColor(255, 255, 255)
            shade(c, "B81D33")
        for ri, row in enumerate(rows):
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                r = cells[i].paragraphs[0].add_run(str(v))
                r.font.size = Pt(8.5)
                if ri % 2 == 1:
                    shade(cells[i], "F7F8FA")
        if widths:
            t.autofit = False
            for i, w in enumerate(widths):
                t.columns[i].width = w
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = w
        doc.add_paragraph()
        return t

    doc.core_properties.title = meta["title"]
    doc.core_properties.author = "Titra"
    doc.core_properties.subject = f'{meta["file_name"]} · {meta["device"]} · session {meta["session_date"]}'

    def pic(blob, width):
        doc.add_picture(io.BytesIO(blob), width=width)
        pp = doc.paragraphs[-1]
        pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pp.paragraph_format.keep_with_next = True  # keep caption with its figure
        pp.paragraph_format.space_after = Pt(2)
        return pp

    if infographic_png:
        # Page 1 = the Typst infographic (it carries its own masthead) sized to fit one page
        iw, ih = _png_size(infographic_png)
        max_h_cm = 29.7 - 2 * 1.5 - 1.6
        w_cm = min(21.0 - 3.2, max_h_cm * iw / max(1, ih))
        doc.add_picture(io.BytesIO(infographic_png), width=Cm(w_cm))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.paragraphs[-1].paragraph_format.space_after = Pt(0)
    else:
        # No infographic image → plain title block + KPI table
        p = doc.add_paragraph()
        r = p.add_run("Titra")
        r.bold, r.font.size = True, Pt(22)
        r = p.add_run(f"   {meta['subtitle']}")
        r.font.size, r.font.color.rgb = Pt(11), gray
        p = doc.add_paragraph()
        r = p.add_run(meta["title"])
        r.bold, r.font.size = True, Pt(16)
        p = doc.add_paragraph()
        r = p.add_run(f'{meta["file_name"]} · {meta["device"]} {meta["device_model"]} · session {meta["session_date"]} · β {data["config"]["beta_band"]} · FTG {data["config"]["ftg_band"]}')
        r.font.size, r.font.color.rgb = Pt(9), gray
        add_table(["Indicator", "Value", "Note"], [[k["label"], f'{k["value"]} {k.get("unit", "")}'.strip(), k.get("note", "")] for k in data["kpis"]])

    # Canvas (landscape section)
    if "canvas" in rasters:
        ls = doc.add_section(WD_SECTION.NEW_PAGE)
        ls.orientation = WD_ORIENT.LANDSCAPE
        ls.page_width, ls.page_height = Cm(29.7), Cm(21.0)
        ls.left_margin = ls.right_margin = Cm(1.4)
        doc.add_heading("Analysis DAG canvas", level=1)
        p = doc.add_paragraph()
        r = p.add_run(f'Preset {data["dag"]["preset"]} · {data["dag"]["n_nodes"]} objects · {data["dag"]["n_edges"]} edges')
        r.font.size, r.font.color.rgb = Pt(9), gray
        pw, ph = _png_size(rasters["canvas"])
        ar = ph / max(1, pw)
        avail_w, avail_h = 29.7 - 2.8, 21.0 - 2 * 2.54 - 2.2
        w_cm = min(avail_w, avail_h / max(ar, 1e-6))
        pic(rasters["canvas"], Cm(w_cm))
        ps = doc.add_section(WD_SECTION.NEW_PAGE)
        ps.orientation = WD_ORIENT.PORTRAIT
        ps.page_width, ps.page_height = Cm(21.0), Cm(29.7)
        ps.left_margin = ps.right_margin = Cm(1.6)
    else:
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # Objects
    figs = [it for it in png_items if it["key"].startswith("fig_") and it["key"] in rasters]
    if figs:
        doc.add_heading("Analysis objects", level=1)
        for n, (it, fe) in enumerate(zip(figs, data["figures"]), start=1):
            pic(rasters[it["key"]], Cm(fe["disp_w_mm"] / 10.0))
            cp = doc.add_paragraph()
            cp.paragraph_format.space_after = Pt(10)
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = cp.add_run(f"Figure {n}. ")
            r.bold, r.font.size, r.font.color.rgb = True, Pt(8.5), crimson
            r = cp.add_run(f'{fe["title"]}. ')
            r.bold, r.font.size = True, Pt(8.5)
            r = cp.add_run(fe.get("desc", ""))
            r.font.size = Pt(8.5)

    # Analysis tables
    if data["analysis_tables"]:
        doc.add_heading("Decision analysis", level=1)
        for t in data["analysis_tables"]:
            doc.add_heading(t["title"], level=2)
            if t.get("note"):
                q = doc.add_paragraph()
                rr = q.add_run(t["note"])
                rr.font.size, rr.font.color.rgb = Pt(8.5), gray
            add_table(t["header"], t["rows"])

    # Appendix
    doc.add_heading("Data appendix", level=1)
    if data["meta"].get("sources"):
        q = doc.add_paragraph()
        rr = q.add_run(f'Merged from {len(data["meta"]["sources"].split(", "))} files: {data["meta"]["sources"]}')
        rr.font.size = Pt(9)
    if data["streams"]:
        doc.add_heading("BrainSense streaming recordings", level=2)
        add_table(["Recording", "Time", "β (dB)", "β peak", "FTG (dB)", "FTG peak", "FTG − β", "RMS (µV)", "Dur. (s)"],
                  [[s["label"], s["time"], s["beta_db"], s["beta_peak"], s["ftg_db"], s["ftg_peak"], s["ratio_db"], s["rms_uv"], s["duration"]] for s in data["streams"]],
                  widths=[Cm(4.6), Cm(1.7)] + [Cm(1.5)] * 7)
    if data["survey"]:
        doc.add_heading("Directional montage survey", level=2)
        add_table(["Hemisphere", "Pair", "Kind", "Peak (Hz)", "Peak (µV)", "Artifact"],
                  [[r["hemisphere"], r["pair"], r["kind"], r["peak_hz"], r["peak_uv"], r["artifact"]] for r in data["survey"]])
    if data.get("titration"):
        doc.add_heading("Dose response (titration)", level=2)
        add_table(["Recording", "Amplitude range", "Δβ (dB)", "ΔFTG (dB)"],
                  [[t["row"], t["range_long"], t["d_beta"], t["d_ftg"]] for t in data["titration"]])
    if data["groups"]:
        doc.add_heading("Therapy groups (final)", level=2)
        add_table(["Group", "Active", "Rate", "FTG", "Programming"],
                  [[g["group_id"], "yes" if g["active"] else "", g["rate"], g["ftg"], "\n".join(g["lines"])] for g in data["groups"]],
                  widths=[Cm(1.9), Cm(1.3), Cm(1.5), Cm(1.6), Cm(11.5)])
    if data["impedance"]:
        doc.add_heading("Monopolar impedance (Ω)", level=2)
        add_table(data["impedance_header"], data["impedance"])
    if data["events"]:
        doc.add_heading("Adaptive / event summary", level=2)
        add_table(["Group", "Hemisphere", "Target", "Above %", "Between %", "Below %", "Avg mA"],
                  [[e["group"], e["hemisphere"], e["target"], e["above"], e["between"], e["below"], e["avg_ma"]] for e in data["events"]])
    doc.add_heading("Selection & parameters synced from the DAG", level=2)
    add_table(["Item", "Value"], data["selection_rows"], widths=[Cm(4.5), Cm(13.3)])
    if data["dag"]["edges"]:
        doc.add_heading("DAG topology", level=2)
        add_table(["Source", "Target", "Edge"], [[e["source"], e["target"], e["label"]] for e in data["dag"]["edges"]])

    q = doc.add_paragraph()
    rr = q.add_run(meta["footnote"])
    rr.font.size, rr.font.color.rgb = Pt(8), gray

    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()


# ── public API ───────────────────────────────────────────────────────────
def build_report(
    payload: dict,
    request: dict,
    out_dir: str | pathlib.Path | None = None,
    template: str | pathlib.Path | None = None,
) -> ReportResult:
    """Build the requested export formats.

    request["formats"]: any of "pdf", "docx", "png_canvas", "png_infographic", "svg", "zip"
    (full report) or "obj_png" / "obj_pdf" (the single object in request["objects"]).
    Each format is built independently: if one fails the others are still returned (failures in
    `ReportResult.errors`, details in out_dir/export_errors.log). Raises RuntimeError only if
    nothing could be produced.
    """
    t0 = time.perf_counter()
    fmts = [f for f in (request.get("formats") or ["pdf"]) if f in FORMATS]
    if any(f.startswith("obj_") for f in fmts):
        return _build_object_export(request, [f for f in fmts if f.startswith("obj_")], out_dir, t0)
    if "zip" in fmts:
        fmts = ["zip"]
    want = set(fmts)
    if "zip" in want:
        want |= {"pdf", "docx", "png_canvas", "png_infographic", "svg"}

    meta = payload.get("metadata") or {}
    stem = f'titra_{_safe_stem(request.get("file_stem") or meta.get("file_name"))}_{_dt.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    res = ReportResult()
    produced: dict[str, tuple[str, str, bytes]] = {}
    extras: dict[str, bytes] = {}  # extra ZIP members (LaTeX source, object PNG/SVG)
    tracebacks: list[str] = []

    def attempt(key: str, fn):
        try:
            return fn()
        except Exception as e:
            res.errors[key] = f"{type(e).__name__}: {e}"
            tracebacks.append(f"[{key}]\n{traceback.format_exc()}")
            return None

    work = pathlib.Path(tempfile.mkdtemp(prefix="nr_report_"))
    try:
        fig_entries, canvas_file, png_items = _write_assets(work, request)
        data = _report_data(payload, request, fig_entries, canvas_file)
        shutil.copyfile(pathlib.Path(template) if template else TEMPLATE, work / "report.typ")

        typst = attempt("typst", _require_typst)
        if typst is None:
            # No Typst (e.g. the browser / WebAssembly build): PDF and the infographic are unavailable;
            # Word and the canvas PNG use figures the browser rendered (request["rasters"], base64 PNG).
            err = res.errors.pop("typst")
            import base64 as _b64

            client = {}
            for k, v in (request.get("rasters") or {}).items():
                try:
                    client[k] = _b64.b64decode(v)
                except Exception:
                    pass
            if "docx" in want:
                docx = attempt("docx", lambda: _build_docx(data, None, client, png_items))
                if docx:
                    produced["docx"] = (f"{stem}.docx", MIME["docx"], docx)
            if "png_canvas" in want and "canvas" in client:
                produced["png_canvas"] = (f"{stem}_canvas.png", MIME["png"], client["canvas"])
            for it in png_items:
                if it["key"].startswith("fig_") and it["key"] in client:
                    extras[f'objects/{it["key"]}_{_safe_stem(it.get("title", ""))[:40]}.png'] = client[it["key"]]
            # browser build (figures supplied): PDF / infographic are simply not offered there → no error
            skip = {"pdf", "png_infographic"} if client else set()
            for k in (want - {"svg", "zip"} - set(produced) - skip) or ({"typst"} if not produced else set()):
                if k not in res.errors:
                    res.errors[k] = err
        else:
            if "pdf" in want:
                pdf = attempt("pdf", lambda: _compile_report(typst, work, data, "report"))
                if pdf:
                    produced["pdf"] = (f"{stem}.pdf", MIME["pdf"], pdf)

            infographic_png = None
            if want & {"png_infographic", "docx"}:
                def _info():
                    pages = _compile_report(typst, work, data, "infographic", fmt="png", ppi=200.0)
                    return pages[0] if isinstance(pages, list) else pages
                infographic_png = attempt("png_infographic", _info)
                if infographic_png and "png_infographic" in want:
                    produced["png_infographic"] = (f"{stem}_infographic.png", MIME["png"], infographic_png)

            rasters: dict[str, bytes] = {}
            if want & {"docx", "zip"}:
                rasters = attempt("png_canvas", lambda: _raster(typst, work, png_items)) or {}
            elif "png_canvas" in want:
                rasters = attempt("png_canvas", lambda: _raster(typst, work, [it for it in png_items if it["key"] == "canvas"])) or {}
            if "png_canvas" in want and "canvas" in rasters:
                produced["png_canvas"] = (f"{stem}_canvas.png", MIME["png"], rasters["canvas"])
            elif "png_canvas" in want and "png_canvas" not in res.errors:
                res.errors["png_canvas"] = "No canvas SVG in the request."

            if "docx" in want:
                docx = attempt("docx", lambda: _build_docx(data, infographic_png, rasters, png_items))
                if docx:
                    produced["docx"] = (f"{stem}.docx", MIME["docx"], docx)

            for it in png_items:
                if it["key"].startswith("fig_") and it["key"] in rasters:
                    extras[f'objects/{it["key"]}_{_safe_stem(it.get("title", ""))[:40]}.png'] = rasters[it["key"]]

        if "svg" in want and request.get("canvas_svg"):
            produced["svg"] = (f"{stem}_canvas.svg", MIME["svg"], request["canvas_svg"].encode("utf-8"))
        for i, ob in enumerate(request.get("objects") or []):
            if ob.get("svg"):
                extras[f'objects/fig_{i:02d}_{_safe_stem(ob.get("title", ""))[:40]}.svg'] = ob["svg"].encode("utf-8")

        if "zip" in fmts:
            if produced:
                bio = io.BytesIO()
                with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for name, _m, blob in produced.values():
                        zf.writestr(name, blob)
                    for name, blob in extras.items():
                        zf.writestr(name, blob)
                    if res.errors:
                        zf.writestr("export_errors.txt", "\n".join(f"{k}: {v}" for k, v in res.errors.items()))
                res.files = [(f"{stem}.zip", MIME["zip"], bio.getvalue())]
        else:
            res.files = [produced[f] for f in fmts if f in produced]
    finally:
        shutil.rmtree(work, ignore_errors=True)

    od = pathlib.Path(out_dir) if out_dir is not None else None
    if od is not None:
        od.mkdir(parents=True, exist_ok=True)
        for name, _m, blob in res.files:
            pth = od / name
            pth.write_bytes(blob)
            res.saved.append(str(pth))
        if tracebacks:
            with open(od / "export_errors.log", "a", encoding="utf-8") as fh:
                fh.write(f"\n===== {_dt.datetime.now().isoformat(timespec='seconds')} · formats={fmts} · python={sys.executable}\n")
                fh.write("\n".join(tracebacks))
    res.elapsed_s = round(time.perf_counter() - t0, 2)
    if not res.files:
        uniq: dict[str, list[str]] = {}
        for k, v in res.errors.items():
            uniq.setdefault(v.removeprefix("RuntimeError: "), []).append(k)
        msg = "\n\n".join(m if len(uniq) == 1 else f"[{', '.join(ks)}] {m}" for m, ks in uniq.items()) or "nothing was produced"
        raise RuntimeError(msg)
    return res


def _build_object_export(request: dict, fmts: list[str], out_dir, t0: float) -> ReportResult:
    """Export ONE object (request["objects"][0]["svg"]) as a vector PDF and/or a 2× PNG via Typst."""
    typst = _require_typst()
    objs = [o for o in (request.get("objects") or []) if o.get("svg")]
    if not objs:
        raise RuntimeError("No object SVG in the export request.")
    ob = objs[0]
    res = ReportResult()
    work = pathlib.Path(tempfile.mkdtemp(prefix="nr_obj_"))
    try:
        svg = _prep_svg(ob["svg"])
        (work / "obj.svg").write_text(svg, encoding="utf-8")
        w, _h = _svg_size(svg)
        w_pt = w * _PX_TO_PT
        (work / "obj.typ").write_text(
            "#set page(width: auto, height: auto, margin: 0pt, fill: white)\n"
            f'#image("obj.svg", width: {w_pt:.2f}pt)\n',
            encoding="utf-8",
        )
        stem = (f'titra_{_safe_stem(request.get("file_stem"))}_{_safe_stem(ob.get("type", "object"))}_'
                f'{_safe_stem(ob.get("title", ""))[:40]}_{_dt.datetime.now().strftime("%H%M%S")}')
        for f in fmts:
            try:
                if f == "obj_pdf":
                    res.files.append((f"{stem}.pdf", MIME["pdf"], typst.compile(str(work / "obj.typ"), root=str(work))))
                elif f == "obj_png":
                    ppi = min(192.0, _MAX_RASTER_PX / max(w, 1.0) * 96.0)
                    out = typst.compile(str(work / "obj.typ"), root=str(work), format="png", ppi=ppi)
                    res.files.append((f"{stem}.png", MIME["png"], out[0] if isinstance(out, list) else out))
            except Exception as e:
                res.errors[f] = f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if out_dir is not None:
        od = pathlib.Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        for name, _m, blob in res.files:
            (od / name).write_bytes(blob)
            res.saved.append(str(od / name))
    res.elapsed_s = round(time.perf_counter() - t0, 2)
    if not res.files:
        raise RuntimeError("; ".join(res.errors.values()) or "object export failed")
    return res


# ── environment helpers (used by the widget's Export ▾ menu) ─────────────
def diagnose(out_dir: str | pathlib.Path | None = None) -> str:
    """Human-readable check of everything the exporters need; also written to out_dir/export_diagnostics.txt."""
    lines = [
        f"Titra export diagnostics · {_dt.datetime.now().isoformat(timespec='seconds')}",
        f"Python      : {sys.version.split()[0]} · {sys.executable}",
        f"Platform    : {platform.platform()}",
    ]
    ok = True
    try:
        import typst

        ver = getattr(typst, "__version__", "?")
        good = _version_tuple(ver) >= MIN_TYPST
        ok &= good
        lines.append(f"typst       : {ver} {'✓' if good else '✗ too old (need ≥ 0.11)'}")
        if good:
            t = time.perf_counter()
            typst.compile('#set page(width: 60pt, height: 30pt)\n#text(font: "New Computer Modern")[β ok]'.encode("utf-8"))
            typst.compile(b"#set page(width: 60pt, height: 30pt)\nok", format="png", ppi=72)
            lines.append(f"              PDF + PNG smoke test ✓ ({time.perf_counter() - t:.2f}s)")
    except ImportError:
        ok = False
        lines.append("typst       : ✗ not installed")
    except Exception as e:
        ok = False
        lines.append(f"typst       : ✗ smoke test failed: {e}")
    try:
        import docx  # noqa: F401
        from importlib.metadata import version as _v

        lines.append(f"python-docx : {_v('python-docx')} ✓")
    except Exception:
        ok = False
        lines.append("python-docx : ✗ not installed")
    if not ok:
        lines.append(f"\nFix: Export ▾ → 'Install / update export tools' (then restart the kernel), or run\n  {pip_command()}")
    text = "\n".join(lines)
    if out_dir is not None:
        od = pathlib.Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        (od / "export_diagnostics.txt").write_text(text, encoding="utf-8")
    return text


def install_tools() -> str:
    """Install/upgrade typst + python-docx into THIS kernel's Python (pip, or uv for pip-less venvs)."""
    import importlib

    pkgs = ["typst", "python-docx"]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    def run(cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=900, creationflags=flags)

    before = None
    try:
        import typst as _t

        before = getattr(_t, "__version__", None)
    except ImportError:
        pass

    has_pip = run([sys.executable, "-m", "pip", "--version"]).returncode == 0
    uv = shutil.which("uv")
    if has_pip:
        proc, how = run([sys.executable, "-m", "pip", "install", "-U", *pkgs]), "pip"
    elif uv:  # uv-created venvs have no pip
        proc, how = run([uv, "pip", "install", "--python", sys.executable, "-U", *pkgs]), "uv"
    else:
        boot = run([sys.executable, "-m", "ensurepip", "--upgrade"])
        if boot.returncode != 0:
            raise RuntimeError(
                "This environment has neither pip nor uv. Install the packages from a terminal, e.g.\n"
                f'  uv pip install --python "{sys.executable}" typst python-docx\n'
                f"or\n  {pip_command()}"
            )
        proc, how = run([sys.executable, "-m", "pip", "install", "-U", *pkgs]), "pip (bootstrapped)"
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-8:])
    if proc.returncode != 0:
        raise RuntimeError(f"{how} failed (exit {proc.returncode}):\n{tail}")

    importlib.invalidate_caches()
    from importlib.metadata import version as _v

    try:
        after = _v("typst")
    except Exception:
        after = "?"
    msg = f"Installed typst {after} and python-docx {_v('python-docx')} with {how} into\n{sys.executable}"
    if before and before != after:
        msg += f"\n\ntypst {before} was already loaded — restart the kernel before exporting."
    else:
        msg += "\n\nReady — choose a report in Export ▾ (no restart needed)."
    return msg
