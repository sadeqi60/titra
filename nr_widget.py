"""Titra — DBS programming studio (AnyWidget). Interactive DBS lead & electrophysiology DAG canvas.

Synced traitlets:
  - `payload`: parsed Percept JSON analytical bundle from `nr_parser.parse_percept_json`
  - `selection`: live selectable entities (contacts, bipolar pairs, recordings, bands,
    time/frequency windows, JSON paths) synced back to Python on every user interaction
  - `dag_state`: live DAG topology (nodes with x/y/w/h/collapsed/config, directed edges, viewport)
  - `svg_output`: pure-vector standalone SVG export of the entire DAG canvas
  - `report_request` / `report_result`: Export ▾ menu → Python report builder (nr_report.py):
    PDF (Typst), Word (.docx), PNG (canvas / infographic) or a ZIP of everything; per-object
    PNG / PDF from each object's ⤓ menu; plus setup actions (check setup, install export tools).
  - `input_request` / `input_state`: the Data Input object — scans `data_dir` for JSON files,
    detects + verifies any JSON (nr_ingest.py) and loads usable data into the DAG. Files are sent
    back to the browser for download and saved to `report_dir` (default: ./reports).
"""

from __future__ import annotations

import base64
import pathlib
import sys
import time
from typing import Any

import anywidget
import traitlets

HERE = pathlib.Path(__file__).resolve().parent


class NeuroReaderDAG(anywidget.AnyWidget):
    _esm = HERE / "nr_dag.js"
    _css = HERE / "nr_dag.css"

    payload = traitlets.Dict().tag(sync=True)
    selection = traitlets.Dict().tag(sync=True)
    dag_state = traitlets.Dict().tag(sync=True)
    svg_output = traitlets.Unicode("").tag(sync=True)
    preset = traitlets.Unicode("beta_ftg").tag(sync=True)
    report_request = traitlets.Dict().tag(sync=True)
    report_result = traitlets.Dict().tag(sync=True)
    input_request = traitlets.Dict().tag(sync=True)
    input_state = traitlets.Dict().tag(sync=True)
    # what this Python can build (the browser build has no Typst → no PDF; PNGs are rendered in the browser)
    capabilities = traitlets.Dict().tag(sync=True)

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        preset: str = "beta_ftg",
        initial_dag: dict[str, Any] | None = None,
        report_dir: str | pathlib.Path | None = None,
        data_dir: str | pathlib.Path | None = None,
        source: str | pathlib.Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.report_dir = pathlib.Path(report_dir) if report_dir else HERE / "reports"
        self.last_report = None  # nr_report.ReportResult of the most recent export
        self._last_report_request: dict[str, Any] = {}
        # no data folder by default: the studio waits for a JSON file (drop / browse); pass data_dir to
        # also offer the JSON files of a folder in the LFP Input object
        self.data_dir = pathlib.Path(data_dir) if data_dir else None
        self.store: dict[str, Any] | None = None  # full parser store of the data loaded via the Input object
        self._upload: tuple[str, bytes] | None = None
        self._session: dict[str, Any] | None = None  # files merged into the current dataset (nr_ingest session)
        payload = payload or {}  # no default file: every pipeline starts at the LFP Input object

        super().__init__(
            payload=payload,
            selection=self._default_selection(payload),
            dag_state=initial_dag or {},
            preset=preset,
            **kwargs,
        )
        state: dict[str, Any] = {"data_dir": str(self.data_dir) if self.data_dir else "", "files": self._scan(),
                                 "loaded": "(notebook)" if payload else "", "report": None, "modality": "lfp"}
        self.input_state = state
        self.capabilities = self._capabilities()
        if source:
            try:
                self.load(source)
            except Exception as exc:  # never block widget creation
                self.input_state = {**self.input_state, "error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _capabilities() -> dict[str, Any]:
        import importlib.util

        def has(mod: str) -> bool:
            try:
                return importlib.util.find_spec(mod) is not None
            except Exception:
                return False

        return {"wasm": sys.platform == "emscripten", "typst": has("typst") and sys.platform != "emscripten", "docx": has("docx")}

    @staticmethod
    def _default_selection(payload: dict[str, Any]) -> dict[str, Any]:
        import nr_parser

        recs = payload.get("recordings") or []
        first_stream = next((r for r in recs if r.get("source") == "BrainSenseTimeDomain"), None)
        first_rec = first_stream or (recs[0] if recs else None)
        hemi = first_rec["hemisphere"] if first_rec else "Left"
        all_contacts = nr_parser.all_contacts_with_data(payload)
        all_pairs = [m["id"] for m in (payload.get("montage") or [])]
        all_rec_ids = [r["id"] for r in recs]
        stim_rate = (
            first_rec["stim_rate_hz"]
            if first_rec and first_rec.get("stim_rate_hz")
            else payload.get("config", {}).get("default_stim_rate_hz", 145.0)
        )

        default_selection = {
            "hemisphere": hemi,
            "contacts": all_contacts,
            "pairs": all_pairs,
            "recordings": all_rec_ids,
            "selected_nodes": ["input_main"],
            "selected_band": "both",
            "stim_rate_hz": stim_rate,
            "ftg_half_bw_hz": payload.get("config", {}).get("ftg_half_bw_hz", 4.0),
            "freq_window_hz": None,
            "time_window_s": None,
            "json_paths": ["DeviceInformation.Final", "LeadConfiguration.Final"],
            "gt_params": {
                "hemisphere": hemi,
                "stim_rate_filter": "ALL",
                "wb": 1.0,
                "wf": 0.6,
                "we": 0.5,
                "tau": 2.0,
                "ftg_safe": 4.0,
                "algo": "thompson",
                "budget": 60,
            },
            "gt_selected_arms": [],
            "gt_bandit_live": {},
            "risk_params": {
                "hemisphere": hemi,
                "stim_rate_filter": "ALL",
                "target": "beta",
                "alpha": 0.99,
                "horizon": 60.0,
                "kappa": 0.6,
                "eps": 0.10,
                "metric": "CVaR",
            },
            "risk_selected_arms": [],
            "opt_params": {
                "hemisphere": hemi,
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
            },
            "opt_selected_arms": [],
            "gsp_params": {
                "hemisphere": hemi,
                "stim_rate_filter": "ALL",
                "sigma": 2.0,
                "series_id": "",
                "phase_band": "ftg",
                "node_size_metric": "local_ftg",
            },
            "gsp_selected_node_idx": None,
        }

        return default_selection

    # ── Data Input object (scan · verify · load any JSON) ────────────────
    def _scan(self) -> list[dict[str, Any]]:
        try:
            import nr_ingest

            return nr_ingest.scan_folder(self.data_dir) if self.data_dir else []
        except Exception:
            return []

    @traitlets.observe("input_request")
    def _on_input_request(self, change: dict[str, Any]) -> None:
        req = change.get("new") or {}
        rid = req.get("id")
        if not rid:
            return
        import nr_ingest

        action = req.get("action", "")
        st = dict(self.input_state or {})
        st.update({"id": rid, "action": action, "error": ""})
        t0 = time.perf_counter()
        try:
            if action == "scan":
                if req.get("data_dir"):
                    self.data_dir = pathlib.Path(req["data_dir"])
                st["data_dir"] = str(self.data_dir) if self.data_dir else ""
                st["files"] = self._scan()
            elif action == "clear":
                self.store = None
                self._upload = None
                self._session = None
                st["add_result"] = None
                st["sources"] = []
                self.selection = self._default_selection({})
                self.payload = {}
                st.update({"report": None, "selected": "", "loaded": ""})
            elif action in ("inspect_load", "load", "inspect"):
                # verify first; load automatically only if the file(s) have usable data
                src = str(req.get("path") or "")
                if not src:
                    raise ValueError("Choose a JSON file first.")
                sess = nr_ingest.new_session(nr_ingest.read_items_paths(src.split("|")))
                rep = nr_ingest.inspect_session(sess)
                st.update({"report": rep, "selected": src, "add_result": None})
                if rep.get("ok") and action != "inspect":
                    self._load_session(sess, st, src)
            elif action in ("upload", "upload_inspect", "upload_inspect_again", "upload_load"):
                if req.get("b64"):
                    self._upload = (str(req.get("name") or "upload.json"), base64.b64decode(req["b64"]))
                if not self._upload:
                    raise ValueError("Upload the file again.")
                name, raw = self._upload
                sess = nr_ingest.new_session([nr_ingest.read_item_bytes(raw, name)])
                rep = nr_ingest.inspect_session(sess)
                st.update({"report": rep, "selected": f"upload:{name}", "add_result": None})
                if rep.get("ok") and action in ("upload", "upload_load"):
                    self._load_session(sess, st, f"upload:{name}")
            elif action == "add":
                # merge more file(s) of the same structure into the loaded session
                items = []
                if req.get("path"):
                    items += nr_ingest.read_items_paths(str(req["path"]).split("|"))
                for f in req.get("files") or []:
                    items.append(nr_ingest.read_item_bytes(base64.b64decode(f.get("b64") or ""), str(f.get("name") or "upload.json")))
                if not items:
                    raise ValueError("Choose a JSON file to add.")
                names = [it[0] for it in items]
                if req.get("replace"):
                    self._session = None
                if not self._session:
                    sess = nr_ingest.new_session(items[:1])
                    if len(items) > 1:
                        sess, _info = nr_ingest.add_to_session(sess, items[1:])
                    rep = nr_ingest.inspect_session(sess)
                    src = str(req.get("path") or f"upload:{names[0]}")
                    st.update({"report": rep, "selected": src, "add_result": None})
                    if rep.get("ok"):
                        self._load_session(sess, st, src)
                else:
                    try:
                        sess, info = nr_ingest.add_to_session(self._session, items)
                    except nr_ingest.MergeConflict as mc:
                        st["add_result"] = {"ok": False, "files": names, "conflicts": mc.conflicts,
                                            "message": f"{', '.join(names)} was not added — it conflicts with the loaded data. Nothing was changed."}
                    else:
                        if info["nothing_new"]:
                            self._session = sess
                            st["add_result"] = {"ok": True, "files": names, "nothing_new": True, "warnings": info["warnings"],
                                                "message": f"{', '.join(names)}: everything in it is already loaded ({info['duplicates']} duplicate entries)."}
                            st["report"] = dict(st.get("report") or {}, sources=sess["sources"])
                        else:
                            rep = nr_ingest.inspect_session(sess)
                            if not rep.get("ok"):
                                st["add_result"] = {"ok": False, "files": names, "conflicts": rep.get("errors") or [rep.get("summary", "")],
                                                    "message": "The combined data could not be verified. Nothing was changed."}
                            else:
                                self._load_session(sess, st, st.get("loaded") or st.get("selected") or "session", merged=True)
                                st["add_result"] = {"ok": True, "files": names, "summary": info["summary"], "duplicates": info["duplicates"],
                                                    "warnings": info["warnings"], "message": f"Added {', '.join(names)}: {info['summary']}"}
            else:
                raise ValueError(f"Unknown input action '{action}'")
        except Exception as exc:
            st["error"] = f"{type(exc).__name__}: {exc}" if not isinstance(exc, ValueError) else str(exc)
            if hasattr(exc, "conflicts"):
                st["add_result"] = {"ok": False, "files": [getattr(exc, "name", "")], "conflicts": exc.conflicts,
                                    "message": "These files cannot be combined — they conflict with each other."}
        st["elapsed_s"] = round(time.perf_counter() - t0, 2)
        self.input_state = st

    def _load_session(self, sess: dict, st: dict, src: str, merged: bool = False) -> None:
        import nr_ingest

        ftg = float((self.selection or {}).get("ftg_half_bw_hz") or 4.0)
        store, rep = nr_ingest.load_session(sess, ftg)
        if merged:
            store["payload"].setdefault("metadata", {})["merged"] = True
        self._session = sess
        self._apply_store(store)
        st.update({"report": rep, "loaded": src, "selected": src, "sources": sess.get("sources", [])})

    def _apply_store(self, store: dict[str, Any]) -> None:
        """Swap the DAG's data: new selection first, then payload (the JS reloads on payload change)."""
        payload = store["payload"]
        self.store = store
        self.selection = self._default_selection(payload)
        self.payload = payload

    def load(self, source: str | pathlib.Path) -> dict[str, Any]:
        """Python API: verify + load a JSON file (or 'a.json|b.json' titration set). Returns the report."""
        import nr_ingest

        src = str(source)
        sess = nr_ingest.new_session(nr_ingest.read_items_paths(src.split("|")))
        rep = nr_ingest.inspect_session(sess)
        if not rep.get("ok"):
            raise ValueError(" ".join(rep.get("errors") or []) or rep.get("summary", "not usable"))
        st = {**(self.input_state or {}), "error": "", "add_result": None}
        self._load_session(sess, st, src)
        self.input_state = st
        return st["report"]

    def add(self, source: str | pathlib.Path) -> dict[str, Any]:
        """Python API: merge another file of the same structure into the loaded data.

        Raises nr_ingest.MergeConflict (data unchanged) if the file contradicts what is loaded.
        """
        import nr_ingest

        src = str(source)
        if not self._session:
            return self.load(src)
        sess, info = nr_ingest.add_to_session(self._session, nr_ingest.read_items_paths(src.split("|")))
        st = {**(self.input_state or {}), "error": ""}
        if info["nothing_new"]:
            self._session = sess
        else:
            self._load_session(sess, st, st.get("loaded") or src, merged=True)
        st["add_result"] = {"ok": True, "files": [pathlib.Path(p).name for p in src.split("|")], "summary": info["summary"],
                            "nothing_new": info["nothing_new"], "warnings": info["warnings"],
                            "message": f"Added: {info['summary']}" if not info["nothing_new"] else "Everything in the file was already loaded."}
        self.input_state = st
        return info

    # ── Report export (Export ▾ menu in the widget toolbar) ──────────────
    @traitlets.observe("report_request")
    def _on_report_request(self, change: dict[str, Any]) -> None:
        req = change.get("new") or {}
        rid = req.get("id")
        if not rid:
            return
        t0 = time.perf_counter()
        action = req.get("action")
        try:
            import nr_report

            if action == "diagnose":
                msg = nr_report.diagnose(self.report_dir)
                self.report_result = {"id": rid, "ok": True, "message": msg, "title": "Export setup check",
                                      "elapsed_s": round(time.perf_counter() - t0, 2)}
                return
            if action == "install_tools":
                msg = nr_report.install_tools()
                self.report_result = {"id": rid, "ok": True, "message": msg, "title": "Export tools installed",
                                      "elapsed_s": round(time.perf_counter() - t0, 2)}
                return
            self._last_report_request = req
            res = nr_report.build_report(self.payload, req, out_dir=self.report_dir)
            self.last_report = res
            self.report_result = {
                "id": rid,
                "ok": True,
                "files": [
                    {"name": n, "mime": m, "b64": base64.b64encode(b).decode("ascii"), "bytes": len(b)}
                    for n, m, b in res.files
                ],
                "saved": res.saved,
                "saved_dir": str(self.report_dir),
                "warnings": res.warnings,
                "errors": res.errors,  # partial failures (other formats still delivered)
                "elapsed_s": res.elapsed_s,
            }
        except Exception as exc:  # report errors back to the widget instead of failing silently
            import traceback

            log = self.report_dir / "export_errors.log"
            try:
                self.report_dir.mkdir(parents=True, exist_ok=True)
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write(f"\n===== action={action or req.get('formats')} python={sys.executable}\n{traceback.format_exc()}")
            except OSError:
                pass
            self.report_result = {
                "id": rid,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "log": str(log),
                "elapsed_s": round(time.perf_counter() - t0, 2),
            }

    def export_report(
        self,
        formats: tuple[str, ...] | list[str] = ("pdf",),
        out_dir: str | pathlib.Path | None = None,
    ):
        """Re-export from Python using the SVGs of the last widget export (or the canvas `svg_output`).

        formats: any of "pdf", "docx", "png_canvas", "png_infographic", "svg", "zip".
        Returns nr_report.ReportResult (files as bytes + saved paths).
        """
        import nr_report

        req = dict(self._last_report_request or {})
        if not req.get("canvas_svg") and self.svg_output:
            req["canvas_svg"] = self.svg_output
        req.setdefault("selection", dict(self.selection))
        req.setdefault("dag", {"n_nodes": len((self.dag_state or {}).get("nodes", [])), "n_edges": len((self.dag_state or {}).get("edges", []))})
        req["formats"] = list(formats)
        res = nr_report.build_report(self.payload, req, out_dir=out_dir or self.report_dir)
        self.last_report = res
        return res


# Brand alias: `from nr_widget import Titra`
Titra = NeuroReaderDAG
