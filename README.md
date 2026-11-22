# Titra — DBS programming studio

Interactive DAG studio for DBS sensing data (Medtronic Percept session reports and amplitude-titration
exports): drop a JSON file, Titra verifies it and builds the analysis pipeline — lead model, LFP, PSD,
spectrogram, β/FTG dynamics, game theory, risk, options and graph signal processing — with PDF / Word /
PNG / SVG export.

## Run on your computer

```
pip install -r requirements.txt
marimo run titra.py
```

(or `marimo edit titra.py` to see the code). Titra starts empty and waits for a JSON file.

## Share it as a website

See [TITRA_WEB.md](TITRA_WEB.md): `python build_web.py --serve` builds a browser-only version
(WebAssembly) that you can publish on GitHub Pages; uploaded files never leave the visitor's computer.

## Files

| File | Role |
|---|---|
| `titra.py` | the marimo notebook (desktop and web) |
| `nr_widget.py` | the Titra widget (Python side) |
| `nr_dag.js`, `nr_dag.css` | the Titra widget (browser side) |
| `nr_ingest.py` | detects, verifies and merges JSON inputs |
| `nr_parser.py` | parses Percept-style sessions into the analysis payload |
| `nr_game.py`, `nr_risk.py`, `nr_options.py`, `nr_weave.py` | decision analysis and graph signal processing |
| `nr_report.py`, `nr_report.typ` | report export (Typst PDF/PNG, Word, ZIP) |
| `build_web.py` | builds / previews the website version |
| `.github/workflows/titra-pages.yml` | GitHub Pages workflow |

No patient or sample data is included in this folder.
