# Titra on the web (browser / WebAssembly build)

Titra runs **entirely in the visitor's browser** (Python via Pyodide). Uploaded JSON files are
processed on the visitor's own computer and are never sent to a server.

## 1 · Preview on your computer

```
python build_web.py --serve
```

Open http://localhost:8000. The first load downloads Python + numpy/pandas/scipy (about a minute;
the browser caches it afterwards). Then drop an LFP JSON file as usual.

## 2 · Publish with GitHub Pages (free link to share)

1. Create a GitHub repository (e.g. `titra`) and push this folder to it. The included `.gitignore`
   keeps `clinical_sample/`, `reports/` and PDFs out of the repository — check `git status` before
   the first push.
2. The workflow is configured at `.github/workflows/titra-pages.yml`.
3. On GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
4. Every push to `main` rebuilds and publishes the site. Your link:
   `https://sadeqi60.github.io/titra/`

No git? Run `build_web.py`, then upload the **contents** of `web/` to any static host
(GitHub Pages "Deploy from a branch", Netlify drop, Cloudflare Pages).

## What works in the browser build

| Feature | Browser build | Desktop (marimo) |
|---|---|---|
| Load / verify / merge JSON files, all objects & presets | ✓ | ✓ |
| SVG and PNG export (canvas, objects, infographic) | ✓ (rendered in the browser) | ✓ |
| Word report (.docx), ZIP of everything | ✓ | ✓ |
| PDF report, PDF infographic, object PDF (Typst) | ✓ (typst.ts, loaded on first PDF export) | ✓ |

Files: `titra.py` (the notebook, desktop and web), `build_web.py` (build / preview),
`.github/workflows/titra-pages.yml` (automatic GitHub Pages publishing).
