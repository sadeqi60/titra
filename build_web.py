"""Build the browser version of Titra (WebAssembly) into ./web — static files for GitHub Pages.

    python build_web.py            # build ./web
    python build_web.py --serve    # build, then preview at http://localhost:8000

Everything runs in the visitor's browser (Pyodide): uploaded JSON files never leave their computer.
Only the Titra code is published — clinical_sample/, reports/ and other local files are NOT copied.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
NOTEBOOK = "titra.py"
OUT = HERE / "web"
PUBLIC = HERE / "public" / "titra"
FILES = (
    "nr_widget.py", "nr_ingest.py", "nr_parser.py", "nr_game.py", "nr_options.py",
    "nr_risk.py", "nr_weave.py", "nr_report.py", "nr_report.typ", "nr_dag.js", "nr_dag.css",
)


def build() -> None:
    missing = [f for f in FILES + (NOTEBOOK,) if not (HERE / f).exists()]
    if missing:
        sys.exit(f"Missing files next to build_web.py: {', '.join(missing)}")
    # 1. the Titra modules the notebook fetches at start-up (public/ is copied by marimo's export)
    if PUBLIC.exists():
        shutil.rmtree(PUBLIC)
    PUBLIC.mkdir(parents=True)
    for f in FILES:
        shutil.copy2(HERE / f, PUBLIC / f)
    # 2. WebAssembly export (read-only app, code hidden)
    if OUT.exists():
        shutil.rmtree(OUT)
    cmd = [sys.executable, "-m", "marimo", "export", "html-wasm", NOTEBOOK, "-o", str(OUT), "--mode", "run", "-f"]
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=HERE, check=True)
    if not (OUT / "public" / "titra" / "nr_widget.py").exists():  # older marimo: copy public/ ourselves
        shutil.copytree(HERE / "public", OUT / "public", dirs_exist_ok=True)
    for junk in ("CLAUDE.md",):  # marimo's editor-assistant notes: not needed on the website
        (OUT / junk).unlink(missing_ok=True)
    (OUT / ".nojekyll").write_text("")  # GitHub Pages: serve files/folders that start with "_"
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"\nBuilt {OUT} ({size:.1f} MB). Publish the CONTENTS of this folder (e.g. GitHub Pages).")


def serve(port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(OUT))
    handler.extensions_map[".wasm"] = "application/wasm"
    print(f"Preview: http://localhost:{port}   (Ctrl+C to stop; the first load downloads Python, ~1 min)")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true", help="preview the build locally after building")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-build", action="store_true", help="only serve an existing ./web")
    a = ap.parse_args()
    if not a.no_build:
        build()
    if a.serve or a.no_build:
        serve(a.port)
