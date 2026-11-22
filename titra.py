import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Titra")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
async def _(mo):
    # Titra — DBS programming studio.
    # Desktop: imports the Titra modules that sit next to this notebook.
    # Browser (WebAssembly build, see build_web.py): fetches them from public/titra/ next to the page;
    # numpy / pandas / scipy come from Pyodide, anywidget and python-docx from PyPI.
    import pathlib
    import sys

    TITRA_FILES = (
        "nr_widget.py", "nr_ingest.py", "nr_parser.py", "nr_game.py", "nr_options.py",
        "nr_risk.py", "nr_weave.py", "nr_report.py", "nr_report.typ", "nr_dag.js", "nr_dag.css",
    )

    if sys.platform == "emscripten":
        import micropip

        await micropip.install(["numpy", "pandas", "scipy", "anywidget"])
        try:  # Word export only; Titra works without it
            await micropip.install("python-docx")
        except Exception as _e:
            print("python-docx unavailable in this browser build:", _e)
        import anywidget  # noqa: F401  (also tells marimo which packages to preload)
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import scipy  # noqa: F401
        from pyodide.http import pyfetch

        titra_src = mo.notebook_location() / "public" / "titra"
        titra_root = pathlib.Path("/tmp/titra")
        titra_root.mkdir(parents=True, exist_ok=True)
        for _name in TITRA_FILES:
            _resp = await pyfetch(str(titra_src / _name))
            if not _resp.ok:
                raise RuntimeError(f"Titra could not load {_name} ({_resp.status}) from {titra_src}")
            (titra_root / _name).write_bytes(await _resp.bytes())
    else:
        titra_root = pathlib.Path(mo.notebook_dir() or pathlib.Path(__file__).resolve().parent)

    if str(titra_root) not in sys.path:
        sys.path.insert(0, str(titra_root))

    from nr_widget import Titra

    return (Titra,)


@app.cell(hide_code=True)
def _(Titra, mo):
    titra = mo.ui.anywidget(Titra())
    titra
    return (titra,)


if __name__ == "__main__":
    app.run()
