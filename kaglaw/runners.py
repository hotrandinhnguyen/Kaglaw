"""Run a notebook/script LOCALLY as a quick smoke test before spending Kaggle GPU.

This executes the notebook's source (reconstructed to a plain .py — the `# %%`
markers are just comments) in a subprocess with a timeout, in an isolated working
directory, and captures stdout/stderr. It catches the cheap-to-fix failures —
syntax errors, bad imports, obvious bugs — without burning a Kaggle run.

Caveats (told to the user, not hidden):
  - It runs your code on THIS machine: deps must be installed locally, and any
    `/kaggle/input/...` paths won't exist unless you have them locally.
  - No GPU/sandbox isolation beyond a temp cwd + timeout. Only test code you trust
    (here, code you/the agent wrote).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from . import actions, metrics, notebook_builder
from .config import LOCAL_PYTHON, LOCAL_RUN_TIMEOUT, LOCAL_RUNS_DIR

_TAIL = 4000  # chars of stdout/stderr to keep


def _interpreter() -> str:
    return LOCAL_PYTHON or sys.executable


def run_notebook_local(
    notebook_id: int,
    *,
    timeout: int | None = None,
    keep_workdir: bool = False,
) -> dict:
    nb = actions.get_notebook(notebook_id)
    if not nb:
        return {"ok": False, "error": f"notebook {notebook_id} not found"}
    try:
        source = notebook_builder.read_source(nb["local_path"])
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"cannot read source: {exc}"}

    # Refuse to run a notebook that still has unfilled {{placeholders}}.
    import re
    holes = sorted(set(re.findall(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", source)))
    if holes:
        return {"ok": False, "error": f"source still has unfilled placeholders: {holes}. "
                                      "Fill them (or run a sweep) before a local test."}

    timeout = int(timeout or LOCAL_RUN_TIMEOUT)
    workdir = LOCAL_RUNS_DIR / f"nb{notebook_id}_{int(time.time())}"
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / "main.py"
    script.write_text(source, encoding="utf-8")

    before = {p.name for p in workdir.iterdir()}
    t0 = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            [_interpreter(), "main.py"],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        rc = proc.returncode
        stdout, stderr = proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = None
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")) \
            + f"\n[killed: exceeded {timeout}s timeout]"
    duration = round(time.time() - t0, 1)

    new_files = sorted(p.name for p in workdir.iterdir() if p.name not in before and p.name != "main.py")
    mn, mv = metrics.extract_metric(stdout)
    ok = (rc == 0) and not timed_out

    result = {
        "ok": ok,
        "notebook_id": notebook_id,
        "returncode": rc,
        "timed_out": timed_out,
        "duration_seconds": duration,
        "metric": ({"name": mn, "value": mv} if mv is not None else None),
        "output_files": new_files,
        "stdout_tail": stdout[-_TAIL:],
        "stderr_tail": stderr[-_TAIL:],
        "workdir": str(workdir),
    }
    if not keep_workdir and ok and not new_files:
        # nothing produced and it passed — clean up to avoid clutter
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
        result["workdir"] = None
    return result
