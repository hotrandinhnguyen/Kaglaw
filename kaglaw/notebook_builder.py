"""Materialize Kaggle code files (.ipynb / .py) from plain source text.

This is the piece that lets the chat agent *author* notebooks instead of only
uploading pre-existing files. The agent sends Python source; we turn it into a
valid Jupyter notebook (nbformat 4) that Kaggle accepts, or a flat script.

Cell convention (jupytext-ish, kept deliberately simple):
    # %%                  -> start a new CODE cell
    # %% [markdown]       -> start a new MARKDOWN cell (each following line may
                             optionally be prefixed with "# " which is stripped)

Text before the first marker becomes the first cell. If there is no marker at
all, the whole source is a single code cell.

`source_to_ipynb` and `ipynb_to_source` are inverses (modulo whitespace), so the
agent can read code back, tweak it, and write it again.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .actions import slugify
from .config import NOTEBOOKS_DIR

_MARKER_RE = re.compile(r"^#\s*%%(.*)$")
_SCRIPT_EXT = {"python": ".py", "r": ".R", "rmarkdown": ".Rmd"}


# --------------------------------------------------------------------------- #
# source  <->  cells
# --------------------------------------------------------------------------- #

def split_into_cells(source: str) -> list[dict[str, str]]:
    """Split source on '# %%' markers into [{type: code|markdown, text}]."""
    lines = source.splitlines()
    cells: list[dict[str, Any]] = []
    cur_type = "code"
    cur_lines: list[str] = []
    started = False

    def flush() -> None:
        text = "\n".join(cur_lines).strip("\n")
        # Drop a leading empty pre-marker cell, but keep intentional empties.
        if not started and not text.strip():
            return
        cells.append({"type": cur_type, "text": text})

    for line in lines:
        m = _MARKER_RE.match(line.strip())
        if m:
            flush()
            cur_lines = []
            tag = m.group(1).strip().lower()
            cur_type = "markdown" if "markdown" in tag or "md" == tag else "code"
            started = True
        else:
            cur_lines.append(line)
    flush()

    if not cells:
        cells.append({"type": "code", "text": source.strip("\n")})
    return cells


def _md_strip(text: str) -> str:
    """For markdown cells written as comments, strip one leading '# ' per line."""
    out = []
    for ln in text.splitlines():
        if ln.startswith("# "):
            out.append(ln[2:])
        elif ln == "#":
            out.append("")
        else:
            out.append(ln)
    return "\n".join(out)


def _as_source_list(text: str) -> list[str]:
    """nbformat stores `source` as a list of lines each ending in '\\n'
    except the last. Empty cell -> []."""
    if text == "":
        return []
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def source_to_ipynb(source: str, language: str = "python") -> dict[str, Any]:
    """Build an nbformat-4 notebook dict from delimited source."""
    nb_cells: list[dict[str, Any]] = []
    for cell in split_into_cells(source):
        if cell["type"] == "markdown":
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": _as_source_list(_md_strip(cell["text"])),
            })
        else:
            nb_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _as_source_list(cell["text"]),
            })
    lang = (language or "python").lower()
    kernel = {
        "python": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "r": {"display_name": "R", "language": "R", "name": "ir"},
    }.get(lang, {"display_name": "Python 3", "language": "python", "name": "python3"})
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": kernel,
            "language_info": {"name": kernel["language"]},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def ipynb_to_source(data: dict[str, Any]) -> str:
    """Reconstruct '# %%'-delimited source from a notebook dict."""
    parts: list[str] = []
    for cell in data.get("cells", []):
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else (src or "")
        text = text.rstrip("\n")
        if cell.get("cell_type") == "markdown":
            commented = "\n".join(
                ("# " + ln) if ln else "#" for ln in text.splitlines()
            )
            parts.append("# %% [markdown]\n" + commented)
        else:
            parts.append("# %%\n" + text)
    return "\n\n".join(parts).strip() + "\n"


# --------------------------------------------------------------------------- #
# files on disk
# --------------------------------------------------------------------------- #

def write_source(
    title: str,
    source: str,
    *,
    language: str = "python",
    kernel_type: str = "notebook",
) -> Path:
    """Write the source as a .ipynb (notebook) or script into
    NOTEBOOKS_DIR/<slug>/. Returns the path to the written file."""
    slug = slugify(title)
    folder = NOTEBOOKS_DIR / slug
    folder.mkdir(parents=True, exist_ok=True)
    if kernel_type == "notebook":
        path = folder / f"{slug}.ipynb"
        nb = source_to_ipynb(source, language)
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    else:
        ext = _SCRIPT_EXT.get((language or "python").lower(), ".py")
        path = folder / f"{slug}{ext}"
        path.write_text(source, encoding="utf-8")
    return path


def read_source(local_path: str | Path) -> str:
    """Read a notebook/script file back into '# %%'-delimited source text."""
    p = Path(local_path)
    if p.is_dir():  # registered as a folder: find the first notebook/script
        cand = (
            list(p.glob("*.ipynb"))
            or list(p.glob("*.py"))
            or list(p.glob("*.R"))
            or list(p.glob("*.Rmd"))
        )
        if not cand:
            raise FileNotFoundError(f"No code file in {p}")
        p = cand[0]
    if p.suffix.lower() == ".ipynb":
        data = json.loads(p.read_text(encoding="utf-8"))
        return ipynb_to_source(data)
    return p.read_text(encoding="utf-8")


def overwrite_source(local_path: str | Path, new_source: str) -> Path:
    """Rewrite the file at local_path with new source (keeps format)."""
    p = Path(local_path)
    if p.is_dir():
        cand = (
            list(p.glob("*.ipynb"))
            or list(p.glob("*.py"))
            or list(p.glob("*.R"))
            or list(p.glob("*.Rmd"))
        )
        if not cand:
            raise FileNotFoundError(f"No code file in {p}")
        p = cand[0]
    if p.suffix.lower() == ".ipynb":
        data = json.loads(p.read_text(encoding="utf-8"))
        lang = (data.get("metadata", {}).get("language_info", {}).get("name") or "python")
        nb = source_to_ipynb(new_source, lang)
        p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    else:
        p.write_text(new_source, encoding="utf-8")
    return p
