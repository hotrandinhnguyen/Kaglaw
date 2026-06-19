"""Parse a numeric research metric (CV / AUC / RMSE / score ...) out of a kernel log.

Notebooks usually print their validation metric, e.g. `CV: 0.8345` or `oof auc = 0.91`.
We scan the pulled log text and return the LAST such number (final summary line tends to
be the one that matters). Patterns are configurable via the `metrics.patterns` setting
(JSON list of [name, regex] with one capture group), else the defaults below are used.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import settings_store

_NUM = r"([+-]?[0-9]*\.?[0-9]+)"

DEFAULT_PATTERNS: list[tuple[str, str]] = [
    ("cv",       rf"(?i)\bcv(?:[ _-]?score)?\s*[:=]\s*{_NUM}"),
    ("oof",      rf"(?i)\boof(?:[ _-]?(?:score|metric|auc))?\s*[:=]\s*{_NUM}"),
    ("val",      rf"(?i)\b(?:val|valid|validation)(?:[ _-]?(?:score|metric|loss|acc|auc))?\s*[:=]\s*{_NUM}"),
    ("auc",      rf"(?i)\bauc\s*[:=]\s*{_NUM}"),
    ("rmsle",    rf"(?i)\brmsle\s*[:=]\s*{_NUM}"),
    ("rmse",     rf"(?i)\brmse\s*[:=]\s*{_NUM}"),
    ("mae",      rf"(?i)\bmae\s*[:=]\s*{_NUM}"),
    ("logloss",  rf"(?i)\blog[ _]?loss\s*[:=]\s*{_NUM}"),
    ("f1",       rf"(?i)\bf1(?:[ _-]?score)?\s*[:=]\s*{_NUM}"),
    ("accuracy", rf"(?i)\bacc(?:uracy)?\s*[:=]\s*{_NUM}"),
    ("score",    rf"(?i)\bscore\s*[:=]\s*{_NUM}"),
]

# files in a pulled output dir we treat as logs worth scanning
_LOG_SUFFIXES = {".log", ".txt", ".out", ".err"}
_MAX_SCAN_BYTES = 500_000


def get_patterns() -> list[tuple[str, str]]:
    raw = settings_store.get("metrics.patterns")
    if raw:
        try:
            data = json.loads(raw)
            pats = [(str(n), str(p)) for n, p in data]
            if pats:
                return pats
        except Exception:
            pass
    return DEFAULT_PATTERNS


def extract_metric(text: str) -> tuple[str | None, float | None]:
    """Return (metric_name, value) for the last matching pattern in `text`."""
    if not text:
        return None, None
    best_pos = -1
    best: tuple[str, float] | None = None
    for name, pattern in get_patterns():
        try:
            rgx = re.compile(pattern)
        except re.error:
            continue
        for m in rgx.finditer(text):
            try:
                val = float(m.group(1))
            except (ValueError, IndexError):
                continue
            # last position wins; ties keep the earlier (more specific) pattern
            if m.start() > best_pos:
                best_pos = m.start()
                best = (name, val)
    if best is None:
        return None, None
    return best


def extract_from_dir(output_dir: str | Path) -> tuple[str | None, float | None, str | None]:
    """Scan log-ish files in a pulled output dir. Returns (name, value, source_file)."""
    d = Path(output_dir)
    if not d.exists():
        return None, None, None
    candidates = [
        p for p in sorted(d.iterdir())
        if p.is_file() and (p.suffix.lower() in _LOG_SUFFIXES)
    ]
    # if no obvious log file, fall back to any small text file (e.g. stdout dumped as .csv? skip)
    if not candidates:
        candidates = [
            p for p in sorted(d.iterdir())
            if p.is_file() and p.suffix.lower() not in {".csv", ".parquet", ".pkl", ".bin",
                                                         ".png", ".jpg", ".zip", ".npy"}
            and p.stat().st_size <= _MAX_SCAN_BYTES
        ]
    for p in candidates:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")[:_MAX_SCAN_BYTES]
        except Exception:
            continue
        name, val = extract_metric(text)
        if val is not None:
            return name, val, p.name
    return None, None, None
