"""Thread-safe multi-account wrapper around the official kaggle SDK.

The Kaggle Python SDK reads credentials from `$KAGGLE_CONFIG_DIR/kaggle.json`
(or `KAGGLE_USERNAME` + `KAGGLE_KEY` env vars) at `authenticate()` time. To
support N nicks we serialize calls behind a lock and swap the env var per
account before instantiating a fresh `KaggleApi`.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .accounts import Account

_api_lock = threading.RLock()


@contextmanager
def _kaggle_env(account: Account) -> Iterator[None]:
    """Temporarily set KAGGLE_CONFIG_DIR (+ KAGGLE_USERNAME/KEY) for this account."""
    prev = {
        k: os.environ.get(k)
        for k in ("KAGGLE_CONFIG_DIR", "KAGGLE_USERNAME", "KAGGLE_KEY")
    }
    import json

    creds = json.loads(account.kaggle_json.read_text(encoding="utf-8"))
    os.environ["KAGGLE_CONFIG_DIR"] = str(account.config_dir)
    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def as_account(account: Account):
    """Yield an authenticated `KaggleApi` bound to `account`. Serialized globally."""
    with _api_lock, _kaggle_env(account):
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        yield api


# ---------- Kernels (notebooks) ----------

def kernel_push(account: Account, folder: Path) -> dict[str, Any]:
    """Push a kernel folder (must contain kernel-metadata.json + the notebook)."""
    with as_account(account) as api:
        resp = api.kernels_push(str(folder))
        return _resp_to_dict(resp)


def kernel_status(account: Account, kernel_ref: str) -> dict[str, Any]:
    """kernel_ref is `<username>/<slug>`. Returns dict with `status`, `failureMessage`."""
    with as_account(account) as api:
        resp = api.kernels_status(kernel_ref)
        return _resp_to_dict(resp)


def kernel_pull_output(account: Account, kernel_ref: str, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    with as_account(account) as api:
        api.kernels_output(kernel_ref, str(dest), force=True, quiet=True)
    return [p.name for p in dest.iterdir()]


def kernel_pull(account: Account, kernel_ref: str, dest: Path, metadata: bool = True) -> list[str]:
    """Pull a kernel's source (+ kernel-metadata.json) so it can be edited and re-pushed."""
    dest.mkdir(parents=True, exist_ok=True)
    with as_account(account) as api:
        api.kernels_pull(kernel_ref, str(dest), metadata=metadata, quiet=True)
    return [p.name for p in dest.iterdir()]


def kernel_list_versions(account: Account, kernel_ref: str) -> list[dict[str, Any]]:
    with as_account(account) as api:
        # Newer kaggle SDKs expose kernels_list_files / no list_versions endpoint;
        # fall back to status which carries version number.
        try:
            resp = api.kernels_list_files(kernel_ref)  # type: ignore[attr-defined]
            return [_resp_to_dict(x) for x in resp]
        except AttributeError:
            return []


# ---------- Competitions ----------

def competition_submit(
    account: Account,
    competition: str,
    file_path: Path,
    message: str,
) -> dict[str, Any]:
    with as_account(account) as api:
        resp = api.competition_submit(str(file_path), message, competition, quiet=True)
        return _resp_to_dict(resp)


def competition_submissions(account: Account, competition: str) -> list[dict[str, Any]]:
    with as_account(account) as api:
        try:
            resp = api.competitions_submissions_list(competition)
        except AttributeError:
            resp = api.competition_submissions(competition)  # older alias
        return [_resp_to_dict(x) for x in resp]


def competition_leaderboard(account: Account, competition: str) -> list[dict[str, Any]]:
    """Returns the full public leaderboard as a list of {teamName, score, ...}."""
    with as_account(account) as api:
        try:
            resp = api.competition_view_leaderboard(competition)
        except AttributeError:
            resp = api.competitions_view_leaderboard(competition)
        # API can return a list directly or a wrapper with .submissions
        if isinstance(resp, list):
            entries = resp
        elif hasattr(resp, "submissions"):
            entries = resp.submissions  # type: ignore[attr-defined]
        else:
            entries = list(resp)
        return [_resp_to_dict(x) for x in entries]


def competitions_list(account: Account, search: str | None = None) -> list[dict[str, Any]]:
    with as_account(account) as api:
        resp = api.competitions_list(search=search) if search else api.competitions_list()
        return [_resp_to_dict(x) for x in resp]


# ---------- helpers ----------

def _resp_to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    # ApiResponse objects from kaggle SDK have attributes; pick non-callable, non-private
    out: dict[str, Any] = {}
    for k in dir(obj):
        if k.startswith("_"):
            continue
        try:
            v = getattr(obj, k)
        except Exception:
            continue
        if callable(v):
            continue
        try:
            # Only keep JSON-ish primitives
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
        except Exception:
            pass
    return out
