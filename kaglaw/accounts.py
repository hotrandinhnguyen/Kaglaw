from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import ACCOUNTS_DIR
from . import db


@dataclass
class Account:
    nick: str
    username: str
    config_dir: Path
    notes: str | None = None

    @property
    def kaggle_json(self) -> Path:
        return self.config_dir / "kaggle.json"


def _read_kaggle_json(path: Path) -> tuple[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["username"], data["key"]


def discover_accounts() -> list[Account]:
    """Scan data/accounts/<nick>/kaggle.json on disk."""
    out: list[Account] = []
    if not ACCOUNTS_DIR.exists():
        return out
    for sub in sorted(ACCOUNTS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        kj = sub / "kaggle.json"
        if not kj.exists():
            continue
        try:
            username, _ = _read_kaggle_json(kj)
        except Exception:
            continue
        out.append(Account(nick=sub.name, username=username, config_dir=sub))
    return out


def sync_accounts_to_db() -> list[Account]:
    accs = discover_accounts()
    with db.connect() as con:
        for a in accs:
            con.execute(
                "INSERT INTO accounts(nick, username, config_dir) VALUES(?,?,?) "
                "ON CONFLICT(nick) DO UPDATE SET username=excluded.username, "
                "config_dir=excluded.config_dir",
                (a.nick, a.username, str(a.config_dir)),
            )
    return accs


def list_accounts() -> list[Account]:
    with db.connect() as con:
        rows = con.execute(
            "SELECT nick, username, config_dir, notes FROM accounts ORDER BY nick"
        ).fetchall()
    return [
        Account(
            nick=r["nick"],
            username=r["username"],
            config_dir=Path(r["config_dir"]),
            notes=r["notes"],
        )
        for r in rows
    ]


def get_account(nick: str) -> Account | None:
    with db.connect() as con:
        r = con.execute(
            "SELECT nick, username, config_dir, notes FROM accounts WHERE nick=?",
            (nick,),
        ).fetchone()
    if not r:
        return None
    return Account(
        nick=r["nick"],
        username=r["username"],
        config_dir=Path(r["config_dir"]),
        notes=r["notes"],
    )


def add_account_from_json(nick: str, kaggle_json_bytes: bytes, notes: str | None = None) -> Account:
    """Save a kaggle.json under data/accounts/<nick>/kaggle.json."""
    nick = nick.strip()
    if not nick or "/" in nick or "\\" in nick or nick.startswith("."):
        raise ValueError("Invalid nick")
    target_dir = ACCOUNTS_DIR / nick
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "kaggle.json"
    target.write_bytes(kaggle_json_bytes)
    try:
        username, _ = _read_kaggle_json(target)
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise ValueError(f"Invalid kaggle.json: {exc}") from exc
    with db.connect() as con:
        con.execute(
            "INSERT INTO accounts(nick, username, config_dir, notes) VALUES(?,?,?,?) "
            "ON CONFLICT(nick) DO UPDATE SET username=excluded.username, "
            "config_dir=excluded.config_dir, notes=excluded.notes",
            (nick, username, str(target_dir), notes),
        )
    return Account(nick=nick, username=username, config_dir=target_dir, notes=notes)


def remove_account(nick: str, delete_files: bool = False) -> None:
    with db.connect() as con:
        con.execute("DELETE FROM accounts WHERE nick=?", (nick,))
    if delete_files:
        d = ACCOUNTS_DIR / nick
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
