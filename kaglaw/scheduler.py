from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from . import actions, orchestrator
from .config import DISPATCH_INTERVAL_SECONDS, POLL_INTERVAL_SECONDS

log = logging.getLogger("kaglaw.scheduler")

_scheduler: BackgroundScheduler | None = None


def _poll_runs() -> None:
    try:
        res = actions.sync_all_active_runs()
        if res.get("synced"):
            log.info("poll: synced %s active runs", res["synced"])
    except Exception:
        log.exception("poll_runs failed")


def _poll_submissions() -> None:
    try:
        comps = actions.list_tracked_competitions()
        for comp in comps:
            try:
                actions.sync_submissions_for_competition(comp)
            except Exception:
                log.exception("sync subs failed for %s", comp)
    except Exception:
        log.exception("poll_submissions failed")


def _dispatch() -> None:
    try:
        res = orchestrator.dispatch_jobs()
        if res.get("dispatched"):
            log.info("dispatch: launched %s job(s), %s queued", res["dispatched"], res["queued_remaining"])
        orchestrator.detect_finished_batches()
    except Exception:
        log.exception("dispatch failed")


def start() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    sch = BackgroundScheduler(daemon=True)
    sch.add_job(_poll_runs, "interval", seconds=POLL_INTERVAL_SECONDS, id="poll_runs", max_instances=1)
    sch.add_job(_poll_submissions, "interval", seconds=POLL_INTERVAL_SECONDS * 5, id="poll_subs", max_instances=1)
    sch.add_job(_dispatch, "interval", seconds=DISPATCH_INTERVAL_SECONDS, id="dispatch", max_instances=1)
    sch.start()
    _scheduler = sch
    log.info("scheduler started, poll=%ss dispatch=%ss", POLL_INTERVAL_SECONDS, DISPATCH_INTERVAL_SECONDS)
    return sch


def stop() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
