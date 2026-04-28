# -*- coding: utf-8 -*-
"""Runtime logger: produces summary, events, and errors logs for each run."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_id(stage: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{stage}"


def append_run_error_log(path: Path | None, payload: Dict[str, Any]) -> None:
    """Append a JSON line to run errors log (e.g. <target-dir>/run_errors.jsonl)."""
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(payload)
    if "ts_utc" not in rec:
        rec["ts_utc"] = _utc_now_iso()
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class RunLogger:
    """Logger for a single run, producing summary/events/errors."""

    def __init__(
        self,
        log_root: Path,
        stage: str,
        run_id: str,
        params: Dict[str, Any],
        paths: Dict[str, str],
    ) -> None:
        self.stage = stage
        self.run_id = run_id
        self.start_time_utc = _utc_now_iso()
        self.end_time_utc: str | None = None
        self.log_dir = Path(log_root) / stage / run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.log_dir / "summary.json"
        self.events_path = self.log_dir / "events.jsonl"
        self.errors_path = self.log_dir / "errors.jsonl"
        self.events_path.touch(exist_ok=True)
        self.errors_path.touch(exist_ok=True)
        self.params = params
        self.paths = paths
        self.counts: Dict[str, int] = {}
        self._status = "success"

    def bump(self, key: str, delta: int = 1) -> None:
        self.counts[key] = int(self.counts.get(key, 0)) + int(delta)

    def set_status(self, status: str) -> None:
        if status not in {"success", "partial", "failed"}:
            return
        self._status = status

    def event(self, event_type: str, payload: Dict[str, Any] | None = None) -> None:
        rec = {
            "ts_utc": _utc_now_iso(),
            "run_id": self.run_id,
            "stage": self.stage,
            "event_type": event_type,
            "payload": payload or {},
        }
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def error(
        self,
        error_type: str,
        message: str,
        traceback_text: str | None = None,
        source_file: str | None = None,
        line_index: int | None = None,
        payload: Dict[str, Any] | None = None,
    ) -> None:
        rec: Dict[str, Any] = {
            "ts_utc": _utc_now_iso(),
            "run_id": self.run_id,
            "stage": self.stage,
            "error_type": error_type,
            "source_file": source_file,
            "line_index": line_index,
            "message": message,
            "traceback": traceback_text or "",
            "payload": payload or {},
        }
        with open(self.errors_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.bump("failed_count", 1)
        if self._status == "success":
            self._status = "partial"

    def write_summary(self) -> None:
        self.end_time_utc = _utc_now_iso()
        start_dt = datetime.fromisoformat(self.start_time_utc)
        end_dt = datetime.fromisoformat(self.end_time_utc)
        duration_sec = (end_dt - start_dt).total_seconds()
        summary = {
            "run_id": self.run_id,
            "stage": self.stage,
            "start_time_utc": self.start_time_utc,
            "end_time_utc": self.end_time_utc,
            "duration_sec": duration_sec,
            "status": self._status,
            "params": self.params,
            "paths": self.paths,
            "counts": self.counts,
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
