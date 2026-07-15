"""Single-worker job queue with cancellation + progress events (SPEC §9).

One worker thread runs jobs sequentially (the GPU is the bottleneck). Each job
gets a cancel Event checked via the `cancel()` callback passed to its work fn;
progress is reported via `progress(message, percent)` and fanned out to `on_event`.
"""

from __future__ import annotations

import queue
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from focusstack_server.errors import classify_error

WorkFn = Callable[[Callable[[str, float], None], Callable[[], bool]], Any]
EventFn = Callable[[dict], None]


@dataclass
class Job:
    id: str
    type: str
    params: dict
    status: str = "pending"  # pending | running | done | error | cancelled
    percent: float = 0.0
    message: str = ""
    result: Any = None
    error: str = ""
    error_code: str = ""
    _work: WorkFn | None = None
    _cancel: threading.Event = field(default_factory=threading.Event)
    _done: threading.Event = field(default_factory=threading.Event)


class JobQueue:
    def __init__(self, on_event: EventFn):
        self._on_event = on_event
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._q: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def submit(self, jtype: str, params: dict, work: WorkFn) -> str:
        jid = uuid.uuid4().hex[:12]
        job = Job(id=jid, type=jtype, params=params, _work=work)
        with self._lock:
            self._jobs[jid] = job
            self._order.append(jid)
        self._q.put(jid)
        return jid

    def get(self, jid: str) -> Job | None:
        return self._jobs.get(jid)

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return [self._jobs[j] for j in self._order]

    def cancel(self, jid: str) -> bool:
        job = self._jobs.get(jid)
        if job is None or job.status in ("done", "error", "cancelled"):
            return False
        job._cancel.set()
        if job.status == "pending":  # not yet started -> mark immediately
            job.status = "cancelled"
            job._done.set()
            self._emit(job)
        return True

    def reorder(self, ordered_ids: list[str]) -> None:
        with self._lock:
            pending = [j for j in ordered_ids if self._jobs.get(j) and self._jobs[j].status == "pending"]
            others = [j for j in self._order if j not in set(pending)]
            self._order = others[:1] + pending + others[1:] if others else pending

    def wait(self, jid: str, timeout: float | None = None) -> bool:
        job = self._jobs.get(jid)
        return bool(job and job._done.wait(timeout))

    def shutdown(self) -> None:
        self._stop.set()
        self._q.put("")  # unblock the worker

    def _emit(self, job: Job, frame_index: int | None = None, level: str = "info") -> None:
        self._on_event({"job_id": job.id, "type": job.type, "status": job.status,
                        "stage": job.message, "percent": job.percent,
                        "frame_index": frame_index, "message": job.message, "level": level})

    def _run(self) -> None:
        while not self._stop.is_set():
            jid = self._q.get()
            if self._stop.is_set() or not jid:
                break
            job = self._jobs.get(jid)
            if job is None or job._cancel.is_set():
                if job and job.status == "pending":
                    job.status = "cancelled"
                    job._done.set()
                    self._emit(job)
                continue
            job.status = "running"
            self._emit(job)

            def progress(msg: str, pct: float, _job: Job = job) -> None:
                _job.message = msg
                _job.percent = float(pct)
                self._emit(_job)

            try:
                job.result = job._work(progress, job._cancel.is_set)  # type: ignore[misc]
                job.status = "done"
                job.percent = 1.0
            except InterruptedError:
                job.status = "cancelled"
            except Exception as e:  # noqa: BLE001 - surfaced as structured error
                job.status = "error"
                job.error = str(e)
                job.error_code, _status = classify_error(e)
            finally:
                self._emit(job)
                job._done.set()
