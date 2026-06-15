import threading
import time

from focusstack_server.jobs import JobQueue


def test_jobs_run_sequentially_and_report_progress():
    events = []
    q = JobQueue(on_event=lambda e: events.append(e))

    def work(progress, cancel):
        for i in range(3):
            progress("step", (i + 1) / 3)
        return {"ok": True}

    jid = q.submit("stack", {}, work)
    q.wait(jid, timeout=5)
    job = q.get(jid)
    assert job.status == "done"
    assert job.result == {"ok": True}
    assert any(e["percent"] == 1.0 for e in events if e["job_id"] == jid)
    q.shutdown()


def test_job_cancellation():
    started = threading.Event()
    q = JobQueue(on_event=lambda e: None)

    def work(progress, cancel):
        started.set()
        for _ in range(1000):
            if cancel():
                raise InterruptedError("cancelled")
            time.sleep(0.005)
        return {"ok": True}

    jid = q.submit("stack", {}, work)
    started.wait(2)
    assert q.cancel(jid)
    q.wait(jid, timeout=5)
    assert q.get(jid).status == "cancelled"
    q.shutdown()


def test_failed_job_records_error():
    q = JobQueue(on_event=lambda e: None)

    def work(progress, cancel):
        raise ValueError("boom")

    jid = q.submit("stack", {}, work)
    q.wait(jid, timeout=5)
    job = q.get(jid)
    assert job.status == "error"
    assert "boom" in job.error
    q.shutdown()
