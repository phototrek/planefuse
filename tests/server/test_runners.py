import time

import numpy as np
from fastapi.testclient import TestClient

from focusstack.io import save_image
from focusstack_server.main import create_app


def _proj_with_frames(tmp_path, n=3):
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    pid = c.post("/api/projects", json={"path": str(tmp_path / "proj"), "name": "P"}).json()["id"]
    src = tmp_path / "frames"
    src.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        save_image(rng.uniform(0, 1, (32, 40, 3)).astype(np.float32),
                   src / f"f_{i:03d}.tif", bit_depth=16)
    c.post(f"/api/projects/{pid}/frames/scan", json={"path": str(src)})
    return c, pid


def _wait_job(c, jid, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish within timeout")


def test_stack_result_records_frame_count(tmp_path):
    c, pid = _proj_with_frames(tmp_path, n=3)
    jid = c.post(f"/api/projects/{pid}/jobs",
                 json={"type": "stack", "params": {"method": "pmax", "device": "cpu"}}).json()["id"]
    job = _wait_job(c, jid)
    assert job["status"] == "done", job
    proj = c.get(f"/api/projects/{pid}").json()
    results = [img for img in proj["images"].values() if img.get("kind") == "result"]
    assert results, proj["images"]
    assert results[0]["frames"] == 3


def test_multiple_stack_jobs_each_persist_a_result(tmp_path):
    # Two algorithms enqueued back-to-back must both keep their result image.
    # (Regression: each runner held a stale Project snapshot and clobbered the
    # other's result on save.)
    c, pid = _proj_with_frames(tmp_path, n=3)
    jids = [
        c.post(f"/api/projects/{pid}/jobs",
               json={"type": "stack", "params": {"method": m, "device": "cpu"}}).json()["id"]
        for m in ("pmax", "weighted")
    ]
    for jid in jids:
        assert _wait_job(c, jid)["status"] == "done"
    proj = c.get(f"/api/projects/{pid}").json()
    methods = sorted(img["method"] for img in proj["images"].values() if img.get("kind") == "result")
    assert methods == ["pmax", "weighted"], proj["images"]
