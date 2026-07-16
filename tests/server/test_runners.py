import time

import numpy as np
from fastapi.testclient import TestClient

from planefuse.io import save_image
from planefuse_server.main import create_app
from tests.engine.test_raw_loader import write_test_raw


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


def test_raw_stack_persists_domain_metadata_decoder_and_provenance(tmp_path):
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    pid = c.post("/api/projects", json={"path": str(tmp_path / "proj"), "name": "RAW"}).json()["id"]
    src = tmp_path / "raw-frames"
    src.mkdir()
    for index in range(2):
        write_test_raw(src / f"frame-{index}.dng")
    scan = c.post(f"/api/projects/{pid}/frames/scan", json={"path": str(src)})
    assert scan.status_code == 200, scan.text
    jid = c.post(
        f"/api/projects/{pid}/jobs",
        json={"type": "stack", "params": {"method": "weighted", "device": "cpu"}},
    ).json()["id"]
    job = _wait_job(c, jid)
    assert job["status"] == "done", job
    proj = c.get(f"/api/projects/{pid}").json()
    result = next(img for img in proj["images"].values() if img.get("kind") == "result")
    assert result["domain"] == "scene_linear_camera_rgb"
    assert result["metadata"]["unique_camera_model"] == "PlaneFuse Camera Co SameCam Pro"
    assert result["decoder"]["auto_brightness"] is False
    assert len(result["provenance"]["sources"]) == 2
    assert all("sha256" in source for source in result["provenance"]["sources"])
