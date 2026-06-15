import time

from fastapi.testclient import TestClient

from focusstack.io import save_image
from focusstack_server.main import create_app
from tests.synthetic.generate import generate_stack


def _wait(c, jid, timeout=40):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.1)
    raise AssertionError("timeout")


def test_full_project_lifecycle(tmp_path):
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    # create
    pid = c.post("/api/projects", json={"path": str(tmp_path / "proj"), "name": "Macro"}).json()["id"]
    # import
    src = tmp_path / "frames"
    src.mkdir()
    for i, f in enumerate(generate_stack(h=80, w=100, n_frames=5, max_sigma=4.0, seed=2).frames):
        save_image(f, src / f"f_{i:03d}.tif", bit_depth=16)
    report = c.post(f"/api/projects/{pid}/frames/scan", json={"path": str(src)}).json()
    assert report["ok"]
    # stack
    jid = c.post(f"/api/projects/{pid}/jobs",
                 json={"type": "stack", "params": {"method": "pmax", "device": "cpu"}}).json()["id"]
    job = _wait(c, jid)
    assert job["status"] == "done", job
    result_id = job["result"]["image_id"]
    # view: register a pyramid for the result, fetch a tile
    reg = c.post(f"/api/projects/{pid}/viewer/register",
                 json={"path": str(tmp_path / "proj" / "cache" / f"{result_id}.tif")}).json()
    t = c.get(f"/api/viewer/{reg['image_id']}/tile/{reg['levels'] - 1}/0/0")
    assert t.status_code == 200
    # export
    dest = tmp_path / "out.tif"
    ejid = c.post(f"/api/projects/{pid}/export",
                  json={"image_id": result_id, "dest": str(dest)}).json()["id"]
    assert _wait(c, ejid)["status"] == "done"
    assert dest.exists()


def test_job_cancellation_over_http(tmp_path):
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    pid = c.post("/api/projects", json={"path": str(tmp_path / "proj"), "name": "P"}).json()["id"]
    src = tmp_path / "frames"
    src.mkdir()
    # enough frames that the job runs long enough to cancel mid-flight
    for i, f in enumerate(generate_stack(h=200, w=260, n_frames=30, max_sigma=4.0, seed=3).frames):
        save_image(f, src / f"f_{i:03d}.tif", bit_depth=16)
    c.post(f"/api/projects/{pid}/frames/scan", json={"path": str(src)})
    jid = c.post(f"/api/projects/{pid}/jobs",
                 json={"type": "stack", "params": {"method": "pmax", "device": "cpu",
                                                   "align": {"max_long_edge": 256}}}).json()["id"]
    time.sleep(0.3)
    c.delete(f"/api/jobs/{jid}")
    assert _wait(c, jid)["status"] == "cancelled"
