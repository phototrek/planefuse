import time

from fastapi.testclient import TestClient

from focusstack.io import save_image
from focusstack_server.main import create_app
from tests.synthetic.generate import generate_stack


def _ready_project(tmp_path):
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    pid = c.post("/api/projects", json={"path": str(tmp_path / "proj"), "name": "P"}).json()["id"]
    src = tmp_path / "frames"
    src.mkdir()
    stack = generate_stack(h=64, w=80, n_frames=4, max_sigma=4.0, seed=5)
    for i, f in enumerate(stack.frames):
        save_image(f, src / f"f_{i:03d}.tif", bit_depth=16)
    c.post(f"/api/projects/{pid}/frames/scan", json={"path": str(src)})
    return c, pid


def _wait_job(c, jid, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_enqueue_stack_job_runs_to_completion(tmp_path):
    c, pid = _ready_project(tmp_path)
    r = c.post(f"/api/projects/{pid}/jobs",
               json={"type": "stack", "params": {"method": "pmax", "device": "cpu"}})
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    job = _wait_job(c, jid)
    assert job["status"] == "done", job
    # a result image_id was registered on the project
    proj = c.get(f"/api/projects/{pid}").json()
    assert any(img.get("kind") == "result" for img in proj["images"].values())


def test_jobs_list_and_unknown_404(tmp_path):
    c, pid = _ready_project(tmp_path)
    assert c.get("/api/jobs").status_code == 200
    assert c.get("/api/jobs/nope").status_code == 404
