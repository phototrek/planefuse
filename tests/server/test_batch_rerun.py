import time

from fastapi.testclient import TestClient

from planefuse.io import save_image
from planefuse_server.main import create_app
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


def _wait(c, jid, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_stack_honors_frames_subset(tmp_path):
    # Discriminating test: the project itself has only 1 frame (a stack needs >=2,
    # so it would error), but the job passes an explicit 4-frame group. The job
    # succeeds only if the runner honors params["frames"].
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    pid = c.post("/api/projects", json={"path": str(tmp_path / "proj"), "name": "P"}).json()["id"]
    full = tmp_path / "full"
    full.mkdir()
    one = tmp_path / "one"
    one.mkdir()
    stack = generate_stack(h=64, w=80, n_frames=4, max_sigma=4.0, seed=5)
    group = []
    for i, f in enumerate(stack.frames):
        p = full / f"f_{i:03d}.tif"
        save_image(f, p, bit_depth=16)
        group.append(str(p))
    save_image(stack.frames[0], one / "solo.tif", bit_depth=16)

    c.post(f"/api/projects/{pid}/frames/scan", json={"path": str(one)})  # proj.frames = 1
    r = c.post(f"/api/projects/{pid}/jobs",
               json={"type": "stack", "params": {"method": "pmax", "device": "cpu", "frames": group}})
    jid = r.json()["id"]
    assert _wait(c, jid)["status"] == "done"


def test_job_dict_includes_params_for_rerun(tmp_path):
    c, pid = _ready_project(tmp_path)
    params = {"method": "pmax", "device": "cpu"}
    jid = c.post(f"/api/projects/{pid}/jobs", json={"type": "stack", "params": params}).json()["id"]
    job = c.get(f"/api/jobs/{jid}").json()
    assert "params" in job
    assert job["params"]["method"] == "pmax"


def test_select_job_persists_reviewable_proposal(tmp_path):
    c, pid = _ready_project(tmp_path)
    jid = c.post(
        f"/api/projects/{pid}/jobs",
        json={"type": "select", "params": {"device": "cpu", "select": {}}},
    ).json()["id"]
    job = _wait(c, jid)
    assert job["status"] == "done", job
    proposal = c.get(f"/api/projects/{pid}").json()["ui_state"]["selectionProposal"]
    assert sorted(proposal) == ["coverage", "kept", "params", "redundant", "warning"]
    assert proposal["kept"]
