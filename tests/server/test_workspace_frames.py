import numpy as np
from fastapi.testclient import TestClient

from focusstack.io import save_image
from focusstack_server.main import create_app


def _proj(tmp_path):
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    pid = c.post("/api/projects", json={}).json()["id"]
    return c, pid


def _frames(c, pid):
    return c.get(f"/api/projects/{pid}").json()["frames"]


def test_add_folder_then_dedupe_then_file(tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    for i in range(3):
        save_image(np.zeros((16, 16, 3), np.float32), src / f"f_{i:03d}.tif", bit_depth=16)
    extra = tmp_path / "extra.tif"
    save_image(np.zeros((16, 16, 3), np.float32), extra, bit_depth=16)

    c, pid = _proj(tmp_path)
    r = c.post(f"/api/projects/{pid}/frames/add", json={"paths": [str(src)]})
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True
    assert len(_frames(c, pid)) == 3

    # re-adding the same folder must not duplicate
    c.post(f"/api/projects/{pid}/frames/add", json={"paths": [str(src)]})
    assert len(_frames(c, pid)) == 3

    # an individual file appends
    c.post(f"/api/projects/{pid}/frames/add", json={"paths": [str(extra)]})
    assert len(_frames(c, pid)) == 4


def test_remove_frames(tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    paths = []
    for i in range(3):
        p = src / f"f_{i:03d}.tif"
        save_image(np.zeros((16, 16, 3), np.float32), p, bit_depth=16)
        paths.append(str(p))
    c, pid = _proj(tmp_path)
    c.post(f"/api/projects/{pid}/frames/add", json={"paths": [str(src)]})
    r = c.post(f"/api/projects/{pid}/frames/remove", json={"paths": [paths[0]]})
    assert r.status_code == 200, r.text
    remaining = _frames(c, pid)
    assert paths[0] not in remaining
    assert len(remaining) == 2
