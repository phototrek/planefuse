import numpy as np
from fastapi.testclient import TestClient

from focusstack.io import save_image
from focusstack_server.main import create_app


def _proj(tmp_path):
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    pid = c.post("/api/projects", json={"path": str(tmp_path / "proj"), "name": "P"}).json()["id"]
    return c, pid


def test_scan_imports_and_validates(tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    for i in range(3):
        save_image(np.zeros((16, 16, 3), np.float32), src / f"f_{i:03d}.tif", bit_depth=16)
    c, pid = _proj(tmp_path)
    r = c.post(f"/api/projects/{pid}/frames/scan", json={"path": str(src)})
    assert r.status_code == 200, r.text
    report = r.json()
    assert report["ok"] is True
    assert len(report["files"]) == 3
    # frames recorded in the project
    assert len(c.get(f"/api/projects/{pid}").json()["frames"]) == 3


def test_scan_reports_size_mismatch(tmp_path):
    src = tmp_path / "frames"
    src.mkdir()
    save_image(np.zeros((16, 16, 3), np.float32), src / "a.tif", bit_depth=16)
    save_image(np.zeros((20, 16, 3), np.float32), src / "b.tif", bit_depth=16)
    c, pid = _proj(tmp_path)
    report = c.post(f"/api/projects/{pid}/frames/scan", json={"path": str(src)}).json()
    assert report["ok"] is False
    statuses = {f["name"]: f["status"] for f in report["files"]}
    assert statuses["b.tif"] == "wrong_size"
