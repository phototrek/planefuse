import time

import numpy as np
from fastapi.testclient import TestClient

from focusstack.io import load_image, save_image
from focusstack_server.main import create_app


def _wait(c, jid, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError("timeout")


def test_export_writes_destination(tmp_path):
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    pid = c.post("/api/projects", json={"path": str(tmp_path / "proj"), "name": "P"}).json()["id"]
    src = tmp_path / "proj" / "cache" / "result.tif"
    save_image(np.random.default_rng(0).uniform(0, 1, (32, 40, 3)).astype(np.float32),
               src, bit_depth=16)
    # register the result image_id on the project via the viewer register endpoint
    image_id = c.post(f"/api/projects/{pid}/viewer/register",
                      json={"path": str(src)}).json()["image_id"]
    dest = tmp_path / "out" / "final.jpg"
    r = c.post(f"/api/projects/{pid}/export",
               json={"image_id": image_id, "dest": str(dest), "format": "jpg", "jpeg_quality": 90})
    assert r.status_code == 200, r.text
    job = _wait(c, r.json()["id"])
    assert job["status"] == "done", job
    assert dest.exists()
    assert load_image(dest).pixels.shape == (32, 40, 3)
