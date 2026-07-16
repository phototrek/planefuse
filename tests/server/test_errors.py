import time

import numpy as np
from fastapi.testclient import TestClient

from planefuse.errors import DiskSpaceError
from planefuse.io import save_image
from planefuse_server.main import create_app


def _wait(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"done", "error", "cancelled"}:
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_job_failure_exposes_stable_typed_error_code(tmp_path):
    client = TestClient(create_app(data_dir=tmp_path / "data"))
    pid = client.post(
        "/api/projects", json={"path": str(tmp_path / "project"), "name": "P"}
    ).json()["id"]
    source = tmp_path / "project" / "cache" / "rendered.tif"
    save_image(np.zeros((8, 10, 3), dtype=np.float32), source)
    image_id = client.post(
        f"/api/projects/{pid}/viewer/register", json={"path": str(source)}
    ).json()["image_id"]
    response = client.post(
        f"/api/projects/{pid}/export",
        json={"image_id": image_id, "dest": str(tmp_path / "bad.dng"), "format": "dng"},
    )
    job = _wait(client, response.json()["id"])
    assert job["status"] == "error"
    assert job["error_code"] == "dng_export_failed"
    assert "scene_linear_camera_rgb" in job["error"]


def test_sync_engine_failure_uses_same_structured_error_contract(tmp_path, monkeypatch):
    app = create_app(data_dir=tmp_path / "data")

    def fail_disk_space(*_args, **_kwargs):
        raise DiskSpaceError("disk is full")

    monkeypatch.setattr("planefuse_server.projects.ProjectStore.create", fail_disk_space)
    response = TestClient(app).post("/api/projects", json={"name": "P"})
    assert response.status_code == 507
    assert response.json() == {
        "error": "insufficient_disk_space",
        "detail": "disk is full",
    }
