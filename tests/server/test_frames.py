from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from focusstack.io import save_image
from focusstack_server.main import create_app
from tests.engine.test_raw_loader import write_test_raw


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


def test_scan_reports_raw_domain_camera_decoder_and_compatibility(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    write_test_raw(src / "a.dng")
    write_test_raw(src / "b.dng", model="Different Camera")
    c, pid = _proj(tmp_path)

    report = c.post(f"/api/projects/{pid}/frames/scan", json={"path": str(src)}).json()

    assert report["domain"] == "scene_linear_camera_rgb"
    assert report["camera"] == "FocusStack Camera Co SameCam Pro"
    assert report["decoder"]["demosaic"] == "AHD"
    statuses = {item["name"]: item["status"] for item in report["files"]}
    assert statuses == {"a.dng": "ok", "b.dng": "incompatible_camera"}


def test_auto_group_prefers_exif_capture_time_gaps(tmp_path, monkeypatch):
    src = tmp_path / "frames"
    src.mkdir()
    names = ["image_001.tif", "image_002.tif", "image_003.tif"]
    captures = {
        names[0]: "2026:07:15 12:00:00",
        names[1]: "2026:07:15 12:00:04",
        names[2]: "2026:07:15 12:00:20",
    }
    for name in names:
        save_image(np.zeros((16, 16, 3), np.float32), src / name, bit_depth=16)
    c, pid = _proj(tmp_path)
    c.post(f"/api/projects/{pid}/frames/scan", json={"path": str(src)})

    def fake_capture_time(path):
        return captures[Path(path).name]

    monkeypatch.setattr("focusstack_server.api.frames.capture_time_from_file", fake_capture_time)
    response = c.post(f"/api/projects/{pid}/frames/auto-group?max_seconds=10")

    assert response.status_code == 200
    assert [[Path(path).name for path in group] for group in response.json()["groups"]] == [
        names[:2],
        names[2:],
    ]


def test_auto_group_falls_back_to_filename_gaps_without_capture_time(tmp_path, monkeypatch):
    src = tmp_path / "frames"
    src.mkdir()
    names = ["image_001.tif", "image_002.tif", "image_010.tif"]
    for name in names:
        save_image(np.zeros((16, 16, 3), np.float32), src / name, bit_depth=16)
    c, pid = _proj(tmp_path)
    c.post(f"/api/projects/{pid}/frames/scan", json={"path": str(src)})
    monkeypatch.setattr(
        "focusstack_server.api.frames.capture_time_from_file",
        lambda path: None,
    )

    response = c.post(f"/api/projects/{pid}/frames/auto-group?max_gap=3")

    assert response.status_code == 200
    assert [[Path(path).name for path in group] for group in response.json()["groups"]] == [
        names[:2],
        names[2:],
    ]


def test_auto_group_rejects_negative_gap_thresholds(tmp_path):
    c, pid = _proj(tmp_path)
    assert c.post(f"/api/projects/{pid}/frames/auto-group?max_gap=-1").status_code == 422
    assert c.post(f"/api/projects/{pid}/frames/auto-group?max_seconds=-1").status_code == 422
