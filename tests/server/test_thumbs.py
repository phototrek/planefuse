import numpy as np
from fastapi.testclient import TestClient

from planefuse.io import save_image
from planefuse_server.main import create_app


def _proj(tmp_path):
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    pid = c.post("/api/projects", json={"path": str(tmp_path / "proj"), "name": "P"}).json()["id"]
    return c, pid


def test_frame_thumb_returns_jpeg(tmp_path):
    c, pid = _proj(tmp_path)
    src = tmp_path / "frames"
    src.mkdir()
    fp = src / "f_000.tif"
    save_image(np.random.default_rng(0).uniform(0, 1, (300, 400, 3)).astype(np.float32),
               fp, bit_depth=16)
    c.post(f"/api/projects/{pid}/frames/scan", json={"path": str(src)})
    r = c.get(f"/api/projects/{pid}/frame-thumb", params={"path": str(fp)})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")
    assert r.content[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_frame_thumb_rejects_path_not_in_project(tmp_path):
    c, pid = _proj(tmp_path)
    stray = tmp_path / "stray.tif"
    save_image(np.zeros((8, 8, 3), np.float32), stray, bit_depth=16)
    r = c.get(f"/api/projects/{pid}/frame-thumb", params={"path": str(stray)})
    assert r.status_code == 400


def test_spa_fallback_serves_index_when_ui_present(tmp_path, monkeypatch):
    # When a built UI exists, an unknown non-/api path returns index.html.
    import planefuse_server.main as m
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>PlaneFuse</title>")
    monkeypatch.setattr(m, "_static_dir", lambda: static)
    c = TestClient(m.create_app(data_dir=tmp_path / "data"))
    r = c.get("/viewer")
    assert r.status_code == 200
    assert "PlaneFuse" in r.text
    # /api still 404s as JSON, not the SPA shell
    assert c.get("/api/nope").status_code == 404
