import planefuse_server.api.fs as fs
import numpy as np
from fastapi.testclient import TestClient

from planefuse.io import save_image
from planefuse_server.main import create_app


def test_fs_list_dirs_and_image_counts(tmp_path):
    (tmp_path / "sub").mkdir()
    save_image(np.zeros((8, 8, 3), np.float32), tmp_path / "a.tif", bit_depth=16)
    save_image(np.zeros((8, 8, 3), np.float32), tmp_path / "b.jpg")
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    r = c.get("/api/fs/list", params={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == str(tmp_path)
    names = {e["name"]: e for e in body["entries"]}
    assert "sub" in names and names["sub"]["is_dir"]
    assert body["image_count"] == 2  # 2 images directly in tmp_path


def test_fs_list_missing_path_is_400(tmp_path):
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    r = c.get("/api/fs/list", params={"path": str(tmp_path / "nope")})
    assert r.status_code == 400


# ── /api/fs/pick (native OS dialog, server-side) ─────────────────────────────

def test_pick_returns_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "_run_picker",
                        lambda mode: ["C:/frames/a.tif", "C:/frames/b.tif"])
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    r = c.post("/api/fs/pick", json={"mode": "files"})
    assert r.status_code == 200
    assert r.json()["paths"] == ["C:/frames/a.tif", "C:/frames/b.tif"]


def test_pick_cancelled_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "_run_picker", lambda mode: [])
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    r = c.post("/api/fs/pick", json={"mode": "directory"})
    assert r.status_code == 200
    assert r.json()["paths"] == []


def test_pick_failure_is_structured_error(tmp_path, monkeypatch):
    def boom(mode):
        raise RuntimeError("no display")
    monkeypatch.setattr(fs, "_run_picker", boom)
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    r = c.post("/api/fs/pick", json={"mode": "directory"})
    assert r.status_code == 500
    assert r.json()["error"] == "picker_failed"


# ── /api/fs/list hardening (the OneDrive 500 fix) ────────────────────────────

def test_entry_skips_unreadable():
    # OneDrive cloud placeholders / reparse points raise OSError on is_dir();
    # such entries must be skipped, not crash the whole listing with a 500.
    class Bad:
        name = "placeholder.tif"

        def is_dir(self):
            raise OSError("[WinError 1920] file cannot be accessed by the system")

    assert fs._entry(Bad()) is None  # type: ignore[arg-type]


def test_entry_normal_file(tmp_path):
    f = tmp_path / "a.tif"
    f.write_bytes(b"x")
    e = fs._entry(f)
    assert e is not None
    assert e["name"] == "a.tif"
    assert e["is_dir"] is False
