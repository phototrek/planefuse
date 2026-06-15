import numpy as np
from fastapi.testclient import TestClient

from focusstack.io import save_image
from focusstack_server.main import create_app


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
