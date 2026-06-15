import numpy as np
from fastapi.testclient import TestClient

from focusstack.io import save_image
from focusstack_server.main import create_app
from focusstack_server.tiles import build_pyramid


def test_build_pyramid_levels_and_tiles(tmp_path):
    img = np.random.default_rng(0).uniform(0, 1, (600, 800, 3)).astype(np.float32)
    out = tmp_path / "tiles"
    levels = build_pyramid(img, out)
    assert levels >= 1
    # level 0 is the coarsest single-ish tile; max level is full-res
    assert (out / f"{levels - 1}").is_dir()
    # a known tile file exists at max zoom
    assert any((out / f"{levels - 1}").glob("*.jpg"))


def test_viewer_serves_tile(tmp_path):
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    pid = c.post("/api/projects", json={"path": str(tmp_path / "proj"), "name": "P"}).json()["id"]
    # register an image manually via a result-like file
    proj_dir = tmp_path / "proj"
    img = np.random.default_rng(1).uniform(0, 1, (300, 400, 3)).astype(np.float32)
    save_image(img, proj_dir / "cache" / "img.tif", bit_depth=16)
    r = c.post(f"/api/projects/{pid}/viewer/register",
               json={"path": str(proj_dir / "cache" / "img.tif")})
    assert r.status_code == 200, r.text
    image_id = r.json()["image_id"]
    levels = r.json()["levels"]
    t = c.get(f"/api/viewer/{image_id}/tile/{levels - 1}/0/0")
    assert t.status_code == 200
    assert t.headers["content-type"].startswith("image/")
