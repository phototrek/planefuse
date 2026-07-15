import json

import numpy as np
from fastapi.testclient import TestClient

from focusstack.io import save_image
from focusstack_server.main import create_app
from focusstack_server.tiles import build_pyramid

from tests.engine.test_raw_loader import write_test_raw


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
    again = c.post(
        f"/api/projects/{pid}/viewer/register",
        json={"path": str(proj_dir / "cache" / "img.tif")},
    )
    assert again.json()["image_id"] == image_id
    assert c.get(f"/api/viewer/{image_id}/analysis").status_code == 200
    # Rendered images have no display variant; ?display=1 falls back to the base tile.
    base = c.get(f"/api/viewer/{image_id}/tile/0/0/0")
    display = c.get(f"/api/viewer/{image_id}/tile/0/0/0?display=1")
    assert display.status_code == 200
    assert display.content == base.content


def test_raw_register_builds_display_tonemap_variant(tmp_path):
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    proj_dir = tmp_path / "proj"
    pid = c.post("/api/projects", json={"path": str(proj_dir), "name": "RAW"}).json()["id"]
    frames = proj_dir / "frames"
    frames.mkdir(parents=True)
    raw_path = write_test_raw(frames / "frame.dng")

    r = c.post(f"/api/projects/{pid}/viewer/register", json={"path": str(raw_path)})
    assert r.status_code == 200, r.text
    image_id = r.json()["image_id"]

    display_dirs = list(tmp_path.rglob(f"tiles/{image_id}/display"))
    assert len(display_dirs) == 1 and display_dirs[0].is_dir()
    base = c.get(f"/api/viewer/{image_id}/tile/0/0/0")
    display = c.get(f"/api/viewer/{image_id}/tile/0/0/0?display=1")
    assert base.status_code == 200 and display.status_code == 200
    assert display.content != base.content

    # A registration whose variant predates the current tonemap revision (or
    # the feature entirely) is healed on the next register call.
    project_file = proj_dir / "project.json"
    project = json.loads(project_file.read_text())
    del project["images"][image_id]["display_tonemap"]
    del project["images"][image_id]["display_rev"]
    project_file.write_text(json.dumps(project))
    import shutil

    shutil.rmtree(display_dirs[0])
    again = c.post(f"/api/projects/{pid}/viewer/register", json={"path": str(raw_path)})
    assert again.json()["image_id"] == image_id
    assert display_dirs[0].is_dir()
    healed = c.get(f"/api/viewer/{image_id}/tile/0/0/0?display=1")
    assert healed.content == display.content


def test_raw_frame_thumb_display_toggle(tmp_path):
    c = TestClient(create_app(data_dir=tmp_path / "data"))
    proj_dir = tmp_path / "proj"
    pid = c.post("/api/projects", json={"path": str(proj_dir), "name": "RAW"}).json()["id"]
    frames = proj_dir / "frames"
    frames.mkdir(parents=True)
    raw_path = write_test_raw(frames / "frame.dng")
    added = c.post(f"/api/projects/{pid}/frames/add", json={"paths": [str(raw_path)]})
    assert added.status_code == 200, added.text

    linear = c.get(f"/api/projects/{pid}/frame-thumb", params={"path": str(raw_path)})
    tonemapped = c.get(
        f"/api/projects/{pid}/frame-thumb", params={"path": str(raw_path), "display": 1}
    )
    assert linear.status_code == 200 and tonemapped.status_code == 200
    assert tonemapped.content != linear.content
