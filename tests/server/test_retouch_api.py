import time

import numpy as np
from fastapi.testclient import TestClient

from focusstack.io import load_image, save_image
from focusstack_server.main import create_app
from focusstack_server.projects import ProjectStore
from tests.synthetic.generate import generate_stack


def _wait_job(client, job_id, timeout=30):
    started = time.time()
    while time.time() - started < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"done", "error", "cancelled"}:
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _client_with_results(tmp_path):
    data_dir = tmp_path / "data"
    client = TestClient(create_app(data_dir=data_dir))
    project_id = client.post(
        "/api/projects",
        json={"path": str(tmp_path / "proj"), "name": "P"},
    ).json()["id"]
    source_dir = tmp_path / "frames"
    source_dir.mkdir()
    stack = generate_stack(h=48, w=60, n_frames=4, max_sigma=4.0, seed=8)
    for index, frame in enumerate(stack.frames):
        save_image(frame, source_dir / f"f_{index:03d}.tif", bit_depth=16)
    scan = client.post(
        f"/api/projects/{project_id}/frames/scan",
        json={"path": str(source_dir)},
    )
    assert scan.status_code == 200, scan.text

    jobs = (
        {"method": "pmax", "device": "cpu"},
        {
            "method": "dmap",
            "device": "cpu",
            "algo_params": {
                "estimation_radius": 2,
                "contrast_threshold": 7.0,
                "smoothing_radius": 4,
            },
        },
    )
    for params in jobs:
        response = client.post(
            f"/api/projects/{project_id}/jobs",
            json={"type": "stack", "params": params},
        )
        assert response.status_code == 200, response.text
        job = _wait_job(client, response.json()["id"])
        assert job["status"] == "done", job

    project = client.get(f"/api/projects/{project_id}").json()
    results = [
        image_id
        for image_id, info in project["images"].items()
        if info.get("kind") == "result"
    ]
    assert len(results) == 2
    return client, data_dir, project_id, results


def _stroke(source_id):
    return {
        "source_id": source_id,
        "points": [[30, 24, 1.0]],
        "radius": 12.0,
        "hardness": 0.5,
        "opacity": 1.0,
        "mode": "normal",
    }


def test_retouch_lifecycle_and_determinism(tmp_path):
    client, data_dir, project_id, (target_id, source_id) = _client_with_results(tmp_path)
    project = client.get(f"/api/projects/{project_id}").json()
    target_path = project["images"][target_id]["path"]
    target_before = load_image(target_path).pixels.copy()

    created = client.post(
        f"/api/projects/{project_id}/retouch",
        json={"target_image_id": target_id},
    )
    assert created.status_code == 200, created.text
    session_id = created.json()["session_id"]
    working_id = created.json()["working_image_id"]
    assert any(source["image_id"] == source_id for source in created.json()["sources"])

    project = client.get(f"/api/projects/{project_id}").json()
    level = project["images"][working_id]["levels"] - 1
    tile_url = f"/api/viewer/{working_id}/tile/{level}/0/0"
    tile_before = client.get(tile_url)
    assert tile_before.status_code == 200

    painted = client.post(f"/api/retouch/{session_id}/stroke", json=_stroke(source_id))
    assert painted.status_code == 200, painted.text
    assert painted.json()["rev"] == 1
    assert painted.json()["dirty_tiles"]
    assert client.get(tile_url).content != tile_before.content

    image_a = client.post(
        f"/api/retouch/{session_id}/flatten",
        json={"name": "A"},
    ).json()["image_id"]
    assert client.post(f"/api/retouch/{session_id}/undo").status_code == 200
    assert client.post(f"/api/retouch/{session_id}/redo").status_code == 200
    image_b = client.post(
        f"/api/retouch/{session_id}/flatten",
        json={"name": "B"},
    ).json()["image_id"]

    project = client.get(f"/api/projects/{project_id}").json()
    pixels_a = load_image(project["images"][image_a]["path"]).pixels
    pixels_b = load_image(project["images"][image_b]["path"]).pixels
    assert np.array_equal(pixels_a, pixels_b)
    assert np.array_equal(load_image(target_path).pixels, target_before)
    assert project["images"][image_a]["kind"] == "result"

    restarted = TestClient(create_app(data_dir=data_dir))
    sessions = restarted.get(f"/api/projects/{project_id}/retouch").json()["sessions"]
    assert any(session["id"] == session_id for session in sessions)


def test_retouch_structured_errors(tmp_path):
    data_dir = tmp_path / "data"
    client = TestClient(create_app(data_dir=data_dir))
    assert client.post(
        "/api/retouch/missing/stroke",
        json=_stroke("missing"),
    ).status_code == 404

    project_id = client.post(
        "/api/projects",
        json={"path": str(tmp_path / "proj"), "name": "P"},
    ).json()["id"]
    store = ProjectStore(data_dir)
    project = store.get(project_id)
    for image_id, shape in (("target", (32, 40, 3)), ("source", (16, 20, 3))):
        path = project.cache / f"{image_id}.tif"
        save_image(np.zeros(shape, np.float32), path, bit_depth=16)
        project.images[image_id] = {"kind": "result", "path": str(path), "method": "pmax"}
    store.save(project)

    mismatch = client.post(
        f"/api/projects/{project_id}/retouch",
        json={"target_image_id": "target"},
    )
    assert mismatch.status_code == 400
    assert mismatch.json()["error"] == "bad_request"


def test_working_image_export_fails_cleanly(tmp_path):
    client, _data_dir, project_id, (target_id, _source_id) = _client_with_results(tmp_path)
    working_id = client.post(
        f"/api/projects/{project_id}/retouch",
        json={"target_image_id": target_id},
    ).json()["working_image_id"]

    response = client.post(
        f"/api/projects/{project_id}/export",
        json={"image_id": working_id, "dest": str(tmp_path / "working.tif")},
    )
    assert response.status_code == 200
    job = _wait_job(client, response.json()["id"])
    assert job["status"] == "error"
    assert "not exportable" in job["error"]
