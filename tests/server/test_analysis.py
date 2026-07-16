import numpy as np
from fastapi.testclient import TestClient

from planefuse.io import save_float_tiff, save_image
from planefuse_server.main import create_app


def _register(client: TestClient, project_id: str, path) -> str:
    response = client.post(
        f"/api/projects/{project_id}/viewer/register", json={"path": str(path)}
    )
    assert response.status_code == 200, response.text
    return response.json()["image_id"]


def test_histogram_has_256_bins_and_per_channel_clipping_counts(tmp_path):
    client = TestClient(create_app(data_dir=tmp_path / "data"))
    pid = client.post(
        "/api/projects", json={"path": str(tmp_path / "project"), "name": "P"}
    ).json()["id"]
    pixels = np.zeros((4, 5, 3), dtype=np.float32)
    pixels[..., 0] = np.linspace(0.0, 1.0, 20).reshape(4, 5)
    pixels[..., 1] = 0.5
    pixels[..., 2] = 1.0
    path = tmp_path / "project" / "cache" / "analysis.tif"
    save_image(pixels, path, bit_depth=16)
    image_id = _register(client, pid, path)

    response = client.get(f"/api/viewer/{image_id}/analysis")
    assert response.status_code == 200, response.text
    data = response.json()
    assert set(data["histograms"]) == {"red", "green", "blue", "luminance"}
    assert all(len(bins) == 256 for bins in data["histograms"].values())
    assert all(sum(bins) == 20 for bins in data["histograms"].values())
    assert data["clipping"]["shadows"] == {"red": 1, "green": 0, "blue": 0}
    assert data["clipping"]["highlights"] == {"red": 1, "green": 0, "blue": 20}


def test_analysis_and_viewer_accept_unclamped_float_working_result(tmp_path):
    client = TestClient(create_app(data_dir=tmp_path / "data"))
    pid = client.post(
        "/api/projects", json={"path": str(tmp_path / "project"), "name": "RAW"}
    ).json()["id"]
    pixels = np.linspace(-0.1, 1.2, 6 * 7 * 3, dtype=np.float32).reshape(6, 7, 3)
    path = tmp_path / "project" / "cache" / "raw-result.tif"
    save_float_tiff(pixels, path)
    project = client.get(f"/api/projects/{pid}").json()
    project["images"]["rawresult"] = {
        "kind": "result",
        "path": str(path),
        "storage": "float32_tiff",
        "domain": "scene_linear_camera_rgb",
    }
    # Use the project file as the persistence seam for this server-level fixture.
    import json

    (tmp_path / "project" / "project.json").write_text(json.dumps(project))
    image_id = _register(client, pid, path)
    analysis = client.get(f"/api/viewer/{image_id}/analysis").json()
    assert analysis["clipping"]["shadows"]["red"] > 0
    assert analysis["clipping"]["highlights"]["blue"] > 0
    tile = client.get(f"/api/viewer/{image_id}/tile/0/0/0")
    assert tile.status_code == 200


def test_display_analysis_tonemaps_scene_linear_results(tmp_path):
    client = TestClient(create_app(data_dir=tmp_path / "data"))
    pid = client.post(
        "/api/projects", json={"path": str(tmp_path / "project"), "name": "RAW"}
    ).json()["id"]
    pixels = np.full((6, 7, 3), 0.02, dtype=np.float32)
    path = tmp_path / "project" / "cache" / "dark-result.tif"
    save_float_tiff(pixels, path)
    project = client.get(f"/api/projects/{pid}").json()
    project["images"]["rawresult"] = {
        "kind": "result",
        "path": str(path),
        "storage": "float32_tiff",
        "domain": "scene_linear_camera_rgb",
        "metadata": {
            "source_path": str(path),
            "as_shot_neutral": [1.0, 1.0, 1.0],
            "color_matrix1": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
    }
    import json

    (tmp_path / "project" / "project.json").write_text(json.dumps(project))
    image_id = _register(client, pid, path)

    def luminance_center(data: dict) -> int:
        bins = data["histograms"]["luminance"]
        return max(range(256), key=lambda index: bins[index])

    linear = client.get(f"/api/viewer/{image_id}/analysis").json()
    display = client.get(f"/api/viewer/{image_id}/analysis?display=1").json()
    assert luminance_center(linear) < 16
    assert luminance_center(display) > 128
    # The two variants are cached under distinct keys and keep coexisting.
    assert client.get(f"/api/viewer/{image_id}/analysis").json() == linear
    assert client.get(f"/api/viewer/{image_id}/analysis?display=1").json() == display
