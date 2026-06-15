from pathlib import Path

from fastapi.testclient import TestClient

from focusstack_server.main import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(data_dir=tmp_path))


def test_system_reports_device_and_versions(tmp_path):
    r = _client(tmp_path).get("/api/system")
    assert r.status_code == 200
    body = r.json()
    assert body["device"] in ("cuda", "mps", "cpu")
    assert "version" in body and "torch" in body


def test_algorithms_lists_registry_with_param_metadata(tmp_path):
    r = _client(tmp_path).get("/api/algorithms")
    assert r.status_code == 200
    algos = r.json()
    names = {a["name"] for a in algos}
    assert {"pmax", "dmap", "weighted", "slab"} <= names
    pmax = next(a for a in algos if a["name"] == "pmax")
    assert any(p["name"] == "selection_smoothing" for p in pmax["params"])
    p0 = pmax["params"][0]
    assert {"name", "label", "type", "default"} <= set(p0)
