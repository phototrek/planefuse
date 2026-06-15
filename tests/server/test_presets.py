from fastapi.testclient import TestClient

from focusstack_server.main import create_app


def _c(tmp_path):
    return TestClient(create_app(data_dir=tmp_path / "data"))


def test_presets_crud(tmp_path):
    c = _c(tmp_path)
    assert c.get("/api/presets").json() == []
    c.post("/api/presets", json={"name": "macro", "params": {"method": "dmap"}})
    presets = c.get("/api/presets").json()
    assert presets[0]["name"] == "macro"
    assert c.delete("/api/presets/macro").status_code == 200
    assert c.get("/api/presets").json() == []


def test_ui_state_patch_persists(tmp_path):
    c = _c(tmp_path)
    pid = c.post("/api/projects", json={"path": str(tmp_path / "p"), "name": "P"}).json()["id"]
    c.patch(f"/api/projects/{pid}/ui-state", json={"screen": "viewer", "zoom": 2})
    assert c.get(f"/api/projects/{pid}").json()["ui_state"]["screen"] == "viewer"
