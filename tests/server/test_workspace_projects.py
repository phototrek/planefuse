from fastapi.testclient import TestClient

from focusstack_server.main import create_app


def _c(tmp_path):
    return TestClient(create_app(data_dir=tmp_path / "data"))


def test_create_scratch_project_with_empty_path(tmp_path):
    c = _c(tmp_path)
    r = c.post("/api/projects", json={})           # no path, no name
    assert r.status_code == 200, r.text
    proj = r.json()
    # scratch dir lives under <data_dir>/scratch/<id>
    assert (tmp_path / "data" / "scratch" / proj["id"]).is_dir()
    assert proj["ui_state"]["saved"] is False
    # still listable/gettable like any project
    assert c.get(f"/api/projects/{proj['id']}").json()["id"] == proj["id"]


def test_create_with_explicit_path_still_works(tmp_path):
    c = _c(tmp_path)
    proj_dir = tmp_path / "mine"
    proj = c.post("/api/projects", json={"path": str(proj_dir), "name": "Mine"}).json()
    assert (proj_dir / "project.json").exists()
    assert proj["name"] == "Mine"
    assert proj["ui_state"]["saved"] is False
