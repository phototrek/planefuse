from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from focusstack_server.main import create_app
from focusstack_server.projects import ProjectStore


def _c(tmp_path):
    return TestClient(create_app(data_dir=tmp_path / "data"))


def test_create_list_get_delete_project(tmp_path):
    c = _c(tmp_path)
    proj_dir = tmp_path / "proj1"
    r = c.post("/api/projects", json={"path": str(proj_dir), "name": "Proj 1"})
    assert r.status_code == 200, r.text
    pid = r.json()["id"]
    assert (proj_dir / "project.json").exists()      # project.json written
    assert (proj_dir / "cache").is_dir()

    assert any(p["id"] == pid for p in c.get("/api/projects").json())
    got = c.get(f"/api/projects/{pid}").json()
    assert got["name"] == "Proj 1"

    # DELETE only unregisters; it must NOT remove user files (SPEC §9)
    assert c.delete(f"/api/projects/{pid}").status_code == 200
    assert not any(p["id"] == pid for p in c.get("/api/projects").json())
    assert (proj_dir / "project.json").exists()       # files preserved


def test_get_unknown_project_404(tmp_path):
    assert _c(tmp_path).get("/api/projects/nope").status_code == 404


def test_projects_persist_across_app_restart(tmp_path):
    c1 = _c(tmp_path)
    pid = c1.post("/api/projects", json={"path": str(tmp_path / "p"), "name": "P"}).json()["id"]
    c2 = _c(tmp_path)  # new app, same data_dir
    assert any(p["id"] == pid for p in c2.get("/api/projects").json())


def test_concurrent_project_creation_never_clobbers_registry_entries(tmp_path):
    store = ProjectStore(tmp_path / "data")
    with ThreadPoolExecutor(max_workers=8) as pool:
        projects = list(
            pool.map(
                lambda index: store.create(tmp_path / f"project-{index}", f"P{index}"),
                range(20),
            )
        )
    assert {project.id for project in store.list()} == {project.id for project in projects}
