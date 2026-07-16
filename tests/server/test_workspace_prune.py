import os
import time

from fastapi.testclient import TestClient

from planefuse_server.main import create_app


def _client(tmp_path):
    return TestClient(create_app(data_dir=tmp_path / "data"))


def _age(data_dir, pid, days):
    pj = data_dir / "scratch" / pid / "project.json"
    old = time.time() - days * 86400
    os.utime(pj, (old, old))


def test_prune_removes_old_unsaved_scratch_only(tmp_path):
    data = tmp_path / "data"
    c = _client(tmp_path)
    old_unsaved = c.post("/api/projects", json={}).json()["id"]
    old_saved = c.post("/api/projects", json={}).json()["id"]
    recent_unsaved = c.post("/api/projects", json={}).json()["id"]

    c.patch(f"/api/projects/{old_saved}", json={"saved": True})
    _age(data, old_unsaved, 30)
    _age(data, old_saved, 30)
    # recent_unsaved keeps its fresh mtime

    c2 = _client(tmp_path)  # new app over same data_dir triggers prune on startup
    ids = {p["id"] for p in c2.get("/api/projects").json()}
    assert old_unsaved not in ids                      # pruned
    assert not (data / "scratch" / old_unsaved).exists()
    assert old_saved in ids                            # saved -> kept
    assert recent_unsaved in ids                       # recent -> kept
