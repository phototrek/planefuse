from fastapi.testclient import TestClient

from planefuse.io import save_image
from planefuse_server.main import create_app
from tests.synthetic.generate import generate_stack


def test_ws_streams_job_progress(tmp_path):
    app = create_app(data_dir=tmp_path / "data")
    c = TestClient(app)
    pid = c.post("/api/projects", json={"path": str(tmp_path / "proj"), "name": "P"}).json()["id"]
    src = tmp_path / "frames"
    src.mkdir()
    for i, f in enumerate(generate_stack(h=48, w=48, n_frames=3, max_sigma=4.0, seed=1).frames):
        save_image(f, src / f"f_{i:03d}.tif", bit_depth=16)
    c.post(f"/api/projects/{pid}/frames/scan", json={"path": str(src)})

    with c.websocket_connect("/ws") as ws:
        jid = c.post(f"/api/projects/{pid}/jobs",
                     json={"type": "stack", "params": {"method": "pmax", "device": "cpu"}}).json()["id"]
        done_event = None
        for _ in range(200):
            evt = ws.receive_json()
            if evt.get("job_id") == jid and evt.get("status") == "done":
                done_event = evt
                break
        assert done_event is not None
        assert done_event["params"] == {"method": "pmax", "device": "cpu"}
