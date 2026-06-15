import numpy as np
import pytest

from focusstack.io import load_image, save_image
from focusstack_server.projects import ProjectStore
from focusstack_server.retouch import RetouchManager
from focusstack_server.tiles import tile_path


def _project_with_two_results(tmp_path, source_shape=(64, 80, 3)):
    store = ProjectStore(tmp_path / "data")
    project = store.create(tmp_path / "proj", "P")
    images = (
        ("res_target", np.zeros((64, 80, 3), np.float32), "pmax"),
        ("res_source", np.ones(source_shape, np.float32), "dmap"),
    )
    for image_id, pixels, method in images:
        path = project.cache / f"{image_id}.tif"
        save_image(pixels, path, bit_depth=16)
        project.images[image_id] = {
            "kind": "result",
            "path": str(path),
            "method": method,
        }
    store.save(project)
    return store, project


def _stroke(source_id="res_source", **kw):
    stroke = {
        "source_id": source_id,
        "points": [[40, 32, 1.0]],
        "radius": 14.0,
        "hardness": 0.5,
        "opacity": 1.0,
        "mode": "normal",
    }
    stroke.update(kw)
    return stroke


def test_create_session_lists_sources_and_builds_working_pyramid(tmp_path):
    _store, project = _project_with_two_results(tmp_path)
    result = RetouchManager(tmp_path / "data").create(project.id, "res_target")
    assert any(source["image_id"] == "res_source" for source in result["sources"])
    assert tile_path(project.cache / "tiles", result["working_image_id"], 0, 0, 0).exists()


def test_stroke_returns_dirty_tiles_and_persists(tmp_path):
    _store, project = _project_with_two_results(tmp_path)
    manager = RetouchManager(tmp_path / "data")
    session_id = manager.create(project.id, "res_target")["session_id"]

    result = manager.stroke(session_id, _stroke())

    assert result["dirty_tiles"]
    assert result["rev"] == 1
    reloaded = ProjectStore(tmp_path / "data").get(project.id)
    record = next(record for record in reloaded.retouch if record["id"] == session_id)
    assert len(record["strokes"]) == 1


def test_empty_mutations_do_not_advance_revision(tmp_path):
    _store, project = _project_with_two_results(tmp_path)
    manager = RetouchManager(tmp_path / "data")
    session_id = manager.create(project.id, "res_target")["session_id"]

    result = manager.stroke(
        session_id,
        _stroke(points=[[-99, -99, 1.0]], radius=4.0),
    )

    assert result == {"rev": 0, "dirty_tiles": []}
    assert manager.undo(session_id) == {"rev": 0, "dirty_tiles": []}
    assert manager.redo(session_id) == {"rev": 0, "dirty_tiles": []}


def test_undo_redo_flatten_determinism(tmp_path):
    _store, project = _project_with_two_results(tmp_path)
    manager = RetouchManager(tmp_path / "data")
    session_id = manager.create(project.id, "res_target")["session_id"]
    manager.stroke(session_id, _stroke())
    image_a = manager.flatten(session_id, "A")["image_id"]
    manager.undo(session_id)
    manager.redo(session_id)
    image_b = manager.flatten(session_id, "B")["image_id"]

    reloaded = ProjectStore(tmp_path / "data").get(project.id)
    pixels_a = load_image(reloaded.images[image_a]["path"]).pixels
    pixels_b = load_image(reloaded.images[image_b]["path"]).pixels
    assert np.array_equal(pixels_a, pixels_b)


def test_session_rehydrates_after_restart_and_preserves_revision(tmp_path):
    _store, project = _project_with_two_results(tmp_path)
    manager = RetouchManager(tmp_path / "data")
    session_id = manager.create(project.id, "res_target")["session_id"]
    manager.stroke(session_id, _stroke())

    restarted = RetouchManager(tmp_path / "data")
    assert restarted.stroke(session_id, _stroke())["rev"] == 2


def test_invalid_source_is_rejected(tmp_path):
    _store, project = _project_with_two_results(tmp_path)
    manager = RetouchManager(tmp_path / "data")
    session_id = manager.create(project.id, "res_target")["session_id"]
    with pytest.raises(ValueError, match="not available"):
        manager.stroke(session_id, _stroke("not-a-source"))


def test_dimension_mismatch_is_rejected(tmp_path):
    _store, project = _project_with_two_results(tmp_path, source_shape=(32, 40, 3))
    with pytest.raises(ValueError, match="shape"):
        RetouchManager(tmp_path / "data").create(project.id, "res_target")


def test_unknown_target_and_session_raise_lookup(tmp_path):
    _store, project = _project_with_two_results(tmp_path)
    manager = RetouchManager(tmp_path / "data")
    with pytest.raises(LookupError):
        manager.create(project.id, "missing")
    with pytest.raises(LookupError):
        manager.stroke("missing", _stroke())
