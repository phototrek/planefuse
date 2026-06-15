import numpy as np

from focusstack.retouch import RetouchSession, Stroke


def _session():
    base = np.zeros((64, 80, 3), np.float32)
    source = np.ones((64, 80, 3), np.float32)
    return RetouchSession(base, {"white": source})


def _stroke(**kw):
    base = {
        "source_id": "white",
        "points": [(40.0, 32.0, 1.0)],
        "radius": 12.0,
        "hardness": 0.5,
        "opacity": 1.0,
        "mode": "normal",
    }
    base.update(kw)
    return Stroke(**base)


def test_normal_blends_source_in():
    session = _session()
    session.apply(_stroke())
    assert session.composite[32, 40, 0] > 0.5
    assert session.composite[0, 0, 0] == 0.0


def test_erase_restores_base():
    session = _session()
    session.apply(_stroke())
    session.apply(_stroke(mode="erase"))
    assert session.composite[32, 40, 0] < 0.5


def test_empty_stroke_not_recorded():
    session = _session()
    bbox = session.apply(_stroke(points=[(-99.0, -99.0, 1.0)], radius=4.0))
    assert bbox == (0, 0, 0, 0)
    assert session.strokes == []


def test_undo_redo_determinism_across_checkpoint():
    session = _session()
    rng = np.random.default_rng(0)
    for _ in range(25):
        x, y = rng.uniform(5, 75), rng.uniform(5, 59)
        session.apply(_stroke(points=[(float(x), float(y), 1.0)]))
    snapshot = session.composite.copy()

    for _ in range(25):
        session.undo()
    assert np.array_equal(session.composite, session.base)

    for _ in range(25):
        session.redo()
    assert np.array_equal(session.composite, snapshot)


def test_undo_to_middle_equals_fresh_replay():
    base = np.zeros((64, 80, 3), np.float32)
    sources = {"white": np.ones((64, 80, 3), np.float32)}
    strokes = [Stroke("white", [(float(10 + i), 32.0, 1.0)], 8.0) for i in range(23)]

    undone = RetouchSession(base, sources)
    for stroke in strokes:
        undone.apply(stroke)
    for _ in range(8):
        undone.undo()

    replayed = RetouchSession(base, sources)
    for stroke in strokes[:15]:
        replayed.apply(stroke)

    assert np.array_equal(undone.composite, replayed.composite)
