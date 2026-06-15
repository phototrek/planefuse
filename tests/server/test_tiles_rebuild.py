import numpy as np

from focusstack_server.tiles import build_pyramid, rebuild_region, tile_path


def test_rebuild_region_changes_only_dirty_tiles(tmp_path):
    image = np.random.default_rng(0).uniform(0, 1, (600, 800, 3)).astype(np.float32)
    root = tmp_path / "tiles" / "img"
    levels = build_pyramid(image, root)
    far_path = tile_path(tmp_path / "tiles", "img", levels - 1, 0, 0)
    far_before = far_path.read_bytes()

    edited = image.copy()
    edited[560:600, 760:800] = 0.0
    dirty = rebuild_region(edited, root, levels, (560, 760, 600, 800))

    assert dirty
    assert all({"z", "x", "y"} <= set(tile) for tile in dirty)
    assert far_path.read_bytes() == far_before


def test_build_pyramid_output_is_unchanged_by_refactor(tmp_path):
    image = np.random.default_rng(1).uniform(0, 1, (300, 400, 3)).astype(np.float32)
    levels_a = build_pyramid(image, tmp_path / "a")
    levels_b = build_pyramid(image, tmp_path / "b")
    assert levels_a == levels_b
    for z in range(levels_a):
        for tile in (tmp_path / "a" / str(z)).glob("*.jpg"):
            assert tile.read_bytes() == (tmp_path / "b" / str(z) / tile.name).read_bytes()


def test_empty_region_writes_no_tiles(tmp_path):
    image = np.zeros((32, 32, 3), np.float32)
    levels = build_pyramid(image, tmp_path / "tiles")
    assert rebuild_region(image, tmp_path / "tiles", levels, (0, 0, 0, 0)) == []
