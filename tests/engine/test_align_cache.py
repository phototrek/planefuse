import numpy as np
import pytest

from planefuse.align.cache import AlignedCache
from planefuse.errors import ValidationError
from planefuse.io import ImageMetadata, ProcessingDomain


def test_cache_roundtrip_frame_and_mask(tmp_path):
    cache = AlignedCache(tmp_path)
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, (32, 40, 3)).astype(np.float32)
    mask = np.ones((32, 40), dtype=bool)
    mask[:, -3:] = False
    cache.write(0, img, mask)

    frames = cache.frame_source()
    masks = cache.mask_source()
    assert len(frames) == 1
    np.testing.assert_allclose(frames.read(0), img, atol=1e-3)  # 16-bit tiff tolerance
    got_mask = masks.read(0)
    m2 = got_mask[..., 0] if got_mask.ndim == 3 else got_mask
    assert not m2[:, -1].any()
    assert m2[:, 0].all()


def test_cache_region_read(tmp_path):
    cache = AlignedCache(tmp_path)
    img = np.full((20, 30, 3), 0.5, dtype=np.float32)
    cache.write(0, img, np.ones((20, 30), dtype=bool))
    crop = cache.frame_source().read(0, region=(0, 0, 8, 8))
    assert crop.shape == (8, 8, 3)


def test_scene_linear_cache_preserves_float_headroom_and_manifest(tmp_path):
    metadata = ImageMetadata(source_path=tmp_path / "source.dng", decoder={"demosaic": "AHD"})
    cache = AlignedCache(
        tmp_path,
        domain=ProcessingDomain.SCENE_LINEAR_CAMERA_RGB,
        metadata=metadata,
        source_hash="raw-stack-a",
    )
    image = np.full((12, 14, 3), 1.25, dtype=np.float32)
    cache.write(0, image, np.ones((12, 14), dtype=bool))
    np.testing.assert_array_equal(cache.frame_source().read(0), image)
    assert cache.manifest["domain"] == "scene_linear_camera_rgb"
    assert cache.manifest["source_hash"] == "raw-stack-a"
    assert cache.manifest["decoder"] == {"demosaic": "AHD"}


def test_cache_rejects_cross_domain_reuse(tmp_path):
    AlignedCache(tmp_path, domain=ProcessingDomain.RENDERED_RGB, source_hash="rendered")
    with pytest.raises(ValidationError, match="cache manifest"):
        AlignedCache(
            tmp_path,
            domain=ProcessingDomain.SCENE_LINEAR_CAMERA_RGB,
            source_hash="raw",
        )
