from pathlib import Path

import numpy as np
import pytest

from focusstack.io import ImageMetadata, load_image, save_image


@pytest.mark.parametrize(
    ("suffix", "bit_depth"),
    [(".tif", 16), (".jpg", 8), (".png", 16)],
)
def test_rendered_export_roundtrips_portable_metadata(tmp_path: Path, suffix: str, bit_depth: int):
    pixels = np.linspace(0.05, 0.95, 18 * 24 * 3, dtype=np.float32).reshape(18, 24, 3)
    metadata = ImageMetadata(
        source_path=tmp_path / "source.nef",
        icc=b"\x00\x00\x02\x00focusstack-test-icc",
        exif={
            "Exif.Image.Make": "FocusStack Camera Co",
            "Exif.Image.Model": "SameCam Pro",
            "Exif.Photo.SubjectDistance": "1/2",
        },
        xmp={
            "Xmp.dc.creator": "FocusStack Test",
            "Xmp.aux.ApproximateFocusDistance": "0.5",
        },
    )
    out = tmp_path / f"result{suffix}"

    save_image(pixels, out, bit_depth=bit_depth, metadata=metadata, provenance={"method": "pmax"})
    reopened = load_image(out)

    assert reopened.metadata.icc == metadata.icc
    assert reopened.metadata.exif["Exif.Image.Make"] == "FocusStack Camera Co"
    assert reopened.metadata.exif["Exif.Image.Model"] == "SameCam Pro"
    assert reopened.metadata.xmp["Xmp.dc.creator"] == "FocusStack Test"
    assert "Exif.Photo.SubjectDistance" not in reopened.metadata.exif
    assert "Xmp.aux.ApproximateFocusDistance" not in reopened.metadata.xmp
    assert reopened.metadata.xmp["Xmp.FocusStack.Provenance"]
