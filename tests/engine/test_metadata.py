from pathlib import Path

from planefuse.io.metadata import ImageMetadata, ProcessingDomain


def test_processing_domains_are_stable_serialized_values():
    assert ProcessingDomain.RENDERED_RGB.value == "rendered_rgb"
    assert ProcessingDomain.SCENE_LINEAR_CAMERA_RGB.value == "scene_linear_camera_rgb"


def test_stack_safe_metadata_removes_single_exposure_focus_fields(tmp_path: Path):
    metadata = ImageMetadata(
        source_path=tmp_path / "frame.nef",
        exif={
            "Exif.Image.Make": "Nikon",
            "Exif.Photo.SubjectDistance": "0.42 m",
            "Exif.Photo.FocusDistance": "420 mm",
        },
        xmp={
            "Xmp.dc.creator": "Photographer",
            "Xmp.aux.ApproximateFocusDistance": "0.42",
        },
    )

    safe = metadata.without_single_exposure_fields()

    assert safe.exif == {"Exif.Image.Make": "Nikon"}
    assert safe.xmp == {"Xmp.dc.creator": "Photographer"}
    assert metadata.exif["Exif.Photo.SubjectDistance"] == "0.42 m"
