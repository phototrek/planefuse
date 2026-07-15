"""FocusStack provenance payloads embedded in exported files."""

from __future__ import annotations

from typing import Any

from focusstack import __version__
from focusstack.io.metadata import ImageMetadata


def dng_provenance(
    metadata: ImageMetadata,
    supplied: dict[str, Any],
    *,
    encoding_offset: float,
    encoding_span: float,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "application": "FocusStack",
        "version": __version__,
        "processing_domain": "scene_linear_camera_rgb",
        "no_bake": {
            "white_balance": True,
            "exposure_style": True,
            "tone_curve": True,
            "denoise": True,
            "sharpen": True,
            "output_color_conversion": True,
        },
        "decoder": dict(metadata.decoder),
        "source_calibration": {
            "black_level": metadata.black_level,
            "white_level": metadata.white_level,
            "color_matrix1": metadata.color_matrix1,
            "color_matrix2": metadata.color_matrix2,
            "calibration_illuminant1": metadata.calibration_illuminant1,
            "calibration_illuminant2": metadata.calibration_illuminant2,
            "as_shot_neutral": metadata.as_shot_neutral,
        },
        "integer_mapping": {
            "offset": encoding_offset,
            "span": encoding_span,
            "formula": "scene_linear = code / 65535 * span + offset",
        },
        **supplied,
    }
