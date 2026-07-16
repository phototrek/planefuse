"""Stable API error codes for typed engine and workflow failures."""

from __future__ import annotations

from planefuse.errors import (
    AlignmentError,
    BackendError,
    DiskSpaceError,
    DngExportError,
    PlaneFuseError,
    RawDecodeError,
    ValidationError,
)

_ERRORS: tuple[tuple[type[Exception], str, int], ...] = (
    (DiskSpaceError, "insufficient_disk_space", 507),
    (DngExportError, "dng_export_failed", 422),
    (RawDecodeError, "raw_decode_failed", 422),
    (AlignmentError, "alignment_failed", 422),
    (ValidationError, "validation_failed", 422),
    (BackendError, "backend_failed", 503),
    (PlaneFuseError, "planefuse_failed", 500),
    (ValueError, "invalid_request", 400),
    (OSError, "io_failed", 500),
)


def classify_error(error: Exception) -> tuple[str, int]:
    for error_type, code, status in _ERRORS:
        if isinstance(error, error_type):
            return code, status
    return "internal_error", 500
