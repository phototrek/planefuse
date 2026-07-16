"""Typed engine exceptions (SPEC §12)."""


class PlaneFuseError(Exception):
    """Base for all engine errors."""


class ValidationError(PlaneFuseError):
    """Input frames failed validation (size/bit-depth/format)."""


class BackendError(PlaneFuseError):
    """Device unavailable or device-level failure."""


class AlignmentError(PlaneFuseError):
    """Alignment failed (M2; defined now so the hierarchy is complete)."""


class RawDecodeError(PlaneFuseError):
    """A camera RAW file could not be decoded without changing the RAW recipe."""


class DngExportError(PlaneFuseError):
    """A Linear DNG could not be written or failed post-write conformance checks."""


class DiskSpaceError(PlaneFuseError):
    """There is not enough free space for an atomic output."""
