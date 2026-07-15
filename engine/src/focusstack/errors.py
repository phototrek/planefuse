"""Typed engine exceptions (SPEC §12)."""


class FocusStackError(Exception):
    """Base for all engine errors."""


class ValidationError(FocusStackError):
    """Input frames failed validation (size/bit-depth/format)."""


class BackendError(FocusStackError):
    """Device unavailable or device-level failure."""


class AlignmentError(FocusStackError):
    """Alignment failed (M2; defined now so the hierarchy is complete)."""


class RawDecodeError(FocusStackError):
    """A camera RAW file could not be decoded without changing the RAW recipe."""


class DngExportError(FocusStackError):
    """A Linear DNG could not be written or failed post-write conformance checks."""


class DiskSpaceError(FocusStackError):
    """There is not enough free space for an atomic output."""
