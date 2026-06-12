"""Typed engine exceptions (SPEC §12)."""


class FocusStackError(Exception):
    """Base for all engine errors."""


class ValidationError(FocusStackError):
    """Input frames failed validation (size/bit-depth/format)."""


class BackendError(FocusStackError):
    """Device unavailable or device-level failure."""


class AlignmentError(FocusStackError):
    """Alignment failed (M2; defined now so the hierarchy is complete)."""
