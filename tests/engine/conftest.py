import pytest
import torch

from planefuse.backend import get_device


def _available_kinds() -> list[str]:
    kinds = ["cpu"]
    if torch.cuda.is_available():
        kinds.append("cuda")
    if torch.backends.mps.is_available():
        kinds.append("mps")
    return kinds


@pytest.fixture(params=_available_kinds())
def device(request):
    """Parametrizes a test over every device present on this machine."""
    return get_device(request.param)


@pytest.fixture(params=[k for k in _available_kinds() if k != "cpu"])
def accel_device(request):
    """Accelerators only; tests using this auto-skip on CPU-only machines."""
    return get_device(request.param)
