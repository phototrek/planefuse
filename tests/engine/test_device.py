import pytest
import torch

from focusstack.backend import Device, free_memory, get_device
from focusstack.errors import BackendError


def test_cpu_device_always_available():
    d = get_device("cpu")
    assert d.kind == "cpu"
    assert d.torch_device == torch.device("cpu")
    assert d.name


def test_auto_returns_a_device():
    d = get_device("auto")
    assert d.kind in ("cuda", "mps", "cpu")


def test_unknown_preference_raises():
    with pytest.raises(ValueError):
        get_device("tpu")


def test_explicit_unavailable_accelerator_raises():
    if not torch.cuda.is_available():
        with pytest.raises(BackendError):
            get_device("cuda")
    if not torch.backends.mps.is_available():
        with pytest.raises(BackendError):
            get_device("mps")


def test_free_memory_positive():
    assert free_memory(get_device("cpu")) > 0


def test_free_memory_on_auto_device():
    assert free_memory(get_device("auto")) > 0
