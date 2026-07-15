"""Generate tiny redistributable same-camera DNG frames for browser tests."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.engine.test_raw_loader import write_test_raw  # noqa: E402


def main() -> None:
    destination = pathlib.Path(sys.argv[1])
    destination.mkdir(parents=True, exist_ok=True)
    incompatible = len(sys.argv) > 2 and sys.argv[2] == "incompatible"
    write_test_raw(destination / "frame-001.dng")
    write_test_raw(
        destination / "frame-002.dng",
        model="DifferentCam" if incompatible else "SameCam Pro",
    )
    print(destination)


if __name__ == "__main__":
    main()
