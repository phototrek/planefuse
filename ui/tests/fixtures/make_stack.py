"""Write a small synthetic stack into a directory for the Playwright smoke."""

import sys
from pathlib import Path

# Make the repo root importable so `tests.synthetic` resolves when run as a script.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from planefuse.io import save_image  # noqa: E402
from tests.synthetic.generate import generate_stack  # noqa: E402


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stack = generate_stack(h=64, w=80, n_frames=4, max_sigma=4.0, seed=7)
    for i, f in enumerate(stack.frames):
        save_image(f, out / f"f_{i:03d}.tif", bit_depth=16)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
