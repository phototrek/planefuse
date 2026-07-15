#!/usr/bin/env python3
"""Small reproducible FocusStack throughput benchmark; results are informational."""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import torch

from focusstack.backend import get_device
from focusstack.stack import ArrayFrameSource, get_algorithm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--method", default="weighted", choices=("weighted", "pmax", "dmap"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    frames = [
        rng.uniform(0.0, 1.0, (args.height, args.width, 3)).astype(np.float32)
        for _ in range(args.frames)
    ]
    source = ArrayFrameSource(frames)
    device = get_device(args.device)
    params = {
        "temperature": 0.05,
        "sharpness_radius": 8,
        "selection_smoothing": 1,
        "estimation_radius": 8,
        "contrast_threshold": 7.0,
        "smoothing_radius": 16,
    }
    started = time.perf_counter()
    result = get_algorithm(args.method).run(source, device, params)
    elapsed = time.perf_counter() - started
    megapixels = args.height * args.width * args.frames / 1_000_000
    print(
        json.dumps(
            {
                "schema": 1,
                "method": args.method,
                "device": device.kind,
                "device_name": device.name,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "height": args.height,
                "width": args.width,
                "frames": args.frames,
                "elapsed_seconds": round(elapsed, 6),
                "input_megapixels_per_second": round(megapixels / elapsed, 3),
                "output_min": float(result.image.min()),
                "output_max": float(result.image.max()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
