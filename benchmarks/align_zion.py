"""Alignment benchmark on the real Zion stack — time + registration quality.

Runs alignment twice (proxy-only baseline vs full-res refinement) on a real
high-MP stack, recording, per config:
  - wall-clock alignment time,
  - the recovered per-frame transforms + ECC correlations (regression anchor),
  - a sub-pixel residual: how far each aligned frame still sits from the
    reference, measured by a full-res ECC translation solve (lower = better).

Writes a JSON report. Re-run later and compare against the committed baseline to
catch regressions in either time or accuracy.

Usage (use the CUDA extra so the torch ops match real GPU usage):
    uv run --extra cu12x python benchmarks/align_zion.py
    uv run --extra cu12x python benchmarks/align_zion.py --out benchmarks/align_zion_latest.json

Point at a different stack with FOCUSSTACK_REAL_STACK_DIR or --stack.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from focusstack.align import AlignParams, align_stack
from focusstack.backend import get_device
from focusstack.stack.sources import ArrayFrameSource

_DEFAULT_STACK = r"G:\BackUpPhoto\USA_2018\2018-10-13 - Zion\STACK 2026"
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _luma(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame.astype(np.float32) @ _LUMA)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _residual_px(ref_lum: np.ndarray, aligned_lum: np.ndarray) -> float:
    """Leftover translation (px) between an aligned frame and the reference, via a
    full-res ECC translation solve warm-started at identity. Sub-pixel."""
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        _, out = cv2.findTransformECC(ref_lum, aligned_lum, warp, cv2.MOTION_TRANSLATION, crit)
        return float(np.hypot(out[0, 2], out[1, 2]))
    except cv2.error:
        return float("nan")


def _run_config(frames: list[np.ndarray], cache_dir: Path, device, refine: bool) -> dict:
    src = ArrayFrameSource(frames)
    t0 = time.perf_counter()
    report = align_stack(src, cache_dir=cache_dir, device=device,
                         params=AlignParams(max_long_edge=2048, refine_full_res=refine))
    align_time = time.perf_counter() - t0

    ref = report.reference
    cached = report.cache.frame_source()
    ref_lum = _luma(cached.read(ref))
    residuals = {}
    for k in range(len(frames)):
        if k == ref:
            continue
        residuals[k] = _residual_px(ref_lum, _luma(cached.read(k)))

    vals = [v for v in residuals.values() if not np.isnan(v)]
    return {
        "align_time_s": round(align_time, 3),
        "proxy_factor": report.proxy_factor,
        "reference": ref,
        "correlations": {str(k): round(v, 5) for k, v in sorted(report.correlations.items())},
        "matrices": [np.round(m, 5).tolist() for m in report.matrices],
        "residual_px": {
            "max": round(max(vals), 4) if vals else None,
            "mean": round(float(np.mean(vals)), 4) if vals else None,
            "per_frame": {str(k): round(v, 4) for k, v in sorted(residuals.items())},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default=os.environ.get("FOCUSSTACK_REAL_STACK_DIR", _DEFAULT_STACK))
    ap.add_argument("--out", default="benchmarks/align_zion_latest.json")
    ap.add_argument("--cache", default=None, help="scratch dir for aligned caches")
    args = ap.parse_args()

    from focusstack.io import load_image  # local import: heavy

    stack_dir = Path(args.stack)
    paths = sorted(stack_dir.glob("stack-pure-*.tif"))
    if not paths:
        raise SystemExit(f"no stack-pure-*.tif frames in {stack_dir}")
    frames = [load_image(p).pixels for p in paths]

    device = get_device("auto")
    cache_root = Path(args.cache) if args.cache else Path(os.environ.get("TEMP", "/tmp")) / "fs-bench"
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"stack: {stack_dir}")
    print(f"frames: {len(frames)}  shape: {frames[0].shape}  device: {device.torch_device}")

    configs = {}
    for name, refine in (("baseline", False), ("full_res", True)):
        print(f"\n== {name} (refine_full_res={refine}) ==")
        res = _run_config(frames, cache_root / name, device, refine)
        print(f"  align_time: {res['align_time_s']}s  "
              f"residual max/mean: {res['residual_px']['max']}/{res['residual_px']['mean']} px")
        configs[name] = res

    report = {
        "benchmark": "align_zion",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "device": str(device.torch_device),
        "stack_dir": str(stack_dir),
        "n_frames": len(frames),
        "frame_shape": list(frames[0].shape),
        "frames": [p.name for p in paths],
        "configs": configs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
