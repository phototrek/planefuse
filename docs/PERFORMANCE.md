# Performance

`scripts/benchmark.py` reports JSON for a deterministic synthetic in-memory stack.
It measures algorithm throughput only; RAW decode, alignment, disk I/O, viewer tiles,
and export require separate measurements. Results are informational, not CI gates.

Run:

```bash
uv run --frozen --extra cpu --extra raw python scripts/benchmark.py \
  --height 1024 --width 1536 --frames 12 --method weighted --device cpu
```

## Recorded environment

- Date: 2026-07-15
- Host: Apple silicon, macOS 14.6.1
- Python: 3.12.3
- Torch: 2.13.0
- Fixture: weighted, 8 × 512 × 768 deterministic random float32 RGB, seed 20260715

| Device | Elapsed | Input throughput | Output range |
|---|---:|---:|---:|
| CPU | 0.080264 s | 39.192 MP/s | 0.002559276–0.991504908 |
| Apple silicon MPS | 0.182619 s | 17.226 MP/s | 0.002559275–0.991504908 |

This tiny fixture is dominated by accelerator dispatch overhead, so the CPU/MPS ratio
must not be projected to real full-resolution stacks. Compare results only with the
same dimensions, frame count, method, seed, versions, and device. The 24 MP targets in
[SPEC.md](SPEC.md) remain release-hardware targets; CPU has no speed target.
