# Benchmarks

Reproducible performance + quality benchmarks on real stacks, with committed
baselines to catch regressions in **time** and **accuracy**.

## align_zion — alignment time + registration quality

```bash
# Use the CUDA extra so torch ops run on the GPU (matches real usage).
uv run --extra cu12x python benchmarks/align_zion.py --out benchmarks/align_zion_latest.json
```

Runs alignment twice on the Zion stack (7 frames, 7795×5199) — proxy-only
`baseline` vs `full_res` (`refine_full_res=True`) — and records, per config:

- `align_time_s` — wall-clock alignment time.
- `correlations` — per-pair ECC correlation (regression anchor).
- `matrices` — recovered per-frame transforms (regression anchor).
- `residual_px` — leftover misalignment of each aligned frame vs the reference,
  from a full-res ECC **translation** solve (sub-pixel; lower = better).

Point at another stack with `FOCUSSTACK_REAL_STACK_DIR` or `--stack`.

### Regression check

Re-run to `align_zion_latest.json` and compare against `align_zion_baseline.json`:

- **Time**: `full_res` should stay ~3× `baseline`; a large jump in either = regression.
- **Accuracy**: `residual_px.max` should not grow materially; `matrices` /
  `correlations` should match within tolerance (alignment is not bit-exact across
  devices/driver versions — compare with a small tolerance, not equality).

Baselines are machine/data specific (the JSON records `git_commit`, `device`,
`stack_dir`, `frame_shape`). Regenerate the baseline deliberately when the
algorithm intentionally changes.

### Finding (baseline `fcb0ec7`, RTX 3090)

| config | align time | residual max / mean | pair corr |
|--------|-----------:|--------------------:|----------:|
| baseline (proxy 2048) | **50.1 s** | 0.317 / 0.136 px | 0.99+ |
| full_res refine       | 146.5 s    | 0.505 / 0.210 px | 0.96–0.98 |

**Full-res refinement does not help this real stack — it is ~3× slower and
slightly *worse* by the residual metric.** Reason: focus-stack frames differ by
defocus; downscaling to the 2048 proxy suppresses those high-frequency
differences, so proxy ECC correlates at 0.99+. Full resolution re-exposes the
defocus mismatch (correlation drops to ~0.97), and per-pair errors accumulate
as chain drift on frames far from the reference. The synthetic-data win (pure
geometry, no defocus — see `tests/engine/test_alignment_accuracy.py`) does not
transfer. Keep `refine_full_res` **off** by default; the dominant residual here
is chain drift, which a reference-direct / bundle-adjusted estimate would
address better than full-res refinement.
