# Troubleshooting

## Run is disabled

Read the validation panel. Remove or replace files marked wrong dimensions/bit depth,
mixed domain, different camera/sensor/CFA, missing calibration, or decode failure.
Smart selection must also have a reviewed and accepted proposal.

## RAW support is not installed

Sync/run with `--extra raw`. The optional `metadata` extra is not needed for RAW.

## CUDA or MPS is unavailable

Check `/api/system`. CUDA requires the current NVIDIA driver and the `cu12x` extra.
MPS requires Apple silicon and a compatible macOS/PyTorch build. Explicit device
requests fail clearly; `auto` falls back to CPU.

## Out of memory

FocusStack retries tiled processing, then tiled CPU. The queue message records the
fallback. Close other GPU-heavy apps or lower alignment proxy size if performance is poor.

## DNG export fails

The destination is not published if metadata, space, TIFF conformance, or LibRaw pixel
validation fails. Ensure all RAW frames validate, the output folder has roughly twice
the uncompressed output size free, and no other process locks the destination.

## Viewer histogram is unavailable

Re-select the result. Viewer registration is idempotent; if source/result files were
moved outside FocusStack, restore them or rescan. Clearing only `cache/tiles` while the
app is closed is safe; tiles rebuild on next view.

## Port 8425 is busy

Native serving probes subsequent ports and opens the selected URL. Docker uses the
fixed localhost mapping; stop the conflicting container/process or change both sides
of the mapping deliberately without widening the host address.
