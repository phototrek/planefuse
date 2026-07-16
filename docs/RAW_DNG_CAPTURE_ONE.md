# No-bake RAW and Capture One Pro

## What RAW mode does

RAW mode accepts current camera RAW families supported by the installed LibRaw. A
stack must use the same camera, sensor mode, active dimensions, orientation, CFA,
precision, and calibration. Mixing developed files and RAW is rejected.

The only irreversible sensor operation is AHD demosaic, which is necessary because
focus alignment and stacking combine RGB samples from different focal planes. The
decoder applies unit white balance, linear gamma, no auto brightness or auto scale,
no denoise/median filtering, no highlight reconstruction, and no color-space conversion.

PlaneFuse does not bake a photographic look. Alignment/focus metrics use normalized
measurement proxies; fusion always reads the original scene-linear camera RGB values.

## Why the output is Linear DNG

A stacked image cannot honestly be exported as the original Bayer/other sensor mosaic:
its pixels come from multiple captures and spatial transforms. PlaneFuse therefore
writes a demosaiced, three-sample `LinearRaw` DNG—the DNG representation that retains
the most real stacked information without fabricating sensor samples.

The file is 16-bit lossless JPEG-compressed and retains camera identity, color matrices,
illuminants, as-shot neutral, black/white reference levels, orientation, safe EXIF/XMP,
and PlaneFuse provenance. Negative values and highlights above scene-linear 1.0 are
encoded with a reversible mapping recorded in XMP and DNG levels. The optional float
TIFF companion preserves the exact float32 working numbers.

Before publication, PlaneFuse reopens the DNG with tifffile and LibRaw and verifies
shape, required tags, compression, and pixels within one 16-bit code.

## Capture One target

The current full desktop target at the time of this document is Capture One Pro 16.7.5.
Capture One’s official DNG documentation says Pro supports DNG and uses a generic DNG
profile where a camera model is not natively supported. PlaneFuse keeps the reference
camera identity/calibration, but Capture One—not embedded third-party adjustments—owns
the initial rendering and all edits.

Official references:

- [Capture One DNG support](https://support.captureone.com/hc/en-us/articles/360003336397-DNG-Support)
- [Capture One 16.7.5 release notes](https://support.captureone.com/hc/en-us/articles/34563165007517)

## Import checklist

1. Export `result.dng`; optionally enable the `-scene-linear-float.tif` companion.
2. Confirm the PlaneFuse job shows “LibRaw validated”.
3. Import the DNG into a Capture One Pro 16.7.5 Catalog or Session.
4. Confirm dimensions/orientation, camera/generic DNG profile, neutral editability,
   highlight/shadow behavior, and absence of an unexpected pre-applied style.
5. Apply white balance, exposure, curve, color, noise reduction, and sharpening in
   Capture One as desired; export a TIFF and visually compare it to the viewer.

Automated conformance is a release gate. Manual import is also recorded separately in
[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) because Capture One is proprietary and its
behavior cannot be asserted by CI.

For importer-format checks, the repository includes a redistributable
[synthetic Linear DNG acceptance file](assets/planefuse-linear-dng-acceptance.dng).
It was produced by the real two-frame RAW stack and DNG writer and contains no
proprietary photograph. It validates the container and editing path only; final manual
acceptance must still use a real supported same-camera photographic stack.

## Limitations

- This is Linear DNG, not the original mosaic and not a container of the original RAWs.
- Per-capture focus distance is removed because it has no single correct merged value.
- Capture One does not promise to honor another application’s embedded adjustments;
  PlaneFuse embeds provenance and calibration, not a baked look.
- Same-camera rules intentionally reject stacks whose calibration cannot be represented
  by one honest DNG metadata set.
