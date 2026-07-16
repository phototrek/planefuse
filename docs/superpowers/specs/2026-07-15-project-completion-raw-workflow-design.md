# PlaneFuse Completion and RAW Workflow Design

**Date:** 2026-07-15  
**Status:** Approved  
**Scope:** Complete the existing product specification, repair audited correctness and security issues, and add a no-bake RAW-to-Linear-DNG workflow for current Capture One Pro.

## 1. Goals

1. Restore a green, reproducible CPU and Apple-silicon baseline and preserve CUDA compatibility.
2. Remove known vulnerable dependency versions and pin supported development runtimes.
3. Complete the unfinished behavior promised by `docs/SPEC.md`, except features explicitly described there as optional and unnecessary for acceptance (for example, custom CUDA kernels when portable Torch performance meets the targets).
4. Accept supported proprietary camera RAW files directly, stack same-camera captures without aesthetic development, and export the merged result as a lossless Linear DNG usable in the current full Capture One Pro edition.
5. Make the README, specification, API reference, user guide, and screenshots describe verified behavior rather than intended behavior.

## 2. Non-goals

- Writing manufacturer-native CR3, NEF, ARW, RAF, or other proprietary RAW formats.
- Preserving the original CFA mosaic after geometric alignment. Subpixel warping and compositing require demosaiced scene-linear samples.
- Importing or reproducing Capture One catalog/session adjustments.
- Supporting mixed-camera, mixed-sensor-mode, or mixed-CFA RAW stacks.
- Baking white balance, exposure styling, tone curves, denoising, sharpening, output profiles, or creative looks into RAW-mode output.
- Shipping optional custom CUDA kernels unless profiling shows that the portable implementation cannot meet the documented performance targets.

## 3. Delivery strategy

Work is split into four independently testable phases. Existing rendered-image workflows remain available throughout.

### Phase 1: Correctness and security

- Remove float64 operations from MPS tensor paths and add a regression test for pixel-coordinate warp conversion.
- Repair low-confidence alignment handling so a bad pair cannot silently contaminate the transform chain. The quality gate must either recover a credible transform or return a flagged, actionable result.
- Upgrade locked Python and UI dependencies, resolve known advisories, pin supported Python, Node, and uv versions, and remove current framework deprecation warnings.
- Run unit, type, lint, build, API, and browser smoke checks in CI. CPU browser tests are mandatory; CUDA and MPS suites remain real-hardware release gates.

### Phase 2: Engine and professional I/O completion

- Implement perspective alignment with homography output and validation.
- Implement portable Lanczos-3 warping and parity tests.
- Preserve ICC, EXIF, and XMP; write PlaneFuse provenance without copying invalid per-exposure fields.
- Support verified 16-bit PNG, TIFF compression choices, JPEG metadata, and atomic export.
- Finish typed error mapping, OOM fallback tests, device parity, golden fixtures, and performance probes.
- Add RAW ingestion, scene-linear processing, and Linear-DNG output as specified below.

### Phase 3: Product UI completion

- Add histogram and per-channel clipping indicators.
- Add synchronized result/source and result/result compare, depth overlay, and alignment blink.
- Complete keyboard shortcuts, validation reporting, alignment-quality review, smart-selection proposal review, batch confirmation, queue controls, estimates, and all export controls.
- Expose RAW compatibility, camera validation, decoder settings, no-bake guarantees, and DNG export status without requiring technical knowledge.

### Phase 4: Distribution and documentation

- Harden native launchers, Docker, and accelerator verification.
- Publish an accurate architecture/API reference, install guide, user guide, RAW/DNG and Capture One workflow, troubleshooting, screenshots, demo assets, performance results, and known limitations.
- Update milestone/status reporting from fresh verification evidence.

## 4. RAW architecture

RAW mode is a parallel input/output path around the existing alignment and fusion algorithms:

```text
camera RAW
  -> LibRaw/rawpy metadata and sensor validation
  -> deterministic AHD demosaic to scene-linear camera RGB
  -> temporary neutral luminance proxies for alignment/focus analysis
  -> full-resolution scene-linear alignment and fusion
  -> lossless 16-bit Linear DNG
```

The rendered TIFF/JPEG/PNG path retains its source-gamma behavior. A frame carries an explicit processing domain (`rendered_rgb` or `scene_linear_camera_rgb`) so caches and algorithms cannot accidentally mix the two.

### 4.1 Decoder behavior

LibRaw is accessed through rawpy. Decode settings must disable:

- automatic brightness;
- camera or automatic white balance multiplication;
- output gamma;
- output color-space conversion;
- denoising and chroma smoothing;
- sharpening;
- highlight styling or reconstruction that changes measured sensor values.

The baseline decoder uses rawpy's cross-platform AHD demosaic with `output_color=raw`, `gamma=(1, 1)`, `no_auto_bright=True`, no automatic/camera white balance, and unit user white-balance multipliers. These settings and the rawpy/LibRaw versions are recorded in provenance metadata. If that path is unavailable for a camera, the job fails with a camera-specific message rather than silently selecting a different algorithm or stylistic fallback.

### 4.2 Stack compatibility

Every RAW frame in one stack must have compatible:

- camera make and model;
- sensor/capture mode and active dimensions;
- orientation;
- CFA layout and channel description;
- sample precision;
- black and white calibration;
- color calibration metadata.

The import report names every incompatible file and the differing fields. Originals are opened read-only and never modified.

### 4.3 Analysis and fusion

Alignment and focus selection use normalized luminance proxies derived from scene-linear samples. Proxy normalization may compensate exposure for measurement only and never changes output samples. Full-resolution alignment and fusion operate on scene-linear camera RGB using float32 with explicit valid-pixel masks.

Exposure metadata and normalization factors are recorded. No per-frame aesthetic development is applied. Values outside the nominal reference white are retained through processing and mapped into the DNG's integer range using documented black/white and baseline-exposure tags rather than clipped silently.

## 5. Linear DNG output

The primary RAW-mode output is a lossless, demosaiced, 16-bit Linear DNG conforming to the current public Adobe DNG specification and sized within Capture One's documented limits. The DNG container and required TIFF/DNG tags are written directly with `tifffile`; `pyexiv2` supplies portable EXIF/XMP extraction and writing. The Adobe DNG SDK is the conformance reference, not a runtime dependency.

The reference frame supplies:

- camera make, model, unique camera model, and serial where permitted;
- orientation, capture time, lens information, and relevant EXIF/XMP;
- color calibration matrices, illuminants, analog balance, and as-shot neutral metadata;
- black/white calibration and baseline exposure;
- ICC/profile information when semantically valid.

PlaneFuse adds a private XMP namespace containing app/engine versions, source hashes and ordered paths, reference frame, decoder/version/settings, alignment transforms and quality scores, exclusions, algorithm and parameters, processing-domain declaration, and export timestamp.

Invalid single-exposure fields such as focus distance and original depth of field are removed or replaced with aggregate provenance. The writer uses atomic temporary-file replacement and then validates the result by:

1. reopening and checking required TIFF/DNG tags;
2. reopening through LibRaw/rawpy;
3. comparing decoded pixels and metadata against the in-memory result within the 16-bit quantization tolerance.

Because Capture One exposes no headless conformance API, the repository includes a generated acceptance fixture and a documented manual Capture One Pro verification checklist. An optional 32-bit floating-point TIFF companion may be exported for archival precision, but it is not represented as RAW and is not a substitute for the Linear DNG.

## 6. Existing feature completion

The following documented gaps are implementation requirements:

- portable Lanczos-3 and perspective alignment;
- EXIF/XMP/ICC round-trip and PlaneFuse provenance;
- 16-bit PNG output;
- complete import validation and alignment-quality decisions;
- histogram, clipping, compare modes, overlays, shortcuts, queue controls, batch review, selection review, estimates, and export controls;
- golden image/metadata fixtures and accelerator release checks;
- user-facing documentation, screenshots, demos, troubleshooting, and performance reporting.

Repository-layout prose will describe real modules. Optional or future components will be labeled explicitly rather than shown as existing files.

## 7. Error handling

RAW and rendered workflows use typed errors that the server converts to stable JSON codes and the UI converts to actionable messages. Required cases include:

- unsupported or undecodable camera RAW;
- mixed camera/sensor/CFA stacks;
- inconsistent dimensions, precision, orientation, or calibration;
- missing metadata required for a valid Linear DNG;
- invalid alignment or insufficient reliable overlap;
- GPU OOM with tiled retry and CPU fallback;
- insufficient disk space or atomic-write failure;
- post-write DNG conformance or pixel-validation failure.

Cancellation leaves no partial public output. Temporary files and cache entries are cleaned or remain safely resumable.

## 8. Verification and acceptance

Every behavior change follows red-green-refactor testing. Acceptance requires:

- all Python unit/API/quality tests pass on CPU;
- MPS regression and browser workflow pass on Apple silicon;
- CUDA parity and OOM paths pass on NVIDIA release hardware;
- Ruff, mypy, Svelte diagnostics, production build, dependency audits, and Docker smoke pass;
- RAW fixtures prove deterministic no-bake decoding and same-camera rejection rules;
- DNG tag, metadata, pixel round-trip, and malformed-output tests pass;
- browser tests cover RAW import, stack, viewer, Linear-DNG export, validation errors, comparison, retouch, and rendered export;
- the generated Linear DNG opens and remains editable in the current full Capture One Pro edition according to the manual acceptance checklist;
- README and guides match the verified feature matrix and commands.

## 9. External compatibility basis

- Adobe DNG 1.7.1.0 and the current Adobe DNG SDK define the output container and tags.
- LibRaw/rawpy provide proprietary RAW decoding and metadata extraction.
- Current Capture One Pro supports Linear DNG; recognized source-camera metadata receives the native camera color path when available, otherwise Capture One uses Generic DNG Standard.

The implementation plan must pin concrete tested versions and must not claim support beyond the automated and manual compatibility matrix.
