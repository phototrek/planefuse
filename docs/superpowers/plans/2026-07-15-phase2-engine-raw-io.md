# Phase 2 Engine, Metadata, RAW, and DNG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete professional engine/I/O promises and add deterministic same-camera RAW stacking with Capture One-compatible Linear-DNG export.

**Architecture:** Introduce explicit frame metadata and processing domains, keep rendered and scene-linear paths separate, add portable alignment/warp capabilities, and isolate RAW decode/DNG write logic in focused modules. Existing algorithms continue consuming streamed float32 RGB frames.

**Tech Stack:** PyTorch, OpenCV, tifffile, Pillow, pyexiv2, rawpy/LibRaw, NumPy, FastAPI, pytest.

## Global Constraints

- RAW mode performs AHD demosaic only; it applies no white balance, gamma, auto-brightness, tone curve, denoise, sharpen, or output-color conversion.
- One RAW stack must share camera model, sensor mode, active dimensions, orientation, CFA layout, precision, and calibration.
- Linear DNG is 16-bit, lossless, atomic, metadata-rich, and validated after writing.
- Rendered-image source-gamma behavior remains unchanged.
- Optional custom CUDA kernels remain out of scope unless profiling proves them necessary.

---

### Task 1: Add explicit metadata and processing domains

**Files:**
- Create: `engine/src/planefuse/io/metadata.py`
- Modify: `engine/src/planefuse/io/loader.py`
- Modify: `engine/src/planefuse/io/__init__.py`
- Modify: `tests/engine/test_loader.py`
- Create: `tests/engine/test_metadata.py`

**Interfaces:**
- Produces: `ProcessingDomain`, `ImageMetadata`, and `Frame(metadata=..., domain=...)`.

- [ ] Write failing tests for rendered-domain defaults, ICC/EXIF/XMP extraction, orientation normalization, and invalid single-exposure tag filtering.
- [ ] Implement `ProcessingDomain(str, Enum)` with `RENDERED_RGB` and `SCENE_LINEAR_CAMERA_RGB`.
- [ ] Implement immutable `ImageMetadata` fields for camera/lens identity, EXIF bytes/map, XMP bytes, ICC, color calibration, black/white levels, exposure, decoder provenance, and source path.
- [ ] Use `pyexiv2` behind a narrow adapter; translate missing/unsupported metadata into empty optional fields rather than import-time failure.
- [ ] Run loader/metadata tests and commit with `feat: add explicit image metadata domains`.

### Task 2: Implement complete rendered export metadata and 16-bit PNG

**Files:**
- Modify: `engine/src/planefuse/io/writer.py`
- Create: `engine/src/planefuse/io/atomic.py`
- Modify: `tests/engine/test_writer.py`
- Modify: `tests/engine/test_writer_depth.py`
- Create: `tests/engine/test_metadata_roundtrip.py`

**Interfaces:**
- Produces: `save_image(..., metadata: ImageMetadata | None, provenance: dict | None)` with atomic replacement.

- [ ] Write failing tests that reopen 16-bit RGB PNG without 8-bit quantization, round-trip ICC/EXIF/XMP, remove focus-distance fields, and leave no destination on simulated write failure.
- [ ] Add an `atomic_output(destination)` context manager that writes beside the destination, fsyncs, validates, and replaces with `Path.replace`.
- [ ] Write 16-bit RGB PNG without Pillow's 8-bit limitation, using imagecodecs/tifffile-supported PNG encoding or a small dedicated encoder selected by a round-trip test.
- [ ] Apply TIFF/JPEG/PNG metadata consistently through `pyexiv2` after pixel encoding.
- [ ] Run writer and metadata suites and commit with `feat: complete metadata-safe image export`.

### Task 3: Implement portable Lanczos-3 warping

**Files:**
- Modify: `engine/src/planefuse/backend/ops.py`
- Modify: `tests/engine/test_align_ops.py`
- Modify: `engine/src/planefuse/align/pipeline.py`
- Modify: `engine/src/planefuse/cli.py`

**Interfaces:**
- Produces: `ops.warp(..., interp="lanczos3")` on CPU/CUDA/MPS.

- [ ] Write impulse, identity, integer-shift, edge-clamp, and device-parity tests comparing against a NumPy Lanczos-3 reference.
- [ ] Verify the new tests fail because `warp` currently falls back to bilinear.
- [ ] Implement separable six-tap windowed-sinc sampling with tensor gather operations and normalized weights; keep grid construction float32 on accelerators.
- [ ] Make `lanczos3` the full-resolution default while retaining bilinear for estimation.
- [ ] Run warp, tiled-equivalence, and alignment tests; commit with `feat: add portable Lanczos-3 warp`.

### Task 4: Implement true perspective alignment

**Files:**
- Modify: `engine/src/planefuse/align/refine.py`
- Modify: `engine/src/planefuse/align/estimate.py`
- Modify: `engine/src/planefuse/align/warp.py`
- Modify: `engine/src/planefuse/backend/ops.py`
- Modify: `tests/engine/test_refine.py`
- Create: `tests/engine/test_homography.py`

**Interfaces:**
- Produces: `model="perspective"` using `cv2.MOTION_HOMOGRAPHY` and projective sampling.

- [ ] Add synthetic homography recovery and full-resolution warp tests.
- [ ] Refactor ECC refinement to return affine similarity for similarity mode and an unprojected 3x3 homography for perspective mode.
- [ ] Add projective grid construction to `ops.warp` when the last row differs from `[0, 0, 1]`.
- [ ] Validate finite/invertible homographies, overlap, corner displacement, and correlation before accepting them.
- [ ] Expose perspective through CLI/registry and run full alignment parity; commit with `feat: add perspective alignment`.

### Task 5: Decode RAW files without aesthetic development

**Files:**
- Create: `engine/src/planefuse/io/raw.py`
- Modify: `engine/src/planefuse/io/loader.py`
- Modify: `engine/src/planefuse/io/__init__.py`
- Modify: `engine/pyproject.toml`
- Create: `tests/fixtures/raw/README.md`
- Create: `tests/engine/test_raw_loader.py`

**Interfaces:**
- Produces: `load_raw(path) -> Frame` and `RAW_EXTENSIONS`.

- [ ] Add redistributable tiny DNG fixtures with two same-camera frames and incompatible metadata variants; record their provenance/license.
- [ ] Write failing tests for `SCENE_LINEAR_CAMERA_RGB`, float32 range/headroom, decoder settings, no white-balance multiplication, no gamma, and camera/CFA metadata.
- [ ] Add `rawpy` as the `raw` extra and implement AHD postprocess with `output_color=raw`, `gamma=(1, 1)`, `no_auto_bright=True`, unit `user_wb`, 16-bit output, and all denoise/sharpen options disabled.
- [ ] Normalize black-subtracted sensor samples without clipping highlight headroom and record the mapping needed for DNG re-encoding.
- [ ] Return typed `RawDecodeError` with source/camera details for LibRaw failures.
- [ ] Run RAW loader tests on macOS/Linux/Windows-compatible wheels and commit with `feat: decode scene-linear camera RAW`.

### Task 6: Validate same-camera RAW stacks

**Files:**
- Modify: `engine/src/planefuse/io/loader.py`
- Modify: `server/src/planefuse_server/api/frames.py`
- Modify: `tests/engine/test_loader.py`
- Modify: `tests/server/test_frames.py`

**Interfaces:**
- Produces: per-file `incompatible_camera`, `incompatible_sensor_mode`, `incompatible_cfa`, and `missing_raw_calibration` statuses plus report-level `domain` and `camera`.

- [ ] Write table-driven failing tests for each incompatible field and mixed rendered/RAW inputs.
- [ ] Compare normalized metadata keys against the reference frame and include expected/actual values in messages.
- [ ] Expose compatibility and no-bake decoder summary in scan responses.
- [ ] Run engine/server validation suites and commit with `feat: validate same-camera RAW stacks`.

### Task 7: Preserve processing domain through pipeline and cache

**Files:**
- Modify: `engine/src/planefuse/stack/sources.py`
- Modify: `engine/src/planefuse/align/cache.py`
- Modify: `engine/src/planefuse/pipeline.py`
- Modify: `engine/src/planefuse/stack/base.py`
- Modify: `tests/engine/test_pipeline.py`
- Create: `tests/engine/test_raw_pipeline.py`

**Interfaces:**
- Produces: `StackResult.metadata`, `StackResult.domain`, and cache manifests that reject cross-domain reuse.

- [ ] Write failing tests showing rendered behavior unchanged, RAW result remains scene-linear, and a rendered cache cannot satisfy a RAW job.
- [ ] Extend frame sources with read-only metadata/domain properties without changing algorithm pixel APIs.
- [ ] Store cache manifest version, domain, source hash, decoder settings, and calibration hash.
- [ ] Use neutral normalized proxies for RAW alignment/focus analysis while full-resolution fusion reads scene-linear frames.
- [ ] Run pipeline/algorithm/tile parity and commit with `feat: preserve scene-linear RAW domain`.

### Task 8: Write and validate Capture One-compatible Linear DNG

**Files:**
- Create: `engine/src/planefuse/io/dng.py`
- Create: `engine/src/planefuse/io/provenance.py`
- Modify: `engine/src/planefuse/io/writer.py`
- Modify: `engine/src/planefuse/io/__init__.py`
- Create: `tests/engine/test_dng_writer.py`
- Create: `tests/fixtures/dng/README.md`

**Interfaces:**
- Produces: `save_linear_dng(image, destination, metadata, provenance)` and `validate_linear_dng(path, expected)`.

- [ ] Write failing tests for required DNG version/backward version, `PhotometricInterpretation=LinearRaw`, 16-bit RGB samples, camera identity, color matrices/illuminants, as-shot neutral, black/white levels, baseline exposure, orientation, EXIF/XMP, and lossless compression.
- [ ] Implement TIFF/DNG tags directly with tifffile using Adobe DNG 1.7.1 as the reference.
- [ ] Encode PlaneFuse provenance in a registered private XMP namespace with ordered source hashes, transforms, quality, exclusions, method, parameters, decoder, and versions.
- [ ] Reopen through tifffile and rawpy, compare decoded pixels within `1/65535`, and fail atomic publication if validation fails.
- [ ] Add malformed metadata, truncated write, over-limit dimensions, and insufficient-space tests.
- [ ] Commit with `feat: export validated Linear DNG`.

### Task 9: Add RAW/DNG CLI and server workflows

**Files:**
- Modify: `engine/src/planefuse/cli.py`
- Modify: `server/src/planefuse_server/runners.py`
- Modify: `server/src/planefuse_server/api/jobs.py`
- Modify: `server/src/planefuse_server/projects.py`
- Modify: `tests/engine/test_cli.py`
- Modify: `tests/server/test_export.py`
- Modify: `tests/server/test_runners.py`

**Interfaces:**
- Produces: RAW extensions in directory discovery; `.dng` output dispatch; project image metadata for domain/camera/provenance.

- [ ] Write failing CLI/API end-to-end tests for RAW input and DNG export.
- [ ] Make output type domain-aware: `.dng` requires a scene-linear RAW result; rendered results receive a clear validation error.
- [ ] Persist result domain, reference metadata, provenance, decoder settings, and DNG validation state in `project.json`.
- [ ] Add optional 32-bit float TIFF companion without labeling it RAW.
- [ ] Run CLI/server lifecycle tests and commit with `feat: expose RAW and Linear-DNG workflows`.

### Task 10: Finish typed failures, OOM, golden, and performance gates

**Files:**
- Modify: `engine/src/planefuse/errors.py`
- Modify: `server/src/planefuse_server/main.py`
- Create: `server/src/planefuse_server/errors.py`
- Modify: `tests/engine/test_pipeline.py`
- Create: `tests/server/test_errors.py`
- Create: `tests/golden/README.md`
- Create: `tests/engine/test_golden.py`
- Create: `scripts/benchmark.py`

- [ ] Add typed RAW, metadata, alignment-quality, export, disk-space, and conformance errors with stable API codes.
- [ ] Add direct→tiled→CPU OOM tests for CUDA-like and MPS-like exceptions.
- [ ] Add small redistributable rendered and RAW golden fixtures with perceptual/metadata tolerances.
- [ ] Add benchmark output for documented operations without turning hardware targets into flaky CI failures.
- [ ] Run all engine/server tests and commit with `test: complete engine reliability gates`.

### Task 11: Phase 2 verification

- [ ] Run all Python checks and rendered/RAW/DNG tests on CPU and MPS.
- [ ] Run CUDA parity/release commands when NVIDIA hardware is available; otherwise record the unverified gate explicitly.
- [ ] Generate a DNG acceptance fixture, validate it with rawpy/tifffile, and perform the documented current Capture One Pro manual import/edit/export check.
