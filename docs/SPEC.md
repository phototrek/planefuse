# PlaneFuse — Professional GPU-Accelerated Focus Stacking Software

**Specification v1.1 — 2026-07-15**

This is the implemented product contract for PlaneFuse. The original milestone order is retained as project history; current operational details are linked from the root README and the API, architecture, installation, user, and RAW workflow guides.

---

## 1. Product overview

PlaneFuse merges a series of photographs taken at different focus distances ("a stack") into a single image that is sharp everywhere. Target users are professional macro, product, and landscape photographers who today use Zerene Stacker or Helicon Focus.

**Core promises:**

1. **Quality** — alignment and stacking results competitive with Zerene Stacker, including handling of focus breathing (scale change between frames), halo control, and full interactive retouching.
2. **Speed** — GPU acceleration for every heavy operation: CUDA on NVIDIA (primary), Metal/MPS on Apple M-series. A 50-frame 24 MP stack aligns and stacks (PMax) in under 60 seconds on an RTX 3080-class GPU.
3. **Robustness** — never crashes on bad input or GPU memory exhaustion; every failure mode degrades gracefully (tiling → CPU fallback → clear error).
4. **Professional workflow** — 16-bit pipeline end to end, ICC/EXIF/XMP preservation, batch processing, presets, CLI automation, full retouching, and a same-camera camera-RAW workflow.
5. **No-bake RAW** — stack scene-linear camera RGB without photographic development and export a lossless, validated 16-bit Linear DNG plus optional float32 TIFF for Capture One Pro.

**Explicit non-goals:** mixing cameras/sensor modes in one RAW stack, reconstructing or inventing a Bayer/other mosaic after demosaic and spatial stacking, exposure/HDR stacking, astro stacking, panorama stitching, tethered capture/camera control, cloud processing, mobile support, and user accounts.

---

## 2. Tech stack and constraints

| Layer | Technology |
|---|---|
| Processing engine | Python 3.12+, **PyTorch 2.x as the single tensor/compute layer** — one implementation, three devices: **CUDA** (NVIDIA, primary), **MPS** (Apple M-series), **CPU** (fallback + CI). Optional CUDA-only custom kernels (CuPy `RawKernel`) may accelerate hot paths, but every op must have a portable torch implementation behind the same interface |
| Image I/O | `tifffile`/`imagecodecs` (TIFF/DNG/PNG), `Pillow` (JPEG/portable metadata), `rawpy`/LibRaw (RAW), ExifRead (portable metadata), optional `pyexiv2` enrichment |
| Alignment math | `opencv-python-headless` (ECC refinement on downscaled luminance, CPU), `torch.fft` phase correlation (any device), GPU warping via torch grid ops |
| Tooling | **uv** for everything Python: a uv workspace (root `pyproject.toml` + committed `uv.lock`) containing both packages; `uv sync`, `uv run pytest`, `uv run ruff`, `uv run mypy`. Never pip/poetry. `npm` for the UI |
| Server | FastAPI + uvicorn, WebSocket progress streaming |
| Frontend | Svelte 5 + Vite + TypeScript, no heavyweight UI framework |
| Packaging | Two Python packages with their own `pyproject.toml`, joined as a uv workspace: `planefuse` (engine) and `planefuse-server` (depends on engine, serves the built UI). Releases provide a visual HTML/JavaScript setup assistant and platform-specific ready folders containing the compiled UI plus the official uv executable; Docker images remain available per §16 |
| Platforms | Windows 11, Linux, macOS on Apple silicon. NVIDIA GPU (CUDA 12.x) is the primary target; Apple M-series accelerates via MPS; everything must also run on CPU for machines without a GPU and for CI. |

**Hard rules:**

- The engine package (`planefuse.engine`) must have **zero imports from server or UI code**. It is importable and fully usable from a plain Python script.
- Every algorithm runs identically (within float tolerance) on every device (CUDA, MPS, CPU). The device is selected at runtime, never at install time.
- Rendered-image math is float32 in source gamma. RAW fusion is float32 scene-linear camera RGB; only normalized neutral proxies may be used for alignment and focus measurements. Integer math exists only at I/O boundaries.
- Processing domains never mix. RAW stacks require one honest camera/sensor/CFA/calibration contract.
- No global mutable state in the engine. A stacking job is a pure function of (frames, parameters) → result.

---

## 3. Repository layout

```
focus-stacker/
├── docs/
│   └── SPEC.md                  # this document
├── engine/
│   ├── pyproject.toml           # package: planefuse
│   └── src/planefuse/
│       ├── backend/             # device abstraction over torch, memory budgeting
│       │   ├── __init__.py      # get_device(), Device info dataclass
│       │   ├── ops.py           # all image ops, device-agnostic torch (the one true implementation)
│       │   ├── cuda_fast.py     # OPTIONAL CuPy RawKernel fast paths, same op interface
│       │   └── kernels/         # .cu kernel source files (loaded as text)
│       ├── io/                  # rendered/RAW load, metadata, TIFF/PNG/DNG export
│       ├── align/               # registration pipeline
│       ├── select/              # smart frame selection / stack thinning (§7.0)
│       ├── stack/               # algorithm registry: pmax.py, dmap.py, weighted.py, slab.py
│       ├── retouch/             # stroke compositing engine
│       ├── post/                # halo suppression, contrast, sharpening
│       ├── pipeline.py          # orchestrates load→align→select→stack→post
│       ├── tiles.py             # tiled processing + viewer tile pyramid
│       ├── project.py           # project file model (JSON on disk)
│       └── cli.py               # `planefuse` CLI entry point
├── server/
│   ├── pyproject.toml           # package: planefuse-server (depends on planefuse)
│   └── src/planefuse_server/
│       ├── main.py              # FastAPI app factory, static UI serving
│       ├── jobs.py              # job queue, cancellation, progress events
│       ├── api/                 # routers: projects, frames, jobs, viewer, retouch, export, system
│       └── ws.py                # WebSocket progress hub
├── ui/                          # Svelte 5 + Vite
│   └── src/
│       ├── routes/              # screens (see §10)
│       ├── lib/viewer/          # tiled deep-zoom viewer component
│       ├── lib/api.ts           # typed API client
│       └── lib/stores/          # project, jobs, viewer state
├── tests/
│   ├── engine/                  # unit + golden tests
│   ├── server/                  # API tests
│   └── synthetic/               # synthetic stack generator (§13.1)
├── pyproject.toml               # uv workspace root
├── uv.lock                      # committed
├── Dockerfile                   # multi-stage, cuda + cpu targets (§16)
└── docker-compose.yml           # §16
```

---

## 4. Engine: device abstraction

The engine has **one implementation of every image operation**, written in device-agnostic PyTorch in `backend/ops.py`; the device is a parameter, not a code path.

```python
def get_device(prefer: str = "auto") -> Device:
    # "auto": cuda if torch.cuda.is_available(), else mps if torch.backends.mps.is_available(), else cpu.
    # prefer="cpu"/"cuda"/"mps" forces a device (cpu is used in CI). Choice is logged and in /api/system.

# backend/ops.py — all float32 tensors, NCHW or HW layouts documented per fn.
# Non-exhaustive: also median_filter2d (unfold + median — no native torch 2D median),
# guided_filter, masked iterative box blur (§7.2), and the streaming fold helpers (§7.1/7.3):
def gaussian_blur(img, sigma) -> Tensor        # separable conv2d
def resize(img, shape, interp) -> Tensor       # F.interpolate
def warp(img, matrix, out_shape, interp) -> Tensor   # affine_grid/grid_sample; Lanczos-3 = explicit gather + windowed-sinc weights in torch (portable)
def sharpness_map(gray, radius) -> Tensor      # windowed local Laplacian energy via conv2d
def fft2 / ifft2 / log_polar_remap(...)        # torch.fft + grid_sample
def free_memory(device) -> int                 # cuda: torch.cuda.mem_get_info; mps: recommended_max_memory − driver_allocated; cpu: psutil
```

- **Optional CUDA fast paths** (`backend/cuda_fast.py`, CuPy `RawKernel`s in `backend/kernels/*.cu`, zero-copy via DLPack): Lanczos-3 warp, pyramid max-energy fold (§7.1), DMap blend. Each registers over the same op interface and is used only when the device is CUDA *and* CuPy imports; the torch implementation is the reference and always exists. A parity test pins each fast path to its torch reference (`atol=1e-3`).
- **Device parity requirement:** for every op and every full algorithm, tests assert CPU output agrees with each available accelerator (CUDA, MPS) within `atol=1e-3` on random inputs.
- **MPS specifics:** float32 throughout (already the working format; MPS has no float64). `torch.fft` on MPS requires macOS 14+; on older macOS the FFT ops take the CPU round-trip below. Do **not** set the global `PYTORCH_ENABLE_MPS_FALLBACK`; if an op is missing on MPS (known candidates: FFT pre-macOS-14, `grid_sample` bicubic), route that op explicitly through a CPU round-trip inside `ops.py` and log it once — deterministic and visible, not silent.
- **Memory discipline:** before each GPU stage, the engine estimates required device memory. If the estimate exceeds 80% of free memory, it switches that stage to tiled mode (§8). On `torch.cuda.OutOfMemoryError` (or the MPS allocation `RuntimeError`) despite tiling, it empties the cache and reruns the stage on CPU, emitting a warning event. A job never fails due to OOM.

---

## 5. Image I/O and color

**Input:** 8/16-bit RGB TIFF (including compressed), JPEG, PNG, and camera-RAW families supported by the installed LibRaw. Grayscale is rejected. Rendered frames must share dimensions and bit depth. RAW frames must additionally share camera identity, active sensor dimensions, orientation, sample precision, CFA layout, black/white levels, color matrices, calibration illuminants, and sensor mode. Mixed rendered/RAW stacks are rejected. Natural filename order is the stack order; batch proposals prefer EXIF capture-time gaps and fall back to filename-number gaps.

**Rendered working space (`rendered_rgb`):** decode to float32 in `[0, 1]`, preserving source gamma and reference ICC. No color conversion is performed; working pixels remain unclamped until export.

**RAW working space (`scene_linear_camera_rgb`):** LibRaw performs AHD demosaic using unit white balance, linear gamma, no auto brightness/scale, no denoise or median filtering, highlight clipping instead of reconstruction, and raw camera color output. No look, working color space, tone curve, sharpening, or aesthetic white balance is baked. Full-resolution alignment and fusion use the original scene-linear values; robust neutral normalization is measurement-only.

**Metadata/provenance:** portable metadata is always available; optional pyexiv2 enriches supported formats. Single-exposure focus/DoF fields are removed. Results record ordered source SHA-256 hashes, parameters, transforms and quality, exclusions, decoder recipe, device, and versions.

**Output:** rendered results export TIFF (none/LZW/ZIP), true 8/16-bit RGB PNG, or 8-bit JPEG with quality control. DMap can export a 16-bit grayscale depth map. Any result can include an unclamped float32 TIFF. RAW results additionally export an atomic, lossless JPEG-compressed, 16-bit RGB LinearRaw DNG with camera calibration and reversible negative/headroom mapping. The DNG is reopened with tifffile and LibRaw and checked tag-by-tag and pixel-by-pixel before publication. It is not a reconstructed sensor mosaic.

---

## 6. Alignment

Misaligned frames are the #1 cause of bad stacks. Focus stacks shift mainly by **scale** (focus breathing) plus small translation/rotation.

**Pipeline (per stack):**

1. **Reference frame:** middle frame by default (minimizes accumulated scale error); user-selectable.
2. **Pairwise estimation, chained:** estimate the transform between each *consecutive* pair (small inter-frame motion → reliable), then compose transforms to map every frame to the reference. Direct-to-reference estimation is wrong here — do not do it.
3. **Per pair:**
   a. Convert both frames to luminance, downscale so the long edge ≤ 2048 px (configurable 1024–4096).
   b. **Initial guess, in this order (both on GPU):** first **scale + rotation via log-polar phase correlation** (FFT magnitude spectra are translation-invariant → log-polar remap → phase correlation; scale/rotation read off the peak — this recovers focus breathing directly); then warp one frame by that correction and estimate **translation via phase correlation** (`torch.fft` with Hann window) on the corrected pair. Translation must come second: phase correlation on an uncorrected pair is biased when scale/rotation are present.
   c. **Refinement:** OpenCV `findTransformECC` with `MOTION_AFFINE`, warm-started from the initial guess, over a 3-level pyramid (ECC at /4, /2, /1 of the downscaled image, each level initializing the next). The resulting affine is then **projected to the nearest similarity transform** (translation + rotation + uniform scale) via orthogonal Procrustes on the 2×2 block — the similarity model is the output; the affine is only an optimization vehicle. (Note: `findTransformECC` has no native similarity model — do not look for one.) Optional `MOTION_HOMOGRAPHY` mode, used as-is without projection, for hand-held stacks (UI toggle: "Perspective alignment").
   d. **Brightness/flicker normalization** (toggle, default on): for metric evaluation, match the pair's luminance mean/std inside the *consecutive pair's* overlap region; the optional gain applied to output frames is computed *relative to the reference frame* (chained gains composed, like the transforms).
4. **Quality gate:** record final ECC correlation per pair. Pairs below a threshold (default 0.90) are flagged; the UI shows them and offers exclude/keep. CLI flag `--drop-misaligned`. When frame *k* is excluded, the pairwise transform is **re-estimated directly between frames k−1 and k+1** so the chained composition never includes the unreliable link.
5. **Full-resolution warp on GPU:** scale the estimated similarity/homography to full resolution and warp each frame once with the chosen interpolation — Bilinear / Bicubic / **Lanczos-3 (default)** via the portable Lanczos warp op (optional CUDA fast path, §4). Output canvas = reference frame size; out-of-frame areas filled by edge clamp and recorded in a per-frame validity mask. Stacking honors the mask by excluding invalid pixels from selection: energy forced to −∞ (PMax), sharpness/weight forced to 0 (DMap, weighted). So soft borders, not black edges, appear in results.

**Parameters (all exposed in UI + CLI + presets):** transform model (translation / similarity / perspective — the translation model skips the log-polar stage and uses ECC `MOTION_TRANSLATION`), max alignment resolution, interpolation, brightness normalization on/off, correlation threshold, reference frame.

**Skip option:** "Frames are pre-aligned" bypasses alignment entirely (rail + telecentric setups).

---

## 7. Stacking algorithms

Algorithms register themselves in a registry: `@register("pmax")` with a typed `Params` dataclass (name, type, default, range, UI label, tooltip). The UI and CLI build their parameter forms from this metadata — adding an algorithm requires no UI changes.

All algorithms consume: an iterator of aligned float32 frames + validity masks (frames are **streamed from disk**, never all held in memory), and emit: result image + optional auxiliary outputs (depth map, selection map) + progress callbacks.

### 7.0 Smart frame selection (stack thinning) — optional pre-stacking stage — Milestone 3

Adapted from Choi, Pazylbekova, Zhou & van Beek, *Improved Image Selection for Focus Stacking in Digital Photography* (Univ. of Waterloo, <https://cs.uwaterloo.ca/~vanbeek/Publications/focusStacking.pdf>). The paper describes an end-to-end acquisition system; its acquisition half (live-view lens sweep, tethered capture, depth-of-field equations requiring lens distance markings) is out of scope per §1. This stage applies its **selection algorithm** to an already-captured, aligned stack: pick the minimal subset of frames that still covers everything that is in focus somewhere, so fusion is faster and accumulates less noise — the paper measured up to 10× faster fusion at equal quality on over-sampled stacks. Not a fusion method: it runs before any §7.1–7.4 algorithm, is off by default, and is recommended in the UI when a stack exceeds ~40 frames.

Selection runs **after** the §6 step 4 quality-gate exclusions are resolved, over the surviving frames in original focus order; all frame indices below refer to that surviving sequence.

Operates on low-res luminance proxies of the aligned frames (long edge ≈ 1024 px):

1. **Grid focus measures:** overlay a cell grid (default 32×48, configurable). Per cell (i,j) and frame p: φᵢⱼ(p) = Σ |−f(x, y−1) + 2·f(x, y) − f(x, y+1)| over the cell (absolute second difference of luminance, the paper's measure), **normalized by the cell's valid-pixel count** (a per-valid-pixel mean, so frames with differing valid support compare fairly). Pixels outside the frame's §6 validity mask are excluded; a cell with > 25% invalid pixels in any frame is treated as unreliable (mirrors the §6 rule that invalid regions never drive selection).
2. **Curve smoothing:** replace each cell's focus-measure curve (φ across frames) with the sum of its own and its 8 neighbors' curves (fewer at borders) to reduce depth-estimate noise.
3. **Reliability classification:** compute the **kurtosis** of each smoothed curve — pinned convention: Fisher (excess) kurtosis with the population (biased) estimator, so devices and the threshold agree bit-for-bit; cells below a kurtosis threshold are unreliable (textureless or multi-peaked — blank walls, occlusion boundaries) and are excluded. The paper trained a decision tree on ~60 features and found it collapses to this single kurtosis test, dominating the standard-deviation rule of prior work. The threshold is an advanced parameter; its default is calibrated against the synthetic generator (calibration script lives in the test suite) so that known textureless regions are rejected.
4. **Per-cell depth and in-focus interval:** depth = argmax of the smoothed curve. The cell's in-focus interval is the maximal run of consecutive frames around the peak whose smoothed measure ≥ **focus tolerance** × peak (relative threshold, default 0.85, exposed parameter). *Deliberate deviation:* the paper criticizes its predecessor's absolute tolerance and replaces it with depth-of-field equations — which need lens metadata unavailable post-capture — so we use a relative threshold instead, which fixes the same flaw (small-but-distinct peaks) without that metadata.
5. **Coverage rows + peak-set augmentation:** the set-covering matrix gets one row per reliable cell, with that cell's in-focus interval as its coverage. Then, following the paper's §2.3: let L = the sorted set of distinct reliable-peak frame indices; for each maximal run of consecutive indices in L, add **one synthetic row** for the index immediately before and one for the index immediately after the run (clipped to the valid frame range). *Adaptation detail (the paper derives these rows' coverage from DoF equations, which we don't have):* a synthetic row's interval is the union of the in-focus intervals of all reliable cells peaking at the adjacent run endpoint, shifted by ∓1 — intervals clipped to the frame range likewise; rows whose interval becomes empty are dropped. These extra constraints smooth grid discretization across steep depth changes and may modestly increase the number of frames kept — that is the intent.
6. **Set covering:** rows as above; columns = frames. Every row's coverage is a contiguous frame interval (consecutive-ones property), so the minimum cover is exact via classic interval stabbing: sort intervals by right endpoint; repeatedly select the right-endpoint frame of the first uncovered interval. O(n log n), no approximation.
7. **Degenerate-result floor:** if no rows survive reliability classification, or the proposal keeps fewer than 3 frames, the stage keeps **all** frames and emits a warning ("scene too low-contrast for frame selection") — in both UI and CLI. Selection must never silently produce an empty or near-empty stack.
8. **Output:** proposed kept/redundant label per frame + per-cell coverage data, computed by a dedicated `select` job (a full pass over the aligned frames — not instant) whose result is recorded in project.json. The UI (Stack setup screen) reads it from project state and presents the proposal — filmstrip with dropped frames dimmed, coverage grid overlay — for confirmation before stacking; never silently drops frames in interactive use. In batch "Stack all groups" mode and via CLI `--select-frames [--focus-tolerance F]`, proposals are auto-accepted per group, with the selection report kept in job history. The selection and its parameters are recorded in project.json and in the stack job's params for reproducibility. Note: after thinning, kept frames are non-uniformly spaced in focus depth; DMap depth maps (and their exports) index the **kept subset**, and fractional-index blending interpolates between possibly distant focal planes — acceptable, but document it in the depth-map export tooltip.

### 7.1 PMax (Laplacian pyramid, max-energy selection) — Milestone 1

The flagship algorithm, modeled on Zerene PMax.

1. For each frame: build a Laplacian pyramid (Gaussian σ≈1.0 separable blur, downsample ×2; depth = `floor(log2(min(H,W))) − 5`, i.e. coarsest level ≥ 32 px).
2. **Streaming fold:** maintain a running "best" pyramid. For each new frame's pyramid, at every level and pixel, compute local energy = |coefficient| smoothed over a 3×3 window; where the new energy exceeds the running best energy, replace coefficient and energy, and record the frame index in a per-level **winner-index map** (int32 — not uint16, whose torch op coverage is too spotty for the median filter and comparisons below, especially on MPS). The residual (top) level folds by weighted average with energy-derived weights. This makes memory usage **independent of frame count** — required for 200-frame stacks.
3. **Halo control (parameter "selection smoothing", 0–3, default 1):** if > 0, apply a (2s+1)×(2s+1) median filter to each level's winner-index map, then run a **second streaming pass** over the frames — recomputing each frame's pyramid on the fly from the cached aligned frames (per-frame pyramids are never stored) — replacing the folded coefficient wherever the filtered winner differs from the original. Only changed pixels are touched. Setting 0 skips the second pass entirely.
4. Collapse the folded pyramid to the result. Collapse can produce values outside [0, 1]; values are kept unclamped in float32 through post-processing and retouching, and clamped only at the I/O boundary on export (all devices must follow this so exports agree). Document in tooltips that PMax can amplify noise and contrast; that is expected and matches Zerene.

Outputs: result + per-level selection visualization (debug toggle).

### 7.2 DMap (depth map) — Milestone 3

1. Per frame, compute a sharpness map: local Laplacian energy within **estimation radius** (default 8 px, range 2–40).
2. Streaming fold of `argmax` over frames → integer index map + max-sharpness map.
3. **Contrast threshold** (default 7%): pixels of the **folded max-sharpness map** whose value falls below its 7th percentile (i.e., the threshold parameter is a percentile of that map's histogram) are "undecided".
4. Smooth the index map with an edge-aware filter (guided filter, guide = max-sharpness image) with **smoothing radius** (default 16 px). Fill undecided regions by **distance-weighted diffusion from decided neighbors** (iterative masked box blur until converged; do *not* use `cv2.inpaint` — it only accepts 8-bit inputs and the index map is float).
5. **Second pass** over frames: composite output pixels from the frame each pixel's (smoothed, fractional) index points to, linearly blending between adjacent frames for fractional indices.
6. Outputs: result + 16-bit depth map (also drives a "3D preview" in the UI later — out of scope, but the export must exist).

### 7.3 Weighted average — Milestone 3

Softmax blend: `w_i = exp(s_i / T)`, output = Σ wᵢ·pixelᵢ / Σ wᵢ, where `s_i` is the frame's sharpness map **normalized by a single stack-global scale: the reference frame's 99.9th-percentile sharpness** (raw Laplacian energy is unbounded and would overflow `exp`; the scale must be global, not per-frame — per-frame normalization would inflate uniformly blurry frames to full weight). Temperature **T** (default 0.05, lower = closer to hard max) and sharpness radius are parameters. **Single-pass streaming** (plus one upfront read of the cached reference frame to compute the global scale before the pass starts) using the online-softmax trick: maintain per-pixel running max `m`, numerator, and denominator; when a new frame raises `m`, rescale both accumulators by `exp(m_old − m_new)` — numerically stable at any T. Halo-resistant, best for smooth/low-detail subjects; slightly soft.

### 7.4 Slabbing — Milestone 3

For deep stacks: split the ordered frames into overlapping sub-stacks ("slabs") of size **N** (default 10, overlap default 2), stack each slab with a chosen inner method (default PMax), then stack the slab results with an outer method (default DMap). Implemented as a composite job reusing the registry; intermediate slabs are cached in the project workspace and inspectable in the UI.

---

## 8. Tiled processing

`tiles.py` provides `process_tiled(fn, images, tile=2048, overlap=128)`: splits work into overlapping tiles, runs `fn` per tile on GPU, blends overlaps with linear feathering. Used automatically when the VRAM estimate fails (§4). Pyramid algorithms set minimum overlap = 2^(pyramid depth) to avoid seam artifacts. Tiled and untiled outputs must agree within `atol=2e-3` (regression test).

---

## 9. Server and API

Single-user local app. `planefuse serve` starts uvicorn on `127.0.0.1:8425` (configurable) and opens the browser. The server only binds localhost; no auth.

**Job model:** one worker thread executes jobs sequentially (GPU is the bottleneck); queue is inspectable and reorderable. Jobs are cancellable between progress steps (the engine checks a `cancel_event` at each frame/stage boundary). Every job emits typed progress events `{job_id, stage, frame_index?, percent, message, level}` over WebSocket `/ws`.

**Projects:** a project is a directory chosen by the user containing `project.json` (frames list, alignment results, job history, retouch sessions, presets used) plus a `cache/` subdir (aligned frames as compressed npy/TIFF, tile pyramids, slab intermediates). Everything is reproducible from sources + project.json; `cache/` is deletable.

**REST endpoints (prefix `/api`):**

| Route | Purpose |
|---|---|
| `GET/POST /projects`, `GET/DELETE /projects/{id}` | manage projects (POST takes a directory path; DELETE only unregisters — it never deletes user files or the project directory) |
| `POST /projects/{id}/frames/scan` | import frames from a folder path; returns validation report |
| `GET /fs/list?path=` | server-side folder browser (drives, dirs, image counts) |
| `POST /projects/{id}/jobs` | enqueue `{type: align|select|stack|export, params}` (`select` runs §7.0 and records the proposal in project.json) |
| `GET /jobs`, `DELETE /jobs/{id}`, `POST /jobs/reorder` | queue state, cancel, reorder pending jobs (takes the new ordered id list) |
| `PATCH /projects/{id}/ui-state` | persist UI state (current screen, selections, viewer position) as an opaque JSON blob in project.json |
| `GET /viewer/{image_id}/tile/{z}/{x}/{y}` | 256 px JPEG/WebP tiles from cached pyramids (clamped to [0, 1] for display only — stored results stay unclamped float32, §7.1). Every viewable image — source frame, aligned frame, stack result, depth map, flattened retouch result — gets a stable `image_id` recorded in project.json when created |
| `GET/POST /projects/{id}/retouch`, `DELETE /retouch/{session_id}` | list/create/delete retouch sessions (POST takes target result `image_id`; the response includes the session's candidate source images **and a working `image_id` for the live composite**, so the viewer can fetch the image being painted through the normal tile route) |
| `POST /retouch/{session_id}/stroke`, `/undo`, `/redo`, `/flatten` | retouching (§11) |
| `POST /projects/{id}/export` | enqueues an `export` job (large TIFF writes are not instant) with the source `image_id` (any result, depth map, or flattened retouch) and format/naming params |
| `GET /system` | device in use (cuda/mps/cpu), GPU name, VRAM / unified memory, versions |
| `GET /algorithms` | registry metadata → UI builds parameter forms |
| `GET/POST/DELETE /presets` | named parameter sets (global, JSON in user config dir) |

**Batch:** `POST /projects/{id}/frames/auto-group` splits an imported folder into multiple stacks using EXIF timestamp gaps (threshold parameter, default 10 s) and filename patterns; the UI shows proposed groups for confirmation, then "Stack all" enqueues one job per group.

`POST /estimate` provides planning estimates, and `GET /viewer/{image_id}/analysis`
returns server-side RGB/luminance histograms and clipping counts. The complete,
OpenAPI-checked route table is [API.md](API.md).

---

## 10. GUI

Dark, dense, professional tool aesthetic (Lightroom/Capture One register: near-black panels, 1 accent color, generous image canvas, every control tooltipped with photographic context). Keyboard-first: shortcuts for zoom (Z = 100%, F = fit), frame stepping (←/→), before/after (\\), brush size ([ ]), undo (Ctrl+Z).

**Screens:**

1. **Project / Import** — open or create project, folder browser, frame filmstrip with thumbnails, validation badges (size/bit-depth mismatch, low alignment score), auto-group review for batch.
2. **Stack setup** — algorithm picker with parameter forms generated from `/algorithms` metadata, alignment settings, smart frame selection toggle (§7.0) with proposal review (dropped frames dimmed in the filmstrip, coverage grid overlay; suggested automatically above ~40 frames), preset save/load, "Stack" + "Stack all groups" buttons. Estimated VRAM/time display.
3. **Queue / Progress** — job list with per-stage progress bars, live log line, cancel/reorder, history with parameters used (re-run with same/edited params).
4. **Viewer** — tiled deep-zoom canvas (custom Svelte component consuming `/viewer` tiles; smooth wheel zoom centered on cursor, drag pan, 100% toggle). Compare modes: result ↔ any source frame (synced pan/zoom), result A ↔ result B (two methods side by side), depth-map overlay, alignment difference blink. Histogram + clipping indicators.
5. **Retouch** — see §11.
6. **Export** — format, bit depth, TIFF compression, JPEG quality, naming template with live preview, destination, optional depth/float companions, and Linear DNG for RAW-domain results. The UI shows the exact no-bake recipe and explains that Capture One owns all photographic development.

UI state (current project, selections, viewer position) persists across reloads via the project API, not localStorage.

---

## 11. Retouching

Server-side compositing; the UI is a thin client sending strokes and re-fetching invalidated tiles.

- A retouch session targets a stacked result; sources are any aligned frame **or any other stacked result of the same stack** (e.g., paint quiet DMap regions into a crunchy PMax — the classic pro workflow).
- A **stroke** = `{source_id, points: [(x, y, pressure?)], radius, hardness (0–1), opacity (0–1), mode: normal|erase}` in full-image coordinates. Pressure, when present, multiplies opacity (radius is unaffected — deterministic brush footprint). The server rasterizes the stroke mask (Gaussian falloff by hardness), composites source over result at float32, invalidates affected tiles, and responds with the dirty tile list; the viewer refetches only those.
- **History:** the session stores the ordered stroke list (undo/redo = pointer moves + recomposite of affected region from a periodic checkpoint, every 20 strokes, to keep undo O(affected area), not O(all strokes)). Persisted in project.json — sessions reopen intact.
- **UI:** left panel lists candidate sources with thumbnails + sharpness-at-cursor hint ("which frame is sharpest under my cursor" indicator, computed from the DMap index map when available); main canvas paints on the result; optional split view showing the source synced; brush preview circle; pressure support via Pointer Events.
- `flatten` bakes the session into a new named result (the original stack result is never destroyed).

---

## 12. Error handling and robustness rules

- **Validation report** at import: per file → ok / wrong size / wrong bit depth / unreadable / unsupported / RAW decode error / mixed domain / incompatible camera-sensor-calibration contract. Stacking refuses to start until every blocking file is named and resolved.
- Corrupted-mid-job file: skip the frame, emit a warning event, continue; final result notes excluded frames.
- GPU OOM: tile → CPU fallback chain (§4); user is informed, never crashed.
- Cancellation leaves the project consistent (cache writes are atomic: write temp + rename).
- All engine errors surface as typed exceptions (`AlignmentError`, `ValidationError`, `BackendError`); the server maps them to structured JSON errors; the UI shows actionable messages, never a stack trace.
- Server start with port busy → try next port, print/open correct URL.

---

## 13. Testing

### 13.1 Synthetic stack generator (build first — everything depends on it)

`tests/synthetic/generate.py`: renders a ground-truth scene (textured depth ramp + objects at known depths), simulates a focus stack from it: per-frame depth-dependent Gaussian defocus blur, plus configurable per-frame scale (focus breathing, e.g. 0.2%/frame), translation jitter, rotation jitter, brightness flicker, and noise. Returns frames + ground-truth all-in-focus image + ground-truth depth map + ground-truth transforms.

### 13.2 Required test suites

- **Alignment accuracy:** recovered transforms vs ground truth — scale error < 0.05%, translation < 0.5 px at full res, on synthetic stacks with breathing + jitter.
- **Stacking quality:** PMax/DMap/weighted result vs ground-truth sharp image — SSIM > 0.97 on synthetic stacks; DMap depth map correlation > 0.95 with ground truth.
- **Frame selection:** on a synthetic over-sampled stack (e.g., 60 frames where ~12 suffice, derivable from the generator's known per-frame depths of field), every reliable cell's ground-truth depth is in focus in ≥ 1 selected frame, the subset size is ≤ 1.5× the known minimum, and stacking the subset loses < 0.005 SSIM vs stacking all frames. Textureless regions of the synthetic scene must be classified unreliable (kurtosis calibration check).
- **Device parity:** every op and every full algorithm, CPU vs each available accelerator, `atol` as in §4/§8. Accelerator tests auto-skip when the device is absent (CI runs CPU; `uv run pytest -m cuda` on an NVIDIA machine and `uv run pytest -m mps` on an Apple-silicon machine cover the accelerators before release). CUDA fast paths additionally pinned to their torch reference (§4).
- **Tiled = untiled** regression (§8).
- **Golden images:** small real stacks committed to the repo (10 frames, downscaled); results compared by hash-with-tolerance to detect drift.
- **API tests:** full project lifecycle against a temp dir; job cancellation; retouch undo/redo determinism.
- **UI:** Playwright smoke — import synthetic stack, run PMax, see viewer tiles, paint one retouch stroke, export, assert output file exists and opens.
- **RAW/DNG:** generated RAW fixtures pin the no-bake LibRaw parameters, same-camera rejection, scene-linear fusion, source hashes, reversible mapping, required DNG tags, lossless compression, and tifffile/LibRaw decoded-code agreement within one 16-bit code. Browser coverage includes RAW import → validation → stack → DNG export on CPU and targeted MPS.

---

## 14. Milestones (implement strictly in order)

**M1 — Engine core + PMax + CLI.** uv workspace, device abstraction, I/O, synthetic generator, PMax (streaming fold, device-agnostic), tiling, `planefuse stack DIR -o out.tif --method pmax [--device cuda|mps|cpu]`. ✓ when: 13.2 parity + PMax-quality + tiled tests pass; CLI stacks a real pre-aligned stack.

**M2 — Alignment.** Full §6 pipeline, cache of aligned frames, `--align` CLI flags. ✓ when: alignment accuracy tests pass; a real handheld stack visibly aligns (difference preview).

**M3 — DMap, weighted, slabbing, smart frame selection.** Registry metadata complete; §7.0 selection stage with CLI flags and kurtosis-threshold calibration script. ✓ when: quality tests pass for all methods; depth map exports; frame-selection coverage test passes.

**M4 — Server + UI (import → stack → view → export).** Screens 1–4 + 6, jobs/WebSocket, tile viewer, presets, batch auto-group. ✓ when: Playwright smoke (minus retouch) passes; a full stack runs end-to-end from the browser.

**M5 — Retouching.** §11 complete. ✓ when: retouch Playwright + undo/redo determinism tests pass.

**M6 — Polish + distribution (implemented).** Keyboard shortcuts, histogram, synchronized compare modes, export naming templates, validation UX, Docker/Compose/CI, distribution launchers, screenshots, and operational documentation. CUDA hardware and current Capture One manual acceptance remain explicit release-machine gates rather than CI claims.

**Post-M6 RAW extension (implemented).** Explicit processing domains, same-camera RAW validation, no-aesthetic-development LibRaw decode, domain-aware float caching, no-bake UI, and atomic Linear DNG/float companion export for Capture One Pro.

Every milestone ends with: all tests green, `uv run ruff check` + `uv run mypy` clean, a short demo script/GIF.

---

## 15. Performance targets (RTX 3080, 24 MP frames)

| Operation | Target |
|---|---|
| Align 50 frames | < 25 s |
| PMax 50 frames | < 30 s |
| DMap 50 frames | < 45 s |
| Viewer tile response (cached) | < 50 ms |
| Retouch stroke round-trip | < 150 ms |

Apple silicon (M3 Pro-class, MPS): indicative target within ~3× of the RTX 3080 numbers — not a gate, but a >10× gap signals an MPS op silently falling back to CPU and must be investigated. CPU has no speed targets but must complete a 50-frame stack without exceeding 16 GB RAM (streaming design guarantees this).

---

## 16. Tooling, Docker, CI

**uv (mandatory for all Python workflows):** the repo root is a uv workspace (`[tool.uv.workspace] members = ["engine", "server"]`) with one committed lock. All setup and execution use `--frozen`. The mutually exclusive `cpu` and `cu12x` extras select the Torch index; `raw` adds rawpy/LibRaw and ExifRead. Native CPU/MPS setup is `uv sync --frozen --extra cpu --extra raw`; CUDA 12.8 setup is `uv sync --frozen --extra cu12x --extra raw`. The optional `metadata` extra adds pyexiv2 but is not required by core RAW or metadata handling.

**Guided setup:** GitHub Pages hosts a static HTML/JavaScript assistant that
detects macOS or Windows, presents four accessible steps, and remembers local
progress. Tagged releases build the UI once and attach ordinary ZIP folders for
Apple silicon and Windows x64. Each folder contains the source, static UI, and
a checksum-verified official uv binary. The launchers prefer that local binary;
uv downloads a managed Python and the frozen dependency set on first launch.
Node is needed only to prepare the release. There is no frozen executable,
native installer project, signing/notarization pipeline, or system-wide install.

**Docker:** one multi-stage `Dockerfile` with two final targets:

- `cuda`: pinned Node 22 UI and uv stages → `nvidia/cuda:12.8.1-runtime-ubuntu24.04`, Python 3.12.11, and `uv sync --frozen --no-dev --extra cu12x --extra raw`. Run with NVIDIA container support.
- `cpu`: pinned Node 22 UI and uv stages → `python:3.12.11-slim-bookworm` with `uv sync --frozen --no-dev --extra cpu --extra raw`, used by CI and non-NVIDIA hosts.

`docker-compose.yml` runs the cuda target with `gpus: all`, publishes the port as literally `127.0.0.1:8425:8425` (the bare `8425:8425` form would bind 0.0.0.0 and expose the auth-less server to the LAN, violating §9's localhost-only model), and mounts two volumes: the user's photo directory (read-only) and a projects/cache directory (read-write); the in-container server binds `0.0.0.0` (reachable only through the localhost-mapped port). Healthcheck: `GET /api/system`. **MPS is not reachable from containers** — Docker on macOS has no GPU passthrough; Apple-silicon users run natively via uv (documented in the README; the cpu image works on macOS but is the slow path).

**CI (GitHub Actions):** every push and pull request verifies the frozen
CPU+RAW graph (the CUDA wheel never downloads), Ruff, mypy, Python tests,
dependency audits, UI typecheck/build/Playwright, documentation/API parity,
Docker/Compose, and reproducible ready-folder assembly. Clean Apple-silicon
macOS and Windows x64 runners then boot the exact ZIP artifacts with a
uv-managed Python and validate the live API and UI. Every green `main` commit
refreshes a rolling `continuous` release; `v*` tags publish only after all gates
pass and receive provenance attestations when the repository is public. Third-
party Actions are commit-pinned and Dependabot maintains Actions, uv, npm, and
Docker inputs. MPS and CUDA algorithm suites remain real-hardware release gates
per §13.2.
