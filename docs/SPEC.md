# FocusStack — Professional GPU-Accelerated Focus Stacking Software

**Specification v1.0 — 2026-06-12**

This document is a complete, self-contained specification for building FocusStack, a professional focus-stacking application. It is written to be handed to an LLM (or a human team) for implementation. Follow the milestones in order; each has explicit acceptance criteria.

---

## 1. Product overview

FocusStack merges a series of photographs taken at different focus distances ("a stack") into a single image that is sharp everywhere. Target users are professional macro, product, and landscape photographers who today use Zerene Stacker or Helicon Focus.

**Core promises:**

1. **Quality** — alignment and stacking results competitive with Zerene Stacker, including handling of focus breathing (scale change between frames), halo control, and full interactive retouching.
2. **Speed** — CUDA acceleration for every heavy operation; a 50-frame 24 MP stack aligns and stacks (PMax) in under 60 seconds on an RTX 3080-class GPU.
3. **Robustness** — never crashes on bad input or GPU memory exhaustion; every failure mode degrades gracefully (tiling → CPU fallback → clear error).
4. **Professional workflow** — 16-bit pipeline end to end, ICC profile and EXIF preservation, batch processing, presets, CLI for automation, full retouching.

**Explicit non-goals (do not build):** RAW decoding (input is developed TIFF/JPEG/PNG), exposure/HDR stacking, astro stacking, panorama stitching, tethered capture / camera control, cloud processing, mobile support, user accounts.

---

## 2. Tech stack and constraints

| Layer | Technology |
|---|---|
| Processing engine | Python 3.12+, CuPy (CUDA 12.x) with custom `RawKernel`s for hot paths, NumPy CPU fallback |
| Image I/O | `tifffile` (TIFF), `Pillow` (JPEG/PNG), `pyexiv2` (EXIF/XMP/ICC metadata) |
| Alignment math | `opencv-python-headless` (ECC refinement on downscaled luminance), CuPy FFT (phase correlation), GPU warping via custom kernels |
| Server | FastAPI + uvicorn, WebSocket progress streaming |
| Frontend | Svelte 5 + Vite + TypeScript, no heavyweight UI framework |
| Packaging | Two Python packages with their own `pyproject.toml`: `focusstack` (engine) and `focusstack-server` (depends on engine, bundles the built UI). The engine CLI's `serve` subcommand lazily imports the server package and prints an install hint if absent. `npm` workspace for UI |
| Platforms | Windows 11 and Linux. NVIDIA GPU (CUDA 12.x) is the primary target; everything must also run on CPU (NumPy) for machines without CUDA and for CI. |

**Hard rules:**

- The engine package (`focusstack.engine`) must have **zero imports from server or UI code**. It is importable and fully usable from a plain Python script.
- Every algorithm runs identically (within float tolerance) on the CuPy and NumPy backends. The backend is selected at runtime, never at install time.
- All image math is float32 in the source gamma space (see §5 — no linearization anywhere). Integer math only at I/O boundaries.
- No global mutable state in the engine. A stacking job is a pure function of (frames, parameters) → result.

---

## 3. Repository layout

```
focus-stacker/
├── docs/
│   └── SPEC.md                  # this document
├── engine/
│   ├── pyproject.toml           # package: focusstack
│   └── src/focusstack/
│       ├── backend/             # xp abstraction: cupy/numpy, kernel registry, memory pool
│       │   ├── __init__.py      # get_backend(), Backend protocol
│       │   ├── cuda.py          # CuPy backend + RawKernel sources
│       │   ├── cpu.py           # NumPy backend (mirrors cuda.py API exactly)
│       │   └── kernels/         # .cu kernel source files (loaded as text)
│       ├── io/                  # load/save, metadata, color profiles
│       ├── align/               # registration pipeline
│       ├── select/              # smart frame selection / stack thinning (§7.0)
│       ├── stack/               # algorithm registry: pmax.py, dmap.py, weighted.py, slab.py
│       ├── retouch/             # stroke compositing engine
│       ├── post/                # halo suppression, contrast, sharpening
│       ├── pipeline.py          # orchestrates load→align→select→stack→post
│       ├── tiles.py             # tiled processing + viewer tile pyramid
│       ├── project.py           # project file model (JSON on disk)
│       └── cli.py               # `focusstack` CLI entry point
├── server/
│   ├── pyproject.toml           # package: focusstack-server (depends on focusstack)
│   └── src/focusstack_server/
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
└── tests/
    ├── engine/                  # unit + golden tests
    ├── server/                  # API tests
    └── synthetic/               # synthetic stack generator (§13.1)
```

---

## 4. Engine: backend abstraction

```python
class Backend(Protocol):
    name: str                    # "cuda" | "cpu"
    xp: ModuleType               # cupy or numpy
    def to_device(self, arr: np.ndarray) -> Array: ...
    def to_host(self, arr: Array) -> np.ndarray: ...
    def gaussian_blur(self, img, sigma) -> Array: ...
    def resize(self, img, shape, interp) -> Array: ...
    def warp(self, img, matrix, out_shape, interp) -> Array: ...   # affine or homography
    def sharpness_map(self, gray, radius) -> Array: ...            # local Laplacian energy
    def free_memory(self) -> int: ...                              # bytes available
```

- `get_backend(prefer: str = "auto")` returns the CUDA backend if CuPy initializes and a device is present, else CPU. The choice is logged and surfaced in the UI/system API. `prefer="cpu"` forces CPU (used in tests).
- Custom CUDA kernels (in `backend/kernels/*.cu`, compiled with `cp.RawKernel`): Lanczos-3 warp (affine + homography), windowed Laplacian energy, pyramid coefficient selection (max-energy fold, §7.1), DMap blend. Everything else uses CuPy array ops / `cupyx.scipy.ndimage`.
- CPU backend implements the same operations with NumPy/SciPy/OpenCV. **Parity requirement:** for every backend operation, a test asserts CPU and GPU outputs agree within `atol=1e-3` on random inputs.
- **Memory discipline:** before each GPU stage, the engine estimates required VRAM. If the estimate exceeds 80% of free VRAM, it switches that stage to tiled mode (§8). On `cupy.cuda.memory.OutOfMemoryError` despite tiling, it frees the pool and reruns the stage on CPU, emitting a warning event. A job never fails due to OOM.

---

## 5. Image I/O and color

**Input:** 8/16-bit RGB TIFF (incl. compressed), JPEG, PNG. Grayscale input is rejected with a clear validation message (RGB only in v1). A stack's frames must share dimensions and bit depth; violations produce a per-file validation report (§12). **Frame order** within a stack defaults to natural filename sort, with an EXIF capture-time option; the UI allows reversing (DMap depth semantics and slabbing depend on focus order — near-to-far vs far-to-near must be consistent, which is all that matters).

**Working space:** images are decoded to float32 in [0, 1], **keeping the source gamma** (no linearization — stacking sharpness metrics behave better in gamma space, and this matches Zerene/Helicon behavior). The ICC profile bytes of the reference frame are carried through untouched and embedded in the output. No color conversion is ever performed.

**Metadata:** EXIF and XMP are copied from the reference frame to the output via `pyexiv2`. Then the `Software` tag is set to `FocusStack <version>`, and tags that no longer apply to a merged image (focus distance, depth-of-field) are dropped. Add an XMP namespace `focusstack:` recording method, parameter hash, frame count, and app version.

**Output:** 16-bit TIFF (default, with chosen compression: none/LZW/ZIP), 8-bit JPEG (quality slider), 16-bit PNG. File naming template with tokens: `{stack_name}`, `{method}`, `{frames}`, `{date}`, `{seq}`. DMap mode can additionally export the depth map as 16-bit grayscale TIFF/PNG.

---

## 6. Alignment

Misaligned frames are the #1 cause of bad stacks. Focus stacks shift mainly by **scale** (focus breathing) plus small translation/rotation.

**Pipeline (per stack):**

1. **Reference frame:** middle frame by default (minimizes accumulated scale error); user-selectable.
2. **Pairwise estimation, chained:** estimate the transform between each *consecutive* pair (small inter-frame motion → reliable), then compose transforms to map every frame to the reference. Direct-to-reference estimation is wrong here — do not do it.
3. **Per pair:**
   a. Convert both frames to luminance, downscale so the long edge ≤ 2048 px (configurable 1024–4096).
   b. **Initial guess, in this order (both on GPU):** first **scale + rotation via log-polar phase correlation** (FFT magnitude spectra are translation-invariant → log-polar remap → phase correlation; scale/rotation read off the peak — this recovers focus breathing directly); then warp one frame by that correction and estimate **translation via phase correlation** (CuPy FFT with Hann window) on the corrected pair. Translation must come second: phase correlation on an uncorrected pair is biased when scale/rotation are present.
   c. **Refinement:** OpenCV `findTransformECC` with `MOTION_AFFINE`, warm-started from the initial guess, over a 3-level pyramid (ECC at /4, /2, /1 of the downscaled image, each level initializing the next). The resulting affine is then **projected to the nearest similarity transform** (translation + rotation + uniform scale) via orthogonal Procrustes on the 2×2 block — the similarity model is the output; the affine is only an optimization vehicle. (Note: `findTransformECC` has no native similarity model — do not look for one.) Optional `MOTION_HOMOGRAPHY` mode, used as-is without projection, for hand-held stacks (UI toggle: "Perspective alignment").
   d. **Brightness/flicker normalization** (toggle, default on): for metric evaluation, match the pair's luminance mean/std inside the *consecutive pair's* overlap region; the optional gain applied to output frames is computed *relative to the reference frame* (chained gains composed, like the transforms).
4. **Quality gate:** record final ECC correlation per pair. Pairs below a threshold (default 0.90) are flagged; the UI shows them and offers exclude/keep. CLI flag `--drop-misaligned`. When frame *k* is excluded, the pairwise transform is **re-estimated directly between frames k−1 and k+1** so the chained composition never includes the unreliable link.
5. **Full-resolution warp on GPU:** scale the estimated similarity/homography to full resolution and warp each frame once with the chosen interpolation — Bilinear / Bicubic / **Lanczos-3 (default)** via the custom kernel. Output canvas = reference frame size; out-of-frame areas filled by edge clamp and recorded in a per-frame validity mask. Stacking honors the mask by excluding invalid pixels from selection: energy forced to −∞ (PMax), sharpness/weight forced to 0 (DMap, weighted). So soft borders, not black edges, appear in results.

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
3. **Reliability classification:** compute the **kurtosis** of each smoothed curve — pinned convention: Fisher (excess) kurtosis with the population (biased) estimator, so backends and the threshold agree bit-for-bit; cells below a kurtosis threshold are unreliable (textureless or multi-peaked — blank walls, occlusion boundaries) and are excluded. The paper trained a decision tree on ~60 features and found it collapses to this single kurtosis test, dominating the standard-deviation rule of prior work. The threshold is an advanced parameter; its default is calibrated against the synthetic generator (calibration script lives in the test suite) so that known textureless regions are rejected.
4. **Per-cell depth and in-focus interval:** depth = argmax of the smoothed curve. The cell's in-focus interval is the maximal run of consecutive frames around the peak whose smoothed measure ≥ **focus tolerance** × peak (relative threshold, default 0.85, exposed parameter). *Deliberate deviation:* the paper criticizes its predecessor's absolute tolerance and replaces it with depth-of-field equations — which need lens metadata unavailable post-capture — so we use a relative threshold instead, which fixes the same flaw (small-but-distinct peaks) without that metadata.
5. **Coverage rows + peak-set augmentation:** the set-covering matrix gets one row per reliable cell, with that cell's in-focus interval as its coverage. Then, following the paper's §2.3: let L = the sorted set of distinct reliable-peak frame indices; for each maximal run of consecutive indices in L, add **one synthetic row** for the index immediately before and one for the index immediately after the run (clipped to the valid frame range). *Adaptation detail (the paper derives these rows' coverage from DoF equations, which we don't have):* a synthetic row's interval is the union of the in-focus intervals of all reliable cells peaking at the adjacent run endpoint, shifted by ∓1 — intervals clipped to the frame range likewise; rows whose interval becomes empty are dropped. These extra constraints smooth grid discretization across steep depth changes and may modestly increase the number of frames kept — that is the intent.
6. **Set covering:** rows as above; columns = frames. Every row's coverage is a contiguous frame interval (consecutive-ones property), so the minimum cover is exact via classic interval stabbing: sort intervals by right endpoint; repeatedly select the right-endpoint frame of the first uncovered interval. O(n log n), no approximation.
7. **Degenerate-result floor:** if no rows survive reliability classification, or the proposal keeps fewer than 3 frames, the stage keeps **all** frames and emits a warning ("scene too low-contrast for frame selection") — in both UI and CLI. Selection must never silently produce an empty or near-empty stack.
8. **Output:** proposed kept/redundant label per frame + per-cell coverage data, computed by a dedicated `select` job (a full pass over the aligned frames — not instant) whose result is recorded in project.json. The UI (Stack setup screen) reads it from project state and presents the proposal — filmstrip with dropped frames dimmed, coverage grid overlay — for confirmation before stacking; never silently drops frames in interactive use. In batch "Stack all groups" mode and via CLI `--select-frames [--focus-tolerance F]`, proposals are auto-accepted per group, with the selection report kept in job history. The selection and its parameters are recorded in project.json and in the stack job's params for reproducibility. Note: after thinning, kept frames are non-uniformly spaced in focus depth; DMap depth maps (and their exports) index the **kept subset**, and fractional-index blending interpolates between possibly distant focal planes — acceptable, but document it in the depth-map export tooltip.

### 7.1 PMax (Laplacian pyramid, max-energy selection) — Milestone 1

The flagship algorithm, modeled on Zerene PMax.

1. For each frame: build a Laplacian pyramid (Gaussian σ≈1.0 separable blur, downsample ×2; depth = `floor(log2(min(H,W))) − 5`, i.e. coarsest level ≥ 32 px).
2. **Streaming fold:** maintain a running "best" pyramid. For each new frame's pyramid, at every level and pixel, compute local energy = |coefficient| smoothed over a 3×3 window; where the new energy exceeds the running best energy, replace coefficient and energy, and record the frame index in a per-level **winner-index map** (uint16). The residual (top) level folds by weighted average with energy-derived weights. This makes memory usage **independent of frame count** — required for 200-frame stacks.
3. **Halo control (parameter "selection smoothing", 0–3, default 1):** if > 0, apply a (2s+1)×(2s+1) median filter to each level's winner-index map, then run a **second streaming pass** over the frames — recomputing each frame's pyramid on the fly from the cached aligned frames (per-frame pyramids are never stored) — replacing the folded coefficient wherever the filtered winner differs from the original. Only changed pixels are touched. Setting 0 skips the second pass entirely.
4. Collapse the folded pyramid to the result. Collapse can produce values outside [0, 1]; values are kept unclamped in float32 through post-processing and retouching, and clamped only at the I/O boundary on export (both backends must follow this so exports agree). Document in tooltips that PMax can amplify noise and contrast; that is expected and matches Zerene.

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

Single-user local app. `focusstack serve` starts uvicorn on `127.0.0.1:8425` (configurable) and opens the browser. The server only binds localhost; no auth.

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
| `GET /system` | backend in use, GPU name, VRAM, versions |
| `GET /algorithms` | registry metadata → UI builds parameter forms |
| `GET/POST/DELETE /presets` | named parameter sets (global, JSON in user config dir) |

**Batch:** `POST /projects/{id}/frames/auto-group` splits an imported folder into multiple stacks using EXIF timestamp gaps (threshold parameter, default 10 s) and filename patterns; the UI shows proposed groups for confirmation, then "Stack all" enqueues one job per group.

---

## 10. GUI

Dark, dense, professional tool aesthetic (Lightroom/Capture One register: near-black panels, 1 accent color, generous image canvas, every control tooltipped with photographic context). Keyboard-first: shortcuts for zoom (Z = 100%, F = fit), frame stepping (←/→), before/after (\\), brush size ([ ]), undo (Ctrl+Z).

**Screens:**

1. **Project / Import** — open or create project, folder browser, frame filmstrip with thumbnails, validation badges (size/bit-depth mismatch, low alignment score), auto-group review for batch.
2. **Stack setup** — algorithm picker with parameter forms generated from `/algorithms` metadata, alignment settings, smart frame selection toggle (§7.0) with proposal review (dropped frames dimmed in the filmstrip, coverage grid overlay; suggested automatically above ~40 frames), preset save/load, "Stack" + "Stack all groups" buttons. Estimated VRAM/time display.
3. **Queue / Progress** — job list with per-stage progress bars, live log line, cancel/reorder, history with parameters used (re-run with same/edited params).
4. **Viewer** — tiled deep-zoom canvas (custom Svelte component consuming `/viewer` tiles; smooth wheel zoom centered on cursor, drag pan, 100% toggle). Compare modes: result ↔ any source frame (synced pan/zoom), result A ↔ result B (two methods side by side), depth-map overlay, alignment difference blink. Histogram + clipping indicators.
5. **Retouch** — see §11.
6. **Export** — format, bit depth, compression, quality, naming template with live preview, destination, "also export depth map".

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

- **Validation report** at import: per file → ok / wrong size / wrong bit depth / unreadable / unsupported format. Stacking refuses to start on mixed dimensions; the message names the offending files.
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
- **CPU/GPU parity:** every backend op and every full algorithm, `atol` as in §4/§8. GPU tests auto-skip without CUDA (CI runs CPU; a local `pytest -m gpu` run covers CUDA before release).
- **Tiled = untiled** regression (§8).
- **Golden images:** small real stacks committed to the repo (10 frames, downscaled); results compared by hash-with-tolerance to detect drift.
- **API tests:** full project lifecycle against a temp dir; job cancellation; retouch undo/redo determinism.
- **UI:** Playwright smoke — import synthetic stack, run PMax, see viewer tiles, paint one retouch stroke, export, assert output file exists and opens.

---

## 14. Milestones (implement strictly in order)

**M1 — Engine core + PMax + CLI.** Backend abstraction, I/O, synthetic generator, PMax (streaming fold, both backends), tiling, `focusstack stack DIR -o out.tif --method pmax [--cpu]`. ✓ when: 13.2 parity + PMax-quality + tiled tests pass; CLI stacks a real pre-aligned stack.

**M2 — Alignment.** Full §6 pipeline, cache of aligned frames, `--align` CLI flags. ✓ when: alignment accuracy tests pass; a real handheld stack visibly aligns (difference preview).

**M3 — DMap, weighted, slabbing, smart frame selection.** Registry metadata complete; §7.0 selection stage with CLI flags and kurtosis-threshold calibration script. ✓ when: quality tests pass for all methods; depth map exports; frame-selection coverage test passes.

**M4 — Server + UI (import → stack → view → export).** Screens 1–4 + 6, jobs/WebSocket, tile viewer, presets, batch auto-group. ✓ when: Playwright smoke (minus retouch) passes; a full stack runs end-to-end from the browser.

**M5 — Retouching.** §11 complete. ✓ when: retouch Playwright + undo/redo determinism tests pass.

**M6 — Polish.** Keyboard shortcuts complete, histogram, compare modes, export naming templates, validation UX, docs (`README` with screenshots, user guide).

Every milestone ends with: all tests green, `ruff` + `mypy` clean, a short demo script/GIF.

---

## 15. Performance targets (RTX 3080, 24 MP frames)

| Operation | Target |
|---|---|
| Align 50 frames | < 25 s |
| PMax 50 frames | < 30 s |
| DMap 50 frames | < 45 s |
| Viewer tile response (cached) | < 50 ms |
| Retouch stroke round-trip | < 150 ms |

CPU fallback has no targets but must complete a 50-frame stack without exceeding 16 GB RAM (streaming design guarantees this).
