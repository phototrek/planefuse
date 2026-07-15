# Architecture

FocusStack is one local application with four boundaries: the engine, persistent
projects, a localhost API, and a browser UI. The engine never imports server or UI
code and can be used from Python or the CLI.

## Processing domains

`rendered_rgb` preserves the source gamma and ICC profile of developed TIFF/JPEG/PNG
files. Export is the only place that clamps working pixels to `[0, 1]`.

`scene_linear_camera_rgb` is used only for camera RAW. LibRaw performs AHD demosaic
with unit white balance, linear gamma, no auto brightness/scale, no denoise, no
median filter, highlight clipping rather than reconstruction, and no output-color
conversion. Full-resolution fusion stays in this domain. Robustly normalized,
neutral luminance is used only for alignment and focus measurement proxies.

Mixed-domain stacks are rejected. RAW stacks additionally require the same camera,
active sensor dimensions, orientation, precision, CFA layout, black/white levels,
color matrices, and calibration illuminants.

## Data flow

```text
ordered source files
  -> metadata/domain validation
  -> optional alignment + validity masks (float cache for RAW)
  -> optional reviewed smart selection
  -> streaming stack algorithm
  -> unclamped float32 StackResult + provenance
  -> rendered export OR validated Linear DNG
```

Algorithms consume a `FrameSource`, so frame count does not multiply working-set
memory. PMax and DMap use a second streaming pass where required. If accelerator
memory is exhausted, the orchestration sequence is direct → tiled → tiled CPU.

## Metadata, provenance, and DNG

Every frame carries immutable `ImageMetadata` and a `ProcessingDomain`. Rendered
exports copy safe ICC/EXIF/XMP fields from the reference frame and remove per-focus
fields. RAW results retain the reference camera calibration and the exact decoder
recipe.

Linear DNG output is a 16-bit, contiguous RGB `LinearRaw` IFD compressed with
lossless JPEG. Negative working values and highlight headroom are mapped reversibly
to integer codes; DNG BlackLevel/WhiteLevel and private FocusStack XMP record the
inverse mapping. The XMP also records ordered source SHA-256 hashes, alignment
transforms/quality, exclusions, algorithm parameters, device, and versions.

Publication is atomic. Before replacement, the file is reopened with tifffile and
LibRaw and the decoded codes must agree within one 16-bit code.

## Projects and caches

A project owns `project.json` and `cache/`. `project.json` is the durable record:
sources, results, domain, reference metadata, decoder, provenance, jobs, retouch
state, and UI state. The cache contains aligned frames, masks, result working TIFFs,
and viewer tiles. It is rebuildable and may be deleted while the app is closed.

Project JSON writes and exported files use temporary siblings plus atomic replace.
Registry and viewer registration are serialized so parallel browser activity cannot
clobber projects or return an image id absent from persistent state.

## Server and UI

FastAPI exposes the routes in [API.md](API.md). A single worker executes compute
jobs; WebSocket `/ws` carries progress while REST polling reconciles durable result
and error state. Typed engine errors map to stable API codes.

Image pixels stay server-side. The browser uses 256-pixel deep-zoom tiles and compact
256-bin analysis payloads for histograms. Compare viewers share one transform, so
pan and zoom remain synchronized.

## Security

Native serving is localhost-only. The Docker process listens on `0.0.0.0` inside the
container, but Compose publishes only `127.0.0.1`. Photo-library mounts are read-only;
projects/cache are read-write. There are no accounts or remote/cloud operations.
