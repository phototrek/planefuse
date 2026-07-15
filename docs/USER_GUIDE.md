# User guide

## Import and validation

Create/open a project, then add a folder or individual files. Files remain in place.
The validation panel names every blocking file. Rendered stacks must share size and
bit depth; RAW stacks must also satisfy the same-camera/calibration contract.

![RAW validation](assets/raw-workspace.png)

Run remains disabled until validation passes. A stack above 40 frames shows a smart-
selection recommendation.

## Stack setup

Choose one or several algorithms. Multiple algorithms produce comparison results.
Alignment defaults to similarity + Lanczos-3; translation, perspective, brightness
matching, quality threshold, and automatic low-quality exclusion are available.
Use “frames are pre-aligned” only for genuinely registered rail/telecentric captures.

Smart selection runs a dedicated proposal job. Review kept/redundant frames and
coverage, then accept before stacking. Auto-group also shows groups before “Stack all”.

Memory/time badges are planning estimates, not benchmark promises.

## Queue and results

The render drawer separates active jobs from finished history. Pending jobs can be
reordered, active jobs cancelled, and completed stack parameters inspected/re-run.
Typed errors show an actionable message rather than a traceback.

## Viewer

The viewer streams tiles. Available modes are single, result/source split,
result/result side-by-side, and hold-to-view before/after. Compare panes share pan
and zoom. Histogram data and independent RGB shadow/highlight counts are computed
server-side.

Keyboard: `F` fit, `Z` 100%, arrows step frames, `\` before/after, `Esc` exits
compare, and `?` opens help. Text fields suppress workspace shortcuts.

![Synchronized comparison and histogram](assets/compare-histogram.png)

## Retouch

Open Retouch on a result, choose another result/source, and paint in full-image
coordinates. `[`/`]` change brush size; Ctrl/Cmd+Z undoes and redo is available.
Flatten creates a new result and never destroys the original.

## Export

TIFF supports none/LZW/ZIP compression, PNG supports true 8/16-bit RGB, and JPEG is
8-bit with quality control. DMap results can include a 16-bit depth map. Any result
can include an unclamped 32-bit float TIFF companion.

RAW results additionally enable validated Linear DNG. See the dedicated
[RAW/Capture One guide](RAW_DNG_CAPTURE_ONE.md).

![Linear DNG export](assets/dng-export.png)

## Project safety

Removing a project only unregisters it. Source files are never deleted. `cache/` is
rebuildable; close FocusStack before clearing it. Keep `project.json` with the source
files for provenance and repeatability.
