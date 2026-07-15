# Phase 1 Stability and Security Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore a green CPU/MPS baseline, remove known actionable advisories, and make CI reproduce the supported runtime matrix.

**Architecture:** Keep all public APIs stable. Repair device math at the shared warp boundary, recover unreliable chained alignment with a validated direct-to-reference fallback, then modernize toolchains and make CPU browser smoke mandatory in CI.

**Tech Stack:** Python 3.12, PyTorch, OpenCV, pytest, uv, Svelte 5, Vite 8, Playwright, GitHub Actions.

## Global Constraints

- MPS tensor operations remain float32; NumPy transform bookkeeping may remain float64 on CPU.
- An alignment transform is usable only when ECC correlation meets the configured quality threshold.
- Existing TIFF/JPEG/PNG behavior and CLI/API shapes remain compatible.
- Dependency audit exceptions require a source-backed explanation and may not hide an actionable fixed advisory.
- Every behavior change follows red-green-refactor.

---

### Task 1: Make pixel-coordinate warping MPS-safe

**Files:**
- Modify: `tests/engine/test_align_ops.py`
- Modify: `engine/src/focusstack/backend/ops.py:107-118`

**Interfaces:**
- Consumes: `ops.warp(img, matrix, out_shape, interp)`
- Produces: `_px_to_norm(...)` operating entirely in float32 when the matrix is float32/MPS.

- [ ] **Step 1: Add an explicit dtype regression test**

```python
def test_px_to_norm_preserves_float32():
    matrix = torch.eye(3, dtype=torch.float32)
    theta = ops._px_to_norm(matrix, 64, 80, 64, 80)
    assert theta.dtype == torch.float32
```

- [ ] **Step 2: Run the existing MPS parity test and record the current failure**

Run on Apple silicon: `.venv/bin/pytest tests/engine/test_align_ops.py::test_warp_parity -q`

Expected before implementation: `Cannot convert a MPS Tensor to float64 dtype`.

- [ ] **Step 3: Keep normalization math on the input floating dtype**

Replace the forced `torch.float64` with:

```python
dt = m.dtype if m.is_floating_point() else torch.float32
work = m.to(dtype=dt)
```

Build `s_out` and `s_in` with `dtype=dt`, and return `norm[:2].to(torch.float32)`.

- [ ] **Step 4: Run CPU and MPS alignment-op tests**

Run: `.venv/bin/pytest tests/engine/test_align_ops.py -q`

Expected: all available-device cases pass.

- [ ] **Step 5: Commit**

```bash
git add engine/src/focusstack/backend/ops.py tests/engine/test_align_ops.py
git commit -m "fix: keep warp normalization MPS-safe"
```

### Task 2: Recover unreliable chain links directly to the reference

**Files:**
- Modify: `engine/src/focusstack/align/pipeline.py`
- Modify: `tests/engine/test_alignment_accuracy.py`
- Modify: `tests/engine/test_align_pipeline.py`

**Interfaces:**
- Consumes: consecutive `PairResult` objects and `AlignParams.correlation_threshold`.
- Produces: reliable per-frame matrices; `AlignReport.recovered: set[int]`; `flagged` contains only unrecovered frames.

- [ ] **Step 1: Extend the failing seed regression to assert recovery**

After the existing transform assertions add:

```python
if seed == 41:
    assert {0, 1} <= report.recovered
    assert 0 not in report.flagged
    assert 1 not in report.flagged
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest tests/engine/test_alignment_accuracy.py -q`

Expected: seed 41 fails with the existing 2.3% scale error and `AlignReport` has no `recovered` field.

- [ ] **Step 3: Add direct fallback for paths crossing bad links**

Add `recovered: set[int]` to `AlignReport`. After the normal chain is built, compute whether the reference-to-frame path crosses any pair below threshold. For each affected non-reference frame call:

```python
direct = estimate_pair(
    source.read(ref), source.read(frame_idx), device,
    max_long_edge=params.max_long_edge,
    model=params.model,
    normalize_brightness=params.normalize_brightness,
)
if direct.correlation >= params.correlation_threshold:
    matrices[frame_idx] = direct.matrix
    recovered.add(frame_idx)
```

Remove recovered frames from `flagged`. Preserve the existing explicit drop-and-bridge behavior for frames whose direct fallback is still unreliable.

- [ ] **Step 4: Add a mocked unrecoverable-frame test**

In `test_align_pipeline.py`, monkeypatch `estimate_pair` so both the chain link and direct fallback remain below threshold; assert the frame stays flagged and is dropped only when `drop_misaligned=True`.

- [ ] **Step 5: Run alignment suites**

Run: `.venv/bin/pytest tests/engine/test_alignment_accuracy.py tests/engine/test_align_pipeline.py tests/engine/test_chain.py -q`

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add engine/src/focusstack/align/pipeline.py tests/engine/test_alignment_accuracy.py tests/engine/test_align_pipeline.py
git commit -m "fix: recover low-confidence alignment links"
```

### Task 3: Pin supported runtimes and upgrade secure dependencies

**Files:**
- Create: `.python-version`
- Create: `.nvmrc`
- Modify: `pyproject.toml`
- Modify: `engine/pyproject.toml`
- Modify: `server/pyproject.toml`
- Modify: `ui/package.json`
- Modify: `ui/package-lock.json`
- Modify: `uv.lock`

**Interfaces:**
- Produces: Python 3.12 and Node >=22.12 reproducible environments with no fixed advisories left unresolved.

- [ ] **Step 1: Add runtime pins**

`.python-version` contains `3.12`; `.nvmrc` contains `22`. Add to `ui/package.json`:

```json
"engines": { "node": "^22.12 || >=24" }
```

- [ ] **Step 2: Raise direct secure floors**

Set Pillow to `>=12.3`, retain OpenCV `<5` for this phase, and add explicit comments/constraints for packages whose major upgrade is deferred to Phase 2 compatibility testing.

- [ ] **Step 3: Upgrade the UI toolchain as one compatible set**

Use Vite `^8.1.4`, Svelte plugin `^7.2.0`, SvelteKit `^2.69.3`, Svelte `^5.56.5`, adapter-static `^3.0.10`, svelte-check `^4.7.2`, Playwright `^1.61.1`, and TypeScript `^6.0.3`. Regenerate with `npm install`.

- [ ] **Step 4: Refresh the Python lock**

Run: `uv lock --upgrade`

Then sync each supported extra independently with `--frozen` and confirm CPU/CUDA conflict resolution remains valid.

- [ ] **Step 5: Run audits**

Run: `npm audit --audit-level=high`

Run: `uvx pip-audit --path .venv/lib/python3.12/site-packages`

Document CVE-2025-3000 as not applicable only if the installed Torch is not the affected 2.6.0 and the project never calls `torch.jit.script`; do not suppress any advisory with a published applicable fix.

- [ ] **Step 6: Run compatibility checks**

Run: `.venv/bin/pytest -q`

Run: `.venv/bin/ruff check .`

Run: `.venv/bin/mypy engine/src server/src`

Run: `npm run check && npm run build && npm run test:e2e`

Expected: all pass on the current Apple-silicon host.

- [ ] **Step 7: Commit**

```bash
git add .python-version .nvmrc pyproject.toml engine/pyproject.toml server/pyproject.toml uv.lock ui/package.json ui/package-lock.json
git commit -m "build: update secure supported toolchains"
```

### Task 4: Make device choice and browser CI deterministic

**Files:**
- Modify: `engine/src/focusstack/backend/device.py`
- Modify: `tests/engine/test_device.py`
- Modify: `ui/playwright.config.ts`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `FOCUSSTACK_DEVICE=cpu|mps|cuda|auto` as the default used only when callers request `auto`.

- [ ] **Step 1: Write environment-preference tests**

Use `monkeypatch.setenv("FOCUSSTACK_DEVICE", "cpu")` and assert `get_device("auto").kind == "cpu"`; assert an explicit `get_device("mps")` is not overridden.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest tests/engine/test_device.py -q`

Expected: auto still selects the host accelerator.

- [ ] **Step 3: Implement the environment default**

At the start of `get_device`, resolve `auto` through `FOCUSSTACK_DEVICE`, validate it with the same `_VALID` set, and leave explicit arguments untouched.

- [ ] **Step 4: Pin Playwright webServer to CPU in CI/local deterministic mode**

Set the web server environment to `FOCUSSTACK_DEVICE=cpu`; add a separate `test:e2e:mps` script that sets MPS for the release gate.

- [ ] **Step 5: Add browser smoke to CI**

Install Chromium with `npx playwright install --with-deps chromium` and run `npm run test:e2e`. Pin Python 3.12 and Node 22 in the workflow.

- [ ] **Step 6: Verify**

Run the device tests, CPU Playwright suite, and MPS Playwright suite.

- [ ] **Step 7: Commit**

```bash
git add engine/src/focusstack/backend/device.py tests/engine/test_device.py ui/playwright.config.ts ui/package.json .github/workflows/ci.yml
git commit -m "ci: verify deterministic browser workflows"
```

### Task 5: Phase 1 verification

**Files:** No production edits.

- [ ] Run all Python tests, Ruff, mypy, Svelte diagnostics, production build, CPU browser tests, MPS browser tests, Docker Compose validation, npm audit, and pip-audit.
- [ ] Confirm `git diff --check` and inspect `git status --short`.
- [ ] Record exact pass/skip counts in the Phase 4 verification document.
