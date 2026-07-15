# Phase 4 Distribution and Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship reproducible cross-platform workflows and documentation that matches verified behavior, including RAW-to-Capture-One guidance.

**Architecture:** Treat commands, screenshots, API tables, and feature matrices as tested artifacts. Generate screenshots from deterministic Playwright fixtures and derive API documentation from the running FastAPI schema where practical.

**Tech Stack:** Bash/Batch, Docker, GitHub Actions, Markdown, FastAPI OpenAPI, Playwright.

## Global Constraints

- Documentation claims require a corresponding passing verification item.
- CUDA/MPS claims identify the exact hardware and software version tested.
- Proprietary user photos are never committed; demos and screenshots use redistributable fixtures.
- Localhost-only security remains explicit in native and Docker instructions.

---

### Task 1: Harden launchers and containers

**Files:**
- Modify: `scripts/start-macos.sh`
- Modify: `scripts/start-windows-gpu.bat`
- Modify: `scripts/start-windows-cpu.bat`
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`
- Modify: `.github/workflows/ci.yml`

- [ ] Add launcher smoke tests or shell-level assertions for runtime/version checks, locked install, first-build behavior, device selection, and propagated exit codes.
- [ ] Use `npm ci`, `uv sync --frozen`, pinned image/tool versions, and clear RAW-extra installation.
- [ ] Validate CPU image health and CUDA Compose configuration without exposing the server beyond localhost.
- [ ] Commit with `build: harden distribution workflows`.

### Task 2: Publish accurate architecture and API references

**Files:**
- Modify: `docs/SPEC.md`
- Create: `docs/ARCHITECTURE.md`
- Create: `docs/API.md`
- Create: `scripts/check_docs.py`

- [ ] Mark `SPEC.md` requirements with verified implementation status and replace nonexistent layout entries with the real tree.
- [ ] Document processing domains, RAW/DNG data flow, cache/provenance model, fallback chain, and security boundaries.
- [ ] Generate/compare endpoint names from FastAPI OpenAPI so stale API rows fail CI.
- [ ] Add local-link and command-reference checks.
- [ ] Commit with `docs: publish verified architecture and API`.

### Task 3: Write installation, user, RAW, and troubleshooting guides

**Files:**
- Modify: `README.md`
- Create: `docs/INSTALL.md`
- Create: `docs/USER_GUIDE.md`
- Create: `docs/RAW_DNG_CAPTURE_ONE.md`
- Create: `docs/TROUBLESHOOTING.md`
- Create: `docs/RELEASE_CHECKLIST.md`

- [ ] Test every documented setup/launch/check command on its applicable environment or label it as a release-hardware command.
- [ ] Document rendered and RAW workflows, same-camera constraints, no-bake behavior, Linear-DNG limitations, Capture One Pro acceptance, compare/retouch/export, project/cache safety, and recovery paths.
- [ ] Include dependency/security audit and accelerator release gates.
- [ ] Commit with `docs: add complete user and RAW guides`.

### Task 4: Generate screenshots, demo assets, and performance report

**Files:**
- Create: `docs/assets/` generated PNG/GIF artifacts
- Create: `scripts/capture_docs.ts`
- Modify: `scripts/benchmark.py`
- Create: `docs/PERFORMANCE.md`
- Modify: `README.md`
- Modify: `docs/USER_GUIDE.md`

- [ ] Add deterministic Playwright capture states for import, RAW validation, stack setup, progress, compare, histogram, retouch, and DNG export.
- [ ] Generate accessible screenshots at fixed viewport/DPR and verify they contain no local/private paths.
- [ ] Run benchmark fixtures, record hardware/software, and distinguish measurements from targets.
- [ ] Commit with `docs: add verified screenshots and performance results`.

### Task 5: Final release verification

**Files:**
- Modify: `README.md`
- Modify: `docs/RELEASE_CHECKLIST.md`

- [ ] Run the complete CPU, MPS, UI, API, audit, build, Docker, documentation, and DNG validation matrix.
- [ ] Run CUDA release gates when hardware is available; leave milestone status incomplete if not verified.
- [ ] Perform current Capture One Pro manual acceptance on the generated Linear DNG and record version/result.
- [ ] Update the README feature matrix and milestones only from those results.
- [ ] Confirm clean tracked state, review the full branch diff, and commit with `docs: record verified release status`.
