# One-click launchers design

**Date:** 2026-07-15  
**Status:** Approved for implementation

## Goal

Place one obvious, double-clickable launcher for macOS and one for Windows at
the repository root. They must reuse the existing locked launch scripts rather
than duplicate environment setup, dependency installation, UI building, or
server startup.

## User-facing files

- `Launch FocusStack.command` — macOS Finder launcher.
- `Launch FocusStack.bat` — Windows Explorer launcher.

The existing scripts under `scripts/` remain the implementation authority and
continue to support explicit CPU/GPU troubleshooting.

## macOS flow

The `.command` file resolves its own directory, changes to the project root,
and delegates to `scripts/start-macos.sh`. It is committed as executable so it
can be opened directly from Finder.

If startup succeeds, the delegated process owns the Terminal window. If it
fails before handoff, the wrapper prints a short failure message and waits for
Return so the user can read the original error.

## Windows flow

The `.bat` file resolves the project root and checks `nvidia-smi`:

1. When `nvidia-smi` exists and reports a usable NVIDIA driver, delegate to
   `scripts\start-windows-gpu.bat`.
2. Otherwise, explain that CPU fallback is being used and delegate to
   `scripts\start-windows-cpu.bat`.

GPU is therefore the default without making machines lacking NVIDIA hardware
unusable. The wrapper propagates the delegated launcher's exit code. On
failure, it prints a concise message and pauses so a double-click user can read
the diagnostic; successful startup does not add an extra prompt.

## Documentation

The README quick-start and installation guide list the two root launchers first
and describe Windows GPU preference with automatic CPU fallback. The lower-level
scripts remain documented for explicit device selection and troubleshooting.

## Verification

Distribution tests read the wrappers and assert:

- the macOS file delegates to `scripts/start-macos.sh` and preserves failure
  visibility;
- the Windows file probes `nvidia-smi`, prefers the GPU launcher, contains the
  CPU fallback, and propagates failures;
- documentation names both one-click files;
- the macOS launcher has executable permission.

Existing launcher, documentation-contract, static, and full project tests must
remain green. Windows batch behavior is additionally syntax-reviewed because
the current release machine is macOS.

## Non-goals

This change does not create signed `.app`/`.exe` bundles, installers, desktop
shortcuts outside the repository, automatic updates, or new dependency/setup
logic.
