# One-click Launchers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one double-clickable root launcher for macOS and one GPU-first, CPU-fallback root launcher for Windows.

**Architecture:** Thin root wrappers delegate all setup and startup work to the existing scripts under `scripts/`. The wrappers only select the platform/device path, preserve delegated exit codes, and keep failure diagnostics visible; distribution tests pin that contract and documentation makes the two entry points obvious.

**Tech Stack:** Bash, Windows batch, pytest, Markdown, existing uv/npm launch scripts.

## Global Constraints

- Create exactly `Launch PlaneFuse.command` and `Launch PlaneFuse.bat` at the repository root.
- Keep `scripts/start-macos.sh`, `scripts/start-windows-gpu.bat`, and `scripts/start-windows-cpu.bat` as the setup/startup authority.
- Windows must prefer a working NVIDIA `nvidia-smi` path and automatically fall back to CPU.
- Preserve the delegated launcher's exit status and pause only after a non-zero exit.
- Commit the macOS `.command` file with executable permission.
- Do not add installers, signed application bundles, desktop shortcuts, dependency logic, or automatic updates.

---

### Task 1: Root one-click launchers

**Files:**
- Create: `Launch PlaneFuse.command`
- Create: `Launch PlaneFuse.bat`
- Modify: `tests/test_distribution.py`

**Interfaces:**
- Consumes: `scripts/start-macos.sh`, `scripts/start-windows-gpu.bat`, and `scripts/start-windows-cpu.bat` as executable entry points.
- Produces: two root files that can be double-clicked and that return the delegated script's exit status.

- [ ] **Step 1: Write failing launcher contract tests**

Append to `tests/test_distribution.py`:

```python
import os


def test_root_macos_launcher_delegates_and_keeps_failures_visible():
    launcher = ROOT / "Launch PlaneFuse.command"
    text = launcher.read_text()
    assert os.access(launcher, os.X_OK)
    assert 'scripts/start-macos.sh' in text
    assert 'status=$?' in text
    assert 'read -r' in text
    assert 'exit "$status"' in text


def test_root_windows_launcher_prefers_gpu_and_falls_back_to_cpu():
    text = (ROOT / "Launch PlaneFuse.bat").read_text().lower()
    probe = text.index("nvidia-smi")
    gpu = text.index("scripts\\start-windows-gpu.bat")
    cpu = text.index("scripts\\start-windows-cpu.bat")
    assert probe < gpu < cpu
    assert "if errorlevel 1 goto cpu" in text
    assert "pause" in text
    assert "exit /b %planefuse_exit%" in text
```

- [ ] **Step 2: Run the tests and verify the missing-file failure**

Run:

```bash
.venv/bin/pytest -q tests/test_distribution.py -k root_
```

Expected: both tests fail with `FileNotFoundError` for the two root launchers.

- [ ] **Step 3: Implement the macOS wrapper**

Create `Launch PlaneFuse.command`:

```bash
#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

status=0
scripts/start-macos.sh || status=$?
if [ "$status" -ne 0 ]; then
  echo
  echo "PlaneFuse could not start (exit code $status)."
  read -r -p "Press Return to close this window…" _
fi
exit "$status"
```

Then set the executable bit:

```bash
chmod +x "Launch PlaneFuse.command"
```

- [ ] **Step 4: Implement the Windows GPU-first wrapper**

Create `Launch PlaneFuse.bat`:

```bat
@echo off
setlocal
cd /d "%~dp0"

where nvidia-smi >nul 2>nul
if errorlevel 1 goto cpu
nvidia-smi >nul 2>nul
if errorlevel 1 goto cpu

echo Starting PlaneFuse with the NVIDIA GPU...
call "scripts\start-windows-gpu.bat"
goto done

:cpu
echo No working NVIDIA GPU was detected. Starting PlaneFuse on CPU...
call "scripts\start-windows-cpu.bat"

:done
set "planefuse_exit=%errorlevel%"
if not "%planefuse_exit%"=="0" (
  echo.
  echo PlaneFuse could not start ^(exit code %planefuse_exit%^).
  pause
)
exit /b %planefuse_exit%
```

- [ ] **Step 5: Run launcher contract and distribution tests**

Run:

```bash
.venv/bin/pytest -q tests/test_distribution.py
```

Expected: all tests in `tests/test_distribution.py` pass.

- [ ] **Step 6: Review shell syntax and file modes**

Run:

```bash
bash -n "Launch PlaneFuse.command"
test -x "Launch PlaneFuse.command"
git diff --check
```

Expected: all commands exit 0. Review the batch labels, `call` targets, and exit-code capture directly because Windows is not available on this machine.

- [ ] **Step 7: Commit the launchers**

```bash
git add "Launch PlaneFuse.command" "Launch PlaneFuse.bat" tests/test_distribution.py
git commit -m "feat: add one-click platform launchers"
```

---

### Task 2: One-click documentation and final verification

**Files:**
- Modify: `tests/test_distribution.py`
- Modify: `README.md`
- Modify: `docs/INSTALL.md`
- Verify: `scripts/check_docs.py`

**Interfaces:**
- Consumes: the exact root filenames from Task 1.
- Produces: discoverable one-click instructions while retaining explicit device launchers for troubleshooting.

- [ ] **Step 1: Write the failing documentation contract test**

Append to `tests/test_distribution.py`:

```python
def test_docs_advertise_both_root_one_click_launchers():
    for document in (ROOT / "README.md", ROOT / "docs/INSTALL.md"):
        text = document.read_text()
        assert "Launch PlaneFuse.command" in text
        assert "Launch PlaneFuse.bat" in text
    install = (ROOT / "docs/INSTALL.md").read_text()
    assert "NVIDIA" in install
    assert "fall" in install.lower() and "CPU" in install
```

- [ ] **Step 2: Run the documentation test and verify it fails**

Run:

```bash
.venv/bin/pytest -q tests/test_distribution.py -k docs_advertise
```

Expected: fail because the root launcher names are not yet present in both documents.

- [ ] **Step 3: Update the README quick start**

Insert before the command-line setup in `README.md`:

```markdown
### One click

- **macOS:** double-click `Launch PlaneFuse.command`.
- **Windows:** double-click `Launch PlaneFuse.bat`; it prefers NVIDIA GPU and
  automatically falls back to CPU.

The first launch performs the locked setup and builds the interface. Keep the
project folder in place; the launchers delegate to the scripts under `scripts/`.
```

Retain the existing native command-line and explicit device instructions below it.

- [ ] **Step 4: Update the installation guide**

Replace the opening of the launcher section in `docs/INSTALL.md` with:

```markdown
## One-click launchers

- On macOS, double-click `Launch PlaneFuse.command` in Finder.
- On Windows, double-click `Launch PlaneFuse.bat` in Explorer. A working NVIDIA
  driver selects the GPU launcher by default; otherwise it falls back to CPU.

The root launchers delegate to the scripts in [scripts](../scripts), which check
required commands, use `npm ci`, build the UI only when absent, and launch from
the frozen lockfile. Their process exit code is propagated to the terminal.

For explicit device selection or troubleshooting, run
`scripts/start-windows-gpu.bat`, `scripts/start-windows-cpu.bat`, or
`scripts/start-macos.sh` directly.
```

- [ ] **Step 5: Run documentation and distribution checks**

Run:

```bash
.venv/bin/pytest -q tests/test_distribution.py
.venv/bin/python scripts/check_docs.py
git diff --check
```

Expected: distribution tests and the documentation contract pass; diff check exits 0.

- [ ] **Step 6: Run full release verification**

Run:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/mypy engine/src server/src
cd ui
env PATH=/Users/adrien/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:/Users/adrien/.local/bin:/usr/local/bin:/usr/bin:/bin npm run check
env PATH=/Users/adrien/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:/Users/adrien/.local/bin:/usr/local/bin:/usr/bin:/bin npm run build
```

Expected: all Python tests pass with only declared hardware/fixture skips, Ruff and mypy report no issues, Svelte reports zero errors/warnings, and the production build exits 0.

- [ ] **Step 7: Commit documentation and verification**

```bash
git add README.md docs/INSTALL.md tests/test_distribution.py
git commit -m "docs: advertise one-click launchers"
```
