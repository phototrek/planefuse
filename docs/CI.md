# Continuous integration and releases

PlaneFuse uses one production workflow: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
It runs on every branch push, pull request, `v*` tag, manual dispatch, and every
Monday to catch dependency or hosted-runner drift even when the source has not
changed.

## Required gates

| Gate | What it proves |
|---|---|
| Python quality, tests, and audit | Workflow lint and launcher syntax, frozen lock, Ruff, mypy, full CPU/RAW tests, pinned pip-audit, and documentation/API parity |
| UI typecheck, audit, build, and browser tests | Reproducible npm install, npm audit, Svelte diagnostics, production build, and complete Playwright workflows |
| Reproducible ready-folder ZIPs | The tested UI is packaged with checksum-verified official uv binaries; a second build must be byte-identical |
| macOS ready-folder first-launch smoke | The released Apple-silicon ZIP extracts, downloads a uv-managed Python, installs frozen dependencies, boots the real server, and serves the API and UI |
| Windows ready-folder first-launch smoke | The released Windows x64 ZIP passes the same first-launch test on Windows |
| Docker build and health | CPU image build, CUDA Compose validation, localhost boundary, and live `/api/system` health |

The workflow uses least-privilege job permissions, cancels superseded runs,
sets explicit timeouts, pins third-party actions to full commit SHAs, pins
downloaded build tools to repository-reviewed SHA-256 digests, retains test
reports, and generates `SHA256SUMS` beside every pair of ready folders.
Each ZIP contains `RELEASE.json` with its project version, exact commit,
platform, uv version, and source date.

## Release channels

### Latest tested build

After every fully green push to `main`, CI moves the `continuous` tag to that
commit and replaces the assets on the **PlaneFuse — latest tested build**
prerelease. The installation assistant links directly to these assets:

- `PlaneFuse-macOS-Apple-Silicon.zip`
- `PlaneFuse-Windows-x64.zip`
- `SHA256SUMS`

A failed or cancelled run never changes this release, so the download always
means “latest main commit that passed every production gate.”

### Versioned release

Push a tag matching the project version exactly, for example `v0.1.0` when
`pyproject.toml` contains `version = "0.1.0"`. CI reruns every gate before
creating or updating that GitHub Release. When the repository is public, the
workflow also creates GitHub/Sigstore build-provenance attestations for the
ZIPs and checksum manifest.

Do not manually replace release assets. Rerun the workflow from the matching
commit instead.

## Repository settings

The repository is currently private on a plan that does not expose branch
protection or rulesets. GitHub's API therefore cannot enforce required checks
on `main`. Once the repository is public or upgraded, configure:

1. Require a pull request before merging.
2. Require all six gates in the table above.
3. Require branches to be up to date before merging.
4. Block force pushes and deletion of `main`.
5. Require conversation resolution.

The guided setup page deploys only when the repository is public, because the
current private-repository plan does not provide the required Pages surface.
The release downloads are private for the same reason. Before inviting
photographers, either make this repository public or publish the setup page and
release assets through a separate public distribution repository.

## Dependency maintenance

`.github/dependabot.yml` checks GitHub Actions, uv/Python, npm, and Docker
dependencies every Monday. Dependency pull requests must pass the same complete
pipeline; they are never auto-published before merge.

## Local release checks

Run the same core checks before publishing:

```bash
uv lock --check
uv run --frozen --extra cpu --extra raw ruff check .
uv run --frozen --extra cpu --extra raw mypy engine/src server/src scripts/build_ready_bundle.py scripts/smoke_ready_bundle.py
PLANEFUSE_DEVICE=cpu uv run --frozen --extra cpu --extra raw pytest -q
cd ui
npm ci --no-fund
npm audit --audit-level=high
npm run check
npm run build
npm run test:e2e
```

The native first-launch and Docker gates intentionally remain in GitHub Actions
because they validate clean machines and the actual uploaded archives.
