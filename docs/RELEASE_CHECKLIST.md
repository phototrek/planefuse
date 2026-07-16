# Release checklist

Status recorded on 2026-07-16. “Pending” means the implementation is present but the
named external hardware/application gate was not available in the verified environment.

| Gate | Result | Evidence/command |
|---|---|---|
| Python engine/server suite | Pass: 312 passed | `.venv/bin/pytest -q` |
| Ruff | Pass | `.venv/bin/ruff check .` |
| Mypy | Pass: 66 files | `.venv/bin/mypy engine/src server/src` |
| UI typecheck/build | Pass | `npm run check`, `npm run build` |
| CPU browser workflows | Pass: 12 passed, 2 intentional skips (docs capture, proprietary fixture) | `npm run test:e2e` |
| MPS RAW→DNG browser | Pass | `npm run test:e2e:mps -- raw.spec.ts -g "stacks without"` |
| DNG tifffile + LibRaw pixel conformance | Pass | `tests/engine/test_dng_writer.py` |
| Python dependency audit | Pass at recorded lock | `uvx pip-audit --path .venv/lib/python3.12/site-packages` |
| npm audit | Pass at recorded lock | `npm audit --audit-level=high` |
| CPU Docker build/health | Pass: pinned image built; localhost `/api/system` healthy | `docker build --target cpu -t planefuse:cpu .` |
| Guided setup page | Pass locally: platform selection, progress, reset, and download paths verified | `node --check installer/setup.js`, browser check |
| Ready-folder assembly | Pass locally: deterministic ZIP, release identity, compiled UI, bundled uv, launch modes, and archive layout verified | `pytest tests/test_distribution.py` |
| Native ready-folder smoke | Pending: macOS and Windows jobs are implemented but unpushed | Run `CI` workflow |
| Continuous release refresh | Pending: gated rolling-release job is implemented but unpushed | Green push to `main` |
| CI enforcement on `main` | Pending: private repository plan does not expose branch protection/rulesets | Make public or upgrade plan, then require six CI gates |
| CUDA engine/browser on NVIDIA hardware | Pending: no NVIDIA hardware in this run | Release-machine gate below |
| Capture One Pro 16.7.5 manual import/edit/export | Pending: installed local app is 16.2.3.32 | Follow RAW guide checklist |

## Production release automation

The single [`CI` workflow](../.github/workflows/ci.yml) builds both folders on
every push and pull request:

- `PlaneFuse-macOS-Apple-Silicon.zip`
- `PlaneFuse-Windows-x64.zip`

It verifies official uv checksums, assembles each ZIP twice to require
byte-for-byte reproducibility, publishes `SHA256SUMS`, and boots the actual
archives on clean Apple-silicon macOS and Windows x64 runners. Each archive has
a `RELEASE.json` source identity.

A fully green push to `main` refreshes the rolling `continuous` prerelease used
by the setup page. A `v*` tag publishes only after the same gates pass and must
match the version in `pyproject.toml`. Public versioned releases also receive a
GitHub/Sigstore provenance attestation. See [CI and releases](CI.md).

GitHub Pages deployment is gated to public repositories. Enable Pages with
**GitHub Actions** as its source when the repository becomes public.

## NVIDIA release-machine gate

```bash
uv sync --frozen --extra cu12x --extra raw
PLANEFUSE_DEVICE=cuda uv run pytest -q -m cuda
cd ui && npm run test:e2e && cd ..
docker compose config --quiet
docker compose up --build
```

Record GPU model/driver, CUDA runtime, Torch version, parity results, benchmark output,
and `/api/system` response. Do not convert the pending row to pass from emulation.

## Capture One manual gate

Use a DNG generated from a real supported same-camera stack, not only the synthetic
acceptance fixture. Record Capture One exact version/edition, OS, import profile,
dimensions/orientation, editability, highlight behavior, export success, and a screenshot.
Regenerate the redistributable format fixture with `scripts/make-acceptance-dng.sh`.

## Release hygiene

- [ ] All rows required for the intended release platform are Pass.
- [x] `python scripts/check_docs.py` passes.
- [x] `git diff --check` is clean and the branch diff has been reviewed.
- [x] No proprietary photos, local paths, tokens, Catalogs, or Sessions are tracked.
- [x] Compose publishes localhost only and photo mounts stay read-only.
