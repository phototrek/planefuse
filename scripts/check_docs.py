#!/usr/bin/env python3
"""Check local Markdown links and keep the documented API synchronized."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from planefuse_server.main import create_app

ROOT = Path(__file__).parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def local_links() -> list[str]:
    errors: list[str] = []
    pattern = re.compile(r"!?\[[^]]*]\(([^)]+)\)")
    for document in DOCS:
        text = document.read_text(encoding="utf-8")
        for raw in pattern.findall(text):
            target = raw.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"{document.relative_to(ROOT)}: missing link target {raw}")
    return errors


def api_contract() -> list[str]:
    api_doc = (ROOT / "docs/API.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"`(GET|POST|PATCH|DELETE) (/api/[^`]+)`", api_doc))
    app = create_app(Path(tempfile.mkdtemp(prefix="planefuse-docs-")))
    openapi = app.openapi()
    actual = {
        (method.upper(), path)
        for path, operations in openapi["paths"].items()
        if path.startswith("/api/")
        for method in operations
        if method in {"get", "post", "patch", "delete"}
    }
    errors = [f"docs/API.md missing {method} {path}" for method, path in sorted(actual - documented)]
    errors += [f"docs/API.md documents absent {method} {path}" for method, path in sorted(documented - actual)]
    return errors


def stale_claims() -> list[str]:
    checks = {
        ROOT / "README.md": ("M6 —", "UI polish pending"),
        ROOT / "engine/src/planefuse/cli.py": ("coming in milestone M4",),
    }
    errors = []
    for path, phrases in checks.items():
        text = path.read_text(encoding="utf-8")
        errors.extend(
            f"{path.relative_to(ROOT)} contains stale phrase: {phrase}"
            for phrase in phrases
            if phrase in text
        )
    return errors


def main() -> None:
    errors = [*local_links(), *api_contract(), *stale_claims()]
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"documentation contract passed ({len(DOCS)} Markdown files)")


if __name__ == "__main__":
    main()
