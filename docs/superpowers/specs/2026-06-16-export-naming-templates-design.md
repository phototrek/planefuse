# Export Naming Templates — Design

> M6 polish slice. Status: approved 2026-06-16.

## Goal

Let the export form build the output filename from a template of named tokens
with a live preview, per SPEC §130 and §256. Replaces the current plain stem
field (`stacked` + extension).

## Tokens

The vocabulary is fixed by SPEC §130 — five tokens, no more:

| token | resolves to |
|---|---|
| `{stack_name}` | `appState.project.name` |
| `{method}` | the result's stacking method (`pmax`, `dmap`, …) |
| `{frames}` | number of frames that went into the stack |
| `{date}` | local date, `YYYY-MM-DD` |
| `{seq}` | 1-based index of the result among `appState.results`, zero-padded to 3 (`001`) |

Unknown `{tokens}` are left literal (a typo stays visible rather than vanishing).

## Architecture

Substitution is **client-side**. The UI already assembles the full `dest` path
it POSTs to `/projects/{id}/export`; the server is unchanged on the export path.
No template-engine dependency — a pure helper does string replacement of the
five known tokens.

### Server (one-line addition)

The stack runner records `proj.images[image_id] = {kind, path, method}`. Add the
frame count so `{frames}` is reachable:

```python
proj.images[image_id] = {"kind": "result", "path": str(out_path),
                         "method": method, "frames": len(paths)}
```

Result discovery in the UI surfaces it as `WorkspaceResult.frames` (optional —
results created before this change have no `frames`, and `{frames}` then resolves
to an empty string).

### Client

`applyTemplate(template: string, ctx: TemplateCtx): string` — pure function:

1. Replace each known `{token}` with its `ctx` value (stringified).
2. Sanitize the *whole result*: filesystem-illegal chars `\ / : * ? " < > |`
   become `_`, so a token value containing a space or colon can never produce a
   broken path. (`{date}` uses `-`, already safe.)

`TemplateCtx` is built at preview/export time from `appState`:
`{ stack_name, method, frames, date, seq }`.

The resolved string is the filename **stem**; the chosen format's extension is
appended (`.tif` / `.png` / `.jpg`) exactly as today.

## UI changes (Toolbar export popover)

- Replace the "Filename" stem `<input>` with a **"Name template"** text input.
  Default value `{stack_name}_{method}`. Bound to a `exportTemplate` state var.
- Add a **live preview** line below it showing the resolved filename
  (e.g. `Zion_pmax.tif`), recomputed reactively as the template or selected
  result changes.
- The existing **"Full path"** override field stays and still wins when edited
  (`exportDestOverride`), so the e2e's direct `export-dest` fill keeps working.

## Persistence

- On project open, load `project.ui_state.exportTemplate` into `exportTemplate`
  (fall back to the default if absent).
- PATCH `ui_state.exportTemplate` on export (when we already hit the server),
  not on every keystroke.

## Testing

- One assert-based unit check for `applyTemplate`: token substitution, a missing
  token left literal, illegal-char sanitization, and `{seq}` zero-padding. This
  is the only non-trivial logic.
- The existing Playwright export smoke fills `export-dest` directly (override
  path) and stays green unchanged.

## Out of scope (YAGNI)

- Per-token insert buttons / dropdown — users type `{...}` directly.
- PATCH-on-keystroke — PATCH-on-export is enough.
- Server-side templating or a new endpoint — the client owns the path.
- Depth-map "also export" naming — that path already exists; it reuses the same
  resolved stem with a suffix if/when wired, not part of this slice.
