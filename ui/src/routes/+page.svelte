<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/api';
  import { appState, type WorkspaceResult } from '$lib/stores.svelte';
  import Toolbar from '$lib/workspace/Toolbar.svelte';
  import InputList from '$lib/workspace/InputList.svelte';
  import ViewerPane from '$lib/workspace/ViewerPane.svelte';
  import RenderDrawer from '$lib/workspace/RenderDrawer.svelte';
  import { isEditableTarget, plainShortcut } from '$lib/shortcuts';

  let initError = $state('');
  let shortcutHelp = $state(false);

  function stepFrame(direction: -1 | 1) {
    if (appState.inputs.length === 0) return;
    const activePath = appState.viewer?.kind === 'input' ? appState.viewer.path : '';
    const current = Math.max(0, appState.inputs.findIndex((frame) => frame.path === activePath));
    const next = (current + direction + appState.inputs.length) % appState.inputs.length;
    appState.viewer = { kind: 'input', path: appState.inputs[next].path };
  }

  function shortcuts(event: KeyboardEvent) {
    if (isEditableTarget(event.target)) return;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      stepFrame(event.key === 'ArrowLeft' ? -1 : 1);
    } else if (event.key === 'Escape') {
      appState.compareMode = 'single';
      shortcutHelp = false;
    } else if (plainShortcut(event, '?')) {
      shortcutHelp = !shortcutHelp;
    }
  }

  function buildResults(images: Record<string, Record<string, unknown>>): WorkspaceResult[] {
    return Object.entries(images)
      .filter(([, img]) => img.kind === 'result')
      .map(([id, img]) => ({
        id,
        label: String(img.name ?? img.method ?? id),
        method: String(img.method ?? ''),
        path: String(img.path ?? ''),
        frames: typeof img.frames === 'number' ? img.frames : undefined,
        imageId: undefined,
        thumb: undefined,
        levels: undefined,
        width: undefined,
        height: undefined,
        domain: String(img.domain ?? 'rendered_rgb'),
        storage: String(img.storage ?? 'rendered_16bit'),
        metadata: typeof img.metadata === 'object' && img.metadata
          ? img.metadata as Record<string, unknown>
          : undefined,
        decoder: typeof img.decoder === 'object' && img.decoder
          ? img.decoder as Record<string, unknown>
          : {},
        provenance: typeof img.provenance === 'object' && img.provenance
          ? img.provenance as Record<string, unknown>
          : {}
      }));
  }

  onMount(async () => {
    try {
      if (!appState.project) {
        appState.project = await api.createScratchProject();
      }
      if (appState.project.frames.length > 0 && appState.inputs.length === 0) {
        const report = await api.addFrames(appState.project.id, []);
        appState.inputs = report.files;
        appState.scanReport = report;
      }
      if (appState.results.length === 0) {
        appState.results = buildResults(appState.project.images);
      }
    } catch (e) {
      initError = (e as Error).message;
    }
  });

  // Result discovery: when a tracked stack job finishes, re-fetch the project
  // and rebuild results. This MUST be async (API call) so $derived cannot be used.
  $effect(() => {
    const trackedIds = appState.stackJobIds;
    if (!appState.project || trackedIds.length === 0) return;

    for (const id of trackedIds) {
      const job = appState.jobs[id];
      if (job?.status !== 'done') continue;
      const projectId = appState.project.id;
      // Remove from tracked set immediately to prevent re-triggering.
      appState.stackJobIds = appState.stackJobIds.filter((jid) => jid !== id);
      // Re-fetch project to pick up the new result image.
      void api.getProject(projectId).then((project) => {
        appState.project = project;
        const newResults = buildResults(project.images);
        appState.results = newResults;
        if (newResults.length > 0) {
          appState.viewer = { kind: 'result', id: newResults[newResults.length - 1].id };
        }
      });
      break; // Handle one at a time; effect re-runs if more remain.
    }
  });
</script>

<svelte:window onkeydown={shortcuts} />

<div class="workspace">
  <div class="ws-toolbar">
    <Toolbar />
  </div>
  <div class="ws-inputs panel">
    <InputList />
  </div>
  <div class="ws-viewer">
    <ViewerPane />
  </div>
  <div class="ws-drawer">
    <RenderDrawer />
  </div>
</div>

{#if initError}
  <div class="init-error mono">Init error: {initError}</div>
{/if}

{#if shortcutHelp}
  <aside class="shortcut-help panel" aria-label="Keyboard shortcuts">
    <strong>Keyboard</strong>
    <span><kbd>F</kbd> Fit</span><span><kbd>Z</kbd> 100%</span>
    <span><kbd>←</kbd><kbd>→</kbd> Step frames</span><span><kbd>\</kbd> Before / after</span>
    <span><kbd>Esc</kbd> Exit compare</span><span><kbd>?</kbd> Close help</span>
  </aside>
{/if}

<style>
  .workspace {
    display: grid;
    grid-template-columns: 240px 1fr;
    grid-template-rows: auto 1fr auto;
    grid-template-areas:
      "toolbar toolbar"
      "inputs viewer"
      "drawer drawer";
    height: 100%;
    overflow: hidden;
  }
  .ws-toolbar {
    grid-area: toolbar;
    display: flex;
    flex-direction: column;
    border-bottom: 1px solid var(--line);
    background: linear-gradient(180deg, var(--panel-2), var(--panel));
    min-height: 48px;
  }
  .ws-inputs {
    grid-area: inputs;
    border-right: 1px solid var(--line);
    border-radius: 0;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .ws-viewer { grid-area: viewer; overflow: hidden; min-width: 0; min-height: 0; }
  .ws-drawer { grid-area: drawer; min-height: 0; }
  .init-error {
    position: fixed;
    bottom: 16px;
    left: 50%;
    transform: translateX(-50%);
    background: var(--panel);
    border: 1px solid var(--bad);
    color: var(--bad);
    padding: 8px 16px;
    border-radius: var(--radius);
    font-size: 12px;
    z-index: 9999;
  }
  .shortcut-help { position: fixed; right: 16px; top: 70px; z-index: 100; display: grid; grid-template-columns: auto auto; gap: 8px 14px; padding: 12px; font-size: 11px; box-shadow: var(--shadow); }
  .shortcut-help strong { grid-column: 1 / -1; }
</style>
