<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/api';
  import { appState, type WorkspaceResult } from '$lib/stores.svelte';
  import Toolbar from '$lib/workspace/Toolbar.svelte';
  import InputList from '$lib/workspace/InputList.svelte';
  import ViewerPane from '$lib/workspace/ViewerPane.svelte';
  import RenderDrawer from '$lib/workspace/RenderDrawer.svelte';

  let initError = $state('');

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
        height: undefined
      }));
  }

  const PROJECT_KEY = 'focusstack:projectId';

  onMount(async () => {
    try {
      if (!appState.project) {
        // Reload the last project across a page refresh; only make a new scratch
        // project if there's no saved id or it no longer exists server-side.
        const savedId = localStorage.getItem(PROJECT_KEY);
        if (savedId) {
          try { appState.project = await api.getProject(savedId); } catch { /* stale id */ }
        }
        if (!appState.project) appState.project = await api.createScratchProject();
      }
      if (appState.project.frames.length > 0 && appState.inputs.length === 0) {
        const report = await api.addFrames(appState.project.id, []);
        appState.inputs = report.files;
      }
      if (appState.results.length === 0) {
        appState.results = buildResults(appState.project.images);
      }
      // Re-attach to jobs from before the refresh (running stack, exports) so the
      // drawer shows them, and re-arm result discovery for any still-running stack.
      if (Object.keys(appState.jobs).length === 0) {
        const jobs = await api.listJobs();
        for (const j of jobs) appState.jobs[j.id] = j;
        appState.stackJobIds = jobs
          .filter((j) => j.type === 'stack' && (j.status === 'running' || j.status === 'pending'))
          .map((j) => j.id);
      }
    } catch (e) {
      initError = (e as Error).message;
    }
  });

  // Remember the active project so a refresh restores it instead of starting blank.
  $effect(() => {
    if (appState.project?.id) localStorage.setItem(PROJECT_KEY, appState.project.id);
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
</style>
