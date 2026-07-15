<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { isEditableTarget, primaryShortcut } from '$lib/shortcuts';
  import {
    api,
    type RetouchMutation,
    type RetouchSession,
    type RetouchStroke
  } from '$lib/api';
  import { appState } from '$lib/stores.svelte';
  import DeepZoom from '$lib/viewer/DeepZoom.svelte';

  interface WorkingImage {
    levels: number;
    width: number;
    height: number;
  }

  interface Source {
    id: string;
    label: string;
    thumb: string;
  }

  let session = $state<RetouchSession | null>(null);
  let working = $state<WorkingImage | null>(null);
  let sources = $state<Source[]>([]);
  let selectedSource = $state('');
  let mode = $state<'normal' | 'erase'>('normal');
  let radius = $state(20);
  let hardness = $state(0.5);
  let opacity = $state(1);
  let flattenName = $state('Retouched');
  let baseRev = $state(0);
  let tileRevs = $state<Record<string, number>>({});
  let busy = $state(false);
  let loading = $state(true);
  let error = $state('');

  onMount(async () => {
    if (!appState.project) {
      loading = false;
      return;
    }
    try {
      const projectId = appState.project.id;
      appState.project = await api.getProject(projectId);
      const sessionId = appState.project.ui_state.retouchSessionId;
      if (typeof sessionId !== 'string') throw new Error('Open Retouch from Viewer first.');
      const listed = await api.listRetouch(projectId);
      session = listed.sessions.find((item) => item.id === sessionId) ?? null;
      if (!session) throw new Error('Retouch session not found. Open it again from Viewer.');
      baseRev = session.rev;

      const info = appState.project.images[session.working_image_id];
      if (!info) throw new Error('Retouch working image is missing.');
      working = {
        levels: Number(info.levels),
        width: Number(info.width),
        height: Number(info.height)
      };

      const loaded: Source[] = [];
      for (const id of session.sources) {
        const sourceInfo = appState.project.images[id];
        if (typeof sourceInfo?.path !== 'string') continue;
        const view = await api.registerView(projectId, sourceInfo.path);
        loaded.push({
          id,
          label: String(sourceInfo.name ?? sourceInfo.method ?? id),
          thumb: api.tileUrl(view.image_id, 0, 0, 0)
        });
      }
      sources = loaded;
      selectedSource = loaded[0]?.id ?? '';
    } catch (e) {
      error = (e as Error).message;
    } finally {
      loading = false;
    }
  });

  let versionForTile = $derived.by(() => {
    const revisions = tileRevs;
    return (z: number, x: number, y: number) => revisions[`${z}/${x}/${y}`] ?? baseRev;
  });

  function applyMutation(result: RetouchMutation) {
    const next = { ...tileRevs };
    for (const tile of result.dirty_tiles) next[`${tile.z}/${tile.x}/${tile.y}`] = result.rev;
    tileRevs = next;
    if (session) session.rev = result.rev;
  }

  async function mutate(operation: () => Promise<RetouchMutation>) {
    if (busy) return;
    busy = true;
    error = '';
    try {
      applyMutation(await operation());
    } catch (e) {
      error = (e as Error).message;
    } finally {
      busy = false;
    }
  }

  function paint(points: [number, number, number][]) {
    if (!session || (mode === 'normal' && !selectedSource)) return;
    const stroke: RetouchStroke = {
      source_id: mode === 'normal' ? selectedSource : session.target_image_id,
      points,
      radius,
      hardness,
      opacity,
      mode
    };
    void mutate(() => api.retouchStroke(session!.id, stroke));
  }

  function undo() {
    if (session) void mutate(() => api.retouchUndo(session!.id));
  }

  function redo() {
    if (session) void mutate(() => api.retouchRedo(session!.id));
  }

  async function flatten() {
    if (!session || !appState.project || busy) return;
    busy = true;
    error = '';
    try {
      const projectId = appState.project.id;
      const { image_id } = await api.flattenRetouch(session.id, flattenName || 'Retouched');
      // Refresh project to pick up the new result image.
      appState.project = await api.getProject(projectId);
      const info = appState.project.images[image_id];
      if (typeof info?.path !== 'string') throw new Error('Flattened result path is missing.');
      // Rebuild results from project.images so the workspace drawer has it.
      appState.results = Object.entries(appState.project.images)
        .filter(([, img]) => img.kind === 'result')
        .map(([id, img]) => ({
          id,
          label: String(img.name ?? img.method ?? id),
          method: String(img.method ?? ''),
          path: String(img.path ?? ''),
          frames: typeof img.frames === 'number' ? img.frames : undefined,
          imageId: undefined as string | undefined,
          thumb: undefined as string | undefined,
          levels: undefined as number | undefined,
          width: undefined as number | undefined,
          height: undefined as number | undefined,
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
      // Auto-select the new flattened result in the viewer.
      appState.viewer = { kind: 'result', id: image_id };
      await goto('/');
    } catch (e) {
      error = (e as Error).message;
      busy = false;
    }
  }

  function shortcuts(e: KeyboardEvent) {
    if (isEditableTarget(e.target)) return;
    if (primaryShortcut(e, 'z')) {
      e.preventDefault();
      e.shiftKey ? redo() : undo();
    } else if (primaryShortcut(e, 'y')) {
      e.preventDefault();
      redo();
    } else if (e.key === '[') {
      radius = Math.max(1, radius - 2);
    } else if (e.key === ']') {
      radius = Math.min(500, radius + 2);
    }
  }
</script>

<svelte:window onkeydown={shortcuts} />

<div class="screen">
  {#if !appState.project}
    <p class="dim">Open a project on the <a href="/">Import</a> screen first.</p>
  {:else if loading}
    <p class="dim">Opening retouch session…</p>
  {:else if !session || !working}
    <p class="error">{error}</p>
    <a class="back" href="/">Back to Workspace</a>
  {:else}
    <header class="toolbar panel">
      <div>
        <p class="eyebrow">Screen 5</p>
        <h1>Retouch</h1>
      </div>
      <div class="modes" aria-label="Brush mode">
        <button class:active={mode === 'normal'} onclick={() => (mode = 'normal')}>Normal</button>
        <button class:active={mode === 'erase'} onclick={() => (mode = 'erase')}>Erase</button>
      </div>
      <button data-testid="retouch-undo" disabled={busy} onclick={undo}>Undo</button>
      <button data-testid="retouch-redo" disabled={busy} onclick={redo}>Redo</button>
      <input
        class="name"
        data-testid="retouch-flatten-name"
        bind:value={flattenName}
        aria-label="Flattened result name"
      />
      <button
        class="primary"
        data-testid="retouch-flatten"
        disabled={busy}
        onclick={flatten}
      >Flatten</button>
    </header>

    {#if error}<p class="error mono">{error}</p>{/if}

    <div class="workspace">
      <aside class="sources panel">
        <p class="eyebrow">Paint source</p>
        {#if sources.length}
          {#each sources as source (source.id)}
            <button
              class="source"
              class:selected={selectedSource === source.id}
              data-testid="retouch-source"
              onclick={() => {
                selectedSource = source.id;
                mode = 'normal';
              }}
            >
              <img src={source.thumb} alt="" />
              <span>{source.label}</span>
            </button>
          {/each}
        {:else}
          <p class="dim">Create another stack result to use as a source.</p>
        {/if}

        <div class="controls">
          <label>
            <span>Radius <b class="mono">{radius}px</b></span>
            <input type="range" min="1" max="500" bind:value={radius} />
          </label>
          <label>
            <span>Hardness <b class="mono">{Math.round(hardness * 100)}%</b></span>
            <input type="range" min="0" max="1" step="0.05" bind:value={hardness} />
          </label>
          <label>
            <span>Opacity <b class="mono">{Math.round(opacity * 100)}%</b></span>
            <input type="range" min="0" max="1" step="0.05" bind:value={opacity} />
          </label>
        </div>
        <p class="hint"><span class="kbd">Space</span> pan · <span class="kbd">[</span><span class="kbd">]</span> brush</p>
      </aside>

      <main class="canvas panel" data-testid="retouch-canvas">
        <DeepZoom
          imageId={session.working_image_id}
          levels={working.levels}
          width={working.width}
          height={working.height}
          tileVersion={versionForTile}
          brushRadius={radius}
          onstroke={paint}
        />
      </main>
    </div>
  {/if}
</div>

<style>
  .screen { padding: 18px 22px; height: 100%; display: flex; flex-direction: column; gap: 12px; }
  .toolbar { min-height: 64px; padding: 10px 14px; display: flex; align-items: center; gap: 8px; }
  .toolbar > div:first-child { margin-right: auto; }
  .toolbar h1 { font-size: 20px; margin-top: 2px; }
  .modes { display: flex; }
  .modes button { border-radius: 0; }
  .modes button:first-child { border-radius: var(--radius) 0 0 var(--radius); }
  .modes button:last-child { border-radius: 0 var(--radius) var(--radius) 0; margin-left: -1px; }
  button.active { color: #1a0d04; background: var(--accent); border-color: var(--accent); }
  .name { width: 150px; }
  .workspace { flex: 1; min-height: 0; display: grid; grid-template-columns: 220px 1fr; gap: 12px; }
  .sources { padding: 14px; display: flex; flex-direction: column; gap: 10px; overflow: auto; }
  .source { padding: 0; overflow: hidden; text-align: left; }
  .source img { display: block; width: 100%; height: 92px; object-fit: cover; background: var(--bg); }
  .source span { display: block; padding: 7px 9px; text-transform: uppercase; font-family: var(--font-mono); font-size: 11px; }
  .source.selected { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
  .controls { display: flex; flex-direction: column; gap: 12px; margin-top: 6px; padding-top: 12px; border-top: 1px solid var(--line); }
  .controls label { display: flex; flex-direction: column; gap: 5px; color: var(--text-dim); font-size: 12px; }
  .controls label span { display: flex; justify-content: space-between; }
  .controls b { color: var(--text); font-weight: 400; }
  .hint { margin-top: auto; color: var(--text-faint); font-size: 11px; }
  .canvas { min-width: 0; min-height: 0; overflow: hidden; padding: 0; }
  .error { color: var(--bad); margin: 0; }
  .back { color: var(--accent-bright); }
</style>
