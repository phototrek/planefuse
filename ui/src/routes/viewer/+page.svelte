<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { api } from '$lib/api';
  import { appState } from '$lib/stores.svelte';
  import DeepZoom from '$lib/viewer/DeepZoom.svelte';

  interface View {
    srcId: string;
    imageId: string;
    levels: number;
    width: number;
    height: number;
  }

  let view = $state<View | null>(null);
  let error = $state('');
  let loading = $state(false);
  let retouching = $state(false);
  let pos: { scale: number; tx: number; ty: number } | null = null;
  let saveTimer: ReturnType<typeof setTimeout> | null = null;

  function latestResult(images: Record<string, Record<string, unknown>>): [string, string] | null {
    let found: [string, string] | null = null;
    for (const [id, info] of Object.entries(images)) {
      if (info.kind === 'result' && typeof info.path === 'string') found = [id, info.path];
    }
    return found;
  }

  onMount(async () => {
    if (!appState.project) return;
    loading = true;
    error = '';
    try {
      // Refresh to pick up results produced after the project was opened.
      appState.project = await api.getProject(appState.project.id);
      const cached = appState.project.ui_state.view as View | undefined;
      const latest = latestResult(appState.project.images);
      if (!latest) {
        error = 'No stack result yet — run a stack first.';
        return;
      }
      const [srcId, path] = latest;
      if (cached && cached.srcId === srcId) {
        view = cached;
      } else {
        const reg = await api.registerView(appState.project.id, path);
        view = { srcId, imageId: reg.image_id, levels: reg.levels, width: reg.width, height: reg.height };
        appState.project.ui_state.view = view;
        await api.patchUiState(appState.project.id, { view });
      }
    } catch (e) {
      error = (e as Error).message;
    } finally {
      loading = false;
    }
  });

  function onmove(v: { scale: number; tx: number; ty: number }) {
    pos = v;
    if (!appState.project) return;
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      if (appState.project) api.patchUiState(appState.project.id, { viewerPos: pos }).catch(() => {});
    }, 500);
  }

  async function retouch() {
    if (!appState.project || !view) return;
    retouching = true;
    error = '';
    try {
      const { sessions } = await api.listRetouch(appState.project.id);
      const existing = [...sessions].reverse().find((s) => s.target_image_id === view?.srcId);
      const sessionId = existing?.id ??
        (await api.createRetouch(appState.project.id, view.srcId)).session_id;
      appState.project.ui_state.retouchSessionId = sessionId;
      await api.patchUiState(appState.project.id, { retouchSessionId: sessionId });
      await goto('/retouch');
    } catch (e) {
      error = (e as Error).message;
    } finally {
      retouching = false;
    }
  }
</script>

<div class="screen">
  <header class="head">
    <div>
      <p class="eyebrow">Screen 4</p>
      <h1>Viewer</h1>
    </div>
    {#if view}
      <button
        data-testid="retouch-this-result"
        disabled={retouching}
        onclick={retouch}
      >{retouching ? 'Opening…' : 'Retouch this result'}</button>
    {/if}
  </header>

  {#if !appState.project}
    <p class="dim">Open a project on the <a href="/">Import</a> screen first.</p>
  {:else if loading}
    <p class="dim">Building tile pyramid…</p>
  {:else}
    {#if error}<p class="err mono">{error}</p>{/if}
    {#if view}
      <div class="canvas panel">
        <DeepZoom imageId={view.imageId} levels={view.levels} width={view.width} height={view.height} {onmove} />
      </div>
    {/if}
  {/if}
</div>

<style>
  .screen { padding: 28px 32px; height: 100%; display: flex; flex-direction: column; gap: 16px; }
  .head { display: flex; align-items: end; justify-content: space-between; gap: 16px; }
  .head h1 { font-size: 26px; margin-top: 4px; }
  .canvas { flex: 1; min-height: 0; overflow: hidden; padding: 0; }
  .err { color: var(--bad); margin: 0; }
</style>
