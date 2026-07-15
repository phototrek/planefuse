<script lang="ts">
  import { api, type ImageAnalysis } from '$lib/api';
  import { appState } from '$lib/stores.svelte';
  import DeepZoom from '$lib/viewer/DeepZoom.svelte';
  import CompareViewer from '$lib/viewer/CompareViewer.svelte';
  import Histogram from '$lib/viewer/Histogram.svelte';
  import ViewerControls from '$lib/viewer/ViewerControls.svelte';
  import ProgressBar from '$lib/components/ProgressBar.svelte';

  interface RegisteredView {
    imageId: string;
    levels: number;
    width: number;
    height: number;
  }

  const viewCache: Record<string, RegisteredView> = {};

  let loadedView = $state<RegisteredView | null>(null);
  let secondaryView = $state<RegisteredView | null>(null);
  let analysis = $state<ImageAnalysis | null>(null);
  let analysisError = $state('');
  let loading = $state(false);
  let viewError = $state('');
  // Plain (non-reactive) variable so it doesn't cause re-entrancy in $effect.
  let lastTargetKey = '';

  let saveTimer: ReturnType<typeof setTimeout> | null = null;

  function targetKey(): string {
    const t = appState.viewer;
    if (!t) return '';
    if (t.kind === 'result') return `result:${t.id}`;
    if (t.kind === 'input') return `input:${t.path}`;
    return `job:${t.id}`;
  }

  async function loadView(key: string) {
    const target = appState.viewer;
    if (!target || !appState.project || target.kind === 'job') {
      loadedView = null;
      loading = false;
      viewError = '';
      return;
    }

    let path: string | undefined;
    if (target.kind === 'result') {
      const result = appState.results.find((r) => r.id === target.id);
      if (!result) { loadedView = null; return; }
      if (result.imageId) {
        loadedView = {
          imageId: result.imageId,
          levels: result.levels ?? 1,
          width: result.width ?? 100,
          height: result.height ?? 100
        };
        return;
      }
      path = result.path;
    } else {
      path = target.path;
    }

    if (!path) { loadedView = null; return; }

    if (viewCache[path]) {
      const cached = viewCache[path];
      if (target.kind === 'result') {
        const result = appState.results.find((r) => r.id === target.id);
        if (result) {
          result.imageId = cached.imageId;
          result.levels = cached.levels;
          result.width = cached.width;
          result.height = cached.height;
        }
      }
      loadedView = cached;
      return;
    }

    loading = true;
    viewError = '';
    try {
      const reg = await api.registerView(appState.project.id, path);
      const view: RegisteredView = {
        imageId: reg.image_id,
        levels: reg.levels,
        width: reg.width,
        height: reg.height
      };
      viewCache[path] = view;
      if (targetKey() === key) {
        if (target.kind === 'result') {
          const result = appState.results.find((r) => r.id === target.id);
          if (result) {
            result.imageId = view.imageId;
            result.levels = view.levels;
            result.width = view.width;
            result.height = view.height;
          }
        }
        loadedView = view;
      }
    } catch (e) {
      if (targetKey() === key) {
        viewError = (e as Error).message;
      }
    } finally {
      if (targetKey() === key) loading = false;
    }
  }

  async function ensureView(path: string): Promise<RegisteredView> {
    if (viewCache[path]) return viewCache[path];
    if (!appState.project) throw new Error('No open project');
    const reg = await api.registerView(appState.project.id, path);
    const view = {
      imageId: reg.image_id,
      levels: reg.levels,
      width: reg.width,
      height: reg.height
    };
    viewCache[path] = view;
    return view;
  }

  function comparisonPath(): string {
    if (appState.compareMode === 'single') return '';
    const target = appState.viewer;
    if (appState.compareMode === 'side-by-side') {
      const requested = appState.results.find((result) => result.id === appState.compareResultId);
      const currentId = target?.kind === 'result' ? target.id : '';
      return requested?.path
        || appState.results.find((result) => result.id !== currentId)?.path
        || '';
    }
    if (target?.kind === 'result') return appState.inputs[0]?.path ?? '';
    return appState.results[appState.results.length - 1]?.path ?? '';
  }

  // Trigger async image registration when viewer target changes.
  // This is intentionally async-in-effect because registration is async
  // and cannot be expressed as $derived.
  $effect(() => {
    // Subscribe to viewer and results to detect changes.
    const viewer = appState.viewer;
    const resultsLen = appState.results.length;
    void resultsLen; // consumed for reactivity
    const key = viewer
      ? (viewer.kind === 'result' ? `result:${viewer.id}` : viewer.kind === 'input' ? `input:${viewer.path}` : `job:${viewer.id}`)
      : '';
    if (key === lastTargetKey) return;
    lastTargetKey = key;
    void loadView(key);
  });

  let lastComparisonPath = '';
  $effect(() => {
    const mode = appState.compareMode;
    const resultChoice = appState.compareResultId;
    const inputCount = appState.inputs.length;
    const resultCount = appState.results.length;
    void mode; void resultChoice; void inputCount; void resultCount;
    const path = comparisonPath();
    if (!path) {
      secondaryView = null;
      lastComparisonPath = '';
      return;
    }
    if (path === lastComparisonPath && secondaryView) return;
    lastComparisonPath = path;
    void ensureView(path).then((view) => {
      if (comparisonPath() === path) secondaryView = view;
    }).catch((error) => {
      if (comparisonPath() === path) viewError = (error as Error).message;
    });
  });

  let lastAnalysisId = '';
  $effect(() => {
    const imageId = loadedView?.imageId ?? '';
    if (!imageId || imageId === lastAnalysisId) return;
    lastAnalysisId = imageId;
    analysis = null;
    analysisError = '';
    void api.imageAnalysis(imageId).then((value) => {
      if (loadedView?.imageId === imageId) analysis = value;
    }).catch((error) => {
      if (loadedView?.imageId === imageId) {
        analysis = null;
        analysisError = (error as Error).message;
      }
    });
  });

  function onmove(v: { scale: number; tx: number; ty: number }) {
    appState.viewerTransform = v;
    if (!appState.project) return;
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      if (appState.project) {
        api.patchUiState(appState.project.id, { viewerPos: v }).catch(() => {});
      }
    }, 500);
  }

  let currentJob = $derived.by(() => {
    const target = appState.viewer;
    if (target?.kind === 'job') return appState.jobs[target.id] ?? null;
    return null;
  });

  let showViewer = $derived(!!loadedView && !currentJob);
</script>

<div class="viewer-pane">
  {#if !appState.project || !appState.viewer}
    <div class="empty">
      <span class="faint">Add frames and Run to see a result</span>
    </div>
  {:else if currentJob}
    <!-- Job progress view -->
    <div class="job-view">
      <div class="job-card panel">
        <div class="job-top">
          <span class="jtype mono">{currentJob.type}</span>
          <span class="jid mono faint">{currentJob.id}</span>
          <span
            class="jstatus"
            class:done={currentJob.status === 'done'}
            class:error={currentJob.status === 'error'}
          >{currentJob.status}</span>
        </div>
        <ProgressBar percent={currentJob.percent} status={currentJob.status} />
        <div class="jmsg mono faint">
          {currentJob.message || (currentJob.error ? `error: ${currentJob.error}` : '')}
        </div>
        {#if currentJob.params}
          <div class="jparams mono faint">
            method: {String(currentJob.params.method ?? '—')}
          </div>
        {/if}
      </div>
    </div>
  {:else if viewError && !loadedView}
    <div class="empty"><span class="err mono">{viewError}</span></div>
  {:else if loading && !loadedView}
    <div class="empty"><span class="faint">Loading…</span></div>
  {:else if showViewer && loadedView}
    <ViewerControls />
    {#if appState.compareMode !== 'single' && secondaryView}
      <CompareViewer
        primary={loadedView}
        secondary={secondaryView}
        mode={appState.compareMode}
        transform={appState.viewerTransform}
        {onmove}
      />
    {:else}
      <DeepZoom
        imageId={loadedView.imageId}
        levels={loadedView.levels}
        width={loadedView.width}
        height={loadedView.height}
        view={appState.viewerTransform}
        {onmove}
      />
    {/if}
    {#if appState.showHistogram && analysis}
      <Histogram {analysis} />
    {:else if appState.showHistogram && analysisError}
      <span class="analysis-error mono">Histogram unavailable: {analysisError}</span>
    {/if}
  {:else}
    <div class="empty"><span class="faint">Select a frame or result to view</span></div>
  {/if}
</div>

<style>
  .viewer-pane { width: 100%; height: 100%; overflow: hidden; position: relative; }
  .empty { display: flex; align-items: center; justify-content: center; height: 100%; }
  .job-view { display: flex; align-items: center; justify-content: center; height: 100%; padding: 32px; }
  .job-card { width: 100%; max-width: 560px; padding: 18px 20px; display: flex; flex-direction: column; gap: 12px; }
  .job-top { display: flex; align-items: center; gap: 12px; }
  .jtype { text-transform: uppercase; font-size: 12px; letter-spacing: 0.06em; color: var(--accent-bright); }
  .jid { font-size: 11px; }
  .jstatus { margin-left: auto; font-size: 12px; font-weight: 600; color: var(--text-dim); text-transform: uppercase; }
  .jstatus.done { color: var(--good); }
  .jstatus.error { color: var(--bad); }
  .jmsg { font-size: 11px; min-height: 14px; }
  .jparams { font-size: 11px; color: var(--text-faint); }
  .err { color: var(--bad); }
  .analysis-error { position: absolute; z-index: 8; top: 50px; right: 12px; padding: 8px; background: var(--panel); color: var(--bad); font-size: 10px; }
</style>
