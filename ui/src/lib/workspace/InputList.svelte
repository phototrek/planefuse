<script lang="ts">
  import { api, type ScanReport } from '$lib/api';
  import { appState } from '$lib/stores.svelte';
  import ValidationPanel from './ValidationPanel.svelte';

  let showAdd = $state(false);
  let folder = $state('');        // paste-a-path fallback (no copy, also the e2e seam)
  let busy = $state(false);
  let error = $state('');
  let lastReport = $state<ScanReport | null>(null);

  // Ingest absolute paths in place — the server reads them from disk, no copy.
  async function addPaths(paths: string[]) {
    if (!appState.project || paths.length === 0) return;
    busy = true;
    error = '';
    try {
      const report = await api.addFrames(appState.project.id, paths);
      appState.inputs = report.files;
      appState.scanReport = report;
      lastReport = report;
      appState.project = await api.getProject(appState.project.id);
      // Auto-select the first input into the viewer on first add.
      if (report.files.length > 0 && appState.viewer === null) {
        appState.viewer = { kind: 'input', path: report.files[0].path };
      }
      showAdd = false;
      folder = '';
    } catch (e) {
      error = (e as Error).message;
    } finally {
      busy = false;
    }
  }

  // Native OS dialog (server-side). Returns real absolute paths — no upload.
  async function pick(mode: 'directory' | 'files') {
    if (!appState.project) return;
    error = '';
    try {
      const { paths } = await api.pick(mode);
      if (paths.length > 0) await addPaths(paths);
    } catch (e) {
      error = (e as Error).message;
    }
  }

  function addPasted() {
    if (folder) addPaths([folder]);
  }

  async function removeFrames(paths: string[]) {
    if (!appState.project || paths.length === 0) return;
    error = '';
    busy = true;
    try {
      const report = await api.removeFrames(appState.project.id, paths);
      appState.inputs = report.files;
      appState.scanReport = report;
      appState.project = await api.getProject(appState.project.id);
      // If the active viewer target was removed, fall back to the first frame.
      if (appState.viewer?.kind === 'input' && paths.includes(appState.viewer.path)) {
        appState.viewer = appState.inputs.length > 0
          ? { kind: 'input', path: appState.inputs[0].path }
          : null;
      }
    } catch (e) {
      error = (e as Error).message;
    } finally {
      busy = false;
    }
  }

  // Drop every frame the server flagged (wrong_size, wrong_bit_depth, etc.) in one call.
  let badCount = $derived(appState.inputs.filter((f) => f.status !== 'ok').length);
  function removeInvalid() {
    removeFrames(appState.inputs.filter((f) => f.status !== 'ok').map((f) => f.path));
  }

  function selectInput(path: string) {
    appState.viewer = { kind: 'input', path };
  }
</script>

<div class="input-list">
  <div class="list-header">
    <span class="eyebrow">Inputs</span>
    <button
      class="ghost add-btn"
      data-testid="ws-add"
      onclick={() => (showAdd = !showAdd)}
    >
      {showAdd ? 'Cancel' : '+ Add…'}
    </button>
  </div>

  {#if showAdd}
    <div class="add-panel panel">
      <div class="pick-row">
        <button
          class="primary"
          data-testid="pick-folder"
          disabled={busy}
          onclick={() => pick('directory')}
        >📁 Pick folder…</button>
        <button
          class="primary"
          data-testid="pick-files"
          disabled={busy}
          onclick={() => pick('files')}
        >🖼 Pick files…</button>
      </div>
      <p class="hint faint">Opens a native dialog — frames are used in place, not copied.</p>
      <div class="paste-row">
        <input
          type="text"
          bind:value={folder}
          data-testid="scan-path"
          placeholder="…or paste a folder / file path"
          onkeydown={(e) => e.key === 'Enter' && addPasted()}
        />
        <button
          class="ghost"
          data-testid="ws-add-confirm"
          disabled={busy || !folder}
          onclick={addPasted}
        >{busy ? 'Adding…' : 'Add'}</button>
      </div>
      {#if error}<p class="err mono">{error}</p>{/if}
    </div>
  {/if}

  {#if !showAdd && error}
    <p class="err mono">{error}</p>
  {/if}

  {#if lastReport && appState.inputs.length > 0}
    <div class="report-row">
      <span
        class="scan-status"
        class:ok={lastReport.ok}
      >{lastReport.ok ? '✓' : '✗'}</span>
      <span class="report-info mono faint">
        {appState.inputs.length} frames{lastReport.width && lastReport.height ? ` · ${lastReport.width}×${lastReport.height}` : ''}{lastReport.bit_depth ? ` · ${lastReport.bit_depth}-bit` : ''}
      </span>
    </div>
  {/if}

  <ValidationPanel />

  {#if badCount > 0}
    <div class="report-row">
      <button
        class="ghost remove-invalid"
        data-testid="remove-invalid"
        disabled={busy}
        onclick={removeInvalid}
      >✕ Remove {badCount} mismatched {badCount === 1 ? 'frame' : 'frames'}</button>
    </div>
  {/if}

  <div class="frames">
    {#each appState.inputs as f (f.path)}
      <div
        id={'input-' + encodeURIComponent(f.path)}
        class="frame-row"
        class:active={appState.viewer?.kind === 'input' && appState.viewer.path === f.path}
        class:invalid={f.status !== 'ok'}
        data-testid="ws-input"
        role="button"
        tabindex="0"
        onclick={() => selectInput(f.path)}
        onkeydown={(e) => e.key === 'Enter' && selectInput(f.path)}
      >
        <span class="fname" title={f.path}>
          {f.name}
        </span>
        {#if f.status !== 'ok'}
          <span class="status-badge" title={f.message}>{f.status}</span>
        {/if}
        <button
          class="ghost remove-btn"
          title="Remove"
          onclick={(e) => { e.stopPropagation(); removeFrames([f.path]); }}
        >✕</button>
      </div>
    {:else}
      <div class="empty faint">No frames — add a folder above.</div>
    {/each}
  </div>
</div>

<style>
  .input-list {
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
  }
  .list-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px 8px;
    border-bottom: 1px solid var(--line);
    flex-shrink: 0;
  }
  .add-btn { font-size: 12px; padding: 5px 10px; }
  .add-panel {
    margin: 8px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    flex-shrink: 0;
  }
  .pick-row { display: flex; gap: 8px; }
  .pick-row button { flex: 1; font-size: 12px; padding: 8px 6px; }
  .hint { font-size: 11px; margin: 0; }
  .paste-row { display: flex; gap: 8px; }
  .paste-row input {
    flex: 1;
    font-family: var(--font-mono);
    font-size: 11px;
  }
  .frames {
    margin: 0;
    padding: 4px 0;
    overflow-y: auto;
    flex: 1;
  }
  .frame-row {
    display: grid;
    grid-template-columns: 1fr auto auto;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    cursor: pointer;
    border-left: 3px solid transparent;
    transition: background 0.1s;
  }
  .frame-row:hover { background: var(--panel-2); }
  .frame-row.active {
    background: var(--accent-dim);
    border-left-color: var(--accent);
  }
  .frame-row.invalid { border-left-color: var(--bad); }
  .fname {
    font-family: var(--font-mono);
    font-size: 11px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .status-badge {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--bad);
    text-transform: uppercase;
    flex-shrink: 0;
  }
  .remove-invalid { font-size: 11px; padding: 5px 10px; color: var(--bad); width: 100%; }
  .remove-btn {
    font-size: 11px;
    padding: 3px 6px;
    opacity: 0;
    transition: opacity 0.1s;
    flex-shrink: 0;
  }
  .frame-row:hover .remove-btn { opacity: 1; }
  .empty { padding: 16px 12px; font-size: 12px; }
  .err { color: var(--bad); font-size: 12px; margin: 0; }
  .report-row { display: flex; align-items: center; gap: 8px; padding: 4px 12px; flex-shrink: 0; }
  .scan-status { font-weight: 600; color: var(--bad); font-size: 12px; }
  .scan-status.ok { color: var(--good); }
  .report-info { font-size: 11px; }
</style>
