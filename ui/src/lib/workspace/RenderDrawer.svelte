<script lang="ts">
  import { goto } from '$app/navigation';
  import { api } from '$lib/api';
  import { appState } from '$lib/stores.svelte';
  import ProgressBar from '$lib/components/ProgressBar.svelte';

  const ACTIVE = new Set(['pending', 'running']);

  let retouchBusy = $state<Record<string, boolean>>({});
  let retouchError = $state('');

  async function openRetouch(resultId: string) {
    if (!appState.project) return;
    const result = appState.results.find((r) => r.id === resultId);
    if (!result) return;
    retouchBusy[resultId] = true;
    retouchError = '';
    try {
      const projectId = appState.project.id;
      const { sessions } = await api.listRetouch(projectId);
      const existing = [...sessions].reverse().find((s) => s.target_image_id === resultId);
      const sessionId = existing?.id ??
        (await api.createRetouch(projectId, resultId)).session_id;
      appState.project.ui_state.retouchSessionId = sessionId;
      await api.patchUiState(projectId, { retouchSessionId: sessionId });
      await goto('/retouch');
    } catch (e) {
      retouchError = (e as Error).message;
    } finally {
      retouchBusy[resultId] = false;
    }
  }

  // Lazily register result thumbnails.
  async function ensureThumb(resultId: string) {
    const result = appState.results.find((r) => r.id === resultId);
    if (!result || result.thumb || !appState.project) return;
    try {
      const reg = await api.registerView(appState.project.id, result.path);
      result.imageId = reg.image_id;
      result.levels = reg.levels;
      result.width = reg.width;
      result.height = reg.height;
      result.thumb = api.tileUrl(reg.image_id, 0, 0, 0);
    } catch {
      // Non-fatal: thumb just stays missing.
    }
  }

  let activeJobs = $derived(
    Object.values(appState.jobs).filter((j) => ACTIVE.has(j.status))
  );

  let exportJob = $derived(
    appState.exportJobId ? appState.jobs[appState.exportJobId] : null
  );
</script>

<div class="render-drawer" class:closed={!appState.drawerOpen}>
  <div class="drawer-header">
    <span class="eyebrow">Renders</span>
    <button
      class="ghost toggle-btn"
      data-testid="ws-drawer-toggle"
      onclick={() => (appState.drawerOpen = !appState.drawerOpen)}
    >{appState.drawerOpen ? '▾ Hide' : '▸ Show'}</button>

    {#if exportJob?.status === 'done'}
      <span class="done-msg" data-testid="export-done">
        ✓ exported {typeof exportJob.result === 'object' && exportJob.result && 'dest' in exportJob.result
          ? (exportJob.result as { dest: string }).dest
          : ''}
      </span>
    {:else if exportJob?.status === 'error'}
      <span class="err mono">export error: {exportJob.error}</span>
    {:else if exportJob?.status === 'running' || exportJob?.status === 'pending'}
      <span class="faint mono">exporting… {Math.round(exportJob.percent * 100)}%</span>
    {/if}

    {#if retouchError}<span class="err mono">{retouchError}</span>{/if}
  </div>

  {#if appState.drawerOpen}
    <div class="drawer-content">
      <!-- Result thumbnails -->
      {#each appState.results as result (result.id)}
        {void ensureThumb(result.id)}
        <div
          class="result-chip"
          class:active={appState.viewer?.kind === 'result' && (appState.viewer?.id === result.id)}
          data-testid="ws-result"
          role="button"
          tabindex="0"
          onclick={() => (appState.viewer = { kind: 'result', id: result.id })}
          onkeydown={(e) => e.key === 'Enter' && (appState.viewer = { kind: 'result', id: result.id })}
        >
          {#if result.thumb}
            <img class="thumb" src={result.thumb} alt={result.label} />
          {:else}
            <div class="thumb thumb-placeholder">
              <span class="faint mono">…</span>
            </div>
          {/if}
          <span class="result-label mono">{result.label}</span>
          <div class="result-actions">
            <button
              class="ghost action-btn"
              data-testid="retouch-this-result"
              disabled={retouchBusy[result.id]}
              onclick={(e) => { e.stopPropagation(); openRetouch(result.id); }}
            >
              {retouchBusy[result.id] ? 'Opening…' : 'Retouch'}
            </button>
          </div>
        </div>
      {/each}

      <!-- Active jobs -->
      {#each activeJobs as job (job.id)}
        <div
          class="job-chip"
          class:active={appState.viewer?.kind === 'job' && (appState.viewer?.id === job.id)}
          data-testid="ws-job"
          role="button"
          tabindex="0"
          onclick={() => (appState.viewer = { kind: 'job', id: job.id })}
          onkeydown={(e) => e.key === 'Enter' && (appState.viewer = { kind: 'job', id: job.id })}
        >
          <div class="job-info">
            <span class="jtype mono">{job.type}</span>
            <span class="jstatus mono">{job.status}</span>
          </div>
          <ProgressBar percent={job.percent} status={job.status} />
          <div class="jmsg mono faint">{job.message}</div>
        </div>
      {/each}

      {#if appState.results.length === 0 && activeJobs.length === 0}
        <div class="empty faint">No results yet — run a stack to see results here.</div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .render-drawer {
    display: flex;
    flex-direction: column;
    border-top: 1px solid var(--line);
    background: var(--panel);
    min-height: 0;
    max-height: 200px;
  }
  .render-drawer.closed { max-height: none; }
  .drawer-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 12px;
    border-bottom: 1px solid var(--line);
    flex-shrink: 0;
    flex-wrap: wrap;
  }
  .toggle-btn { font-size: 11px; padding: 4px 8px; }
  .drawer-content {
    display: flex;
    gap: 10px;
    padding: 8px 12px;
    overflow-x: auto;
    overflow-y: hidden;
    flex: 1;
    align-items: flex-start;
  }
  .result-chip {
    display: flex;
    flex-direction: column;
    gap: 4px;
    width: 120px;
    flex-shrink: 0;
    cursor: pointer;
    border-radius: var(--radius);
    border: 2px solid transparent;
    padding: 4px;
    transition: border-color 0.12s;
  }
  .result-chip:hover { border-color: var(--line-strong); }
  .result-chip.active { border-color: var(--accent); }
  .thumb {
    display: block;
    width: 100%;
    height: 80px;
    object-fit: cover;
    border-radius: 4px;
    background: var(--bg);
  }
  .thumb-placeholder {
    height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg);
    border-radius: 4px;
  }
  .result-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .result-actions { display: flex; gap: 4px; }
  .action-btn { font-size: 10px; padding: 3px 6px; }
  .job-chip {
    display: flex;
    flex-direction: column;
    gap: 6px;
    width: 160px;
    flex-shrink: 0;
    cursor: pointer;
    border-radius: var(--radius);
    border: 2px solid transparent;
    padding: 8px;
    background: var(--panel-2);
    transition: border-color 0.12s;
  }
  .job-chip:hover { border-color: var(--line-strong); }
  .job-chip.active { border-color: var(--accent); }
  .job-info { display: flex; justify-content: space-between; align-items: center; }
  .jtype { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--accent-bright); }
  .jstatus { font-size: 10px; text-transform: uppercase; color: var(--text-dim); }
  .jmsg { font-size: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .empty { padding: 16px; font-size: 12px; }
  .done-msg { color: var(--good); font-size: 12px; font-weight: 500; }
  .err { color: var(--bad); font-size: 12px; }
</style>
