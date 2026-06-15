<script lang="ts">
  import { onMount } from 'svelte';
  import { api, type Project, type ScanReport } from '$lib/api';
  import { appState } from '$lib/stores.svelte';
  import FolderBrowser from '$lib/components/FolderBrowser.svelte';
  import Filmstrip from '$lib/components/Filmstrip.svelte';

  let projects = $state<Project[]>([]);
  let showCreate = $state(false);
  let newPath = $state('');
  let newName = $state('');

  let showImport = $state(false);
  let scanPath = $state('');
  let report = $state<ScanReport | null>(null);
  let groups = $state<string[][] | null>(null);

  let busy = $state(false);
  let error = $state('');

  async function refresh() {
    try {
      projects = await api.listProjects();
    } catch (e) {
      error = (e as Error).message;
    }
  }
  onMount(refresh);

  async function create() {
    if (!newPath || !newName) return;
    busy = true;
    error = '';
    try {
      const p = await api.createProject(newPath, newName);
      appState.project = p;
      showCreate = false;
      report = null;
      groups = null;
      await refresh();
    } catch (e) {
      error = (e as Error).message;
    } finally {
      busy = false;
    }
  }

  async function open(id: string) {
    busy = true;
    error = '';
    try {
      appState.project = await api.getProject(id);
      report = null;
      groups = null;
    } catch (e) {
      error = (e as Error).message;
    } finally {
      busy = false;
    }
  }

  async function scan() {
    if (!appState.project || !scanPath) return;
    busy = true;
    error = '';
    try {
      report = await api.scan(appState.project.id, scanPath);
      appState.project = await api.getProject(appState.project.id);
      groups = null;
    } catch (e) {
      error = (e as Error).message;
    } finally {
      busy = false;
    }
  }

  async function autoGroup() {
    if (!appState.project) return;
    busy = true;
    try {
      groups = (await api.autoGroup(appState.project.id)).groups;
      appState.project!.ui_state.groups = groups;
      await api.patchUiState(appState.project.id, { groups });
    } catch (e) {
      error = (e as Error).message;
    } finally {
      busy = false;
    }
  }
</script>

<div class="screen">
  <header class="head">
    <p class="eyebrow">Screen 1</p>
    <h1>Project &amp; Import</h1>
  </header>

  {#if error}<p class="err mono">{error}</p>{/if}

  <section class="panel block">
    <div class="block-head">
      <h2>Projects</h2>
      <button class="primary" data-testid="new-project" onclick={() => (showCreate = !showCreate)}>
        New project
      </button>
    </div>

    {#if showCreate}
      <div class="create">
        <label>
          <span class="eyebrow">Folder</span>
          <FolderBrowser bind:value={newPath} inputTestid="project-path" />
        </label>
        <label>
          <span class="eyebrow">Name</span>
          <input type="text" bind:value={newName} data-testid="project-name" placeholder="My macro stack" />
        </label>
        <button class="primary" data-testid="create-project" disabled={busy} onclick={create}>Create</button>
      </div>
    {/if}

    <ul class="plist">
      {#each projects as p (p.id)}
        <li class:active={appState.project?.id === p.id}>
          <span class="pn">{p.name}</span>
          <span class="pd mono faint">{p.directory}</span>
          <button class="ghost" onclick={() => open(p.id)}>Open</button>
        </li>
      {:else}
        <li class="empty faint">No projects yet — create one above.</li>
      {/each}
    </ul>
  </section>

  {#if appState.project}
    <section class="panel block">
      <div class="block-head">
        <h2>Import frames</h2>
        <button data-testid="import-frames" onclick={() => (showImport = !showImport)}>
          {showImport ? 'Hide' : 'Import from folder'}
        </button>
      </div>

      {#if showImport}
        <div class="create">
          <label>
            <span class="eyebrow">Source folder</span>
            <FolderBrowser bind:value={scanPath} inputTestid="scan-path" />
          </label>
          <button class="primary" data-testid="scan-go" disabled={busy} onclick={scan}>Scan</button>
        </div>
      {/if}

      {#if report}
        <div class="report">
          <span
            class="status"
            class:ok={report.ok}
            data-testid={report.ok ? 'scan-ok' : 'scan-bad'}
          >
            {report.ok ? '✓ all frames valid' : '✗ validation issues'}
          </span>
          <span class="faint mono">
            {report.files.length} frames · {report.width}×{report.height} · {report.bit_depth}-bit
          </span>
          <button class="ghost" onclick={autoGroup}>Auto-group</button>
        </div>
        <Filmstrip projectId={appState.project.id} files={report.files} />
      {/if}

      {#if groups}
        <div class="groups">
          <p class="eyebrow">Proposed groups ({groups.length})</p>
          {#each groups as g, i (i)}
            <div class="grp mono">Group {i + 1}: {g.length} frames</div>
          {/each}
        </div>
      {/if}
    </section>
  {/if}
</div>

<style>
  .screen { padding: 28px 32px; max-width: 1100px; display: flex; flex-direction: column; gap: 20px; }
  .head h1 { font-size: 26px; margin-top: 4px; }
  .block { padding: 18px 20px; }
  .block-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
  .block-head h2 { font-size: 16px; }
  .create { display: flex; flex-direction: column; gap: 14px; padding: 4px 0 14px; max-width: 640px; }
  .create label { display: flex; flex-direction: column; gap: 6px; }
  .plist { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; }
  .plist li {
    display: grid;
    grid-template-columns: 1fr auto auto;
    align-items: center;
    gap: 14px;
    padding: 11px 4px;
    border-top: 1px solid var(--line);
  }
  .plist li.active { box-shadow: inset 3px 0 0 var(--accent); padding-left: 12px; }
  .plist .pn { font-weight: 600; }
  .plist .pd { font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 460px; }
  .plist .empty { display: block; grid-template-columns: none; }
  .report { display: flex; align-items: center; gap: 16px; padding: 6px 0; }
  .status { font-weight: 600; color: var(--bad); }
  .status.ok { color: var(--good); }
  .groups { margin-top: 8px; display: flex; flex-direction: column; gap: 4px; }
  .grp { font-size: 12px; color: var(--text-dim); }
  .err { color: var(--bad); font-size: 13px; }
</style>
