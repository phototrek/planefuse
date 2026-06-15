<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/api';
  import { appState } from '$lib/stores.svelte';
  import FolderBrowser from '$lib/components/FolderBrowser.svelte';

  let resultId = $state<string | null>(null);
  let depthId = $state<string | null>(null);

  let folder = $state('');
  let stem = $state('stacked');
  let format = $state<'tif' | 'jpg' | 'png'>('tif');
  let bitDepth = $state(16);
  let jpegQuality = $state(95);
  let alsoDepth = $state(false);

  let jobId = $state<string | null>(null);
  let error = $state('');

  let filename = $derived(`${stem || 'stacked'}.${format}`);
  // Auto path from folder+filename, overridable by editing the full-path field.
  let autoDest = $derived.by(() => {
    if (!folder) return '';
    const sep = folder.includes('\\') ? '\\' : '/';
    return `${folder.replace(/[\\/]+$/, '')}${sep}${filename}`;
  });
  let destOverride = $state<string | null>(null);
  let dest = $derived(destOverride ?? autoDest);

  let job = $derived(jobId ? appState.jobs[jobId] : null);

  function findImages(images: Record<string, Record<string, unknown>>) {
    for (const [id, info] of Object.entries(images)) {
      if (info.kind === 'result') resultId = id;
      if (info.kind === 'depth') depthId = id;
    }
  }

  onMount(async () => {
    if (!appState.project) return;
    try {
      appState.project = await api.getProject(appState.project.id);
      findImages(appState.project.images);
    } catch (e) {
      error = (e as Error).message;
    }
  });

  async function doExport() {
    if (!appState.project || !resultId || !dest) return;
    error = '';
    try {
      const body = {
        image_id: resultId,
        dest,
        format,
        bit_depth: bitDepth,
        jpeg_quality: jpegQuality
      };
      const r = await api.export(appState.project.id, body);
      jobId = r.id;
      if (alsoDepth && depthId) {
        const depthDest = dest.replace(/(\.[^.]+)$/, `_depth$1`);
        await api.export(appState.project.id, { ...body, image_id: depthId, dest: depthDest });
      }
    } catch (e) {
      error = (e as Error).message;
    }
  }
</script>

<div class="screen">
  <header class="head">
    <p class="eyebrow">Screen 6</p>
    <h1>Export</h1>
  </header>

  {#if !appState.project}
    <p class="dim">Open a project on the <a href="/">Import</a> screen first.</p>
  {:else if !resultId}
    <p class="dim">No stack result yet — run a stack first.</p>
  {:else}
    {#if error}<p class="err mono">{error}</p>{/if}

    <section class="panel block">
      <div class="grid">
        <label><span class="lbl">Format</span>
          <select bind:value={format}>
            <option value="tif">TIFF</option>
            <option value="png">PNG</option>
            <option value="jpg">JPEG</option>
          </select>
        </label>
        <label><span class="lbl">Bit depth</span>
          <select bind:value={bitDepth}>
            <option value={8}>8-bit</option>
            <option value={16}>16-bit</option>
          </select>
        </label>
        {#if format === 'jpg'}
          <label><span class="lbl">JPEG quality</span>
            <input type="number" bind:value={jpegQuality} min="1" max="100" />
          </label>
        {/if}
        <label><span class="lbl">Filename</span>
          <input type="text" bind:value={stem} />
        </label>
      </div>

      <div class="destrow">
        <span class="lbl">Destination folder</span>
        <FolderBrowser bind:value={folder} />
      </div>

      <label class="destrow">
        <span class="lbl">Full path</span>
        <input
          class="mono"
          type="text"
          value={dest}
          data-testid="export-dest"
          oninput={(e) => (destOverride = e.currentTarget.value)}
          placeholder="C:\\out\\stacked.tif"
        />
      </label>
      <p class="preview faint mono">→ {filename}</p>

      {#if depthId}
        <label class="toggle">
          <input type="checkbox" bind:checked={alsoDepth} />
          <span>Also export depth map</span>
        </label>
      {/if}

      <div class="actions">
        <button class="primary big" data-testid="export-go" onclick={doExport}>Export</button>
      </div>
    </section>

    {#if job}
      <section class="panel block status-block">
        <span class="st" class:done={job.status === 'done'} class:error={job.status === 'error'}>{job.status}</span>
        <span class="mono faint">{job.message}</span>
        {#if job.status === 'done'}
          <span class="done-msg" data-testid="export-done">
            ✓ wrote {typeof job.result === 'object' && job.result && 'dest' in job.result
              ? (job.result as { dest: string }).dest
              : dest}
          </span>
        {/if}
        {#if job.status === 'error'}<span class="bad mono">{job.error}</span>{/if}
      </section>
    {/if}
  {/if}
</div>

<style>
  .screen { padding: 28px 32px; max-width: 820px; display: flex; flex-direction: column; gap: 16px; }
  .head h1 { font-size: 26px; margin-top: 4px; }
  .block { padding: 18px 20px; display: flex; flex-direction: column; gap: 16px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  label { display: flex; flex-direction: column; gap: 6px; }
  .lbl { font-size: 13px; color: var(--text-dim); }
  .destrow input { width: 100%; }
  .preview { font-size: 12px; margin: 0; }
  .toggle { flex-direction: row; align-items: center; gap: 10px; }
  .toggle input { width: 16px; height: 16px; accent-color: var(--accent); }
  .actions { display: flex; justify-content: flex-end; }
  .big { padding: 11px 28px; font-size: 15px; }
  .status-block { flex-direction: row; align-items: center; gap: 14px; flex-wrap: wrap; }
  .st { font-weight: 600; text-transform: uppercase; color: var(--text-dim); }
  .st.done { color: var(--good); }
  .st.error { color: var(--bad); }
  .done-msg { color: var(--good); font-weight: 500; }
  .err { color: var(--bad); }
</style>
