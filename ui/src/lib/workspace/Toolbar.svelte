<script lang="ts">
  import { onMount } from 'svelte';
  import { api, type Algorithm } from '$lib/api';
  import { appState } from '$lib/stores.svelte';
  import { applyTemplate, todayISO, type TemplateCtx } from './template';
  import ParamForm from '$lib/components/ParamForm.svelte';
  import FolderBrowser from '$lib/components/FolderBrowser.svelte';

  // ── Algorithm + params ──────────────────────────────────────────────────────
  let algos = $state<Algorithm[]>([]);
  let selected = $state<string[]>([]);   // algos toggled to run; ≥1 required
  let values = $state<Record<string, unknown>>({});  // tuned params, only when exactly one selected
  let useAlign = $state(true);
  let maxLongEdge = $state(2048);
  let useSelect = $state(false);
  let presets = $state<{ name: string; params: Record<string, unknown> }[]>([]);
  let presetName = $state('');

  // ── UI toggles ──────────────────────────────────────────────────────────────
  let showOptions = $state(false);
  let showExport = $state(false);

  // ── Export popover state ─────────────────────────────────────────────────────
  let exportFolder = $state('');
  let exportTemplate = $state('{stack_name}_{method}');
  let exportFormat = $state<'tif' | 'jpg' | 'png'>('tif');
  let exportBitDepth = $state(16);
  let exportJpegQuality = $state(95);
  let exportDestOverride = $state<string | null>(null);

  // ── Save state ───────────────────────────────────────────────────────────────
  // Use override pattern: user edit overrides, else derive from project
  let projectNameOverride = $state<string | null>(null);
  let saveBusy = $state(false);
  let saveError = $state('');
  let saveDone = $state(false);

  // ── Error state ──────────────────────────────────────────────────────────────
  let runError = $state('');
  let exportError = $state('');

  // Param tuning is offered only when exactly one algo is selected.
  let current = $derived(selected.length === 1 ? algos.find((a) => a.name === selected[0]) : undefined);
  let projectName = $derived(projectNameOverride ?? appState.project?.name ?? '');

  function resetValues(a: Algorithm | undefined) {
    const v: Record<string, unknown> = {};
    for (const p of a?.params ?? []) v[p.name] = p.default;
    values = v;
  }

  function toggleAlgo(name: string) {
    selected = selected.includes(name)
      ? selected.filter((n) => n !== name)
      : [...selected, name];
    // When exactly one is selected, load its defaults so it can be tuned.
    if (selected.length === 1) resetValues(algos.find((a) => a.name === selected[0]));
  }

  function defaultsFor(name: string): Record<string, unknown> {
    const v: Record<string, unknown> = {};
    for (const p of algos.find((a) => a.name === name)?.params ?? []) v[p.name] = p.default;
    return v;
  }

  onMount(async () => {
    try {
      algos = await api.algorithms();
      presets = await api.listPresets();
    } catch (e) {
      runError = (e as Error).message;
    }
  });

  // One algo's job params. Tuned `values` apply only when it's the sole selected
  // algo; in multi-select every algo runs with its defaults.
  function buildParamsFor(name: string): Record<string, unknown> {
    const params: Record<string, unknown> = {
      method: name,
      device: appState.system?.device ?? 'auto',
      algo_params: selected.length === 1 && selected[0] === name ? values : defaultsFor(name)
    };
    if (useAlign) params.align = { max_long_edge: maxLongEdge };
    if (useSelect) params.select = {};
    return params;
  }

  async function savePreset() {
    if (!presetName) return;
    await api.addPreset(presetName, {
      methods: selected,
      algo_params: values,
      align: useAlign ? { max_long_edge: maxLongEdge } : undefined,
      select: useSelect ? {} : undefined
    });
    presets = await api.listPresets();
    presetName = '';
  }

  function loadPreset(p: { params: Record<string, unknown> }) {
    // New presets store `methods` (array); legacy presets store a single `method`.
    selected = Array.isArray(p.params.methods)
      ? (p.params.methods as string[])
      : typeof p.params.method === 'string'
        ? [p.params.method]
        : [];
    if (selected.length === 1) {
      resetValues(algos.find((a) => a.name === selected[0]));
      if (p.params.algo_params) values = { ...(p.params.algo_params as Record<string, unknown>) };
    }
    useAlign = !!p.params.align;
    if (p.params.align) maxLongEdge = (p.params.align as { max_long_edge: number }).max_long_edge;
    useSelect = !!p.params.select;
  }

  async function deletePreset(name: string) {
    await api.deletePreset(name);
    presets = await api.listPresets();
  }

  // ── Run ──────────────────────────────────────────────────────────────────────
  async function run() {
    if (!appState.project || selected.length === 0) return;
    runError = '';
    try {
      let firstId: string | null = null;
      for (const name of selected) {
        const res = await api.enqueueStack(appState.project.id, buildParamsFor(name));
        appState.stackJobIds = [...appState.stackJobIds, res.id];
        if (!firstId) firstId = res.id;
      }
      if (firstId) appState.viewer = { kind: 'job', id: firstId };
      appState.drawerOpen = true;
    } catch (e) {
      runError = (e as Error).message;
    }
  }

  // ── Auto-group ───────────────────────────────────────────────────────────────
  async function autoGroup() {
    if (!appState.project) return;
    runError = '';
    try {
      const { groups } = await api.autoGroup(appState.project.id);
      for (const g of groups) {
        for (const name of selected) {
          const res = await api.enqueueStack(appState.project.id, { ...buildParamsFor(name), frames: g });
          appState.stackJobIds = [...appState.stackJobIds, res.id];
        }
      }
      appState.drawerOpen = true;
    } catch (e) {
      runError = (e as Error).message;
    }
  }

  // ── Export ───────────────────────────────────────────────────────────────────

  // Load the persisted template when the project changes.
  $effect(() => {
    const t = appState.project?.ui_state?.exportTemplate;
    if (typeof t === 'string' && t) exportTemplate = t;
  });

  function exportCtx(): TemplateCtx {
    const id = getExportImageId();
    const idx = appState.results.findIndex((r) => r.id === id);
    const result = idx >= 0 ? appState.results[idx] : undefined;
    return {
      stack_name: projectName || 'stacked',
      method: result?.method ?? '',
      frames: result?.frames,
      date: todayISO(),
      seq: (idx >= 0 ? idx : 0) + 1
    };
  }

  let exportStem = $derived(applyTemplate(exportTemplate, exportCtx()) || 'stacked');
  let exportFilename = $derived(`${exportStem}.${exportFormat}`);

  let exportAutoDest = $derived.by(() => {
    if (!exportFolder) return '';
    const sep = exportFolder.includes('\\') ? '\\' : '/';
    return `${exportFolder.replace(/[\\/]+$/, '')}${sep}${exportFilename}`;
  });

  let exportDest = $derived(exportDestOverride ?? exportAutoDest);

  function getExportImageId(): string | null {
    const viewer = appState.viewer;
    if (viewer?.kind === 'result') {
      const result = appState.results.find((r) => r.id === viewer.id);
      if (result) return result.id;
    }
    if (appState.results.length > 0) return appState.results[appState.results.length - 1].id;
    return null;
  }

  async function doExport() {
    if (!appState.project) return;
    const imgId = getExportImageId();
    if (!imgId || !exportDest) return;
    exportError = '';
    try {
      const body: Record<string, unknown> = {
        image_id: imgId,
        dest: exportDest,
        format: exportFormat,
        bit_depth: exportBitDepth,
        jpeg_quality: exportJpegQuality
      };
      const r = await api.export(appState.project.id, body);
      appState.exportJobId = r.id;
      api.patchUiState(appState.project.id, { exportTemplate }).catch(() => {});
      showExport = false;
    } catch (e) {
      exportError = (e as Error).message;
    }
  }

  // ── Save ─────────────────────────────────────────────────────────────────────
  async function save() {
    if (!appState.project) return;
    saveBusy = true;
    saveError = '';
    saveDone = false;
    try {
      const result = await api.saveProject(appState.project.id, {
        name: projectName,
        saved: true
      });
      appState.project.name = result.name;
      projectNameOverride = null; // Reset so derived picks up new name from project
      saveDone = true;
      setTimeout(() => { saveDone = false; }, 2000);
    } catch (e) {
      saveError = (e as Error).message;
    } finally {
      saveBusy = false;
    }
  }

  let canExport = $derived(appState.results.length > 0);
</script>

<div class="toolbar">
  <!-- Left: project name + save -->
  <div class="section project-section">
    <input
      class="project-name"
      type="text"
      value={projectName}
      oninput={(e) => (projectNameOverride = e.currentTarget.value)}
      data-testid="ws-project-name"
      placeholder="Untitled"
      aria-label="Project name"
    />
    <button
      class="ghost"
      data-testid="ws-save"
      disabled={saveBusy || !appState.project}
      onclick={save}
    >
      {saveBusy ? 'Saving…' : saveDone ? '✓ Saved' : 'Save'}
    </button>
    {#if saveError}<span class="err mono">{saveError}</span>{/if}
  </div>

  <div class="divider"></div>

  <!-- Center: algo + run + auto-group -->
  <div class="section algo-section">
    <div class="algos">
      {#each algos as a (a.name)}
        <button
          class="algo"
          class:sel={selected.includes(a.name)}
          data-testid="algo-{a.name}"
          aria-pressed={selected.includes(a.name)}
          title="Toggle {a.name} — run several at once"
          onclick={() => toggleAlgo(a.name)}
        >{a.name}</button>
      {/each}
    </div>

    <button
      class="ghost"
      onclick={() => (showOptions = !showOptions)}
      title="Algorithm options"
    >
      Options{showOptions ? ' ▲' : ' ▾'}
    </button>

    <button
      class="primary"
      data-testid="ws-run"
      disabled={!appState.project || appState.inputs.length === 0 || selected.length === 0}
      title={selected.length === 0 ? 'Select at least one algorithm' : `Run ${selected.length} algorithm(s)`}
      onclick={run}
    >
      Run{selected.length > 1 ? ` ×${selected.length}` : ''}
    </button>

    <button
      class="ghost"
      data-testid="ws-autogroup"
      disabled={!appState.project || appState.inputs.length === 0 || selected.length === 0}
      onclick={autoGroup}
    >
      Auto-group
    </button>

    {#if runError}<span class="err mono">{runError}</span>{/if}
  </div>

  <div class="divider"></div>

  <!-- Right: export -->
  <div class="section export-section">
    <div class="export-wrap">
      <button
        class="ghost"
        data-testid="ws-export"
        disabled={!canExport}
        onclick={() => (showExport = !showExport)}
        title={canExport ? 'Export result' : 'No result to export yet'}
      >
        Export{showExport ? ' ▲' : ' ▾'}
      </button>

      {#if showExport}
        <div class="popover panel">
          <div class="export-grid">
            <label>
              <span class="lbl">Format</span>
              <select bind:value={exportFormat}>
                <option value="tif">TIFF</option>
                <option value="png">PNG</option>
                <option value="jpg">JPEG</option>
              </select>
            </label>
            <label>
              <span class="lbl">Bit depth</span>
              <select bind:value={exportBitDepth}>
                <option value={8}>8-bit</option>
                <option value={16}>16-bit</option>
              </select>
            </label>
            {#if exportFormat === 'jpg'}
              <label>
                <span class="lbl">Quality</span>
                <input type="number" bind:value={exportJpegQuality} min="1" max="100" />
              </label>
            {/if}
            <label class="tpl">
              <span class="lbl">Name template</span>
              <input type="text" bind:value={exportTemplate} data-testid="export-template" />
            </label>
            <div class="tpl-preview faint mono" data-testid="export-preview">{exportFilename}</div>
            <div class="tpl-hint faint">{`{stack_name} {method} {frames} {date} {seq}`}</div>
          </div>
          <div class="dest-row">
            <span class="lbl">Output folder</span>
            <FolderBrowser bind:value={exportFolder} />
          </div>
          <label class="dest-row">
            <span class="lbl">Full path</span>
            <input
              class="mono dest-input"
              type="text"
              value={exportDest}
              data-testid="export-dest"
              oninput={(e) => (exportDestOverride = e.currentTarget.value)}
              placeholder="C:\out\stacked.tif"
            />
          </label>
          {#if exportError}<p class="err mono">{exportError}</p>{/if}
          <div class="export-actions">
            <button
              class="primary"
              data-testid="export-go"
              disabled={!exportDest}
              onclick={doExport}
            >Export</button>
          </div>
        </div>
      {/if}
    </div>
  </div>
</div>

<!-- Always-visible align + select toggles (below toolbar) -->
<div class="align-row">
  <label class="toggle">
    <input type="checkbox" bind:checked={useAlign} />
    <span>Align frames</span>
    {#if useAlign}
      <span class="faint mono">max long edge</span>
      <input class="num" type="number" bind:value={maxLongEdge} min="128" max="8192" step="1" />
    {/if}
  </label>
  <label class="toggle">
    <input type="checkbox" bind:checked={useSelect} />
    <span>Smart frame selection</span>
  </label>
</div>

<!-- Options panel (full-width, below align row) -->
{#if showOptions}
  <div class="options-panel panel">
    {#if current}
      <div class="params-section">
        <p class="eyebrow">Parameters</p>
        <ParamForm specs={current.params} bind:values />
      </div>
    {:else if selected.length > 1}
      <div class="params-section">
        <p class="eyebrow">Parameters</p>
        <p class="faint">{selected.length} algorithms selected — each runs with its default parameters.</p>
      </div>
    {/if}

    <div class="presets-section">
      <p class="eyebrow">Presets</p>
      <div class="preset-row">
        <input type="text" bind:value={presetName} placeholder="Preset name" />
        <button onclick={savePreset}>Save current</button>
      </div>
      <div class="preset-list">
        {#each presets as p (p.name)}
          <div class="preset-chip">
            <button class="ghost" onclick={() => loadPreset(p)}>{p.name}</button>
            <button class="ghost del" onclick={() => deletePreset(p.name)}>✕</button>
          </div>
        {/each}
      </div>
    </div>
  </div>
{/if}

<style>
  .toolbar {
    display: flex;
    align-items: center;
    gap: 0;
    height: 100%;
    overflow: hidden;
  }
  .section {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 12px;
    height: 100%;
    flex-shrink: 0;
  }
  .divider {
    width: 1px;
    height: 60%;
    background: var(--line);
    flex-shrink: 0;
  }
  .project-section { min-width: 0; }
  .project-name {
    font-size: 13px;
    font-weight: 600;
    width: 160px;
    padding: 5px 8px;
    background: transparent;
    border: 1px solid transparent;
  }
  .project-name:hover { border-color: var(--line-strong); }
  .project-name:focus { background: var(--bg); border-color: var(--accent); }
  .algo-section { gap: 8px; flex-wrap: nowrap; flex-shrink: 0; }
  .algos { display: flex; gap: 4px; }
  .algo {
    text-transform: uppercase;
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.06em;
    padding: 5px 10px;
  }
  .algo.sel { background: var(--accent); color: #1a0d04; border-color: var(--accent); }
  .export-section { margin-left: auto; }
  .export-wrap { position: relative; }
  .popover {
    position: absolute;
    top: calc(100% + 8px);
    right: 0;
    width: 380px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    z-index: 100;
    box-shadow: var(--shadow);
  }
  .export-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .export-grid label { display: flex; flex-direction: column; gap: 4px; }
  .dest-row { display: flex; flex-direction: column; gap: 6px; }
  .dest-input { width: 100%; font-family: var(--font-mono); font-size: 11px; }
  .export-actions { display: flex; justify-content: flex-end; }
  .lbl { font-size: 12px; color: var(--text-dim); }
  .align-row {
    display: flex;
    align-items: center;
    gap: 24px;
    padding: 4px 12px;
    border-top: 1px solid var(--line);
    font-size: 12px;
    flex-shrink: 0;
  }
  .options-panel {
    padding: 14px 16px;
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    border-top: 1px solid var(--line);
  }
  .params-section, .presets-section { display: flex; flex-direction: column; gap: 10px; min-width: 200px; }
  .toggle { display: flex; align-items: center; gap: 8px; padding: 4px 0; font-size: 13px; }
  .toggle input[type='checkbox'] { width: 14px; height: 14px; accent-color: var(--accent); }
  .num { width: 80px; }
  .preset-row { display: flex; gap: 6px; }
  .preset-list { display: flex; flex-wrap: wrap; gap: 6px; }
  .preset-chip { display: flex; align-items: center; border: 1px solid var(--line); border-radius: var(--radius); }
  .del { color: var(--text-faint); padding: 5px 7px; }
  .err { color: var(--bad); font-size: 11px; }
  .tpl-preview { font-size: 11px; font-family: var(--font-mono); color: var(--text-faint); grid-column: 1 / -1; }
  .tpl-hint { font-size: 10px; color: var(--text-faint); grid-column: 1 / -1; letter-spacing: 0.02em; }
</style>
