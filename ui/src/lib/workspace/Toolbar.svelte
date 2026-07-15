<script lang="ts">
  import { onMount } from 'svelte';
  import { api, type Algorithm } from '$lib/api';
  import { appState } from '$lib/stores.svelte';
  import ParamForm from '$lib/components/ParamForm.svelte';
  import ExportPanel from './ExportPanel.svelte';
  import EstimateBadge from './EstimateBadge.svelte';
  import ProjectControls from './ProjectControls.svelte';

  // ── Algorithm + params ──────────────────────────────────────────────────────
  let algos = $state<Algorithm[]>([]);
  let selected = $state<string[]>([]);   // algos toggled to run; ≥1 required
  let values = $state<Record<string, unknown>>({});  // tuned params, only when exactly one selected
  let useAlign = $state(true);
  let maxLongEdge = $state(2048);
  let alignModel = $state('similarity');
  let alignInterpolation = $state('lanczos3');
  let normalizeBrightness = $state(true);
  let correlationThreshold = $state(0.9);
  let dropMisaligned = $state(false);
  let useSelect = $state(false);
  let selectionAccepted = $state(false);
  let selectionJobId = $state<string | null>(null);
  let presets = $state<{ name: string; params: Record<string, unknown> }[]>([]);
  let presetName = $state('');

  // ── UI toggles ──────────────────────────────────────────────────────────────
  let showOptions = $state(false);
  let proposedGroups = $state<string[][]>([]);

  // ── Error state ──────────────────────────────────────────────────────────────
  let runError = $state('');
  let selectionProposal = $derived(
    appState.project?.ui_state?.selectionProposal as {
      kept: number[];
      redundant: number[];
      warning: string | null;
      coverage: { reliable_cells: number; total_cells: number };
    } | undefined
  );

  let inputSignature = '';
  $effect(() => {
    const signature = appState.inputs.map((frame) => frame.path).join('\n');
    if (inputSignature && signature !== inputSignature) selectionAccepted = false;
    inputSignature = signature;
  });

  $effect(() => {
    if (!selectionJobId || appState.jobs[selectionJobId]?.status !== 'done' || !appState.project) return;
    const projectId = appState.project.id;
    selectionJobId = null;
    void api.getProject(projectId).then((project) => {
      appState.project = project;
      selectionAccepted = false;
    });
  });

  // Param tuning is offered only when exactly one algo is selected.
  let current = $derived(selected.length === 1 ? algos.find((a) => a.name === selected[0]) : undefined);

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
    if (useAlign) {
      params.align = {
        max_long_edge: maxLongEdge,
        model: alignModel,
        interp: alignInterpolation,
        normalize_brightness: normalizeBrightness,
        correlation_threshold: correlationThreshold,
        drop_misaligned: dropMisaligned
      };
    }
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
      proposedGroups = groups;
    } catch (e) {
      runError = (e as Error).message;
    }
  }

  async function reviewSelection() {
    if (!appState.project) return;
    const params: Record<string, unknown> = {
      device: appState.system?.device ?? 'auto',
      select: {}
    };
    if (useAlign) params.align = (buildParamsFor(selected[0] ?? 'pmax').align);
    const response = await api.enqueueJob(appState.project.id, 'select', params);
    selectionJobId = response.id;
    appState.drawerOpen = true;
  }

  async function stackGroups() {
    if (!appState.project) return;
    for (const group of proposedGroups) {
      for (const name of selected) {
        const response = await api.enqueueStack(appState.project.id, {
          ...buildParamsFor(name),
          frames: group
        });
        appState.stackJobIds = [...appState.stackJobIds, response.id];
      }
    }
    proposedGroups = [];
    appState.drawerOpen = true;
  }

</script>

<div class="toolbar">
  <!-- Left: project name + save -->
  <div class="section project-section">
    <ProjectControls />
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
      disabled={!appState.project || appState.inputs.length === 0 || selected.length === 0 || !appState.scanReport?.ok || (useSelect && !selectionAccepted)}
      title={!appState.scanReport?.ok ? 'Resolve input validation issues first' : useSelect && !selectionAccepted ? 'Review and accept smart selection first' : selected.length === 0 ? 'Select at least one algorithm' : `Run ${selected.length} algorithm(s)`}
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
    <ExportPanel />
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
      <select bind:value={alignModel} aria-label="Alignment model">
        <option value="translation">Translation</option>
        <option value="similarity">Similarity</option>
        <option value="perspective">Perspective</option>
      </select>
      <select bind:value={alignInterpolation} aria-label="Warp interpolation">
        <option value="lanczos3">Lanczos-3</option>
        <option value="bicubic">Bicubic</option>
        <option value="bilinear">Bilinear</option>
      </select>
      <label class="inline-check"><input type="checkbox" bind:checked={normalizeBrightness} /> Match brightness</label>
      <label class="inline-check"><input type="checkbox" bind:checked={dropMisaligned} /> Exclude low quality</label>
      <span class="faint mono">ECC ≥</span>
      <input class="num threshold" type="number" bind:value={correlationThreshold} min="0" max="1" step="0.01" />
    {/if}
  </label>
  <label class="toggle">
    <input type="checkbox" bind:checked={useSelect} />
    <span>Smart frame selection</span>
  </label>
  {#if useSelect}
    <button class="ghost" disabled={!!selectionJobId || !appState.scanReport?.ok} onclick={reviewSelection}>
      {selectionJobId ? 'Analyzing…' : 'Review proposal'}
    </button>
    {#if selectionAccepted}<span class="good mono">proposal accepted</span>{/if}
  {/if}
  <EstimateBadge method={selected[0] ?? 'pmax'} align={useAlign} />
</div>

{#if useSelect && selectionProposal}
  <div class="selection-proposal panel" data-testid="selection-proposal">
    <div>
      <strong>Smart-selection proposal</strong>
      <span>{selectionProposal.kept.length} kept · {selectionProposal.redundant.length} redundant</span>
      <span class="mono faint">coverage cells {selectionProposal.coverage.reliable_cells}/{selectionProposal.coverage.total_cells}</span>
      {#if selectionProposal.warning}<span class="warn">{selectionProposal.warning}</span>{/if}
    </div>
    <div class="proposal-strip">
      {#each appState.inputs as frame, index}
        <span class:excluded={selectionProposal.redundant.includes(index)} title={frame.name}>{index + 1}</span>
      {/each}
    </div>
    <button class="primary" onclick={() => (selectionAccepted = true)}>Accept proposal</button>
  </div>
{/if}

{#if proposedGroups.length > 0}
  <div class="group-review panel" data-testid="group-review">
    <div>
      <strong>Review batch groups</strong>
      <span class="faint">Nothing is queued until you confirm.</span>
    </div>
    <ol>
      {#each proposedGroups as group, index}
        <li><span>Group {index + 1}</span><span class="mono faint">{group.length} frames · {group[0]?.split(/[\\/]/).pop()} → {group[group.length - 1]?.split(/[\\/]/).pop()}</span></li>
      {/each}
    </ol>
    <div class="group-actions">
      <button onclick={() => (proposedGroups = [])}>Cancel</button>
      <button class="primary" data-testid="stack-all-groups" onclick={stackGroups}>Stack all groups</button>
    </div>
  </div>
{/if}

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
    overflow: visible;
    position: relative;
    z-index: 40;
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
  .threshold { width: 64px; }
  .inline-check { display: flex; align-items: center; gap: 4px; white-space: nowrap; }
  .group-review { display: grid; grid-template-columns: minmax(180px, .5fr) 1fr auto; gap: 14px; align-items: center; padding: 10px 14px; border-top: 1px solid var(--line); font-size: 11px; }
  .group-review > div:first-child { display: flex; flex-direction: column; gap: 3px; }
  .group-review ol { display: flex; gap: 8px; margin: 0; padding: 0; list-style: none; overflow-x: auto; }
  .group-review li { display: flex; flex-direction: column; min-width: 150px; }
  .group-actions { display: flex; gap: 6px; }
  .selection-proposal { display: grid; grid-template-columns: minmax(200px, .5fr) 1fr auto; gap: 14px; align-items: center; padding: 10px 14px; border-top: 1px solid var(--line); font-size: 10px; }
  .selection-proposal > div:first-child { display: flex; flex-direction: column; gap: 2px; }
  .proposal-strip { display: flex; gap: 2px; overflow-x: auto; }
  .proposal-strip span { min-width: 20px; padding: 3px; text-align: center; color: var(--good); background: color-mix(in srgb, var(--good) 15%, var(--panel)); }
  .proposal-strip span.excluded { opacity: .35; color: var(--text-faint); text-decoration: line-through; }
  .good { color: var(--good); font-size: 10px; }
  .warn { color: var(--warn, #e1a857); }
  .preset-row { display: flex; gap: 6px; }
  .preset-list { display: flex; flex-wrap: wrap; gap: 6px; }
  .preset-chip { display: flex; align-items: center; border: 1px solid var(--line); border-radius: var(--radius); }
  .del { color: var(--text-faint); padding: 5px 7px; }
  .err { color: var(--bad); font-size: 11px; }
</style>
