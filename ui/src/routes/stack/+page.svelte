<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { api, type Algorithm } from '$lib/api';
  import { appState } from '$lib/stores.svelte';
  import ParamForm from '$lib/components/ParamForm.svelte';

  let algos = $state<Algorithm[]>([]);
  let method = $state('pmax');
  let values = $state<Record<string, unknown>>({});

  let useAlign = $state(true);
  let maxLongEdge = $state(2048);
  let useSelect = $state(false);

  let presets = $state<{ name: string; params: Record<string, unknown> }[]>([]);
  let presetName = $state('');

  let busy = $state(false);
  let error = $state('');

  let current = $derived(algos.find((a) => a.name === method));

  function resetValues(a: Algorithm | undefined) {
    const v: Record<string, unknown> = {};
    for (const p of a?.params ?? []) v[p.name] = p.default;
    values = v;
  }

  function pickMethod(name: string) {
    method = name;
    resetValues(algos.find((a) => a.name === name));
  }

  onMount(async () => {
    try {
      algos = await api.algorithms();
      resetValues(algos.find((a) => a.name === method));
      presets = await api.listPresets();
    } catch (e) {
      error = (e as Error).message;
    }
  });

  function buildParams(): Record<string, unknown> {
    const params: Record<string, unknown> = {
      method,
      device: appState.system?.device ?? 'auto',
      algo_params: values
    };
    if (useAlign) params.align = { max_long_edge: maxLongEdge };
    if (useSelect) params.select = {};
    return params;
  }

  async function savePreset() {
    if (!presetName) return;
    await api.addPreset(presetName, buildParams());
    presets = await api.listPresets();
    presetName = '';
  }

  function loadPreset(p: { params: Record<string, unknown> }) {
    if (typeof p.params.method === 'string') pickMethod(p.params.method);
    if (p.params.algo_params) values = { ...(p.params.algo_params as Record<string, unknown>) };
    useAlign = !!p.params.align;
    if (p.params.align) maxLongEdge = (p.params.align as { max_long_edge: number }).max_long_edge;
    useSelect = !!p.params.select;
  }

  async function deletePreset(name: string) {
    await api.deletePreset(name);
    presets = await api.listPresets();
  }

  let groups = $derived((appState.project?.ui_state.groups as string[][] | undefined) ?? []);

  async function stack() {
    if (!appState.project) return;
    busy = true;
    error = '';
    try {
      await api.enqueueStack(appState.project.id, buildParams());
      await goto('/queue');
    } catch (e) {
      error = (e as Error).message;
    } finally {
      busy = false;
    }
  }

  async function stackAllGroups() {
    if (!appState.project || groups.length === 0) return;
    busy = true;
    error = '';
    try {
      for (const g of groups) {
        await api.enqueueStack(appState.project.id, { ...buildParams(), frames: g });
      }
      await goto('/queue');
    } catch (e) {
      error = (e as Error).message;
    } finally {
      busy = false;
    }
  }
</script>

<div class="screen">
  <header class="head">
    <p class="eyebrow">Screen 2</p>
    <h1>Stack setup</h1>
  </header>

  {#if !appState.project}
    <p class="dim">Open a project on the <a href="/">Import</a> screen first.</p>
  {:else}
    {#if error}<p class="err mono">{error}</p>{/if}

    <section class="panel block">
      <h2>Algorithm</h2>
      <div class="algos">
        {#each algos as a (a.name)}
          <button
            class="algo"
            class:sel={method === a.name}
            data-testid="algo-{a.name}"
            onclick={() => pickMethod(a.name)}
          >{a.name}</button>
        {/each}
      </div>
      {#if current}
        <div class="params">
          <ParamForm specs={current.params} bind:values />
        </div>
      {/if}
    </section>

    <section class="panel block">
      <h2>Alignment &amp; selection</h2>
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
        <span class="faint">(auto-drop redundant frames)</span>
      </label>
    </section>

    <section class="panel block">
      <h2>Presets</h2>
      <div class="preset-row">
        <input type="text" bind:value={presetName} placeholder="Preset name" />
        <button onclick={savePreset}>Save current</button>
      </div>
      <ul class="presets">
        {#each presets as p (p.name)}
          <li>
            <button class="ghost" onclick={() => loadPreset(p)}>{p.name}</button>
            <button class="ghost del" onclick={() => deletePreset(p.name)}>✕</button>
          </li>
        {/each}
      </ul>
    </section>

    <div class="actions">
      {#if groups.length > 1}
        <button class="big" data-testid="stack-all" disabled={busy} onclick={stackAllGroups}>
          Stack all groups ({groups.length})
        </button>
      {/if}
      <button class="primary big" data-testid="stack-go" disabled={busy} onclick={stack}>Stack</button>
    </div>
  {/if}
</div>

<style>
  .screen { padding: 28px 32px; max-width: 1000px; display: flex; flex-direction: column; gap: 18px; }
  .head h1 { font-size: 26px; margin-top: 4px; }
  .block { padding: 18px 20px; }
  .block h2 { font-size: 15px; margin-bottom: 14px; }
  .algos { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
  .algo { text-transform: uppercase; font-family: var(--font-mono); font-size: 12px; letter-spacing: 0.06em; }
  .algo.sel { background: var(--accent); color: #1a0d04; border-color: var(--accent); }
  .toggle { display: flex; align-items: center; gap: 10px; padding: 7px 0; }
  .toggle input[type='checkbox'] { width: 16px; height: 16px; accent-color: var(--accent); }
  .num { width: 90px; }
  .preset-row { display: flex; gap: 8px; margin-bottom: 12px; }
  .presets { list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 8px; }
  .presets li { display: flex; align-items: center; gap: 2px; border: 1px solid var(--line); border-radius: var(--radius); }
  .presets .del { color: var(--text-faint); padding: 7px 9px; }
  .actions { display: flex; justify-content: flex-end; }
  .big { padding: 11px 28px; font-size: 15px; }
  .err { color: var(--bad); }
</style>
