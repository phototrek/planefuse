<script lang="ts">
  import { appState, type CompareMode } from '$lib/stores.svelte';

  const modes: { id: CompareMode; label: string }[] = [
    { id: 'single', label: 'Single' },
    { id: 'split-source', label: 'Result / source' },
    { id: 'side-by-side', label: 'Result / result' },
    { id: 'before-after', label: 'Before / after' }
  ];
</script>

<div class="viewer-controls panel" aria-label="Viewer controls">
  <div class="mode-group">
    {#each modes as mode}
      <button
        class:active={appState.compareMode === mode.id}
        aria-pressed={appState.compareMode === mode.id}
        onclick={() => (appState.compareMode = mode.id)}
      >{mode.label}</button>
    {/each}
  </div>
  {#if appState.compareMode === 'side-by-side'}
    <select bind:value={appState.compareResultId} aria-label="Comparison result">
      <option value={null}>Choose result</option>
      {#each appState.results as result}
        <option value={result.id}>{result.label}</option>
      {/each}
    </select>
  {/if}
  <button
    class:active={appState.showHistogram}
    aria-pressed={appState.showHistogram}
    onclick={() => (appState.showHistogram = !appState.showHistogram)}
  >Histogram</button>
</div>

<style>
  .viewer-controls { position: absolute; top: 10px; left: 50%; transform: translateX(-50%); z-index: 10; display: flex; gap: 6px; align-items: center; padding: 5px; background: color-mix(in srgb, var(--panel) 90%, transparent); backdrop-filter: blur(8px); }
  .mode-group { display: flex; gap: 2px; }
  button { border: 0; background: transparent; color: var(--text-dim); font-size: 10px; padding: 5px 8px; border-radius: 3px; }
  button.active { background: var(--accent-dim); color: var(--accent-bright); }
  select { max-width: 150px; font-size: 10px; padding: 4px; }
</style>
