<script lang="ts">
  import type { WorkspaceResult } from '$lib/stores.svelte';

  let { result }: { result: WorkspaceResult } = $props();
  let selected = $derived(Array.isArray(result.provenance.selected) ? result.provenance.selected as number[] : []);
  let excluded = $derived(Array.isArray(result.provenance.excluded) ? result.provenance.excluded as number[] : []);
  let sources = $derived(Array.isArray(result.provenance.sources) ? result.provenance.sources : []);
</script>

{#if selected.length > 0}
  <details class="review" data-testid="selection-review">
    <summary>Selection · {selected.length}/{sources.length || selected.length} kept {#if excluded.length}<span>{excluded.length} redundant</span>{/if}</summary>
    <div class="strip" aria-label="Frame selection coverage">
      {#each sources as _source, index}
        <span class:excluded={excluded.includes(index)} title={`Frame ${index + 1}`}>{index + 1}</span>
      {/each}
    </div>
  </details>
{/if}

<style>
  .review { min-width: 150px; font-size: 10px; }
  summary { cursor: pointer; display: flex; gap: 7px; }
  summary span { color: var(--text-faint); }
  .strip { position: absolute; z-index: 30; display: flex; gap: 2px; max-width: 420px; overflow-x: auto; padding: 8px; background: var(--bg); border: 1px solid var(--line); }
  .strip span { min-width: 20px; padding: 3px; text-align: center; color: var(--good); background: color-mix(in srgb, var(--good) 15%, var(--panel)); }
  .strip span.excluded { opacity: .35; color: var(--text-faint); text-decoration: line-through; }
</style>
