<script lang="ts">
  import type { WorkspaceResult } from '$lib/stores.svelte';

  let { result }: { result: WorkspaceResult } = $props();
  let alignment = $derived(
    typeof result.provenance.alignment === 'object' && result.provenance.alignment
      ? result.provenance.alignment as Record<string, unknown>
      : null
  );
  let quality = $derived(
    alignment && typeof alignment.quality === 'object' && alignment.quality
      ? Object.entries(alignment.quality as Record<string, number>)
      : []
  );
  let low = $derived(quality.filter(([, score]) => Number(score) < 0.9));
</script>

{#if alignment && alignment.model !== 'none'}
  <details class="review" data-testid="alignment-review">
    <summary>
      Alignment · {String(alignment.model)}
      {#if low.length}<span class="warn">{low.length} low-quality link{low.length === 1 ? '' : 's'}</span>{:else}<span class="good">quality passed</span>{/if}
    </summary>
    <div class="details mono">
      <span>reference {String(alignment.reference)}</span>
      <span>recovered {JSON.stringify(alignment.recovered ?? [])}</span>
      <span>excluded {JSON.stringify(alignment.excluded ?? [])}</span>
      {#each low as [pair, score]}
        <span class="warn">pair {pair}: {Number(score).toFixed(3)}</span>
      {/each}
    </div>
  </details>
{/if}

<style>
  .review { min-width: 180px; font-size: 10px; }
  summary { cursor: pointer; display: flex; gap: 7px; }
  .details { position: absolute; z-index: 30; display: flex; flex-direction: column; gap: 3px; padding: 8px; border: 1px solid var(--line); background: var(--bg); }
  .warn { color: var(--warn, #e1a857); }
  .good { color: var(--good); }
</style>
