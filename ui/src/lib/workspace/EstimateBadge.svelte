<script lang="ts">
  import { api, type WorkEstimate } from '$lib/api';
  import { appState } from '$lib/stores.svelte';

  let { method, align }: { method: string; align: boolean } = $props();
  let estimate = $state<WorkEstimate | null>(null);
  let requestVersion = 0;

  $effect(() => {
    const report = appState.scanReport;
    const frames = report?.files.length ?? 0;
    const width = report?.width ?? 0;
    const height = report?.height ?? 0;
    const version = ++requestVersion;
    if (!report?.ok || frames < 2 || !width || !height) {
      estimate = null;
      return;
    }
    void api.estimate({ width, height, frames, method, align }).then((value) => {
      if (version === requestVersion) estimate = value;
    }).catch(() => {
      if (version === requestVersion) estimate = null;
    });
  });

  function duration(seconds: number): string {
    if (seconds < 60) return `${Math.max(1, Math.round(seconds))} s`;
    return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  }
</script>

{#if estimate}
  <span class="estimate mono" title={estimate.basis} data-testid="work-estimate">
    ≈ {duration(estimate.seconds)} · ≈ {(estimate.memory_bytes / 1_073_741_824).toFixed(1)} GB
  </span>
{/if}

<style>
  .estimate { color: var(--text-faint); font-size: 10px; white-space: nowrap; }
</style>
