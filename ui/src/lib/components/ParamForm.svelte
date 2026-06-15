<script lang="ts">
  import type { ParamSpec } from '$lib/api';

  let {
    specs,
    values = $bindable({})
  }: { specs: ParamSpec[]; values?: Record<string, unknown> } = $props();
</script>

<div class="pf">
  {#each specs as p (p.name)}
    <label class="field" title={p.tooltip ?? ''}>
      <span class="lbl">{p.label}</span>

      {#if p.type === 'bool'}
        <input type="checkbox" checked={values[p.name] as boolean} oninput={(e) => (values[p.name] = e.currentTarget.checked)} />
      {:else if p.type === 'choice' && p.choices}
        <select value={values[p.name]} onchange={(e) => (values[p.name] = e.currentTarget.value)}>
          {#each p.choices as c (c)}<option value={c}>{c}</option>{/each}
        </select>
      {:else}
        <input
          type="number"
          value={values[p.name] as number}
          min={p.min ?? undefined}
          max={p.max ?? undefined}
          step={p.type === 'int' ? 1 : 'any'}
          oninput={(e) => (values[p.name] = e.currentTarget.valueAsNumber)}
        />
      {/if}
    </label>
  {/each}
</div>

<style>
  .pf { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 18px; }
  .field { display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 10px; }
  .lbl { font-size: 13px; color: var(--text-dim); }
  .field input[type='number'], .field select { width: 120px; }
  .field input[type='checkbox'] { width: 16px; height: 16px; accent-color: var(--accent); justify-self: end; }
</style>
