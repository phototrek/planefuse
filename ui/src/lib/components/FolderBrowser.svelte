<script lang="ts">
  import { api, type FsList } from '$lib/api';

  let {
    value = $bindable(''),
    placeholder = 'C:\\path\\to\\folder',
    inputTestid = ''
  }: { value?: string; placeholder?: string; inputTestid?: string } = $props();

  let listing = $state<FsList | null>(null);
  let err = $state('');
  let open = $state(false);

  async function load() {
    err = '';
    listing = null;
    if (!value) return;
    try {
      listing = await api.fsList(value);
    } catch (e) {
      err = (e as Error).message;
    }
  }

  function up() {
    const p = value.replace(/[\\/]+$/, '');
    const i = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'));
    if (i > 0) {
      value = p.slice(0, i);
      load();
    }
  }

  function into(p: string) {
    value = p;
    load();
  }

  let dirs = $derived(listing?.entries.filter((e) => e.is_dir) ?? []);
</script>

<div class="fb">
  <div class="row">
    <input
      type="text"
      bind:value
      {placeholder}
      data-testid={inputTestid}
      onkeydown={(e) => e.key === 'Enter' && load()}
    />
    <button class="ghost" onclick={() => { open = !open; load(); }} title="Browse subfolders">
      {open ? 'Hide' : 'Browse'}
    </button>
  </div>

  {#if open}
    <div class="tree panel">
      <button class="crumb ghost" onclick={up}>↑ up</button>
      {#if err}
        <p class="bad mono">{err}</p>
      {:else if listing}
        <p class="eyebrow count">{listing.image_count} images here · {dirs.length} folders</p>
        <ul>
          {#each dirs as d (d.path)}
            <li><button class="ghost" onclick={() => into(d.path)}>📁 {d.name}</button></li>
          {/each}
        </ul>
      {/if}
    </div>
  {/if}
</div>

<style>
  .fb { display: flex; flex-direction: column; gap: 8px; }
  .row { display: flex; gap: 8px; }
  .row input { flex: 1; font-family: var(--font-mono); font-size: 12px; }
  .tree { padding: 10px; max-height: 240px; overflow: auto; }
  .crumb { font-family: var(--font-mono); font-size: 12px; margin-bottom: 6px; }
  .count { margin: 4px 2px 8px; }
  ul { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 2px; }
  li button { width: 100%; text-align: left; font-size: 13px; }
  .bad { color: var(--bad); font-size: 12px; }
</style>
