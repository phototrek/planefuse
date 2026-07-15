<script lang="ts">
  import { api } from '$lib/api';
  import { appState } from '$lib/stores.svelte';

  let override = $state<string | null>(null);
  let busy = $state(false);
  let error = $state('');
  let done = $state(false);
  let name = $derived(override ?? appState.project?.name ?? '');

  async function save() {
    if (!appState.project) return;
    busy = true;
    error = '';
    done = false;
    try {
      const result = await api.saveProject(appState.project.id, { name, saved: true });
      appState.project.name = result.name;
      override = null;
      done = true;
      setTimeout(() => (done = false), 2000);
    } catch (cause) {
      error = (cause as Error).message;
    } finally {
      busy = false;
    }
  }
</script>

<div class="project-controls">
  <input
    class="project-name"
    type="text"
    value={name}
    oninput={(event) => {
      override = event.currentTarget.value;
      if (appState.project) appState.project.name = event.currentTarget.value;
    }}
    data-testid="ws-project-name"
    placeholder="Untitled"
    aria-label="Project name"
  />
  <button class="ghost" data-testid="ws-save" disabled={busy || !appState.project} onclick={save}>
    {busy ? 'Saving…' : done ? '✓ Saved' : 'Save'}
  </button>
  {#if error}<span class="err mono">{error}</span>{/if}
</div>

<style>
  .project-controls { display: flex; align-items: center; gap: 8px; min-width: 0; }
  .project-name { font-size: 13px; font-weight: 600; width: 160px; padding: 5px 8px; background: transparent; border: 1px solid transparent; }
  .project-name:hover { border-color: var(--line-strong); }
  .project-name:focus { background: var(--bg); border-color: var(--accent); }
  .err { color: var(--bad); font-size: 10px; }
</style>
