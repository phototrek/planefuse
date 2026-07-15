<script lang="ts">
  import { api, type FileStatus } from '$lib/api';
  import { appState } from '$lib/stores.svelte';

  let {
    projectId,
    files,
    dropped = new Set<string>()
  }: { projectId: string; files: FileStatus[]; dropped?: Set<string> } = $props();

  const BADGE: Record<string, string> = {
    ok: 'ok',
    wrong_size: 'size',
    wrong_bit_depth: 'depth',
    not_rgb: 'rgb',
    unreadable: 'read',
    unsupported: 'type'
  };
</script>

<div class="strip" data-testid="filmstrip">
  {#each files as f (f.path)}
    <figure class="frame" class:bad={f.status !== 'ok'} class:dim={dropped.has(f.path)} title={f.message || f.name}>
      <img src={api.thumbUrl(projectId, f.path, appState.displayTonemap)} alt={f.name} loading="lazy" />
      <figcaption>
        <span class="nm mono">{f.name}</span>
        <span class="badge" class:ok={f.status === 'ok'}>{BADGE[f.status] ?? f.status}</span>
      </figcaption>
    </figure>
  {/each}
</div>

<style>
  .strip {
    display: flex;
    gap: 10px;
    overflow-x: auto;
    padding: 12px 2px;
  }
  .frame {
    margin: 0;
    flex: 0 0 auto;
    width: 132px;
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    overflow: hidden;
  }
  .frame.bad { border-color: var(--bad); }
  .frame.dim { opacity: 0.4; }
  .frame img {
    display: block;
    width: 132px;
    height: 100px;
    object-fit: cover;
    background: #000;
  }
  figcaption {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    padding: 5px 7px;
  }
  .nm { font-size: 11px; color: var(--text-dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .badge {
    font-size: 10px;
    font-family: var(--font-mono);
    padding: 1px 5px;
    border-radius: 4px;
    background: var(--bad);
    color: #240a0a;
    text-transform: uppercase;
  }
  .badge.ok { background: var(--accent-dim); color: var(--accent-bright); }
</style>
